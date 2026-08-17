from __future__ import annotations

import io
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.services.face_quality import QualityLimits, assess
from app.services.face_service import EMBEDDING_MODEL_VERSION, extract_face_embeddings

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
FACE_UPLOAD_DIR = BASE_DIR / "data" / "face_uploads"


# ── Enrollment quality gate ─────────────────────────────────────────────────
# The checks themselves now live in services/face_quality.py, shared with the
# runtime CCTV path. Before that split the two gates disagreed: enrollment
# checked size + detector score + landmark asymmetry + face-crop blur, while the
# camera pipeline checked only a pixel width and a WHOLE-FRAME blur value. A
# face that enrollment would have rejected outright could still name an employee
# and write an attendance row — the strict gate was on the side where mistakes
# are cheap, and absent on the side where they touch payroll.
#
# Enrollment keeps its own limits: an enrolled vector is reused for
# every future match, so a marginal enrolment photo degrades that employee
# permanently, whereas a marginal runtime frame degrades one observation.
_QUALITY_CHECK = os.getenv("FACE_ENROLL_QUALITY_CHECK", "1").lower() in {"1", "true", "yes"}

ENROLL_LIMITS = QualityLimits(
    min_face_px=float(os.getenv("FACE_ENROLL_MIN_FACE_PX", "90")),
    min_det_score=float(os.getenv("FACE_ENROLL_MIN_DET_SCORE", "0.62")),
    # Do not reject side/profile enrollment images. The gallery must contain the
    # poses that CCTV will actually capture. Size, detector confidence, and blur
    # checks remain active so unusable images are still rejected.
    max_yaw_deg=float(os.getenv("FACE_ENROLL_MAX_YAW_DEG", "180")),
    max_pitch_deg=float(os.getenv("FACE_ENROLL_MAX_PITCH_DEG", "180")),
    max_landmark_asym=float(os.getenv("FACE_ENROLL_MAX_FACE_ASYM", "1.0")),
    min_blur_var=float(os.getenv("FACE_ENROLL_MIN_BLUR_VAR", "25.0")),
    good_face_px=160.0,
    good_blur_var=200.0,
)


async def process_face_uploads(files: list[UploadFile]) -> list[dict]:
    """Enrol every USABLE photo; skip the rest instead of failing the whole batch.

    An employee's embeddings are additive (the matcher keeps them all and scores
    against the BEST one), so partial success is strictly better than none. Only
    when NO photo is usable is an error raised, listing every reason.
    """
    prepared_images: list[dict] = []
    skipped: list[str] = []

    for file in files:
        name = file.filename or "image.jpg"
        image_bytes = await file.read()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            skipped.append(f"{name}: not a readable image")
            continue

        faces = extract_face_embeddings(pil_image)
        if not faces:
            skipped.append(
                f"{name}: no detectable face found (the face must still be visible)"
            )
            continue
        if len(faces) > 1:
            skipped.append(f"{name}: multiple faces detected — use one face per image")
            continue

        face = faces[0]
        quality = assess(face, np.asarray(pil_image), limits=ENROLL_LIMITS)
        if _QUALITY_CHECK and not quality.ok:
            skipped.append(f"{name}: {quality.detail}")
            continue

        prepared_images.append(
            {
                "filename": name,
                "bytes": image_bytes,
                "embedding": face["embedding"],
                "aligned": bool(face.get("aligned", True)),
                "quality": quality,
                "skipped": list(skipped),   # carried so the route can report them
            }
        )

    if not prepared_images:
        raise HTTPException(
            status_code=400,
            detail="No usable photo in this upload. " + " | ".join(skipped),
        )

    # Attach the final skip list to the first item so callers can surface it.
    prepared_images[0]["skipped"] = skipped
    return prepared_images


# ---------------------------------------------------------------------------
# Gallery persistence
# ---------------------------------------------------------------------------
def enroll_embeddings(
    employee_id: int,
    observations: Sequence[dict],
    *,
    source: str = "upload",
    detector: Optional[str] = None,
    replace: bool = False,
) -> int:
    """Add embeddings to an employee's gallery, with provenance. Returns count added.

    Each observation is a dict with at least ``embedding``; optionally
    ``camera_id``, ``aligned`` and the quality fields.

    Two things are written, deliberately:

      * one ``employee_face_embeddings`` row per vector, carrying the model
        version, detector, alignment flag, source camera and frame quality. This
        is what makes the gallery auditable — previously an employee's entire
        enrolment was one opaque blob and there was no way to tell a
        studio-photo vector from a CCTV one, or a buffalo_l vector from whatever
        replaced it;

      * a rebuilt ``employees.embedding`` (N, 512) stack, which stays the hot
        path read by the in-memory matcher cache. Keeping it in sync means no
        existing query or route changes.

    ``replace=True`` retires the current gallery first (soft-delete, so the audit
    trail of what the system believed last month survives).
    """
    from app.db.session import SessionLocal
    from app.models.employee import Employee
    from app.models.employee_face import EmployeeFaceEmbedding
    from app.services.embedding_cache import embedding_to_blob, invalidate_embedding_cache

    vectors: list[np.ndarray] = []
    rows: list[dict] = []
    for item in observations:
        try:
            vector = np.asarray(item["embedding"], dtype=np.float32).ravel()
        except (KeyError, TypeError, ValueError):
            continue
        norm = float(np.linalg.norm(vector))
        if vector.size == 0 or norm <= 0:
            continue
        unit = vector / norm
        vectors.append(unit)
        quality = item.get("quality")
        rows.append(
            {
                "embedding": unit.tobytes(),
                "camera_id": item.get("camera_id"),
                "aligned": bool(item.get("aligned", True)),
                "quality_score": _num(item.get("quality_score"), getattr(quality, "score", None)),
                "face_px": _num(item.get("face_px"), getattr(quality, "face_px", None)),
                "yaw": _num(item.get("yaw"), getattr(quality, "yaw", None)),
                "pitch": _num(item.get("pitch"), getattr(quality, "pitch", None)),
                "blur_var": _num(item.get("blur_var"), getattr(quality, "blur_var", None)),
            }
        )

    if not vectors:
        return 0

    if detector is None:
        from app.core.config import get_settings

        detector = (get_settings().face_detector or "insightface").lower()

    with SessionLocal() as db:
        employee = db.query(Employee).filter(Employee.id == int(employee_id)).first()
        if employee is None:
            raise ValueError(f"employee {employee_id} not found")

        if replace:
            (
                db.query(EmployeeFaceEmbedding)
                .filter(
                    EmployeeFaceEmbedding.employee_id == int(employee_id),
                    EmployeeFaceEmbedding.active.is_(True),
                )
                .update({"active": False}, synchronize_session=False)
            )

        for row in rows:
            db.add(
                EmployeeFaceEmbedding(
                    employee_id=int(employee_id),
                    embedding=row["embedding"],
                    model_version=EMBEDDING_MODEL_VERSION,
                    detector=detector,
                    aligned=row["aligned"],
                    source=source,
                    camera_id=row["camera_id"],
                    quality_score=row["quality_score"],
                    face_px=row["face_px"],
                    yaw=row["yaw"],
                    pitch=row["pitch"],
                    blur_var=row["blur_var"],
                    active=True,
                )
            )
        db.flush()

        _rebuild_stack(db, employee, EmployeeFaceEmbedding, embedding_to_blob)
        db.commit()

    invalidate_embedding_cache()
    logger.info(
        "ENROLL employee=%s added=%d source=%s detector=%s model=%s replace=%s",
        employee_id, len(rows), source, detector, EMBEDDING_MODEL_VERSION, replace,
    )
    return len(rows)


def _num(primary, fallback):
    value = primary if primary is not None else fallback
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _rebuild_stack(db, employee, model_cls, embedding_to_blob) -> None:
    """Recompute employees.embedding from the ACTIVE, model-matched rows.

    Rows whose ``model_version`` differs from the running model are excluded.
    Comparing vectors produced by different recognition models is meaningless —
    they occupy different spaces — and silently mixing them is precisely the
    failure this table was added to make visible. A model change therefore
    degrades to "these embeddings are ignored until re-enrolled", which is
    correct and loud, instead of "everyone's scores mysteriously dropped".
    """
    rows = (
        db.query(model_cls)
        .filter(
            model_cls.employee_id == employee.id,
            model_cls.active.is_(True),
            model_cls.model_version == EMBEDDING_MODEL_VERSION,
        )
        .order_by(model_cls.id.asc())
        .all()
    )

    vectors = []
    for row in rows:
        try:
            vector = np.frombuffer(row.embedding, dtype=np.float32)
        except Exception:
            continue
        if vector.size:
            vectors.append(vector)

    if not vectors:
        employee.embedding = None
        employee.sample_count = 0
        return

    stack = np.stack(vectors).astype(np.float32)
    employee.embedding = embedding_to_blob(stack)
    employee.sample_count = int(stack.shape[0])


def gallery_summary(employee_id: int) -> dict:
    """What an employee's gallery is made of — used by the enrolment UI.

    Surfaces the two facts that predict whether this employee will be recognised
    on a fixed ceiling camera: how many embeddings they have, and whether ANY of
    them were captured from the cameras that will have to recognise them.
    """
    from app.db.session import SessionLocal
    from app.models.employee_face import EmployeeFaceEmbedding

    with SessionLocal() as db:
        rows = (
            db.query(EmployeeFaceEmbedding)
            .filter(
                EmployeeFaceEmbedding.employee_id == int(employee_id),
                EmployeeFaceEmbedding.active.is_(True),
            )
            .all()
        )

    cameras = sorted({row.camera_id for row in rows if row.camera_id})
    stale = [row for row in rows if row.model_version != EMBEDDING_MODEL_VERSION]
    unaligned = [row for row in rows if not row.aligned]
    return {
        "employee_id": int(employee_id),
        "total": len(rows),
        "from_upload": sum(1 for row in rows if row.source == "upload"),
        "from_cctv": sum(1 for row in rows if row.source == "cctv"),
        "cameras": cameras,
        "has_camera_enrolment": bool(cameras),
        "stale_model_version": len(stale),
        "unaligned": len(unaligned),
        "model_version": EMBEDDING_MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# Photo files
# ---------------------------------------------------------------------------
def save_employee_photo(employee_id: int, image_bytes: bytes, filename: str) -> str:
    employee_dir = FACE_UPLOAD_DIR / str(employee_id)
    employee_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".jpg"
    image_path = employee_dir / f"photo{suffix}"
    image_path.write_bytes(image_bytes)
    return str(image_path)


def delete_employee_photos(employee_id: int) -> None:
    """Remove all stored face photos for an employee (used when clearing enrollment)."""
    employee_dir = FACE_UPLOAD_DIR / str(employee_id)
    if employee_dir.exists():
        shutil.rmtree(employee_dir, ignore_errors=True)
