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
# System-wide recognition coverage
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------
# `gallery_summary` answers "how is THIS employee enrolled?". Nothing answered
# the question that actually predicts whether the cameras work at all: *how many
# people can the matcher recognise?*
#
# Measured on this deployment when the report was written: 50 employees, 5 of
# them Active, and the matcher's gallery held THREE. The other 47 were
# structurally unrecognisable — every one of them arrived at the entrance as an
# "unknown face", scoring an average of 0.188 against a 0.45 threshold. That is
# not a tuning problem and no threshold change can fix it, but from the outside
# it is indistinguishable from "recognition is broken", which is exactly how it
# was reported.
#
# The exclusions are individually silent by design: `embedding_cache._load_from_db`
# just doesn't SELECT those employees. An employee enrolled entirely under a
# previous recognition model (an ArcFace -> AdaFace switch leaves their rows
# behind) simply stops being recognised, with no error anywhere. This function
# reproduces that query's predicates and reports what each one dropped, so the
# silence becomes a number somebody can act on.
#
# The reasons deliberately mirror `_load_from_db`'s filters one-for-one. If that
# query changes, this must change with it — see test_recognition_coverage.
def recognition_coverage() -> dict:
    """Who the matcher can recognise, and why everyone else is excluded.

    Pure read. Safe to call from an admin endpoint or a health check.
    """
    from app.db.session import SessionLocal
    from app.models.employee import Employee, EmploymentStatus
    from app.models.employee_face import EmployeeFaceEmbedding

    excluded: list[dict] = []
    in_gallery: list[dict] = []
    vectors = 0

    with SessionLocal() as db:
        employees = (
            db.query(Employee)
            .filter(Employee.employment_status == EmploymentStatus.ACTIVE.value)
            .order_by(Employee.id)
            .all()
        )
        for emp in employees:
            rows = (
                db.query(EmployeeFaceEmbedding)
                .filter(
                    EmployeeFaceEmbedding.employee_id == emp.id,
                    EmployeeFaceEmbedding.active.is_(True),
                )
                .all()
            )
            matched = [r for r in rows if r.model_version == EMBEDDING_MODEL_VERSION]
            entry = {
                "employee_id": emp.id,
                "employee_code": emp.employee_code,
                "name": emp.full_name,
                "active_embeddings": len(rows),
                "model_matched": len(matched),
                # Lets the UI show a face without a request per employee.
                "has_photo": resolve_employee_photo(emp.id) is not None,
            }

            # Same order as the filters in embedding_cache._load_from_db, so the
            # reason names the FIRST predicate that dropped them.
            if not rows:
                entry["reason"] = "no_enrolment"
                entry["detail"] = "No face has ever been enrolled for this employee."
            elif not matched:
                entry["reason"] = "stale_model"
                entry["detail"] = (
                    f"Enrolled under a previous recognition model "
                    f"({', '.join(sorted({r.model_version for r in rows}))}); the system "
                    f"now runs {EMBEDDING_MODEL_VERSION}. Vectors from different models "
                    "are not comparable, so these are ignored. Re-enrol to restore."
                )
            elif emp.embedding is None:
                # Matched rows exist but the aggregated stack was never rebuilt,
                # so the JOIN finds them while `Employee.embedding IS NOT NULL`
                # drops them. Recoverable without new photos.
                entry["reason"] = "no_stack"
                entry["detail"] = (
                    "Model-matched embeddings exist but employees.embedding was "
                    "never rebuilt from them. Re-run enrolment to rebuild the stack."
                )
            else:
                vectors += len(matched)
                in_gallery.append(entry)
                continue
            excluded.append(entry)

    reasons: dict[str, int] = {}
    for item in excluded:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1

    active = len(in_gallery) + len(excluded)
    return {
        "model_version": EMBEDDING_MODEL_VERSION,
        "active_employees": active,
        "in_gallery": len(in_gallery),
        "gallery_vectors": vectors,
        "coverage_pct": round(100.0 * len(in_gallery) / active, 1) if active else 0.0,
        "reasons": reasons,
        "recognisable": in_gallery,
        "excluded": excluded,
    }


# ---------------------------------------------------------------------------
# Photo files
# ---------------------------------------------------------------------------
def resolve_employee_photo(employee_id: int) -> Optional[Path]:
    """Absolute path of an employee's stored enrolment photo, or None.

    Resolved from FACE_UPLOAD_DIR and the employee id, deliberately NOT from
    `employees.photo_path`. That column stores whatever absolute path the
    machine that ran the enrolment happened to have -- every row in this
    deployment reads
    `C:\sakshi folder\application\Resume analyzer\...\data\face_uploads\<id>\photo.jpeg`,
    a developer workstation path that does not exist on the server, where the
    app lives under C:\SoftwizApp. Serving the column directly would work in
    dev and 404 for every employee in production.

    Containment is enforced the same way as `unknown_faces.resolve_crop`: the id
    is coerced to an int and the resolved path must sit inside FACE_UPLOAD_DIR,
    so nothing reachable from a request can escape the data directory.

    NOTE on what is (and is not) here: enrolment stores ONE image per employee.
    `register_employee_face` accepts up to 10 photos and keeps an embedding for
    every one of them, but only calls `save_employee_photo` for
    `prepared_images[0]`, and that writes a fixed `photo<ext>` filename. So an
    employee with 5 gallery vectors still has a single viewable photo, and
    re-enrolling overwrites it. Callers should present the photo as "the cover
    image", never as "the gallery".
    """
    try:
        employee_dir = (FACE_UPLOAD_DIR / str(int(employee_id))).resolve()
        employee_dir.relative_to(FACE_UPLOAD_DIR.resolve())
    except (ValueError, OSError, TypeError):
        logger.warning("rejected face-photo path for employee %r", employee_id)
        return None
    if not employee_dir.is_dir():
        return None
    # `save_employee_photo` keeps the uploader's extension, so the file may be
    # photo.jpg / .jpeg / .png. Newest wins: re-enrolment can leave an older
    # extension behind (employee 5 has both photo.jpg and photo.jpeg), and the
    # stale one is not what the current gallery was built from.
    candidates = sorted(
        (p for p in employee_dir.glob("photo.*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


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
