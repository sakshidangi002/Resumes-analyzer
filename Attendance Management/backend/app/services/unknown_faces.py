"""Capture, cluster and resolve faces the cameras could not name.

THE PROBLEM THIS SOLVES
-----------------------
Camera positions are fixed, so the pixel quality that reaches ArcFace is fixed
too. Once thresholds are calibrated, the remaining recognition failures are not
evenly distributed — they cluster on the handful of employees whose enrolled
photos look nothing like what their camera actually captures. Global tuning
cannot fix that, and without a record of unmatched faces there is no way to even
learn who those people are.

THE LOOP
--------
    unmatched face (that PASSED the quality gate)
        -> stored here with crop, embedding, quality and near-miss score
        -> clustered by appearance: "this same stranger, 14 times today"
        -> HR assigns a cluster to an employee
        -> those embeddings are enrolled AS THAT EMPLOYEE, tagged with the
           camera they came from
        -> that employee now matches from that camera's viewpoint

The last step is the important one. It is viewpoint-matched enrolment obtained
for free from live traffic, which is the single highest-leverage accuracy lever
available when the optics cannot be improved.

WHAT IS DELIBERATELY NOT STORED
-------------------------------
Faces that FAILED the quality gate. A 14px motion-blurred profile that went
unrecognised is not evidence of a recognition failure — it is evidence of a bad
frame. Storing those would bury the actionable rows under thousands of unusable
ones and make the queue too noisy to review, which is the normal way this
feature dies in production.

Sampling is also rate-limited per camera: a person standing in a doorway
produces one usable face every analysis tick, and one row per sighting is enough
to reconstruct "this person was here", while 400 rows is just disk.

PRIVACY
-------
Rows hold a biometric embedding (encrypted at rest) and a face crop on disk,
under the same access-controlled data/ tree as attendance snapshots. Call
``purge_older_than`` from the scheduled cleanup — see attendance_closeout.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.datetime_utils import get_ist_now

logger = logging.getLogger(__name__)

CROP_ROOT = Path(__file__).resolve().parents[2] / "data" / "unknown_faces"

ENABLED = os.getenv("UNKNOWN_FACE_CAPTURE", "true").lower() in {"1", "true", "yes"}
RETENTION_DAYS = int(os.getenv("UNKNOWN_FACE_RETENTION_DAYS", "30"))
_JPEG_QUALITY = int(os.getenv("UNKNOWN_FACE_JPEG_QUALITY", "85"))

# Minimum seconds between stored rows on one camera. See "sampling" above.
_SAMPLE_INTERVAL = float(os.getenv("UNKNOWN_FACE_SAMPLE_SEC", "8.0"))

# Hard cap on pending rows. A camera pointed at a busy corridor would otherwise
# fill the table with strangers and turn the review queue into a chore nobody
# does. When the cap is hit, capture stops and says so — loudly, once — rather
# than silently degrading.
_MAX_PENDING = int(os.getenv("UNKNOWN_FACE_MAX_PENDING", "2000"))

# Two faces belong to the same cluster when their cosine similarity is at least
# this. Higher than the fusion outlier bound (0.35) because the job here is
# different: fusion asks "is this the same track?", clustering asks "is this the
# same PERSON across hours and lighting?" and a wrong merge silently enrols one
# person's face under another's name.
CLUSTER_MIN_COS = float(os.getenv("UNKNOWN_FACE_CLUSTER_COS", "0.55"))

# Bars for "does this cluster look like the employee it is being filed under?"
# Derived from measured gallery coherence: a clean single-person gallery sits
# around 0.50 (Rakhi 0.502, Sakshi 0.541), a contaminated one at 0.248.
_ASSIGN_AGREE_OK = float(os.getenv("UNKNOWN_FACE_ASSIGN_AGREE_OK", "0.35"))
_ASSIGN_AGREE_WARN = float(os.getenv("UNKNOWN_FACE_ASSIGN_AGREE_WARN", "0.22"))

_lock = threading.Lock()
_last_capture: dict[str, float] = {}
_pending_full_logged = False


def _safe_component(value: object) -> str:
    text = str(value)
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:40] or "unknown"


def _embedding_to_bytes(embedding) -> Optional[bytes]:
    try:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            return None
        return vector.tobytes()
    except Exception:
        return None


def _bytes_to_embedding(blob: bytes) -> Optional[np.ndarray]:
    try:
        vector = np.frombuffer(blob, dtype=np.float32)
        return vector if vector.size else None
    except Exception:
        return None


def _should_sample(camera_id: str) -> bool:
    now = time.time()
    with _lock:
        last = _last_capture.get(camera_id, 0.0)
        if now - last < _SAMPLE_INTERVAL:
            return False
        _last_capture[camera_id] = now
        return True


def _save_crop(frame_bgr, box, camera_id: str, when: datetime) -> Optional[str]:
    """Write the face crop; return a path RELATIVE to CROP_ROOT, or None."""
    if frame_bgr is None or box is None or len(box) < 4:
        return None
    try:
        import cv2

        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        pad_x = int(abs(x2 - x1) * 0.35)
        pad_y = int(abs(y2 - y1) * 0.35)
        crop = frame_bgr[
            max(0, min(y1, y2) - pad_y):min(height, max(y1, y2) + pad_y),
            max(0, min(x1, x2) - pad_x):min(width, max(x1, x2) + pad_x),
        ]
        if crop.size == 0:
            return None
        relative = Path(when.strftime("%Y-%m-%d")) / (
            f"{_safe_component(camera_id)}_{when.strftime('%H%M%S_%f')[:-3]}.jpg"
        )
        destination = CROP_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), crop, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]):
            return None
        return relative.as_posix()
    except Exception:
        logger.debug("unknown-face crop write failed", exc_info=True)
        return None


def resolve_crop(relative_path: str) -> Optional[Path]:
    """Absolute path for a stored crop, or None if it escapes the root.

    The stored value round-trips through the database and is fed to a download
    route, so containment is checked here rather than at each call site.
    """
    if not relative_path:
        return None
    try:
        candidate = (CROP_ROOT / relative_path).resolve()
        candidate.relative_to(CROP_ROOT.resolve())
    except (ValueError, OSError):
        logger.warning("rejected unknown-face path outside the root: %r", relative_path)
        return None
    return candidate if candidate.is_file() else None


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def capture(
    *,
    camera_id: str,
    embedding,
    quality,
    frame_bgr=None,
    box=None,
    best_score: Optional[float] = None,
    best_margin: Optional[float] = None,
    best_employee_id: Optional[int] = None,
    when: Optional[datetime] = None,
) -> Optional[int]:
    """Record one unmatched-but-usable face. Returns the row id, or None.

    NEVER raises and never blocks meaningfully: this runs on a camera thread and
    a review-queue feature must not be able to cost the system a recognition
    pass, let alone crash a camera.

    ``quality`` is the FaceQuality result for this observation. Only faces that
    passed the camera's gate reach here — see the module docstring.
    """
    if not ENABLED:
        return None
    if not _should_sample(str(camera_id)):
        return None

    blob = _embedding_to_bytes(embedding)
    if blob is None:
        return None

    when = when or get_ist_now()

    try:
        from app.db.session import SessionLocal
        from app.models.unknown_face import STATUS_PENDING, UnknownFace

        with SessionLocal() as db:
            global _pending_full_logged
            pending = (
                db.query(UnknownFace)
                .filter(UnknownFace.status == STATUS_PENDING)
                .count()
            )
            if pending >= _MAX_PENDING:
                if not _pending_full_logged:
                    _pending_full_logged = True
                    logger.warning(
                        "UNKNOWN-FACE queue is full (%d pending >= %d) — capture "
                        "paused. Review or purge the queue to resume.",
                        pending, _MAX_PENDING,
                    )
                return None
            _pending_full_logged = False

            crop_path = _save_crop(frame_bgr, box, str(camera_id), when)

            row = UnknownFace(
                camera_id=str(camera_id),
                captured_at=when,
                embedding=blob,
                crop_path=crop_path,
                quality_score=getattr(quality, "score", None),
                face_px=getattr(quality, "face_px", None),
                yaw=getattr(quality, "yaw", None),
                pitch=getattr(quality, "pitch", None),
                blur_var=getattr(quality, "blur_var", None),
                best_score=best_score,
                best_margin=best_margin,
                best_employee_id=best_employee_id,
                status=STATUS_PENDING,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            logger.info(
                "UNKNOWN-FACE captured id=%s camera=%s q=%.2f face_px=%.0f "
                "near_miss=%s@%.3f",
                row.id, camera_id, float(getattr(quality, "score", 0.0) or 0.0),
                float(getattr(quality, "face_px", 0.0) or 0.0),
                best_employee_id, float(best_score or 0.0),
            )
            return int(row.id)
    except Exception:
        logger.exception("UNKNOWN-FACE capture failed camera=%s", camera_id)
        return None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def recluster(min_cos: float = CLUSTER_MIN_COS, limit: int = 5000) -> dict:
    """Group pending unknown faces by appearance. Returns a summary dict.

    Greedy single-link agglomeration against cluster CENTROIDS rather than
    against any single member. Single-link on raw members chains badly: A~B and
    B~C merges A with C even when A and C are plainly different people, which in
    this workflow means enrolling a stranger's face under an employee's name.

    Cheap by construction — a few thousand 512-d vectors is a millisecond of
    numpy — so it is safe to run on demand from the admin UI rather than needing
    a scheduled job.
    """
    from app.db.session import SessionLocal
    from app.models.unknown_face import STATUS_PENDING, UnknownFace

    with SessionLocal() as db:
        rows = (
            db.query(UnknownFace)
            .filter(UnknownFace.status == STATUS_PENDING)
            .order_by(UnknownFace.captured_at.asc())
            .limit(limit)
            .all()
        )
        if not rows:
            return {"rows": 0, "clusters": 0}

        vectors: list[np.ndarray] = []
        usable: list = []
        for row in rows:
            vector = _bytes_to_embedding(row.embedding)
            if vector is None:
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            vectors.append(vector / norm)
            usable.append(row)

        if not usable:
            return {"rows": 0, "clusters": 0}

        centroids: list[np.ndarray] = []
        counts: list[int] = []
        assignment: list[int] = []

        for vector in vectors:
            best_idx, best_sim = None, min_cos
            for idx, centroid in enumerate(centroids):
                sim = float(np.dot(vector, centroid))
                if sim >= best_sim:
                    best_idx, best_sim = idx, sim
            if best_idx is None:
                centroids.append(vector.copy())
                counts.append(1)
                assignment.append(len(centroids) - 1)
            else:
                # Running mean, re-normalised so later comparisons stay cosine.
                n = counts[best_idx]
                merged = (centroids[best_idx] * n + vector) / (n + 1)
                norm = float(np.linalg.norm(merged))
                centroids[best_idx] = merged / norm if norm > 0 else merged
                counts[best_idx] = n + 1
                assignment.append(best_idx)

        for row, cluster_idx in zip(usable, assignment):
            row.cluster_id = int(cluster_idx)
        db.commit()

        logger.info(
            "UNKNOWN-FACE reclustered rows=%d clusters=%d (min_cos=%.2f)",
            len(usable), len(centroids), min_cos,
        )
        return {
            "rows": len(usable),
            "clusters": len(centroids),
            "sizes": sorted(counts, reverse=True),
        }


def list_clusters(limit: int = 50) -> list[dict]:
    """Pending clusters, largest first — the review queue.

    Largest first is the right order: a cluster of 40 sightings is an employee
    the cameras are systematically failing on, which is worth an HR minute. A
    cluster of 1 is probably a visitor.
    """
    from sqlalchemy import func

    from app.db.session import SessionLocal
    from app.models.unknown_face import STATUS_PENDING, UnknownFace

    with SessionLocal() as db:
        grouped = (
            db.query(
                UnknownFace.cluster_id,
                func.count(UnknownFace.id).label("size"),
                func.min(UnknownFace.captured_at).label("first_seen"),
                func.max(UnknownFace.captured_at).label("last_seen"),
                func.max(UnknownFace.quality_score).label("best_quality"),
                func.max(UnknownFace.best_score).label("best_score"),
            )
            .filter(
                UnknownFace.status == STATUS_PENDING,
                UnknownFace.cluster_id.isnot(None),
            )
            .group_by(UnknownFace.cluster_id)
            .order_by(func.count(UnknownFace.id).desc())
            .limit(limit)
            .all()
        )

        clusters = []
        for entry in grouped:
            sample = (
                db.query(UnknownFace)
                .filter(
                    UnknownFace.cluster_id == entry.cluster_id,
                    UnknownFace.status == STATUS_PENDING,
                )
                .order_by(UnknownFace.quality_score.desc().nullslast())
                .first()
            )
            cameras = [
                camera_id for (camera_id,) in db.query(UnknownFace.camera_id)
                .filter(
                    UnknownFace.cluster_id == entry.cluster_id,
                    UnknownFace.status == STATUS_PENDING,
                )
                .distinct()
                .all()
            ]
            clusters.append(
                {
                    "cluster_id": int(entry.cluster_id),
                    "size": int(entry.size),
                    "first_seen": entry.first_seen.isoformat() if entry.first_seen else None,
                    "last_seen": entry.last_seen.isoformat() if entry.last_seen else None,
                    "best_quality": float(entry.best_quality or 0.0),
                    "best_near_miss_score": float(entry.best_score or 0.0),
                    "cameras": cameras,
                    "sample_face_id": int(sample.id) if sample else None,
                    "sample_crop": sample.crop_path if sample else None,
                    "near_miss_employee_id": (
                        int(sample.best_employee_id)
                        if sample is not None and sample.best_employee_id is not None
                        else None
                    ),
                }
            )
        return clusters


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def list_cluster_faces(cluster_id: int, limit: int = 8) -> list[dict]:
    """The individual sightings in one cluster, best-quality first.

    `list_clusters` returns a single sample per group, which is enough to show
    that a group exists and not enough to say WHO it is. The captures are person
    boxes from a doorway camera and are frequently tiny — measured on this
    deployment, 54x88 px is the median for some days, holding a face of maybe
    15-20 px. Nobody identifies a colleague from one of those.

    Across seventeen sightings, though, there is usually at least one frame
    where the person turned toward the lens or walked close enough. Showing the
    reviewer several is the difference between a screen they can act on and one
    they cannot, and it costs nothing — the crops are already on disk.
    """
    # STATUS_PENDING lives on the model, not this module — every other function
    # here imports it locally and this one did not, which is a NameError at the
    # first request rather than at import, so it survived a callable() smoke test.
    from app.db.session import SessionLocal
    from app.models.unknown_face import STATUS_PENDING, UnknownFace

    # COLUMNS, not whole ORM rows. `UnknownFace.embedding` is an EncryptedBinary,
    # so loading the entity decrypts a 512-float vector per sighting that this
    # endpoint never looks at. Worse than wasteful: one row that cannot be
    # decrypted — after a SECRET_KEY rotation, say — raises and takes the entire
    # review screen down with it, when all the reviewer wanted was a thumbnail.
    with SessionLocal() as db:
        rows = (
            db.query(
                UnknownFace.id,
                UnknownFace.camera_id,
                UnknownFace.captured_at,
                UnknownFace.quality_score,
                UnknownFace.face_px,
                UnknownFace.crop_path,
            )
            .filter(
                UnknownFace.cluster_id == int(cluster_id),
                UnknownFace.status == STATUS_PENDING,
                UnknownFace.crop_path.isnot(None),
            )
            .order_by(UnknownFace.quality_score.desc().nullslast())
            .limit(max(1, min(int(limit), 24)))
            .all()
        )
        return [
            {
                "face_id": int(row.id),
                "camera_id": row.camera_id,
                "captured_at": row.captured_at.isoformat() if row.captured_at else None,
                "quality_score": float(row.quality_score or 0.0),
                "face_px": float(row.face_px or 0.0),
                "has_crop": True,
            }
            for row in rows
        ]


def cluster_agreement(cluster_id: int, employee_id: int) -> dict:
    """Does this cluster actually look like this employee's existing gallery?

    THE FAILURE THIS PREVENTS
    -------------------------
    Attributing a cluster is a human judgement made from doorway crops that are
    often 54x88 px, holding a face of 15-20 px. That is not reliably
    identifiable by eye, and on 2026-09-02 several clusters were filed under the
    wrong person. The damage is asymmetric and severe:

      * Seema's gallery ended up holding at least three different people. Its
        internal coherence fell to 0.248, against ~0.50 for a clean one.
      * A gallery spanning several people covers a wide region of embedding
        space, so an arbitrary face lands near it. MEASURED: her gallery matched
        **133 of 405** unassigned sightings (32.8%), against 1-20 for everyone
        else — she had become a magnet, and the reported symptom was "it detects
        the wrong person as Seema".

    So an assignment is now checked against what that employee already looks
    like. This does not block anything — a first enrolment has nothing to
    compare against, and the caller decides what to do with a warning — but it
    makes a bad merge visible at the moment it is made rather than a week later
    in the attendance sheet.

    Returns ``agreement`` (mean cosine between the cluster's faces and the
    employee's existing vectors) and a ``verdict``.
    """
    import numpy as np

    from app.db.session import SessionLocal
    from app.models.employee_face import EmployeeFaceEmbedding
    from app.models.unknown_face import STATUS_PENDING, UnknownFace
    from app.services.face_service import EMBEDDING_MODEL_VERSION

    with SessionLocal() as db:
        gallery = [
            np.frombuffer(row.embedding, dtype=np.float32)
            for row in db.query(EmployeeFaceEmbedding).filter(
                EmployeeFaceEmbedding.employee_id == int(employee_id),
                EmployeeFaceEmbedding.active.is_(True),
                EmployeeFaceEmbedding.model_version == EMBEDDING_MODEL_VERSION,
            ).all()
        ]
        faces = [
            _bytes_to_embedding(row.embedding)
            for row in db.query(UnknownFace).filter(
                UnknownFace.cluster_id == int(cluster_id),
                UnknownFace.status == STATUS_PENDING,
            ).all()
        ]

    gallery = [g for g in gallery if g is not None and g.size]
    faces = [f for f in faces if f is not None and getattr(f, "size", 0)]
    if not gallery:
        return {"agreement": None, "verdict": "no_gallery",
                "detail": "First enrolment for this employee — nothing to compare against."}
    if not faces:
        return {"agreement": None, "verdict": "no_faces",
                "detail": "This cluster has no usable embeddings."}

    G, F = np.stack(gallery), np.stack(faces)
    agreement = float(np.mean(F @ G.T))
    if agreement >= _ASSIGN_AGREE_OK:
        verdict, detail = "match", "Consistent with this employee's existing faces."
    elif agreement >= _ASSIGN_AGREE_WARN:
        verdict, detail = "weak", (
            "Only loosely similar to this employee's existing faces. Check the "
            "pictures carefully before enrolling."
        )
    else:
        verdict, detail = "mismatch", (
            "This does NOT look like the same person as the faces already "
            "enrolled for them. Enrolling it would let strangers be recognised "
            "as this employee."
        )
    return {"agreement": round(agreement, 3), "verdict": verdict, "detail": detail,
            "gallery_size": len(gallery), "cluster_size": len(faces)}


def assign_cluster(
    cluster_id: int,
    employee_id: int,
    reviewed_by: Optional[int] = None,
    max_enroll: int = 5,
) -> dict:
    """Attribute a cluster to an employee and enrol its best faces.

    Only the ``max_enroll`` HIGHEST-QUALITY members are enrolled. Adding forty
    near-identical frames would inflate that employee's gallery depth without
    adding information, which the matcher's enrolment-bias correction then has
    to undo — and a deep gallery of mediocre vectors is exactly the condition
    that produced this system's known mislabelling.

    The enrolled rows are tagged with the CAMERA they came from, which is what
    makes this viewpoint-matched enrolment rather than just more photos.
    """
    from app.db.session import SessionLocal
    from app.models.employee import Employee
    from app.models.unknown_face import STATUS_ASSIGNED, STATUS_PENDING, UnknownFace
    from app.services.employee_face_service import enroll_embeddings

    with SessionLocal() as db:
        employee = db.query(Employee).filter(Employee.id == int(employee_id)).first()
        if employee is None:
            raise ValueError(f"employee {employee_id} not found")

        rows = (
            db.query(UnknownFace)
            .filter(
                UnknownFace.cluster_id == int(cluster_id),
                UnknownFace.status == STATUS_PENDING,
            )
            .order_by(UnknownFace.quality_score.desc().nullslast())
            .all()
        )
        if not rows:
            return {"assigned": 0, "enrolled": 0}

        now = get_ist_now()
        enrol_payload = []
        for row in rows[: max(1, int(max_enroll))]:
            vector = _bytes_to_embedding(row.embedding)
            if vector is None:
                continue
            enrol_payload.append(
                {
                    "embedding": vector,
                    "camera_id": row.camera_id,
                    "quality_score": row.quality_score,
                    "face_px": row.face_px,
                    "yaw": row.yaw,
                    "pitch": row.pitch,
                    "blur_var": row.blur_var,
                }
            )

        for row in rows:
            row.status = STATUS_ASSIGNED
            row.assigned_employee_id = int(employee_id)
            row.reviewed_by = reviewed_by
            row.reviewed_at = now
        db.commit()

    enrolled = 0
    if enrol_payload:
        enrolled = enroll_embeddings(
            employee_id=int(employee_id), observations=enrol_payload, source="cctv",
        )

    logger.info(
        "UNKNOWN-FACE cluster=%s assigned to employee=%s (%d rows, %d enrolled)",
        cluster_id, employee_id, len(rows), enrolled,
    )
    return {"assigned": len(rows), "enrolled": enrolled}


def ignore_cluster(cluster_id: int, reviewed_by: Optional[int] = None) -> int:
    """Mark a cluster as 'not an employee' so it stops appearing in the queue."""
    from app.db.session import SessionLocal
    from app.models.unknown_face import STATUS_IGNORED, STATUS_PENDING, UnknownFace

    with SessionLocal() as db:
        rows = (
            db.query(UnknownFace)
            .filter(
                UnknownFace.cluster_id == int(cluster_id),
                UnknownFace.status == STATUS_PENDING,
            )
            .all()
        )
        now = get_ist_now()
        for row in rows:
            row.status = STATUS_IGNORED
            row.reviewed_by = reviewed_by
            row.reviewed_at = now
        db.commit()
        return len(rows)


def purge_older_than(days: int = RETENTION_DAYS) -> int:
    """Delete unknown-face rows and crops older than the retention window.

    Biometric data with no retention policy is a liability, and this table grows
    with every unrecognised passer-by. Intended to be called from the same
    nightly tick that prunes attendance snapshots.
    """
    import shutil

    from app.db.session import SessionLocal
    from app.models.unknown_face import UnknownFace

    cutoff = get_ist_now() - timedelta(days=max(1, int(days)))
    with SessionLocal() as db:
        removed = (
            db.query(UnknownFace)
            .filter(UnknownFace.captured_at < cutoff)
            .delete(synchronize_session=False)
        )
        db.commit()

    if CROP_ROOT.exists():
        cutoff_date = cutoff.date()
        for day_dir in CROP_ROOT.iterdir():
            if not day_dir.is_dir():
                continue
            try:
                folder_date = datetime.fromisoformat(day_dir.name).date()
            except ValueError:
                continue
            if folder_date < cutoff_date:
                shutil.rmtree(day_dir, ignore_errors=True)

    if removed:
        logger.info("UNKNOWN-FACE purged %d row(s) older than %s", removed, cutoff.date())
    return int(removed)
