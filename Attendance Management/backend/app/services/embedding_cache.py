from __future__ import annotations

import io
import threading

import numpy as np

from app.db.session import SessionLocal
from app.models.employee import Employee, EmploymentStatus


_lock = threading.Lock()
_candidates: list[dict] | None = None


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(embedding, dtype=np.float32))
    return buffer.getvalue()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    buffer = io.BytesIO(blob)
    buffer.seek(0)
    return np.load(buffer, allow_pickle=False)


def _load_from_db() -> list[dict]:
    # A query vector and a gallery vector must come from the same recognition
    # model. Employee.embedding is a legacy hot-path blob and has no model
    # version of its own, so it cannot be trusted after an ArcFace -> AdaFace
    # switch. Use the provenance table as the compatibility gate.
    from app.models.employee_face import EmployeeFaceEmbedding
    from app.services.face_service import EMBEDDING_MODEL_VERSION
    with SessionLocal() as db:
        rows = (
            db.query(Employee)
            .join(
                EmployeeFaceEmbedding,
                EmployeeFaceEmbedding.employee_id == Employee.id,
            )
            .filter(
                Employee.embedding.isnot(None),
                Employee.employment_status == EmploymentStatus.ACTIVE.value,
                EmployeeFaceEmbedding.active.is_(True),
                EmployeeFaceEmbedding.model_version == EMBEDDING_MODEL_VERSION,
            )
            .distinct()
            .order_by(Employee.id.desc())
            .all()
        )

    from app.services.match import enrollment_bias_penalty

    candidates: list[dict] = []
    for emp in rows:
        if emp.embedding is None:
            continue
        embedding = blob_to_embedding(emp.embedding)
        candidates.append(
            {
                "employee_id": emp.id,
                "employee_code": emp.employee_code,
                "employee_name": emp.full_name,
                "embedding": embedding,
                # Precomputed here so the matcher pays nothing per comparison.
                # Corrects the max-over-photos bias that otherwise favours
                # whoever was enrolled with the most (or most varied) photos —
                # see match.enrollment_bias_penalty.
                "bias_penalty": enrollment_bias_penalty(embedding),
            }
        )
    return candidates


def get_employee_candidates() -> list[dict]:
    global _candidates
    with _lock:
        if _candidates is None:
            _candidates = _load_from_db()
        return list(_candidates)


def invalidate_embedding_cache() -> None:
    global _candidates
    with _lock:
        _candidates = None


def warm_embedding_cache() -> int:
    global _candidates
    with _lock:
        _candidates = _load_from_db()
        return len(_candidates)
