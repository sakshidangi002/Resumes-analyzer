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
    with SessionLocal() as db:
        rows = (
            db.query(Employee)
            .filter(
                Employee.embedding.isnot(None),
                Employee.employment_status == EmploymentStatus.ACTIVE.value,
            )
            .order_by(Employee.id.desc())
            .all()
        )

    candidates: list[dict] = []
    for emp in rows:
        if emp.embedding is None:
            continue
        candidates.append(
            {
                "employee_id": emp.id,
                "employee_code": emp.employee_code,
                "employee_name": emp.full_name,
                "embedding": blob_to_embedding(emp.embedding),
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
