from __future__ import annotations

import threading

from app.db import blob_to_embedding, db_cursor


_lock = threading.Lock()
_candidates: list[dict] | None = None


def _load_from_db() -> list[dict]:
    with db_cursor() as (_, cur):
        rows = cur.execute(
            """
            SELECT id AS employee_id, (first_name || ' ' || last_name) AS employee_name, embedding
            FROM employees
            WHERE embedding IS NOT NULL
            ORDER BY id DESC
            """
        ).fetchall()

    return [
        {
            "employee_id": row["employee_id"],
            "employee_name": row["employee_name"],
            "embedding": blob_to_embedding(row["embedding"]),
        }
        for row in rows
    ]


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
