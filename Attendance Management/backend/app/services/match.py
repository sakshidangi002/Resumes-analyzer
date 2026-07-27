from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Best cosine similarity between query `a` and reference `b`.

    `b` may be a single embedding (512,) OR a stack of an employee's enrolled
    embeddings (N, 512). Returning the MAX over the stack lets one employee be
    matched from several angles/lighting conditions, which is the key to high
    recognition accuracy — far better than averaging all photos into one vector.
    """
    q = np.asarray(a, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn == 0:
        return -1.0

    ref = np.asarray(b, dtype=np.float32)
    if ref.ndim == 1:
        ref = ref[None, :]           # treat single embedding as a 1-row stack
    if ref.size == 0:
        return -1.0

    norms = np.linalg.norm(ref, axis=1)
    valid = norms > 0
    if not valid.any():
        return -1.0

    sims = (ref[valid] @ q) / (norms[valid] * qn)
    return float(np.max(sims))


def find_best_match(
    query_embedding: np.ndarray,
    candidates: list[dict],
    threshold: float = 0.45,
    min_margin: float = 0.0,
) -> dict:
    per_employee: dict[int, dict] = {}

    for candidate in candidates:
        employee_id = int(candidate["employee_id"])
        employee_code = str(candidate.get("employee_code") or employee_id)
        employee_name = candidate["employee_name"]
        score = cosine_similarity(query_embedding, candidate["embedding"])

        current = per_employee.get(employee_id)
        if current is None or score > current["score"]:
            per_employee[employee_id] = {
                "employee_id": employee_id,
                "employee_code": employee_code,
                "employee_name": employee_name,
                "score": float(score),
            }

    ranked = sorted(per_employee.values(), key=lambda item: item["score"], reverse=True)
    if not ranked:
        return {
            "status": False,
            "employee_id": None,
            "employee_code": None,
            "employee_name": "Unknown",
            "score": -1.0,
            "runner_up_score": -1.0,
            "margin": 0.0,
        }

    best = ranked[0]
    runner_up_score = ranked[1]["score"] if len(ranked) > 1 else -1.0
    margin = float(best["score"] - runner_up_score) if runner_up_score >= 0 else float(best["score"])
    status = float(best["score"]) >= float(threshold) and margin >= float(min_margin)

    if not status:
        return {
            "status": False,
            "employee_id": best["employee_id"],
            "employee_code": best["employee_code"],
            "employee_name": best["employee_name"],
            "score": float(best["score"]),
            "runner_up_score": float(runner_up_score),
            "margin": float(margin),
        }

    return {
        "status": True,
        "employee_id": best["employee_id"],
        "employee_code": best["employee_code"],
        "employee_name": best["employee_name"],
        "score": float(best["score"]),
        "runner_up_score": float(runner_up_score),
        "margin": float(margin),
    }