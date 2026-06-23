from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def find_best_match(
    query_embedding: np.ndarray,
    candidates: list[dict],
    threshold: float = 0.45,
    min_margin: float = 0.0,
) -> dict:
    per_employee: dict[int, dict] = {}

    for candidate in candidates:
        employee_id = int(candidate["employee_id"])
        employee_name = candidate["employee_name"]
        score = cosine_similarity(query_embedding, candidate["embedding"])

        current = per_employee.get(employee_id)
        if current is None or score > current["score"]:
            per_employee[employee_id] = {
                "employee_id": employee_id,
                "employee_name": employee_name,
                "score": float(score),
            }

    ranked = sorted(per_employee.values(), key=lambda item: item["score"], reverse=True)
    if not ranked:
        return {
            "status": False,
            "employee_id": None,
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
            "employee_name": best["employee_name"],
            "score": float(best["score"]),
            "runner_up_score": float(runner_up_score),
            "margin": float(margin),
        }

    return {
        "status": True,
        "employee_id": best["employee_id"],
        "employee_name": best["employee_name"],
        "score": float(best["score"]),
        "runner_up_score": float(runner_up_score),
        "margin": float(margin),
    }
