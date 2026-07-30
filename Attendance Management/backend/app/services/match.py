from __future__ import annotations

import math
from statistics import NormalDist
import os

import numpy as np

# ---------------------------------------------------------------------------
# Enrollment-count bias correction
# ---------------------------------------------------------------------------
# cosine_similarity returns the MAX over an employee's enrolled photos, which is
# right for recall — it lets one employee be matched from several angles. But it
# also means an employee with N photos gets N independent draws at the noise
# distribution, so their maximum is inflated purely by having more photos.
#
# MEASURED on the live gallery, feeding 4000 pure-noise embeddings (which is
# effectively what a 14px face produces). A fair matcher would pick each of the
# five employees 20% of the time:
#
#     Rakhi Channa      5 photos  ->  30.9%
#     Sakshi Dangi      4 photos  ->  21.9%
#     Saloni Pathania   2 photos  ->  18.2%
#     Adarsh Maurya     2 photos  ->  17.3%
#     Seema Chauhan     1 photo   ->  11.7%
#
# Monotonic in photo count: the best-enrolled employee was matched 2.6x more
# often than the worst, on random input. That is almost certainly the mechanism
# behind this system's known mislabelling incidents — the wrong name was simply
# the person with the deepest gallery.
#
# For unit vectors in D dimensions a chance cosine is ~N(0, 1/D), so subtracting
# the EXPECTED MAXIMUM of that employee's draws equalises the null distribution
# while barely touching a genuine match, which scores far above the noise floor.
#
# The photos in a real gallery are NOT independent — they are the same face from
# similar angles, often correlated at r=0.5+. Five near-identical photos are
# worth far less than five independent draws, so using raw N over-penalises a
# well-enrolled employee. The effective sample size
#
#     N_eff = N / (1 + (N-1) * mean_pairwise_correlation)
#
# collapses to 1 for identical photos and to N for genuinely diverse ones, which
# is exactly the behaviour wanted: an employee is penalised for gallery
# DIVERSITY (real extra chances at the noise), not for photo count alone.
_BIAS_SCALE = float(os.getenv("MATCH_ENROLLMENT_BIAS_SCALE", "1.0"))


def _effective_sample_size(stack: np.ndarray) -> float:
    """Number of *independent* draws an employee's photo stack is worth."""
    n = stack.shape[0]
    if n <= 1:
        return float(n)
    norms = np.linalg.norm(stack, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = stack / norms
    sims = unit @ unit.T
    # Mean of the strict upper triangle = mean pairwise correlation.
    iu = np.triu_indices(n, k=1)
    mean_r = float(np.clip(np.mean(sims[iu]), 0.0, 0.999))
    return n / (1.0 + (n - 1) * mean_r)


def _expected_max_z(n: float) -> float:
    """Expected maximum of n standard normal draws (Blom's approximation).

    NOT sqrt(2 ln n): that is the large-n asymptotic and overestimates badly in
    the range that matters here. Measured against 3000 trials at n=8, sqrt(2 ln
    n) predicts 0.090 where the truth is 0.063 — a 1.4x over-penalty that would
    suppress genuine matches for well-enrolled employees.

    Blom's estimate of the largest order statistic is accurate from n=1 upward
    and returns exactly 0 at n=1.
    """
    if n <= 1.0:
        return 0.0
    return NormalDist().inv_cdf((n - 0.375) / (n + 0.25))


def enrollment_bias_penalty(embedding) -> float:
    """Expected inflation of a max-over-N score under the null hypothesis.

    Accepts either an embedding stack (N, D) or a single vector (D,).
    """
    if _BIAS_SCALE <= 0:
        return 0.0
    ref = np.asarray(embedding, dtype=np.float32)
    if ref.ndim == 1 or ref.shape[0] <= 1:
        return 0.0
    n_eff = _effective_sample_size(ref)
    sigma = 1.0 / math.sqrt(ref.shape[1])   # chance cosine ~ N(0, 1/D)
    return _BIAS_SCALE * sigma * _expected_max_z(n_eff)


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
        raw = cosine_similarity(query_embedding, candidate["embedding"])
        # Level the playing field between employees with different gallery
        # depth — see enrollment_bias_penalty. Without this, ranking partly
        # reflects who was photographed most, not who is in the frame.
        # `bias_penalty` is precomputed once at cache load where available.
        penalty = candidate.get("bias_penalty")
        if penalty is None:
            penalty = enrollment_bias_penalty(candidate["embedding"])
        score = raw - float(penalty)

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