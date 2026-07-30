"""Match scores must not depend on how many photos someone was enrolled with.

`cosine_similarity` takes the MAX over an employee's enrolled photos, which is
right for recall — it lets one person match from several angles. But it gives an
employee with N photos N independent draws at the noise distribution, so their
maximum is inflated purely by gallery depth.

MEASURED on the live gallery with 4000 pure-noise embeddings (what a 14px face
effectively produces). A fair matcher picks each of five employees 20% of the
time:

    Rakhi Channa      5 photos -> 30.9%      Adarsh Maurya   2 photos -> 17.3%
    Sakshi Dangi      4 photos -> 21.9%      Seema Chauhan   1 photo  -> 11.7%
    Saloni Pathania   2 photos -> 18.2%

Monotonic in photo count — the best-enrolled employee matched 2.6x more often
than the worst, on random input. That is very likely the mechanism behind this
system's known mislabelling incidents.
"""

import numpy as np
from app.services.match import enrollment_bias_penalty, find_best_match


def _unit(v):
    return v / np.linalg.norm(v)


def _cand(emp_id, name, embeddings):
    arr = np.asarray(embeddings, dtype=np.float32)
    return {"employee_id": emp_id, "employee_code": f"E{emp_id}",
            "employee_name": name, "embedding": arr}


def _stack(rng, n, correlation=0.0):
    """N photos of one person. correlation=0 -> fully diverse (worst case for
    the bias); correlation near 1 -> near-identical shots."""
    base = _unit(rng.normal(size=512))
    out = []
    for _ in range(n):
        v = correlation * base + (1 - correlation) * _unit(rng.normal(size=512))
        out.append(_unit(v))
    return np.asarray(out, dtype=np.float32)


def test_single_photo_is_never_penalised():
    rng = np.random.default_rng(1)
    assert enrollment_bias_penalty(_unit(rng.normal(size=512))) == 0.0
    assert enrollment_bias_penalty(_stack(rng, 1)) == 0.0


def test_penalty_grows_with_gallery_diversity():
    rng = np.random.default_rng(2)
    penalties = [enrollment_bias_penalty(_stack(rng, n)) for n in (1, 2, 4, 8, 16)]
    assert penalties == sorted(penalties)
    assert penalties[-1] > penalties[1] > 0


def test_near_identical_photos_are_barely_penalised():
    """Five copies of the same shot give no real extra chance at the noise, so
    they must not be charged like five independent draws."""
    rng = np.random.default_rng(4)
    diverse = enrollment_bias_penalty(_stack(rng, 6, correlation=0.0))
    duplicates = enrollment_bias_penalty(_stack(rng, 6, correlation=0.95))
    assert duplicates < diverse / 2, (
        f"near-duplicate gallery penalised almost as hard as a diverse one "
        f"({duplicates:.4f} vs {diverse:.4f})"
    )


def test_deep_gallery_no_longer_wins_on_noise():
    """The regression this exists to prevent."""
    rng = np.random.default_rng(3)
    deep = _cand(1, "deep", _stack(rng, 8))
    shallow = _cand(2, "shallow", _unit(rng.normal(size=512)))

    wins = {"deep": 0, "shallow": 0}
    for _ in range(2000):
        q = _unit(rng.normal(size=512))
        wins[find_best_match(q, [deep, shallow], threshold=0.0, min_margin=0.0)["employee_name"]] += 1

    share = wins["deep"] / 2000
    assert 0.35 < share < 0.65, (
        f"8-photo employee won {share:.0%} of pure-noise matches (fair is 50%) "
        f"— enrollment depth is still biasing the result"
    )


def test_a_genuine_match_still_wins_easily():
    """The correction must not cost real recognition.

    A true face scores far above the noise floor, so a penalty of ~0.02-0.05
    is irrelevant to it — but if the coefficient were ever set too high this
    test fails loudly.
    """
    rng = np.random.default_rng(11)
    truth = _unit(rng.normal(size=512))
    # The real person, enrolled 8 times (worst case for the penalty).
    target = _cand(1, "target", np.asarray(
        [truth] + [_unit(rng.normal(size=512)) for _ in range(7)], dtype=np.float32))
    others = [_cand(i, f"other{i}", _unit(rng.normal(size=512))) for i in range(2, 8)]

    probe = _unit(truth + rng.normal(scale=0.08, size=512))   # a solid real match
    best = find_best_match(probe, [target, *others], threshold=0.30, min_margin=0.10)

    assert best["employee_name"] == "target"
    assert best["status"] is True, f"genuine match lost to the penalty: {best['score']:.3f}"


def test_penalty_is_small_relative_to_a_real_match():
    """Sanity bound: even a maximally diverse gallery costs far less than the
    gap between a real match (~0.4+) and the noise floor (~0.1)."""
    rng = np.random.default_rng(9)
    assert enrollment_bias_penalty(_stack(rng, 20)) < 0.15, (
        "the penalty is large enough to suppress real matches"
    )
