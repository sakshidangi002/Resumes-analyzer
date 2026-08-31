"""The gate between "we recognised someone" and "write a payroll row".

Before this module the rule was `confirm_frames = 1`: one matched frame above
0.45 with a 0.10 margin wrote attendance. These tests pin the replacement —
evidence from the whole track, with one narrowly-bounded escape hatch for the
fleeting-face case that made the previous multi-frame attempt unusable.
"""
import numpy as np
import pytest

from app.services import attendance_gate
from app.services.face_quality import QualityLimits
from app.services.camera_profile import CameraProfile
from app.services.face_tracker import FaceTrack


def _profile(purpose="IN", **overrides):
    base = dict(
        camera_id="1",
        purpose=purpose,
        threshold=0.45,
        margin=0.18,
        limits=QualityLimits(),
        min_observations=3,
        min_quality=0.28,
        min_consensus=0.55,
        analysis_interval=0.12,
        face_crop_scale=3,
        attendance_cooldown=20.0,
    )
    base.update(overrides)
    return CameraProfile(**base)


def _track():
    return FaceTrack(track_id=1, centroid=(0.0, 0.0), box=(0, 0, 10, 10))


def _similar(seed, base, jitter=0.05):
    """A vector close to `base` — what the same face looks like frame to frame."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=base.shape).astype(np.float32) * jitter
    vector = base + noise
    return vector / np.linalg.norm(vector)


@pytest.fixture
def identity():
    rng = np.random.default_rng(42)
    vector = rng.normal(size=512).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _load(track, identity, count, quality=0.5):
    for i in range(count):
        track.add_observation(_similar(i, identity), quality=quality)


# ---------------------------------------------------------------------------
# The core requirement
# ---------------------------------------------------------------------------
def test_single_ordinary_observation_does_not_mark(identity):
    """The exact case that shipped: one frame, decent score, attendance written."""
    track = _track()
    _load(track, identity, 1, quality=0.4)

    decision = attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.50, margin=0.20, profile=_profile(),
    )
    assert not decision.allowed
    assert decision.reason == "insufficient_observations"


def test_enough_consistent_observations_marks(identity):
    track = _track()
    _load(track, identity, 4, quality=0.5)
    # Two agreeing identifications, as an attendance camera requires.
    attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.60, margin=0.25, profile=_profile(),
    )
    decision = attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.60, margin=0.25, profile=_profile(),
    )
    assert decision.allowed
    assert decision.path == "evidence"


def test_below_margin_is_refused_even_with_a_high_score(identity):
    """Score alone cannot separate this system's known failure.

    The documented mislabelling had the wrong person at 0.77 while correct
    matches sat at 0.73-0.79. The margin is what distinguishes them.
    """
    track = _track()
    _load(track, identity, 5, quality=0.6)
    decision = attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.77, margin=0.04, profile=_profile(),
    )
    assert not decision.allowed
    assert decision.reason == "below_margin"


def test_low_quality_observations_are_refused(identity):
    track = _track()
    _load(track, identity, 6, quality=0.05)      # plenty of frames, all unusable
    attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.60, margin=0.25, profile=_profile(),
    )
    decision = attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.60, margin=0.25, profile=_profile(),
    )
    assert not decision.allowed
    assert decision.reason == "low_quality"


def test_unstable_identity_is_refused(identity):
    """Named as one employee, then another — the track has not settled."""
    track = _track()
    _load(track, identity, 5, quality=0.6)
    attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.60, margin=0.25, profile=_profile(),
    )
    decision = attendance_gate.evaluate(
        track, employee_id=9, matched=True, score=0.60, margin=0.25, profile=_profile(),
    )
    assert not decision.allowed
    assert decision.reason == "unstable_identity"


def test_no_match_is_reported_as_such(identity):
    track = _track()
    _load(track, identity, 5)
    decision = attendance_gate.evaluate(
        track, employee_id=None, matched=False, score=0.2, margin=0.0, profile=_profile(),
    )
    assert not decision.allowed
    assert decision.reason == "no_match"


def test_already_marked_track_is_refused(identity):
    track = _track()
    _load(track, identity, 5, quality=0.6)
    track.attendance_marked = True
    decision = attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.9, margin=0.5, profile=_profile(),
    )
    assert not decision.allowed
    assert decision.reason == "already_marked"


# ---------------------------------------------------------------------------
# The escape hatch
# ---------------------------------------------------------------------------
def test_one_unambiguous_look_may_mark(identity):
    """The case that killed the previous multi-frame attempt.

    At the check-out camera a face is often detected exactly once per pass. A
    single observation is allowed ONLY when it is unambiguous on every axis.
    """
    track = _track()
    _load(track, identity, 1, quality=0.80)
    decision = attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.62, margin=0.40, profile=_profile(),
    )
    assert decision.allowed
    assert decision.path == "strong_single"


def test_the_known_mislabelling_does_not_qualify_as_a_strong_single(identity):
    """A 0.77 score with an ordinary margin from a mediocre upscaled face.

    These are the numbers from the incident recorded in the code comments. The
    escape hatch must not readmit them.
    """
    track = _track()
    _load(track, identity, 1, quality=0.30)
    decision = attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.77, margin=0.20, profile=_profile(),
    )
    assert not decision.allowed
    assert decision.reason == "insufficient_observations"


def test_strong_single_still_needs_a_real_margin(identity):
    track = _track()
    _load(track, identity, 1, quality=0.90)
    decision = attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.80, margin=0.19, profile=_profile(),
    )
    assert not decision.allowed


# ---------------------------------------------------------------------------
# Track consistency
# ---------------------------------------------------------------------------
def _at_cosine(base, seed, r):
    """A vector whose cosine to `base` is r.

    Built as r*base + sqrt(1-r^2)*perpendicular, NOT by adding Gaussian noise:
    in 512 dimensions a noise vector with per-component sigma s has norm
    s*sqrt(512), so 'small' noise decorrelates almost completely and would make
    a genuine track look like two different people.
    """
    g = np.random.default_rng(seed)
    perp = g.normal(size=base.shape).astype(np.float32)
    perp -= (perp @ base) * base
    perp /= np.linalg.norm(perp)
    vector = r * base + np.sqrt(max(0.0, 1.0 - r * r)) * perp
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def test_intruder_frames_are_rejected_from_a_settled_track():
    """A tracker ID-switch mid-track: the second person's frames never enter.

    This is the common shape of the problem — someone passes behind the tracked
    person and the box jumps. Measured: all three intruder frames are rejected
    and the fused template stays on the original person.
    """
    rng = np.random.default_rng(1)
    person_a = rng.normal(size=512).astype(np.float32)
    person_a /= np.linalg.norm(person_a)
    person_b = rng.normal(size=512).astype(np.float32)
    person_b /= np.linalg.norm(person_b)

    track = _track()
    for i in range(3):
        assert track.add_observation(_at_cosine(person_a, i, 0.8), quality=0.6)
    for i in range(3):
        assert not track.add_observation(_at_cosine(person_b, 10 + i, 0.8), quality=0.6)

    assert track.fuser.accepted == 3
    assert track.fuser.rejected == 3
    fused = track.fused_embedding()
    assert float(fused @ person_a) > 0.8
    assert float(fused @ person_b) < 0.3


def test_interleaved_merge_degrades_the_match_rather_than_faking_one():
    """The hard case: two people alternating from the very first frame.

    Outlier rejection has no settled consensus to reject against, so BOTH
    people's frames get in and the template really is a blend. This is a known
    residual weakness, not something the gate eliminates.

    What makes it survivable is that a blended template resembles NEITHER
    person strongly — measured at 0.68 and 0.63, against 0.96 for a clean
    single-identity track. Its score against the gallery is degraded by roughly
    a third, so it normally lands below the camera threshold: a MISS, which HR
    can correct, rather than a confident write on the wrong person.

    Consensus does not rescue this case and is not set to pretend otherwise —
    a merged track reads 0.64 while a genuine noisy track reads 0.68, and a
    threshold between those two would cost more real recognitions than it saves.
    """
    rng = np.random.default_rng(2)
    person_a = rng.normal(size=512).astype(np.float32)
    person_a /= np.linalg.norm(person_a)
    person_b = rng.normal(size=512).astype(np.float32)
    person_b /= np.linalg.norm(person_b)

    merged = _track()
    for i in range(3):
        merged.add_observation(_at_cosine(person_a, i, 0.8), quality=0.6)
        merged.add_observation(_at_cosine(person_b, 10 + i, 0.8), quality=0.6)

    clean = _track()
    for i in range(6):
        clean.add_observation(_at_cosine(person_a, i, 0.8), quality=0.6)

    merged_fused = merged.fused_embedding()
    clean_fused = clean.fused_embedding()

    # The blend is markedly worse against its best candidate than a clean track.
    assert float(clean_fused @ person_a) > 0.9
    assert float(merged_fused @ person_a) < 0.75
    assert float(merged_fused @ person_b) < 0.75


def test_consensus_backstop_refuses_a_degenerate_template():
    """A template whose own observations barely agree cannot justify payroll.

    Consensus is a BACKSTOP, not the primary merge defence (outlier rejection is
    — see the two tests above). It catches the residual case where a track's
    observations are so scattered that no single identity claim is supportable.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(size=512).astype(np.float32)
    base /= np.linalg.norm(base)

    track = _track()
    # Deliberately scattered. Pushed in past the outlier check (which would
    # otherwise reject them one by one) so the gate itself is what is tested.
    #
    # Measured boundary at min_consensus=0.55: a track is refused below a
    # within-track similarity of about 0.28. That is the honest reach of this
    # backstop — it catches templates with no coherent identity in them, not a
    # tidy two-person merge.
    for i in range(4):
        track.fuser._obs.append((_at_cosine(base, i, 0.15), 0.6))
        track.fuser.accepted += 1
        track.fuser.best_quality = 0.6
    track.fuser._dirty = True

    profile = _profile(min_observations=3, min_consensus=0.55)
    assert track.consensus < 0.55

    attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.60, margin=0.25, profile=profile,
    )
    decision = attendance_gate.evaluate(
        track, employee_id=7, matched=True, score=0.60, margin=0.25, profile=profile,
    )
    assert not decision.allowed
    assert decision.reason == "inconsistent_track"


def test_a_genuine_track_clears_the_consensus_bar():
    """Guards against tightening min_consensus into a recall problem.

    Measured: a genuine track with within-track cosine 0.6 (a realistic floor
    once the pose gate is applied) produces consensus ~0.68. The default of 0.55
    leaves headroom; anything above ~0.65 would start rejecting real people.
    """
    rng = np.random.default_rng(4)
    base = rng.normal(size=512).astype(np.float32)
    base /= np.linalg.norm(base)

    track = _track()
    for i in range(6):
        track.add_observation(_at_cosine(base, i, 0.6), quality=0.6)

    assert track.consensus > 0.60


def test_gate_does_not_set_attendance_marked(identity):
    """The gate is pure: the caller records the write.

    A track blocked by the per-camera cooldown must be able to try again rather
    than being silently flagged as done for the rest of its life.
    """
    track = _track()
    _load(track, identity, 5, quality=0.6)
    for _ in range(2):
        decision = attendance_gate.evaluate(
            track, employee_id=7, matched=True, score=0.60, margin=0.25,
            profile=_profile(),
        )
    assert decision.allowed
    assert track.attendance_marked is False


def test_monitor_profile_never_marks_attendance():
    assert _profile("MONITOR").marks_attendance is False
    assert _profile("IN").marks_attendance is True
    assert _profile("OUT").marks_attendance is True
