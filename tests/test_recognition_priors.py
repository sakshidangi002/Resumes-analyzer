"""Two accuracy mechanisms for fixed, badly-placed cameras.

The cameras cannot be moved, so a face arrives as 16-30px and a single frame's
embedding is largely noise (measured: an 18-20px face scored 0.09 against its
own enrolled identity). Two things recover usable signal without touching the
hardware:

  1. EMBEDDING FUSION — average a track's embeddings, weighted by face size.
     Per-frame noise is roughly independent, so N frames cut it by ~sqrt(N).
  2. CANDIDATE REDUCTION — someone already checked in is not walking in again.
     Fewer candidates means the runner-up is a real alternative rather than an
     employee who is demonstrably elsewhere, so the margin gate tightens.

Both must fail SAFE: fusion must never produce a non-unit vector, and the
narrowing must never make a legitimate match impossible.
"""
import numpy as np
import pytest
from app.services.face_tracker import FaceTrack
from app.services.recognition import _narrow_candidates


def _unit(v):
    return v / np.linalg.norm(v)


def _track():
    return FaceTrack(track_id=1, centroid=(0.0, 0.0), box=(0, 0, 10, 10))


# ---------------------------------------------------------------------------
# Embedding fusion
# ---------------------------------------------------------------------------
def test_fusion_raises_the_true_match_score():
    """The whole point: noise averages out, signal does not.

    A marginal single-frame score must climb materially once several
    observations of the same person are fused.
    """
    rng = np.random.default_rng(11)
    truth = _unit(rng.normal(size=512))

    track = _track()
    first = None
    for _ in range(15):
        observed = _unit(truth + rng.normal(scale=0.118, size=512))
        if first is None:
            first = float(observed @ truth)
        track.add_observation(observed, quality=1.0)

    fused = float(track.fused_embedding() @ truth)
    assert first < 0.45, f"test setup wrong: single frame already strong ({first:.2f})"
    assert fused > first + 0.25, (
        f"fusion did not help: {first:.3f} -> {fused:.3f}"
    )


def test_fused_embedding_is_unit_length():
    """Cosine similarity is taken against unit enrolled vectors. Averaging unit
    vectors shortens them, so the mean MUST be re-normalised or every fused
    score is depressed."""
    rng = np.random.default_rng(3)
    track = _track()
    for _ in range(8):
        track.add_observation(_unit(rng.normal(size=512)), quality=2.0)
    assert np.isclose(np.linalg.norm(track.fused_embedding()), 1.0, atol=1e-5)


def test_quality_weighting_favours_the_larger_face():
    """Weighting by face width makes the close-up frames dominate, which is what
    removes the need to explicitly pick a 'best' frame."""
    near, far = _unit(np.array([1.0, 0.0] + [0.0] * 510)), _unit(np.array([0.0, 1.0] + [0.0] * 510))
    track = _track()
    track.add_observation(far, quality=10.0)     # distant, small face
    track.add_observation(near, quality=90.0)    # close-up
    fused = track.fused_embedding()
    assert fused @ near > fused @ far, "the small distant face dominated the fusion"


def test_empty_track_has_no_fused_embedding():
    assert _track().fused_embedding() is None


@pytest.mark.parametrize("bad", [None, np.zeros((2, 512))])
def test_add_observation_ignores_unusable_input(bad):
    track = _track()
    track.add_observation(bad, quality=10.0)
    assert track.observations == 0
    assert track.fused_embedding() is None


def test_zero_or_negative_quality_is_ignored():
    """A zero-width face carries no information and must not enter the average
    (and must not divide by zero)."""
    track = _track()
    track.add_observation(_unit(np.ones(512)), quality=0.0)
    track.add_observation(_unit(np.ones(512)), quality=-5.0)
    assert track.observations == 0
    assert track.fused_embedding() is None


def test_best_face_px_tracks_the_maximum():
    track = _track()
    for q in (20.0, 55.0, 31.0):
        track.add_observation(_unit(np.ones(512)), quality=q)
    assert track.best_face_px == 55.0


# ---------------------------------------------------------------------------
# Candidate narrowing
# ---------------------------------------------------------------------------
def _candidates(ids):
    return [
        {"employee_id": i, "employee_code": f"E{i}", "employee_name": f"emp{i}",
         "embedding": np.zeros(512)}
        for i in ids
    ]


@pytest.fixture
def present(monkeypatch):
    state = {"ids": set()}

    def fake():
        return set(state["ids"])

    import app.services.presence_cache as pc

    monkeypatch.setattr(pc, "get_present_employee_ids", fake)
    return state


def test_in_camera_excludes_people_already_inside(present):
    present["ids"] = {2, 3}
    out = _narrow_candidates(_candidates([1, 2, 3, 4]), "IN", "cctv")
    assert {c["employee_id"] for c in out} == {1, 4}


@pytest.mark.parametrize("purpose", ["OUT", "MONITOR"])
def test_out_and_monitor_keep_only_people_inside(present, purpose):
    present["ids"] = {2, 3}
    out = _narrow_candidates(_candidates([1, 2, 3, 4]), purpose, "cctv")
    assert {c["employee_id"] for c in out} == {2, 3}


def test_narrowing_never_empties_the_pool(present):
    """An empty pool would make recognition impossible rather than merely
    stricter — fall back to everyone instead."""
    present["ids"] = {1, 2, 3, 4}
    out = _narrow_candidates(_candidates([1, 2, 3, 4]), "IN", "cctv")
    assert len(out) == 4


def test_unknown_presence_is_not_treated_as_nobody_inside(present):
    """An empty presence set means 'nobody inside' OR 'the lookup failed' — the
    two are indistinguishable, so it must be treated as no information. Reading
    it as 'nobody is inside' would block every OUT-camera match."""
    present["ids"] = set()
    out = _narrow_candidates(_candidates([1, 2, 3]), "OUT", "cctv")
    assert len(out) == 3


@pytest.mark.parametrize("source", ["webcam", "upload"])
def test_self_service_sources_are_never_narrowed(present, source):
    """Kiosk / self check-in must work regardless of recorded state."""
    present["ids"] = {1, 2}
    out = _narrow_candidates(_candidates([1, 2, 3]), "IN", source)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# The dev-room (body tracking) path must fuse too
# ---------------------------------------------------------------------------
def test_person_track_supports_fusion():
    """The room cameras need fusion MORE than the entrance does.

    A seated person is in view for minutes, so there are far more observations
    to average, and a ceiling camera's faces are the smallest in the system.
    Fusion was originally wired only into the face-only pipeline, so the dev
    room — the case it helps most — got none of it.
    """
    from app.services.person_tracker import PersonTrack

    rng = np.random.default_rng(5)
    truth = _unit(rng.normal(size=512))
    track = PersonTrack(track_id=1, box=(0, 0, 10, 10))

    first = None
    for _ in range(15):
        observed = _unit(truth + rng.normal(scale=0.118, size=512))
        if first is None:
            first = float(observed @ truth)
        track.add_observation(observed, quality=25.0)

    fused = float(track.fused_embedding() @ truth)
    assert fused > first + 0.25, f"no fusion benefit on the body path: {first:.3f} -> {fused:.3f}"


def test_both_track_types_share_one_fusion_implementation():
    """Face and person tracks must not drift apart — that is exactly how the
    body path ended up without fusion in the first place."""
    from app.services.embedding_fusion import EmbeddingFuser
    from app.services.person_tracker import PersonTrack

    face = FaceTrack(track_id=1, centroid=(0, 0), box=(0, 0, 9, 9))
    person = PersonTrack(track_id=1, box=(0, 0, 9, 9))
    assert isinstance(face.fuser, EmbeddingFuser)
    assert isinstance(person.fuser, EmbeddingFuser)
