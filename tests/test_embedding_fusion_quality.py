"""Bounded, outlier-rejecting embedding fusion.

The original fuser kept an unbounded running weighted sum with no consistency
check, so (a) a tracker ID-switch silently folded a second person into the same
template, and (b) a person in view for a long time could never recover from a
bad start. These tests pin the replacement's behaviour.
"""
import numpy as np
import pytest

from app.services.embedding_fusion import DEFAULT_WINDOW, EmbeddingFuser


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _at_cosine(base, seed, r):
    """Vector whose cosine to `base` is r (correct high-dimensional construction)."""
    g = np.random.default_rng(seed)
    perp = g.normal(size=base.shape).astype(np.float32)
    perp -= (perp @ base) * base
    perp /= np.linalg.norm(perp)
    return _unit(r * base + np.sqrt(max(0.0, 1.0 - r * r)) * perp)


@pytest.fixture
def person():
    rng = np.random.default_rng(7)
    return _unit(rng.normal(size=512))


@pytest.fixture
def stranger():
    rng = np.random.default_rng(99)
    return _unit(rng.normal(size=512))


# ---------------------------------------------------------------------------
# Basics preserved from the original implementation
# ---------------------------------------------------------------------------
def test_fused_is_unit_length(person):
    fuser = EmbeddingFuser()
    for i in range(5):
        fuser.add(_at_cosine(person, i, 0.8), 0.5)
    assert np.isclose(np.linalg.norm(fuser.fused()), 1.0, atol=1e-5)


def test_empty_fuser_returns_none():
    assert EmbeddingFuser().fused() is None
    assert EmbeddingFuser().consensus() == 0.0


@pytest.mark.parametrize("bad", [None, [], np.zeros(512, dtype=np.float32)])
def test_unusable_input_is_ignored(bad):
    fuser = EmbeddingFuser()
    assert fuser.add(bad, 1.0) is False
    assert fuser.accepted == 0


@pytest.mark.parametrize("quality", [0.0, -1.0, None])
def test_non_positive_quality_is_ignored(person, quality):
    fuser = EmbeddingFuser()
    assert fuser.add(person, quality) is False
    assert fuser.fused() is None


def test_averaging_reduces_noise(person):
    """The whole point of fusion on a fixed ceiling camera."""
    fuser = EmbeddingFuser()
    for i in range(10):
        fuser.add(_at_cosine(person, i, 0.6), 0.5)
    single = float(_at_cosine(person, 0, 0.6) @ person)
    fused = float(fuser.fused() @ person)
    assert fused > single


def test_higher_quality_observations_dominate(person, stranger):
    """Quality weighting, not size weighting.

    A sharp frontal frame must outweigh a blurred profile of the same pixel
    width — the distinction the old `quality = face_px` could not express.
    """
    fuser = EmbeddingFuser()
    fuser.add(person, 0.9)          # good frame
    fuser.add(_at_cosine(person, 1, 0.45), 0.05)   # marginal frame
    assert float(fuser.fused() @ person) > 0.9


# ---------------------------------------------------------------------------
# New behaviour
# ---------------------------------------------------------------------------
def test_window_is_bounded(person):
    """An unbounded window let the first minute outvote the present."""
    fuser = EmbeddingFuser()
    for i in range(DEFAULT_WINDOW * 3):
        fuser.add(_at_cosine(person, i, 0.85), 0.5)
    assert len(fuser._obs) == DEFAULT_WINDOW
    assert fuser.accepted == DEFAULT_WINDOW * 3     # counter is not windowed


def test_outlier_is_rejected_once_a_consensus_exists(person, stranger):
    fuser = EmbeddingFuser()
    for i in range(3):
        assert fuser.add(_at_cosine(person, i, 0.85), 0.5) is True
    assert fuser.add(stranger, 0.5) is False
    assert fuser.rejected == 1


def test_early_observations_are_not_outlier_checked(person, stranger):
    """The first frames define the track; there is nothing to compare them to.

    Rejecting against a one-sample 'consensus' would let a single bad opening
    frame decide who the track is for its whole life.
    """
    fuser = EmbeddingFuser()
    assert fuser.add(person, 0.5) is True
    assert fuser.add(stranger, 0.5) is True
    assert fuser.rejected == 0


def test_pose_variation_of_the_same_person_is_not_rejected(person):
    """Outlier rejection must not discard the variation fusion exists to capture."""
    fuser = EmbeddingFuser()
    for i in range(4):
        fuser.add(_at_cosine(person, i, 0.85), 0.5)
    # A pronounced pose change — still clearly the same person.
    assert fuser.add(_at_cosine(person, 50, 0.5), 0.5) is True


def test_consensus_reflects_agreement(person):
    tight = EmbeddingFuser()
    loose = EmbeddingFuser()
    for i in range(5):
        tight.add(_at_cosine(person, i, 0.9), 0.5)
        # Pushed in directly: outlier rejection would otherwise refuse these,
        # and the point here is to compare consensus values, not gating.
        loose._obs.append((_at_cosine(person, i, 0.25), 0.5))
        loose.accepted += 1
    loose._dirty = True
    assert tight.consensus() > loose.consensus()


def test_reset_clears_everything(person):
    fuser = EmbeddingFuser()
    for i in range(4):
        fuser.add(_at_cosine(person, i, 0.85), 0.5)
    fuser.reset()
    assert fuser.fused() is None
    assert (fuser.accepted, fuser.rejected, fuser.observations) == (0, 0, 0)
    assert fuser.best_quality == 0.0


def test_observations_alias_tracks_accepted(person):
    """`observations` is the historical name and must keep its meaning."""
    fuser = EmbeddingFuser()
    for i in range(3):
        fuser.add(_at_cosine(person, i, 0.85), 0.5)
    assert fuser.observations == fuser.accepted == 3
