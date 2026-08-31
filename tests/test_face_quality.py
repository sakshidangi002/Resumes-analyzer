"""Face quality gate — the check that stands between a bad frame and payroll.

The bug that motivated this module: pose was read as `face.yaw`, an attribute
InsightFace never sets, so every face in the system reported 0/0/0 and the pose
gate could not fire. A full profile scored identically to a dead-on frontal.
These tests pin the reading of pose, the hard gates, and the shape of the soft
score used as the fusion weight.
"""
import numpy as np
import pytest

from app.services.face_quality import (
    QualityLimits,
    _landmark_asymmetry,
    _pose_from_face,
    assess,
    face_px_width,
)


LIMITS = QualityLimits(
    min_face_px=30.0,
    min_det_score=0.5,
    max_yaw_deg=40.0,
    max_pitch_deg=35.0,
    max_landmark_asym=0.55,
    min_blur_var=15.0,
    good_face_px=80.0,
    good_blur_var=100.0,
)


def _face(width=60.0, det=0.9, yaw=0.0, pitch=0.0, roll=0.0, kps=None):
    return {
        "box": [10.0, 10.0, 10.0 + width, 10.0 + width],
        "confidence": det,
        "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
        "kps": kps,
    }


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------
def test_pose_is_read_from_the_pose_dict():
    yaw, pitch, roll = _pose_from_face(_face(yaw=12.0, pitch=-4.0, roll=1.5))
    assert (yaw, pitch, roll) == (12.0, -4.0, 1.5)


def test_missing_pose_reads_as_zero_not_as_an_error():
    assert _pose_from_face({}) == (0.0, 0.0, 0.0)
    assert _pose_from_face({"pose": None}) == (0.0, 0.0, 0.0)


def test_unparseable_pose_degrades_to_zero():
    assert _pose_from_face({"pose": {"yaw": "sideways"}}) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------
def test_small_face_is_rejected():
    result = assess(_face(width=20.0), limits=LIMITS)
    assert not result.ok
    assert result.reason == "face_too_small"


def test_upscaled_crop_is_measured_at_its_real_size():
    """A 3x-zoomed 60px box is really a 20px face and must not pass."""
    result = assess(_face(width=60.0), limits=LIMITS, scale=3.0)
    assert not result.ok
    assert result.reason == "face_too_small"


def test_low_detector_confidence_is_rejected():
    result = assess(_face(det=0.2), limits=LIMITS)
    assert not result.ok
    assert result.reason == "low_det_score"


def test_extreme_yaw_is_rejected():
    """The gate that could never fire before the pose fix."""
    result = assess(_face(yaw=70.0), limits=LIMITS)
    assert not result.ok
    assert result.reason == "pose_yaw"


def test_extreme_pitch_is_rejected():
    """Employees looking down at monitors is the normal case on these cameras."""
    result = assess(_face(pitch=-60.0), limits=LIMITS)
    assert not result.ok
    assert result.reason == "pose_pitch"


def test_absent_pose_does_not_reject():
    """A detector that reports no pose must not have all its faces thrown away.

    The YOLO path has no 3D pose model; treating "unknown" as "extreme" would
    disable that backend entirely.
    """
    face = _face()
    face["pose"] = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    assert assess(face, limits=LIMITS).ok


def test_profile_landmarks_are_rejected():
    # Nose sitting on top of the right eye = full profile.
    kps = [[0.0, 0.0], [40.0, 0.0], [40.0, 10.0]]
    result = assess(_face(kps=kps), limits=LIMITS)
    assert not result.ok
    assert result.reason == "landmark_asym"


def test_centred_landmarks_pass():
    kps = [[0.0, 0.0], [40.0, 0.0], [20.0, 10.0]]
    assert assess(_face(kps=kps), limits=LIMITS).ok


def test_missing_landmarks_do_not_reject():
    assert assess(_face(kps=None), limits=LIMITS).ok


def test_blur_is_measured_on_the_face_crop_only():
    """A flat (blurred) face inside a sharp frame must still be rejected.

    Whole-frame blur was what the CCTV path used, and a frame is 'sharp' because
    of the door frame and floor tiles regardless of the face in it.
    """
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    # Sharp noise everywhere EXCEPT the face box.
    rng = np.random.default_rng(0)
    rgb[:, :] = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    rgb[10:70, 10:70] = 128         # the face region is flat -> blurry

    result = assess(_face(width=60.0), rgb, limits=LIMITS)
    assert not result.ok
    assert result.reason == "blurry"


def test_missing_image_skips_the_blur_term_rather_than_guessing():
    result = assess(_face(), None, limits=LIMITS)
    assert result.ok
    assert result.blur_var == 0.0


# ---------------------------------------------------------------------------
# Soft score
# ---------------------------------------------------------------------------
def test_score_is_zero_when_rejected():
    assert assess(_face(width=5.0), limits=LIMITS).score == 0.0


def test_bigger_and_more_frontal_scores_higher():
    small_turned = assess(_face(width=35.0, yaw=30.0), limits=LIMITS)
    large_frontal = assess(_face(width=78.0, yaw=0.0), limits=LIMITS)
    assert large_frontal.score > small_turned.score


def test_score_stays_within_unit_range():
    for width in (31.0, 60.0, 400.0):
        for yaw in (0.0, 20.0, 39.0):
            score = assess(_face(width=width, yaw=yaw), limits=LIMITS).score
            assert 0.0 < score <= 1.0


def test_one_bad_axis_drags_the_whole_score_down():
    """Geometric, not arithmetic: a large sharp near-profile is NOT 'good'.

    Averaging would let size compensate for pose, which is exactly the
    observation whose embedding cannot be trusted.
    """
    big_frontal = assess(_face(width=200.0, yaw=0.0), limits=LIMITS).score
    big_turned = assess(_face(width=200.0, yaw=39.0), limits=LIMITS).score
    assert big_turned < big_frontal * 0.5


@pytest.mark.parametrize("bad", [None, {}, {"box": [1, 2]}])
def test_unusable_input_is_reported_not_raised(bad):
    result = assess(bad, limits=LIMITS)
    assert not result.ok
    assert result.score == 0.0


def test_face_px_width_divides_out_the_scale():
    assert face_px_width({"box": [0, 0, 90, 90]}, scale=3.0) == 30.0
    assert face_px_width({"box": [0, 0, 90, 90]}) == 90.0
    assert face_px_width({}) == 0.0


def test_landmark_asymmetry_is_bounded():
    """The old metric was max/min, which is unbounded and untunable."""
    for kps in (
        [[0, 0], [40, 0], [20, 5]],
        [[0, 0], [40, 0], [39, 5]],
        [[0, 0], [40, 0], [0, 5]],
    ):
        assert 0.0 <= _landmark_asymmetry(kps) <= 1.0
