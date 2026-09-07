"""Landmark asymmetry must be invariant to head ROLL.

Camera 58 is mounted at a steep angle and delivers faces rolled -7 to -44
degrees. The old metric compared nose-to-eye distances along the IMAGE x-axis,
which conflates roll with yaw: rotating a dead-on face in the image plane made
it read as "turned to the side" and it was rejected as `landmark_asym`.
"""
from __future__ import annotations

import math

import pytest

from app.services.face_quality import _landmark_asymmetry


def rot(points, deg, cx=0.0, cy=0.0):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return [((x - cx) * c - (y - cy) * s + cx,
             (x - cx) * s + (y - cy) * c + cy) for x, y in points]


# left_eye, right_eye, nose — nose exactly centred: a dead-on frontal face
FRONTAL = [(-20.0, 0.0), (20.0, 0.0), (0.0, 15.0)]
# nose pushed toward the right eye: a genuine side turn
TURNED = [(-20.0, 0.0), (20.0, 0.0), (14.0, 15.0)]


@pytest.mark.parametrize("roll", [0, -7, -15.8, -22.8, -32.9, -34.6, -43.9, 30, 60])
def test_frontal_face_stays_symmetric_under_any_roll(roll):
    """The exact rolls measured on camera 58, plus extremes."""
    asym = _landmark_asymmetry(rot(FRONTAL, roll))
    assert asym < 0.05, f"roll {roll} made a frontal face read as asym={asym:.3f}"


@pytest.mark.parametrize("roll", [0, -20, -43.9, 35])
def test_genuine_turn_is_still_detected_under_roll(roll):
    """Roll invariance must not blind the metric to a real side-turn."""
    asym = _landmark_asymmetry(rot(TURNED, roll))
    assert asym > 0.4, f"a real turn read as only asym={asym:.3f} at roll {roll}"


def test_roll_does_not_change_the_measurement():
    """Same face, different roll -> same asymmetry (that is the whole point)."""
    base = _landmark_asymmetry(TURNED)
    for roll in (-43.9, -22.8, 17.0, 45.0):
        assert _landmark_asymmetry(rot(TURNED, roll)) == pytest.approx(base, abs=0.02)


def test_regression_camera58_rolled_frontal_face_is_not_rejected():
    """The concrete failure: a frontal face at -43.9 deg scored 0.902 (limit 0.55)."""
    assert _landmark_asymmetry(rot(FRONTAL, -43.9)) <= 0.55


def test_missing_or_short_landmarks_are_not_penalised():
    assert _landmark_asymmetry(None) == 0.0
    assert _landmark_asymmetry([]) == 0.0
    assert _landmark_asymmetry([(0.0, 0.0), (1.0, 1.0)]) == 0.0


def test_degenerate_eyes_do_not_crash():
    assert _landmark_asymmetry([(5.0, 5.0), (5.0, 5.0), (5.0, 9.0)]) == 1.0


def test_malformed_landmarks_do_not_raise():
    assert _landmark_asymmetry([("a", "b"), (1.0, 1.0), (2.0, 2.0)]) == 0.0
