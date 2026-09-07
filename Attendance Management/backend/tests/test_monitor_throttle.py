"""Tests for MONITOR analysis throttling.

The throttle is a few lines inside `_RecognitionThread.run()`, which cannot be
driven in a test without a camera, a DVR and a YOLO model. So these exercise the
DECISION the loop makes, extracted here exactly as the loop performs it, plus a
guard asserting the real loop still contains that decision.

What is being protected:
  * a monitor camera may not analyse again before its interval has elapsed
  * cameras 59 and 60 hold INDEPENDENT schedules
  * a throttled monitor never gates a doorway camera
  * the interval is a FLOOR on the period, not a delay added to the work
  * 0 disables the throttle (previous free-running behaviour)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


class Gate:
    """The throttle decision from _RecognitionThread.run(), in isolation.

    Mirrors the real code: booked from the START of a pass, so the resulting
    period is max(interval, work) rather than interval + work.
    """

    def __init__(self, is_monitor: bool, interval: float):
        self.is_monitor = is_monitor
        self.interval = interval
        self.next_allowed = 0.0
        self.throttled = False

    def may_analyse(self, now: float) -> bool:
        if self.is_monitor and self.interval > 0:
            if now < self.next_allowed:
                self.throttled = True
                return False
            self.next_allowed = now + self.interval
            self.throttled = False
        return True


INTERVAL = 15.0


def test_monitor_can_analyse_when_due():
    g = Gate(True, INTERVAL)
    assert g.may_analyse(100.0) is True
    assert g.throttled is False


def test_monitor_blocked_before_interval_elapses():
    g = Gate(True, INTERVAL)
    g.may_analyse(100.0)
    for t in (100.1, 105.0, 110.0, 114.9):
        assert g.may_analyse(t) is False, f"should still be throttled at {t}"
        assert g.throttled is True


def test_monitor_eligible_again_after_interval():
    g = Gate(True, INTERVAL)
    g.may_analyse(100.0)
    assert g.may_analyse(114.9) is False
    assert g.may_analyse(115.0) is True
    assert g.throttled is False


def test_cameras_59_and_60_are_independent():
    c59, c60 = Gate(True, INTERVAL), Gate(True, INTERVAL)
    assert c59.may_analyse(100.0) is True
    # 60 starts its own schedule 7s later and must not inherit 59's.
    assert c60.may_analyse(107.0) is True
    assert c59.may_analyse(115.0) is True      # 59 due
    assert c60.may_analyse(115.0) is False     # 60 not due until 122
    assert c60.may_analyse(122.0) is True


def test_throttled_monitor_never_blocks_a_doorway_camera():
    """The doorway gate is independent state; a blocked monitor cannot gate it."""
    mon, door = Gate(True, INTERVAL), Gate(False, INTERVAL)
    mon.may_analyse(100.0)
    for t in (100.5, 103.0, 110.0, 114.0):
        assert mon.may_analyse(t) is False
        assert door.may_analyse(t) is True, "doorway must never be throttled"
    assert door.next_allowed == 0.0, "a doorway camera must not book slots"


def test_interval_is_a_floor_on_the_period_not_an_added_delay():
    """A pass slower than the interval must not push the schedule out by its own
    duration -- the next slot is booked from the START of the pass."""
    g = Gate(True, 15.0)
    g.may_analyse(100.0)          # pass starts at 100 and takes 20s (>interval)
    assert g.may_analyse(120.0) is True, (
        "work longer than the interval must leave the camera immediately "
        "eligible, giving period = max(interval, work)")


def test_interval_below_work_time_is_a_no_op():
    """Why 5s does nothing here: a room pass already costs ~6s."""
    g = Gate(True, 5.0)
    g.may_analyse(100.0)
    assert g.may_analyse(106.3) is True, (
        "with a 6.3s pass a 5s floor never binds -- this is measured, not "
        "hypothetical: cameras 59/60 ran at 9.08s and 8.03s periods")


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_zero_or_negative_disables_throttle(interval):
    g = Gate(True, interval)
    for t in (100.0, 100.1, 100.2):
        assert g.may_analyse(t) is True, "0 must restore free-running behaviour"


def test_doorway_camera_is_never_throttled_at_any_interval():
    g = Gate(False, 999.0)
    for t in range(0, 50, 5):
        assert g.may_analyse(float(t)) is True


def test_changing_the_interval_changes_the_behaviour():
    short, long = Gate(True, 2.0), Gate(True, 30.0)
    short.may_analyse(100.0)
    long.may_analyse(100.0)
    assert short.may_analyse(102.0) is True
    assert long.may_analyse(102.0) is False


# --------------------------------------------------------------------------
# Guard: the real loop must still contain this decision, sited before the work.
# --------------------------------------------------------------------------
SRC = (Path(__file__).resolve().parents[1]
       / "app" / "services" / "camera_service.py").read_text(encoding="utf-8")


def test_throttle_is_wired_into_the_recognition_loop():
    assert "_MONITOR_MIN_INTERVAL" in SRC
    assert "self._monitor_next_allowed" in SRC
    assert "w.is_monitor and _MONITOR_MIN_INTERVAL > 0" in SRC, (
        "the throttle must apply to MONITOR cameras only")


def test_throttle_does_not_sleep_inside_the_worker():
    """A blocking sleep in the shared inference path would worsen scheduling."""
    block = SRC[SRC.index("MONITOR analysis throttle"):][:2000]
    assert "time.sleep" not in block, (
        "the throttle must skip the pass via `continue` and let the loop's own "
        "wait handle timing -- never sleep inside the worker")
    assert "continue" in block


def test_throttle_runs_before_the_expensive_work():
    """It must gate BEFORE frame read / motion / inference, or it saves nothing."""
    i_throttle = SRC.index("MONITOR analysis throttle")
    i_analyse = SRC.index("_analyse(", i_throttle)
    assert i_throttle < i_analyse
    # and before the per-pass frame grab it guards
    assert i_throttle < SRC.index("frame_ts = w._latest_frame_ts", i_throttle)


def test_interval_is_configurable_not_hardcoded():
    assert 'os.getenv("CCTV_MONITOR_ANALYSIS_INTERVAL")' in SRC
    assert 'monitor_analysis_interval' in SRC, "must fall back to the setting"
    assert re.search(r"_MONITOR_MIN_INTERVAL\s*=\s*_monitor_min_interval\(\)", SRC)


# --------------------------------------------------------------------------
# Recognition cadence: YOLO/ByteTrack every pass, face stage rate-limited.
# --------------------------------------------------------------------------
class RecogGate:
    """The face-stage decision from _RecognitionThread.run(), in isolation.

    Measured from the END of the last face pass, so a slow face pass is followed
    by a full tracking window rather than being immediately re-triggered.
    """

    def __init__(self, is_monitor: bool, interval: float):
        self.is_monitor = is_monitor
        self.interval = interval
        self.last_end = 0.0

    def run_pass(self, start: float, face_cost: float, track_cost: float) -> bool:
        skip = False
        if not self.is_monitor and self.interval > 0:
            skip = (start - self.last_end) < self.interval
        end = start + (track_cost if skip else track_cost + face_cost)
        if not skip:
            self.last_end = end
        return not skip          # True = face stage ran


RECOG_INTERVAL = 5.0
FACE_COST, TRACK_COST = 6.0, 0.6


def test_face_stage_runs_on_the_first_pass():
    g = RecogGate(False, RECOG_INTERVAL)
    assert g.run_pass(100.0, FACE_COST, TRACK_COST) is True


def test_face_stage_is_skipped_during_the_interval():
    g = RecogGate(False, RECOG_INTERVAL)
    g.run_pass(100.0, FACE_COST, TRACK_COST)   # ends at 106.6
    for t in (106.6, 108.0, 110.0, 111.5):
        assert g.run_pass(t, FACE_COST, TRACK_COST) is False


def test_face_stage_resumes_after_the_interval():
    g = RecogGate(False, RECOG_INTERVAL)
    g.run_pass(100.0, FACE_COST, TRACK_COST)   # ends at 106.6
    assert g.run_pass(111.5, FACE_COST, TRACK_COST) is False   # 4.9s elapsed
    assert g.run_pass(111.6, FACE_COST, TRACK_COST) is True    # 5.0s elapsed


def test_tracking_runs_on_every_pass_including_skipped_ones():
    """The whole point: a skipped pass still costs track time, so YOLO ran."""
    g = RecogGate(False, RECOG_INTERVAL)
    g.run_pass(100.0, FACE_COST, TRACK_COST)
    t, n = 106.6, 0
    while t < 111.6:                     # the tracking window
        assert g.run_pass(t, FACE_COST, TRACK_COST) is False
        t += TRACK_COST; n += 1
    assert n >= 8, f"expected a run of consecutive tracking passes, got {n}"


def test_monitor_cameras_are_not_rate_limited_here():
    """Monitors rely on the face stage to name seated staff and are already
    throttled by _MONITOR_MIN_INTERVAL; their cadence must be untouched."""
    g = RecogGate(True, RECOG_INTERVAL)
    for t in (100.0, 100.1, 100.2):
        assert g.run_pass(t, FACE_COST, TRACK_COST) is True


def test_zero_disables_the_limit():
    g = RecogGate(False, 0.0)
    for t in (100.0, 100.1, 100.2):
        assert g.run_pass(t, FACE_COST, TRACK_COST) is True


def test_recognition_cadence_is_wired_in():
    assert "_RECOG_MIN_INTERVAL" in SRC
    assert "self._last_recog_end" in SRC
    assert "skip_faces=(blurry or _skip_recog)" in SRC, (
        "the face stage must be skipped via skip_faces, which leaves YOLO running"
    )
    assert "not w.is_monitor and _RECOG_MIN_INTERVAL > 0" in SRC, (
        "rate limiting applies to doorway cameras only"
    )
    assert 'os.getenv("CCTV_RECOG_MIN_INTERVAL_SEC"' in SRC
