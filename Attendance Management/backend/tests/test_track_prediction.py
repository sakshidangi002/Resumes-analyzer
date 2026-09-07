"""Tests for display-only bounding-box extrapolation.

These drive BoxPredictor through a fake clock so the timing cases are exact and
the suite does not sleep. The fake PersonTrack mirrors only the three attributes
the predictor actually reads (`track_id`, `box`, `last_seen`) plus a field that
must survive the shallow copy.
"""
from __future__ import annotations

import copy

import pytest

from app.services import track_prediction as tp
from app.services.track_prediction import BoxPredictor


FRAME = (1080, 960)          # (h, w), matching the real camera geometry


class FakeTrack:
    """Stand-in for PersonTrack: same attribute names the predictor reads."""

    def __init__(self, track_id, box, last_seen, employee_name="Person"):
        self.track_id = track_id
        self.box = box
        self.last_seen = last_seen
        self.employee_name = employee_name
        self.matched = False

    def get_display_info(self):
        return {"track_id": self.track_id, "box": self.box,
                "employee_name": self.employee_name, "matched": self.matched}


@pytest.fixture
def clock(monkeypatch):
    """Deterministic monotonic clock."""
    state = {"t": 1000.0}
    monkeypatch.setattr(tp.time, "monotonic", lambda: state["t"])
    return state


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    """Pin tuning constants so a future .env change cannot break these tests."""
    monkeypatch.setattr(tp, "ENABLED", True)
    monkeypatch.setattr(tp, "BLEND", 1.0)          # snap: assert position, not easing
    monkeypatch.setattr(tp, "VEL_SMOOTH", 1.0)     # no velocity lag in assertions
    monkeypatch.setattr(tp, "MAX_EXTRAPOLATION_SEC", 1.0)
    monkeypatch.setattr(tp, "HIDE_STALE_SEC", 0.0)
    monkeypatch.setattr(tp, "MIN_SPEED_FRAC", 0.01)
    monkeypatch.setattr(tp, "MAX_SPEED_FRAC", 1.5)


def cx(box):
    return (box[0] + box[2]) / 2.0


def cy(box):
    return (box[1] + box[3]) / 2.0


def feed(pred, track, clock, dt):
    """Advance the clock and run one display tick."""
    clock["t"] += dt
    return pred.predict([track], FRAME)[0]


# --------------------------------------------------------------------------
# Test 1 / 2: a walker's box moves the way the walker is moving.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("direction,sign", [("right", +1), ("left", -1)])
def test_box_follows_walker(clock, direction, sign):
    pred = BoxPredictor("58")
    box = (400, 300, 500, 700)
    tr = FakeTrack(1, box, last_seen=1.0)
    pred.predict([tr], FRAME)                       # first sighting: anchor only

    # Second measurement 0.5s later, 100px along.
    clock["t"] += 0.5
    tr.box = (400 + sign * 100, 300, 500 + sign * 100, 700)
    tr.last_seen = 2.0
    drawn = pred.predict([tr], FRAME)[0]
    assert cx(drawn.box) == pytest.approx(cx(tr.box))   # measurement wins

    # No new detection; 0.25s of display ticks must carry the box onward.
    clock["t"] += 0.25
    drawn = pred.predict([tr], FRAME)[0]
    moved = cx(drawn.box) - cx(tr.box)
    assert sign * moved > 0, f"box should drift {direction}, moved {moved}"
    assert abs(moved) == pytest.approx(50.0, abs=2.0)   # 200 px/s * 0.25s


# --------------------------------------------------------------------------
# Test 3: a stationary person's box must not creep.
# --------------------------------------------------------------------------
def test_stationary_box_stays_put(clock):
    pred = BoxPredictor("59", is_monitor=True)
    box = (400, 300, 500, 700)
    tr = FakeTrack(7, box, last_seen=1.0)
    pred.predict([tr], FRAME)

    # Two measurements with a few px of detector jitter — under the deadband.
    for i, jitter in enumerate((3, -2), start=2):
        clock["t"] += 0.5
        tr.box = (400 + jitter, 300, 500 + jitter, 700)
        tr.last_seen = float(i)
        pred.predict([tr], FRAME)

    start = cx(pred.predict([tr], FRAME)[0].box)
    for _ in range(4):                       # half a second of display ticks
        clock["t"] += 0.125
        drawn = pred.predict([tr], FRAME)[0]
    assert cx(drawn.box) == pytest.approx(start, abs=1.0)


# --------------------------------------------------------------------------
# Test 4 / 5: predict inside the cap, stop extrapolating past it.
# --------------------------------------------------------------------------
def test_predicts_within_cap_and_stops_after(clock):
    pred = BoxPredictor("58")
    tr = FakeTrack(2, (100, 300, 200, 700), last_seen=1.0)
    pred.predict([tr], FRAME)
    clock["t"] += 0.5
    tr.box = (200, 300, 300, 700)            # +100px in 0.5s -> 200 px/s
    tr.last_seen = 2.0
    measured_cx = cx(pred.predict([tr], FRAME)[0].box)

    clock["t"] += 0.5                        # 0.5s stale: inside the 1.0s cap
    assert cx(pred.predict([tr], FRAME)[0].box) == pytest.approx(
        measured_cx + 100.0, abs=2.0)

    clock["t"] += 1.0                        # 1.5s stale: past the cap
    capped = cx(pred.predict([tr], FRAME)[0].box)
    assert capped == pytest.approx(measured_cx, abs=1.0), (
        "past the cap the box must revert to the MEASURED position, not keep "
        "marching down an empty corridor")


def test_hide_stale_is_opt_in(clock, monkeypatch):
    """With HIDE_STALE_SEC set, a long-unmeasured box is dropped entirely."""
    monkeypatch.setattr(tp, "HIDE_STALE_SEC", 2.0)
    pred = BoxPredictor("58")
    tr = FakeTrack(3, (100, 300, 200, 700), last_seen=1.0)
    pred.predict([tr], FRAME)
    clock["t"] += 1.0
    assert len(pred.predict([tr], FRAME)) == 1
    clock["t"] += 2.0
    assert pred.predict([tr], FRAME) == []


# --------------------------------------------------------------------------
# Test 6: a real detection always overrides the prediction.
# --------------------------------------------------------------------------
def test_new_detection_corrects_prediction(clock):
    pred = BoxPredictor("58")
    tr = FakeTrack(4, (100, 300, 200, 700), last_seen=1.0)
    pred.predict([tr], FRAME)
    clock["t"] += 0.5
    tr.box = (200, 300, 300, 700)
    tr.last_seen = 2.0
    pred.predict([tr], FRAME)

    clock["t"] += 0.5                        # drifts to ~cx 350
    assert cx(pred.predict([tr], FRAME)[0].box) == pytest.approx(350.0, abs=2.0)

    # The person actually stopped: the new measurement disagrees with the drift.
    tr.box = (210, 300, 310, 700)
    tr.last_seen = 3.0
    drawn = pred.predict([tr], FRAME)[0]
    assert cx(drawn.box) == pytest.approx(260.0, abs=2.0), (
        "the measured position must win over the extrapolated one")


# --------------------------------------------------------------------------
# Test 7: velocity re-fits when the walker turns around.
# --------------------------------------------------------------------------
def test_direction_change_updates_velocity(clock):
    pred = BoxPredictor("58")
    tr = FakeTrack(5, (400, 300, 500, 700), last_seen=1.0)
    pred.predict([tr], FRAME)
    for i, x in enumerate((500, 600), start=2):     # walking right
        clock["t"] += 0.5
        tr.box = (x, 300, x + 100, 700)
        tr.last_seen = float(i)
        pred.predict([tr], FRAME)

    # Now walk back left for three measurements.
    for i, x in enumerate((500, 400, 300), start=4):
        clock["t"] += 0.5
        tr.box = (x, 300, x + 100, 700)
        tr.last_seen = float(i)
        drawn = pred.predict([tr], FRAME)[0]
    anchor = cx(drawn.box)
    clock["t"] += 0.25
    assert cx(pred.predict([tr], FRAME)[0].box) < anchor, (
        "after turning, the box must drift LEFT, not continue right")


# --------------------------------------------------------------------------
# Test 8: two people get independent prediction state.
# --------------------------------------------------------------------------
def test_two_tracks_are_independent(clock):
    pred = BoxPredictor("58")
    a = FakeTrack(10, (100, 300, 200, 700), last_seen=1.0)   # moves right
    b = FakeTrack(11, (700, 300, 800, 700), last_seen=1.0)   # stands still
    pred.predict([a, b], FRAME)
    clock["t"] += 0.5
    a.box, a.last_seen = (200, 300, 300, 700), 2.0
    b.last_seen = 2.0
    pred.predict([a, b], FRAME)

    clock["t"] += 0.25
    da, db = pred.predict([a, b], FRAME)
    assert cx(da.box) > cx(a.box), "the walker's box should advance"
    assert cx(db.box) == pytest.approx(cx(b.box), abs=1.0), (
        "the stationary person's box must not inherit the walker's velocity")


# --------------------------------------------------------------------------
# Safety: the real track object is never mutated, and bad input never raises.
# --------------------------------------------------------------------------
def test_original_track_is_never_mutated(clock):
    """The attendance/crossing path reads PersonTrack.box; it must stay real."""
    pred = BoxPredictor("58")
    tr = FakeTrack(6, (100, 300, 200, 700), last_seen=1.0)
    pred.predict([tr], FRAME)
    clock["t"] += 0.5
    tr.box = (200, 300, 300, 700)
    tr.last_seen = 2.0
    pred.predict([tr], FRAME)
    clock["t"] += 0.5
    before = copy.deepcopy(tr.box)
    drawn = pred.predict([tr], FRAME)[0]

    assert tr.box == before, "the measured box was mutated"
    assert drawn is not tr, "a moved box must be drawn from a COPY"
    assert drawn.box != tr.box
    assert drawn.employee_name == tr.employee_name   # identity survives the copy


def test_absurd_velocity_is_clamped(clock):
    """An id swap can imply a teleport; the box must not be flung across frame."""
    pred = BoxPredictor("58")
    tr = FakeTrack(8, (0, 300, 100, 700), last_seen=1.0)
    pred.predict([tr], FRAME)
    clock["t"] += 0.05
    tr.box = (850, 300, 950, 700)            # ~17000 px/s implied
    tr.last_seen = 2.0
    pred.predict([tr], FRAME)
    clock["t"] += 0.5
    drawn = pred.predict([tr], FRAME)[0]
    assert 0 <= drawn.box[0] and drawn.box[2] <= FRAME[1], "box left the frame"


@pytest.mark.parametrize("bad", [
    None,
    [],
    [FakeTrack(1, (), 1.0)],                 # empty box
    [FakeTrack(None, (1, 2, 3, 4), 1.0)],    # no track id
])
def test_bad_input_never_raises(clock, bad):
    pred = BoxPredictor("58")
    pred.predict(bad, FRAME)
    pred.predict(bad, None)                  # frame_shape unusable too


def test_disabled_returns_tracks_untouched(clock, monkeypatch):
    monkeypatch.setattr(tp, "ENABLED", False)
    pred = BoxPredictor("58")
    tr = FakeTrack(9, (100, 300, 200, 700), last_seen=1.0)
    assert pred.predict([tr], FRAME)[0] is tr


def test_recycled_track_id_does_not_inherit_velocity(clock):
    """Adopted ids (1000000+) are minted fresh; a gap means a NEW person."""
    pred = BoxPredictor("58")
    tr = FakeTrack(1000001, (100, 300, 200, 700), last_seen=1.0)
    pred.predict([tr], FRAME)
    clock["t"] += 0.5
    tr.box, tr.last_seen = (300, 300, 400, 700), 2.0
    pred.predict([tr], FRAME)

    clock["t"] += tp.STALE_STATE_SEC + 1.0   # id disappears, then comes back
    tr.box, tr.last_seen = (700, 300, 800, 700), 3.0
    drawn = pred.predict([tr], FRAME)[0]
    assert drawn.box == tr.box, "a recycled id must start from a clean state"


# --------------------------------------------------------------------------
# Latency compensation: the box is drawn on the LIVE frame, but was measured
# on a frame ~1.8s older (camera 58). Without compensating that offset the box
# trails the person permanently, however well velocity is estimated.
# --------------------------------------------------------------------------
@pytest.fixture
def wallclock(monkeypatch):
    state = {"t": 5000.0}
    monkeypatch.setattr(tp.time, "time", lambda: state["t"])
    return state


def test_latency_compensation_advances_box(clock, wallclock):
    """A measurement that is already 1.0s old must be projected forward.

    The pipeline lag is roughly constant pass to pass (it is dominated by the
    inference time), so BOTH measurements here carry the same 1.0s lag. Giving
    only the second one a lag would collapse the capture-time delta to zero and
    make the velocity unmeasurable -- a test artefact, not a real condition.
    """
    LAG = 1.0
    pred = BoxPredictor("58")
    tr = FakeTrack(20, (100, 300, 200, 700), last_seen=1.0)      # cx 150
    pred.predict([tr], FRAME, measured_at=wallclock["t"] - LAG)

    # Second measurement 1.0s later, 200px along -> 200 px/s.
    clock["t"] += 1.0
    wallclock["t"] += 1.0
    tr.box = (300, 300, 400, 700)                                # cx 350
    tr.last_seen = 2.0
    drawn = pred.predict([tr], FRAME, measured_at=wallclock["t"] - LAG)[0]

    # The measured centre is 350, but that frame was captured 1.0s ago and the
    # person kept walking: 350 + 200*1.0 = 550.
    assert cx(drawn.box) == pytest.approx(550.0, abs=15.0), (
        "the box must be projected forward by the measurement lag, not drawn "
        "at the stale measured position")


def test_zero_measured_at_disables_compensation(clock, wallclock):
    pred = BoxPredictor("58")
    tr = FakeTrack(21, (100, 300, 200, 700), last_seen=1.0)
    pred.predict([tr], FRAME, measured_at=0.0)
    clock["t"] += 0.5
    tr.box, tr.last_seen = (200, 300, 300, 700), 2.0
    drawn = pred.predict([tr], FRAME, measured_at=0.0)[0]
    assert cx(drawn.box) == pytest.approx(cx(tr.box), abs=1.0)


def test_absurd_lag_is_ignored(clock, wallclock):
    """A clock step or stale timestamp must not fling the box across frame."""
    pred = BoxPredictor("58")
    tr = FakeTrack(22, (100, 300, 200, 700), last_seen=1.0)
    pred.predict([tr], FRAME, measured_at=wallclock["t"])
    clock["t"] += 0.5
    tr.box, tr.last_seen = (200, 300, 300, 700), 2.0
    # measured_at claims the frame was captured an hour ago.
    drawn = pred.predict([tr], FRAME, measured_at=wallclock["t"] - 3600.0)[0]
    assert 0 <= drawn.box[0] and drawn.box[2] <= FRAME[1]


def test_out_of_order_pass_does_not_flip_velocity(clock, wallclock):
    """A late pass carrying an older capture time must not invert the heading."""
    pred = BoxPredictor("58")
    tr = FakeTrack(23, (100, 300, 200, 700), last_seen=1.0)
    pred.predict([tr], FRAME, measured_at=wallclock["t"])
    clock["t"] += 1.0; wallclock["t"] += 1.0
    tr.box, tr.last_seen = (300, 300, 400, 700), 2.0
    pred.predict([tr], FRAME, measured_at=wallclock["t"])
    before = cx(pred.predict([tr], FRAME, measured_at=wallclock["t"])[0].box)

    clock["t"] += 0.5; wallclock["t"] += 0.5
    tr.box, tr.last_seen = (400, 300, 500, 700), 3.0
    # This pass reports a capture time BEFORE the previous one.
    pred.predict([tr], FRAME, measured_at=wallclock["t"] - 2.0)
    clock["t"] += 0.25; wallclock["t"] += 0.25
    after = cx(pred.predict([tr], FRAME, measured_at=wallclock["t"] - 2.0)[0].box)
    assert after > before, "heading must stay rightward after an out-of-order pass"
