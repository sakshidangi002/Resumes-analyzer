"""The detection funnel — Fix 1 from the September detection investigation.

That investigation found three ways a person can be lost between YOLO emitting
a box and a track existing, and could only measure ONE of them:

  * dropped by the 30 px size floor  — a bare `continue`, no log, no counter
  * dropped by the 0.35 adopt bar    — measurable only by grepping a log line
  * never detected at all            — needs ground truth

The first two are now counted, with the box height and the position in frame,
because "the detector never saw them" and "we discarded them" are
indistinguishable from outside and need opposite fixes.

These tests pin two properties above all:

  * the funnel BALANCES — outcomes are exhaustive and sum to the detection
    count, so a code path that drops a detection without reporting itself shows
    up as a mismatch;
  * counting NEVER changes what the pipeline does with a detection.
"""
import pytest

from app.services.pipeline_metrics import (
    FUNNEL_OUTCOMES,
    PipelineMetrics,
    _box_bucket,
    _pos_bucket,
)


@pytest.fixture
def m():
    return PipelineMetrics()


# ---------------------------------------------------------------------------
# Buckets sit on the system's own thresholds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("px,bucket", [
    (7.0, "<30"),        # the 10x7 decode-corruption fragments on camera 60
    (29.9, "<30"),       # just under _MIN_PERSON_PX
    (30.0, "30-46"),     # exactly the size floor
    (45.9, "30-46"),     # the band where the floor is closest to a real person
    (46.0, "46-80"),     # the smallest hand-labelled real person
    (88.0, "80-150"),
    (300.0, ">=300"),
])
def test_box_height_buckets(px, bucket):
    assert _box_bucket(px) == bucket


@pytest.mark.parametrize("frac,bucket", [
    (0.0, "y0.0-0.2"),   # top of frame = far door on a corridor camera
    (0.19, "y0.0-0.2"),
    (0.5, "y0.4-0.6"),
    (0.99, "y0.8-1.0"),
    (1.0, "y0.8-1.0"),   # clamped, not an IndexError
    (1.7, "y0.8-1.0"),   # out of range must not raise
    (-0.3, "y0.0-0.2"),
])
def test_position_buckets(frac, bucket):
    assert _pos_bucket(frac) == bucket


# ---------------------------------------------------------------------------
# The funnel balances
# ---------------------------------------------------------------------------
def test_outcomes_sum_to_the_detection_count(m):
    """`detections` is derived, not counted separately, on purpose.

    A mismatch means a code path discarded a detection without reporting
    itself — exactly the bug class this funnel exists to expose.
    """
    m.record_detection(57, "tracked_by_id", 120.0, 0.5)
    m.record_detection(57, "adopted", 90.0, 0.4)
    m.record_detection(57, "dropped_adopt_bar", 55.0, 0.2)
    m.record_detection(57, "dropped_size_floor", 12.0, 0.1)

    totals = m.snapshot()["detection_funnel_total"]
    assert totals["detections"] == 4
    assert sum(totals[o] for o in FUNNEL_OUTCOMES if o in totals) == 4


def test_dropped_share_is_reported(m):
    """The headline number: what fraction of detections never became tracks."""
    for _ in range(68):
        m.record_detection(57, "tracked_by_id", 120.0, 0.5)
    for _ in range(32):
        m.record_detection(57, "dropped_adopt_bar", 55.0, 0.2)

    totals = m.snapshot()["detection_funnel_total"]
    assert totals["detections"] == 100
    assert totals["dropped_total"] == 32
    assert totals["dropped_pct"] == pytest.approx(32.0)


def test_no_detections_does_not_divide_by_zero(m):
    totals = m.snapshot()["detection_funnel_total"]
    assert totals["detections"] == 0
    assert totals["dropped_pct"] == 0.0


# ---------------------------------------------------------------------------
# The questions the investigation could not answer
# ---------------------------------------------------------------------------
def test_the_size_floor_becomes_visible(m):
    """Was previously a bare `continue`: unknowable from outside.

    The <30 bucket can ONLY be reached through the size-floor rejection path,
    so if it stays empty in production the floor is costing nothing.
    """
    m.record_detection(58, "dropped_size_floor", 12.0, 0.1)
    m.record_detection(58, "dropped_size_floor", 7.0, 0.05)

    snap = m.snapshot()
    assert snap["detection_funnel"]["58"]["dropped_size_floor"] == 2
    assert snap["detection_box_px"]["58"]["dropped_size_floor"]["<30"] == 2


def test_losses_are_separable_by_distance(m):
    """"Are we losing the DISTANT ones?" — previously unanswerable.

    Nothing recorded WHERE a lost detection had been, so a loss at the far door
    and a loss underfoot were the same number.
    """
    for _ in range(9):                       # far end of the corridor
        m.record_detection(57, "dropped_adopt_bar", 48.0, 0.1)
    for _ in range(1):                       # near the camera
        m.record_detection(57, "dropped_adopt_bar", 260.0, 0.9)

    pos = m.snapshot()["detection_position"]["57"]["dropped_adopt_bar"]
    assert pos["y0.0-0.2"] == 9
    assert pos["y0.8-1.0"] == 1


def test_the_two_drop_reasons_stay_distinct(m):
    """They need opposite fixes: more pixels vs a lower bar."""
    m.record_detection(57, "dropped_size_floor", 20.0, 0.3)
    m.record_detection(57, "dropped_adopt_bar", 70.0, 0.3)

    funnel = m.snapshot()["detection_funnel"]["57"]
    assert funnel["dropped_size_floor"] == 1
    assert funnel["dropped_adopt_bar"] == 1


def test_cameras_are_counted_separately(m):
    m.record_detection(57, "dropped_adopt_bar", 50.0, 0.2)
    m.record_detection(59, "tracked_by_id", 200.0, 0.6)

    snap = m.snapshot()
    assert "dropped_adopt_bar" in snap["detection_funnel"]["57"]
    assert "tracked_by_id" in snap["detection_funnel"]["59"]


# ---------------------------------------------------------------------------
# Robustness — a counter must never break the pipeline
# ---------------------------------------------------------------------------
def test_missing_height_and_position_still_record_the_outcome(m):
    """A caller that cannot compute them must still be able to report."""
    m.record_detection(57, "adopted", None, None)

    snap = m.snapshot()
    assert snap["detection_funnel"]["57"]["adopted"] == 1
    assert snap["detection_box_px"] == {}
    assert snap["detection_position"] == {}


def test_unparseable_values_do_not_lose_the_outcome(m):
    m.record_detection(57, "adopted", "tall", "middle")

    assert m.snapshot()["detection_funnel"]["57"]["adopted"] == 1


def test_buckets_read_in_order_not_hash_order(m):
    for px in (400.0, 10.0, 100.0):
        m.record_detection(57, "tracked_by_id", px, 0.5)

    buckets = list(m.snapshot()["detection_box_px"]["57"]["tracked_by_id"])
    assert buckets == ["<30", "80-150", ">=300"]


def test_key_space_cannot_grow_with_traffic(m):
    """Bounded by construction: cameras x outcomes x buckets."""
    for _ in range(5000):
        m.record_detection(57, "dropped_adopt_bar", 50.0, 0.3)

    snap = m.snapshot()
    assert len(snap["detection_funnel"]["57"]) == 1
    assert len(snap["detection_box_px"]["57"]["dropped_adopt_bar"]) == 1


def test_funnel_is_thread_safe(m):
    """Written from four camera threads concurrently."""
    import threading

    def hammer():
        for _ in range(400):
            m.record_detection(57, "tracked_by_id", 100.0, 0.5)

    threads = [threading.Thread(target=hammer) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert m.snapshot()["detection_funnel_total"]["detections"] == 2000


def test_reset_clears_the_funnel(m):
    m.record_detection(57, "adopted", 90.0, 0.4)
    m.reset()
    assert m.snapshot()["detection_funnel"] == {}


# ---------------------------------------------------------------------------
# Wiring — the engine reports every outcome, and counting is non-fatal
# ---------------------------------------------------------------------------
def test_engine_reports_all_four_outcomes():
    """Each FUNNEL_OUTCOME must have a call site, or the funnel cannot balance."""
    pytest.importorskip("ultralytics", reason="bytetrack_engine imports ultralytics")
    import inspect

    from app.services import bytetrack_engine as bt

    source = inspect.getsource(bt.ByteTrackEngine.update)
    for outcome in FUNNEL_OUTCOMES:
        assert f'"{outcome}"' in source, f"no call site reports {outcome}"


def test_a_metrics_failure_cannot_drop_a_detection(monkeypatch):
    """_count_detection is wrapped: a broken counter must never affect the
    pipeline's handling of a detection."""
    pytest.importorskip("ultralytics", reason="bytetrack_engine imports ultralytics")
    from app.services import bytetrack_engine as bt
    from app.services import pipeline_metrics

    def _explode(*_a, **_kw):
        raise RuntimeError("counter is wedged")

    monkeypatch.setattr(pipeline_metrics.metrics, "record_detection", _explode)

    # Must not raise.
    bt._count_detection("57", "adopted", 90.0, 0.4)
