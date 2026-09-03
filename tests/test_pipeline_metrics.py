"""Pipeline outcome counters and the inference percentiles behind /api/metrics.

Every number in the September 2026 audit was obtained by grepping 110 MB of
log text. These tests pin the counters that replace that, and in particular the
three things a mean could never show:

  * where attendance decisions actually die (reason counts),
  * how big the faces really are (histogram),
  * what share of a camera's cycle is spent QUEUEING rather than working.

Pure unit tests: no cameras, no database, no OpenCV.
"""
import pytest

from app.services import pipeline_metrics
from app.services.pipeline_metrics import PipelineMetrics, _bucket_for


@pytest.fixture
def m():
    return PipelineMetrics()


# ---------------------------------------------------------------------------
# Decision counters
# ---------------------------------------------------------------------------
def test_decisions_are_counted_per_camera_and_reason(m):
    m.record_decision(57, reason="insufficient_observations", allowed=False)
    m.record_decision(57, reason="insufficient_observations", allowed=False)
    m.record_decision(57, reason="allowed", allowed=True)
    m.record_decision(58, reason="no_match", allowed=False)

    snap = m.snapshot()
    assert snap["decisions"]["57"]["insufficient_observations"] == 2
    assert snap["decisions"]["57"]["allowed"] == 1
    assert snap["decisions"]["58"]["no_match"] == 1


def test_per_camera_counts_are_not_averaged_away(m):
    """A single failing camera must stay visible.

    The audit's central finding was that per-camera behaviour diverges sharply.
    A global-only total would hide exactly the camera that needs attention.
    """
    for _ in range(50):
        m.record_decision(58, reason="low_quality", allowed=False)
    for _ in range(2):
        m.record_decision(57, reason="low_quality", allowed=False)

    snap = m.snapshot()
    assert snap["decisions"]["58"]["low_quality"] == 50
    assert snap["decisions"]["57"]["low_quality"] == 2
    assert snap["decisions_total"]["low_quality"] == 52


def test_camera_id_type_does_not_split_a_counter(m):
    """Camera ids arrive as int from the worker and str from elsewhere."""
    m.record_decision(57, reason="allowed", allowed=True)
    m.record_decision("57", reason="allowed", allowed=True)

    assert m.snapshot()["decisions"]["57"]["allowed"] == 2


def test_a_missing_reason_still_counts(m):
    """An allowed decision carries reason=None in some paths."""
    m.record_decision(57, reason=None, allowed=True)
    m.record_decision(57, reason="", allowed=False)

    snap = m.snapshot()
    assert snap["decisions"]["57"]["allowed"] == 1
    assert snap["decisions"]["57"]["unknown"] == 1


# ---------------------------------------------------------------------------
# Face size histogram — the optics signal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("px,bucket", [
    (12.0, "<28"),        # junk detection
    (27.9, "<28"),        # below the quality floor
    (28.0, "28-40"),      # exactly the floor
    (33.0, "28-40"),      # the measured production median
    (46.0, "40-55"),      # the measured p90
    (70.0, "70-112"),     # exactly good_face_px
    (111.9, "70-112"),
    (112.0, ">=112"),     # ArcFace native
    (400.0, ">=112"),
])
def test_face_px_buckets_sit_on_the_systems_own_thresholds(px, bucket):
    assert _bucket_for(px) == bucket


def test_face_sizes_are_histogrammed(m):
    for px in (31.0, 33.0, 35.0, 60.0, 120.0):
        m.record_decision(57, reason="no_match", allowed=False, face_px=px)

    hist = m.snapshot()["face_px_histogram"]["57"]
    assert hist["28-40"] == 3
    assert hist["55-70"] == 1
    assert hist[">=112"] == 1


def test_histogram_reads_in_bucket_order_not_hash_order(m):
    """Reading this by eye in a JSON response is the primary use."""
    m.record_decision(57, reason="x", allowed=False, face_px=200.0)
    m.record_decision(57, reason="x", allowed=False, face_px=10.0)
    m.record_decision(57, reason="x", allowed=False, face_px=50.0)

    assert list(m.snapshot()["face_px_histogram"]["57"]) == ["<28", "40-55", ">=112"]


def test_a_decision_without_a_face_size_is_still_counted(m):
    """Not every rejection reaches the point of measuring a face."""
    m.record_decision(57, reason="role", allowed=False, face_px=None)

    snap = m.snapshot()
    assert snap["decisions"]["57"]["role"] == 1
    assert snap["face_px_histogram"] == {}


def test_an_unparseable_face_size_does_not_lose_the_decision(m):
    m.record_decision(57, reason="no_match", allowed=False, face_px="not-a-number")

    assert m.snapshot()["decisions"]["57"]["no_match"] == 1


# ---------------------------------------------------------------------------
# Attendance writes
# ---------------------------------------------------------------------------
def test_attendance_write_outcomes_are_counted(m):
    m.record_attendance_write(57, "check_in")
    m.record_attendance_write(57, "check_in")
    m.record_attendance_write(58, "lost")

    snap = m.snapshot()
    assert snap["attendance_writes"]["57"]["check_in"] == 2
    assert snap["attendance_writes_total"]["lost"] == 1


def test_lost_writes_are_visible_as_their_own_counter(m):
    """Every 'lost' is a payroll event needing manual entry — it must not be
    folded into a generic failure bucket."""
    m.record_attendance_write(57, "lost")
    m.record_attendance_write(57, "dropped_queue_overflow")
    m.record_attendance_write(57, "retried")

    totals = m.snapshot()["attendance_writes_total"]
    assert totals["lost"] == 1
    assert totals["dropped_queue_overflow"] == 1
    assert totals["retried"] == 1


# ---------------------------------------------------------------------------
# Shape / safety
# ---------------------------------------------------------------------------
def test_empty_snapshot_has_every_key():
    """The endpoint must not KeyError before the first camera pass."""
    snap = PipelineMetrics().snapshot()
    for key in (
        "decisions", "decisions_total",
        "face_px_histogram", "face_px_total",
        "attendance_writes", "attendance_writes_total",
    ):
        assert key in snap
        assert snap[key] == {}


def test_counters_are_thread_safe(m):
    """These are written from four camera threads plus the writer pool."""
    import threading

    def hammer():
        for _ in range(500):
            m.record_decision(57, reason="no_match", allowed=False, face_px=33.0)

    threads = [threading.Thread(target=hammer) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = m.snapshot()
    assert snap["decisions"]["57"]["no_match"] == 3000
    assert snap["face_px_histogram"]["57"]["28-40"] == 3000


def test_the_key_space_cannot_grow_with_traffic(m):
    """Bounded by construction: cameras x reasons, both small and fixed.

    A metrics store that grows with traffic becomes the leak it exists to find.
    """
    for i in range(5000):
        m.record_decision(57, reason="no_match", allowed=False, face_px=33.0)

    snap = m.snapshot()
    assert len(snap["decisions"]["57"]) == 1
    assert len(snap["face_px_histogram"]["57"]) == 1


def test_module_exposes_a_singleton():
    assert isinstance(pipeline_metrics.metrics, PipelineMetrics)
