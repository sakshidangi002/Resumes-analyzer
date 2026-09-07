"""Per-camera inference percentiles and queue-wait share.

`_record_perf` kept a running MEAN and overwrote `wait_ms` with the latest
value. Neither can express the two numbers the audit actually turned on:

  * the doorway p90 cycle (9.8s) against its p50 (3.5s) — a mean of 4.6s hides
    the tail completely, and it is the tail that decides whether a person
    walking past gets sampled at all;
  * the share of a cycle spent QUEUEING rather than inferring (54-76%) — which
    an overwritten `wait_ms` made uncomputable.

These tests pin both, and pin the host-suspend exclusion that already existed
so percentiles do not inherit the bug that poisoned the mean.
"""
import pytest

pytest.importorskip("ultralytics", reason="bytetrack_engine imports ultralytics lazily")

from app.services import bytetrack_engine as bt


@pytest.fixture(autouse=True)
def clean_perf():
    with bt._perf_lock:
        bt._perf.clear()
    yield
    with bt._perf_lock:
        bt._perf.clear()


def test_percentiles_describe_the_tail_a_mean_hides():
    """Ninety fast passes and ten slow ones: the mean looks fine, p90 does not."""
    for _ in range(90):
        bt._record_perf("57", wait_ms=100.0, infer_ms=900.0)      # 1.0s cycle
    for _ in range(10):
        bt._record_perf("57", wait_ms=8000.0, infer_ms=2000.0)    # 10.0s cycle

    stats = bt.get_perf_stats()["57"]
    assert stats["cycle_p50_ms"] == pytest.approx(1000.0, abs=1.0)
    assert stats["cycle_p90_ms"] >= 1000.0
    assert stats["cycle_max_ms"] == pytest.approx(10000.0, abs=1.0)
    # The point: a person crossing in 2000ms is missed on the slow passes, and
    # the mean cycle (1.9s) would have suggested they were not.
    assert stats["cycle_max_ms"] > 2000.0


def test_wait_share_is_computable():
    """The single most useful number about this pipeline, and it was lost.

    `wait_ms` was overwritten every call, so cumulative queue time did not
    exist anywhere.
    """
    for _ in range(10):
        bt._record_perf("57", wait_ms=3000.0, infer_ms=2000.0)

    stats = bt.get_perf_stats()["57"]
    assert stats["wait_total_ms"] == pytest.approx(30000.0)
    assert stats["wait_share_pct"] == pytest.approx(60.0, abs=0.5)


def test_wait_share_is_zero_when_nothing_queues():
    for _ in range(5):
        bt._record_perf("59", wait_ms=0.0, infer_ms=2000.0)

    assert bt.get_perf_stats()["59"]["wait_share_pct"] == 0.0


def test_cameras_are_measured_independently():
    for _ in range(20):
        bt._record_perf("57", wait_ms=3000.0, infer_ms=1000.0)   # doorway, queueing
    for _ in range(20):
        bt._record_perf("59", wait_ms=100.0, infer_ms=3000.0)    # room, working

    stats = bt.get_perf_stats()
    assert stats["57"]["wait_share_pct"] > 50.0
    assert stats["59"]["wait_share_pct"] < 10.0


def test_a_host_suspend_does_not_poison_the_percentiles():
    """The 56-minute gap on camera 60 was the machine sleeping, not a stall.

    It already poisoned the mean (22.3s against a 4.0s median). The percentiles
    must not inherit that: _record_perf zeroes an implausible wait before it
    reaches the ring buffer.
    """
    for _ in range(20):
        bt._record_perf("60", wait_ms=500.0, infer_ms=2500.0)
    bt._record_perf("60", wait_ms=3_357_839.0, infer_ms=2816.0)   # the real log line

    stats = bt.get_perf_stats()["60"]
    assert stats["suspensions"] == 1
    assert stats["cycle_max_ms"] < 10_000.0, "a host suspend leaked into the cycle samples"
    assert stats["wait_share_pct"] < 30.0


def test_sample_buffer_is_bounded():
    """Bounded by construction — this must not grow for the life of the process."""
    for i in range(bt._PERF_SAMPLES * 3):
        bt._record_perf("57", wait_ms=0.0, infer_ms=float(i))

    stats = bt.get_perf_stats()["57"]
    assert stats["samples"] == bt._PERF_SAMPLES
    assert stats["calls"] == bt._PERF_SAMPLES * 3, "calls must still count every pass"


def test_ring_buffers_are_not_leaked_into_the_public_shape():
    """`_cycles`/`_infers` are deques; they must not reach a JSON response."""
    bt._record_perf("57", wait_ms=10.0, infer_ms=20.0)

    stats = bt.get_perf_stats()["57"]
    assert not any(key.startswith("_") for key in stats), stats.keys()

    # _record_perf's own return value feeds the PERF log line.
    returned = bt._record_perf("57", wait_ms=10.0, infer_ms=20.0)
    assert not any(key.startswith("_") for key in returned)
    assert returned["calls"] == 2


def test_no_samples_yet_does_not_divide_by_zero():
    assert bt.get_perf_stats() == {}


@pytest.mark.parametrize("fraction,expected", [
    (0.50, 50.0),
    (0.90, 90.0),
    (0.99, 99.0),
])
def test_percentile_helper(fraction, expected):
    values = [float(n) for n in range(1, 101)]
    assert bt._percentile(values, fraction) == pytest.approx(expected, abs=1.0)


def test_percentile_of_an_empty_sample_is_zero():
    assert bt._percentile([], 0.9) == 0.0
