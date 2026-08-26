"""The inference pool: N workers, N models, no queue.

WHY A POOL AT ALL. This ran on one worker until a benchmark showed why it should
not. On this box (4 physical / 8 logical cores):

    workers   passes/s   vs 1     latency
       1        0.403    1.00x      2.48s
       2        0.587    1.46x      3.40s
       3        0.733    1.82x      3.99s
       4        0.811    2.01x      4.81s
    2 (SHARED)  0.280    0.93x      6.88s

The last row is the one these tests exist to prevent. Two workers sharing ONE
ultralytics model are slower than a single worker -- so a pool whose workers
share a model is worse than no pool, and it fails silently: throughput just
quietly drops while every metric still looks plausible. Hence the tests below
assert that each worker builds its OWN processor, not merely that N threads run.

THE OTHER RULE IS NO QUEUE. Workers pull; nothing is ever pushed at them. A
queue would let a worker start a 3.4s inference on a frame that was newest when
it was enqueued and is stale by the time it runs -- reintroducing, one layer up,
exactly the failure the frame-age cutoff exists to prevent.
"""
import threading
import time

import pytest

from app.cctv_v2.capture.grabber import CameraGrabber
from app.cctv_v2.scheduler.loop import InferenceScheduler

ALL = (57, 58, 59, 60)


def grabbers(cameras=ALL):
    return {cid: CameraGrabber(cid, f"test://{cid}") for cid in cameras}


def publish_all(gs, frame="f"):
    for cid, g in gs.items():
        g.publish(frame, timestamp=time.time())


def drain(sched, gs, seconds=1.0, republish=True):
    """Run the pool for a wall-clock window, keeping the cameras streaming."""
    sched.start()
    end = time.time() + seconds
    while time.time() < end:
        if republish:
            publish_all(gs)
        time.sleep(0.01)
    sched.stop(timeout=5)


# ---------------------------------------------------------------------------
# Pool creation and model isolation
# ---------------------------------------------------------------------------
def test_the_pool_runs_the_requested_number_of_workers():
    gs = grabbers()
    built = []
    sched = InferenceScheduler(
        gs.values(), workers=3,
        process_factory=lambda: (built.append(1), lambda cid, s: None)[1],
    )
    publish_all(gs)
    drain(sched, gs, 0.4)

    assert sched.worker_count == 3
    assert len(built) == 3, f"expected 3 processors built, got {len(built)}"


def test_each_worker_builds_its_OWN_processor():
    """The whole point. A pool whose workers share one model is measurably
    slower than a single worker -- see the module docstring."""
    gs = grabbers()
    made = []

    def factory():
        obj = object()               # stands in for a YOLO instance
        made.append(obj)
        return lambda cid, s: None

    sched = InferenceScheduler(gs.values(), workers=4, process_factory=factory)
    publish_all(gs)
    drain(sched, gs, 0.4)

    assert len(made) == 4
    assert len({id(m) for m in made}) == 4, "workers shared a model instance"


def test_the_factory_is_called_in_the_worker_thread_not_at_construction():
    """A model built on the constructing thread would be one model handed to
    every worker, which is the anti-pattern this pool exists to avoid."""
    gs = grabbers()
    threads = []
    sched = InferenceScheduler(
        gs.values(), workers=2,
        process_factory=lambda: (
            threads.append(threading.current_thread().name),
            lambda cid, s: None,
        )[1],
    )
    assert threads == [], "factory ran before any worker started"

    publish_all(gs)
    drain(sched, gs, 0.4)

    assert len(threads) == 2
    assert all(n.startswith("cctv-infer-") for n in threads), threads
    assert len(set(threads)) == 2, "both processors were built on one thread"


def test_a_scheduler_needs_either_a_process_or_a_factory():
    with pytest.raises(ValueError):
        InferenceScheduler(grabbers().values())


def test_one_worker_still_works_with_the_plain_callback():
    """The single-worker form stays valid -- it is what every existing test and
    the whole Step 6 validation used."""
    gs = grabbers()
    seen = []
    sched = InferenceScheduler(gs.values(), process=lambda cid, s: seen.append(cid))
    publish_all(gs)

    assert sched.worker_count == 1
    assert sched.run_once() is not None
    assert seen


# ---------------------------------------------------------------------------
# No queue, newest frame
# ---------------------------------------------------------------------------
def test_a_busy_pool_does_not_accumulate_frames():
    """Cameras stream throughout a slow inference; the next pass must start on
    what is current, not on the backlog that piled up behind it."""
    gs = grabbers(cameras=(57,))
    processed = []

    def slow(cid, snap):
        processed.append(snap.sequence)
        time.sleep(0.25)             # frames keep arriving during this

    sched = InferenceScheduler(gs.values(), process=slow)
    sched.start()
    end = time.time() + 1.2
    while time.time() < end:
        gs[57].publish("f", timestamp=time.time())
        time.sleep(0.005)
    sched.stop(timeout=5)

    published = gs[57].health().frames_grabbed
    assert len(processed) >= 2, "not enough passes to prove anything"
    # Every processed frame is far into the stream, never near its start.
    assert min(processed[1:]) > len(processed), (
        f"a backlog was worked through: processed {processed} "
        f"out of {published} published"
    )
    assert processed == sorted(processed), "frames went backwards"


def test_a_worker_starts_on_the_frame_current_when_it_begins():
    """`_claim` re-reads the slot, so the gap between deciding and starting
    cannot hand a worker an older frame than the one that was scored."""
    gs = grabbers(cameras=(57,))
    seen = []
    sched = InferenceScheduler(gs.values(), process=lambda cid, s: seen.append(s.sequence))

    gs[57].publish("old", timestamp=time.time())
    for _ in range(30):
        gs[57].publish("new", timestamp=time.time())

    sched.run_once()
    assert seen == [31], f"expected the newest frame, got sequence {seen}"


def test_no_two_workers_take_the_same_camera_at_once():
    """Otherwise three workers spend three inferences on one scene while the
    other cameras wait."""
    gs = grabbers()
    concurrent = {}
    clashes = []
    lock = threading.Lock()

    def slow(cid, snap):
        with lock:
            if concurrent.get(cid):
                clashes.append(cid)
            concurrent[cid] = True
        time.sleep(0.05)
        with lock:
            concurrent[cid] = False

    sched = InferenceScheduler(gs.values(), workers=4, process_factory=lambda: slow)
    publish_all(gs)
    drain(sched, gs, 1.0)

    assert clashes == [], f"cameras processed twice at once: {set(clashes)}"


def test_all_four_cameras_are_served_by_a_pool():
    gs = grabbers()
    seen = set()
    lock = threading.Lock()

    def process(cid, snap):
        with lock:
            seen.add(cid)
        time.sleep(0.02)

    sched = InferenceScheduler(gs.values(), workers=3, process_factory=lambda: process)
    publish_all(gs)
    drain(sched, gs, 1.5)

    assert seen == set(ALL), f"only {sorted(seen)} were served"


def test_a_worker_is_not_tied_to_a_camera():
    """Safe only because YOLO predict is stateless. When tracking arrives the
    tracker must be keyed on CAMERA, never on worker -- see loop.py's SCOPE."""
    gs = grabbers()
    by_worker = {}
    lock = threading.Lock()

    def make():
        wid = object()

        def process(cid, snap):
            with lock:
                by_worker.setdefault(id(wid), set()).add(cid)
            time.sleep(0.02)
        return process

    sched = InferenceScheduler(gs.values(), workers=2, process_factory=make)
    publish_all(gs)
    drain(sched, gs, 2.0)

    assert any(len(v) > 1 for v in by_worker.values()), (
        "no worker ever handled more than one camera; the decoupling is untested"
    )


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------
def test_a_crashing_inference_does_not_stop_the_pool():
    gs = grabbers()
    ok = []
    lock = threading.Lock()

    def process(cid, snap):
        if cid == 57:
            raise RuntimeError("this camera explodes")
        with lock:
            ok.append(cid)

    sched = InferenceScheduler(gs.values(), workers=2, process_factory=lambda: process)
    publish_all(gs)
    drain(sched, gs, 1.0)

    assert set(ok) >= {58, 59, 60}, f"healthy cameras stopped being served: {set(ok)}"


def test_a_camera_is_released_after_its_inference_crashes():
    """Otherwise one exception removes that camera from scheduling forever --
    a leak that looks exactly like a camera quietly going dark."""
    gs = grabbers(cameras=(57,))
    calls = []

    def process(cid, snap):
        calls.append(cid)
        raise RuntimeError("boom")

    sched = InferenceScheduler(gs.values(), process=process)
    gs[57].publish("f", timestamp=time.time())

    sched.run_once()
    gs[57].publish("f", timestamp=time.time())
    sched.run_once()

    assert len(calls) == 2, "the camera was never released after the first crash"
    assert sched._in_flight == set()


def test_a_worker_whose_factory_fails_disables_only_itself():
    gs = grabbers()
    built = []
    served = []
    lock = threading.Lock()

    def factory():
        with lock:
            n = len(built)
            built.append(n)
        if n == 0:
            raise RuntimeError("this worker cannot load its model")
        return lambda cid, s: served.append(cid)

    sched = InferenceScheduler(gs.values(), workers=3, process_factory=factory)
    publish_all(gs)
    drain(sched, gs, 1.0)

    assert len(built) == 3
    assert served, "the surviving workers did no work"


def test_stopping_the_pool_joins_every_worker():
    gs = grabbers()
    sched = InferenceScheduler(
        gs.values(), workers=3, process_factory=lambda: lambda cid, s: time.sleep(0.01)
    )
    publish_all(gs)
    sched.start()
    time.sleep(0.3)
    sched.stop(timeout=5)

    assert all(not t.is_alive() for t in sched._threads)


# ---------------------------------------------------------------------------
# Stale frames still never reach a worker, with several of them
# ---------------------------------------------------------------------------
def test_a_pool_still_refuses_stale_frames():
    gs = grabbers()
    seen = []
    lock = threading.Lock()

    def process(cid, snap):
        with lock:
            seen.append(cid)
        time.sleep(0.02)

    sched = InferenceScheduler(gs.values(), workers=3, process_factory=lambda: process)
    # 59 offers a frame far past the 5s room cutoff and never refreshes it.
    gs[59].publish("ancient", timestamp=time.time() - 90.0)
    for cid in (57, 58, 60):
        gs[cid].publish("f", timestamp=time.time())

    sched.start()
    end = time.time() + 1.0
    while time.time() < end:
        for cid in (57, 58, 60):
            gs[cid].publish("f", timestamp=time.time())
        time.sleep(0.01)
    sched.stop(timeout=5)

    assert 59 not in seen, "a stale frame reached a worker"
    assert set(seen) >= {57, 58, 60}
    assert sched.stats[59].stale_skips > 0


def test_a_pool_recovers_a_camera_automatically():
    gs = grabbers()
    seen = []
    lock = threading.Lock()

    def process(cid, snap):
        with lock:
            seen.append(cid)
        time.sleep(0.02)

    sched = InferenceScheduler(gs.values(), workers=2, process_factory=lambda: process)
    gs[59].publish("ancient", timestamp=time.time() - 90.0)
    sched.start()

    end = time.time() + 0.6
    while time.time() < end:
        for cid in (57, 58, 60):
            gs[cid].publish("f", timestamp=time.time())
        time.sleep(0.01)
    assert 59 not in seen

    end = time.time() + 1.0                      # 59 comes back
    while time.time() < end:
        publish_all(gs)
        time.sleep(0.01)
    sched.stop(timeout=5)

    assert 59 in seen, "camera never re-entered scheduling after recovering"


# ---------------------------------------------------------------------------
# Grabber independence
# ---------------------------------------------------------------------------
def test_a_slow_worker_cannot_block_a_grabber():
    """Capture must not be coupled to inference. In V1 they lived in one module
    and a slow pass froze the live view.

    Synchronised on events rather than sleeps. An earlier version asserted "more
    than 50 frames published in 0.8s", which passed alone and failed under full
    suite load -- it was measuring the machine, not the coupling.
    """
    gs = grabbers(cameras=(57,))
    inference_started = threading.Event()
    may_finish = threading.Event()

    def very_slow(cid, snap):
        inference_started.set()
        may_finish.wait(timeout=5.0)

    sched = InferenceScheduler(gs.values(), process=very_slow)
    gs[57].publish("f", timestamp=time.time())
    sched.start()
    try:
        assert inference_started.wait(timeout=5.0), "inference never started"

        # A pass is now definitely in flight and will not return until released.
        before = gs[57].health().frames_grabbed
        for _ in range(20):
            gs[57].publish("f", timestamp=time.time())
        after = gs[57].health().frames_grabbed
    finally:
        may_finish.set()
        sched.stop(timeout=5)

    assert after - before == 20, (
        "publishing was blocked while an inference was running"
    )


# ---------------------------------------------------------------------------
# V1 isolation
# ---------------------------------------------------------------------------
def test_v1_does_not_import_v2():
    """V2 must remain additive. If a V1 service imports cctv_v2, the rollback
    switch stops being a rollback."""
    import pathlib

    services = pathlib.Path(__file__).resolve().parents[1] / \
        "Attendance Management" / "backend" / "app" / "services"
    offenders = [
        p.name for p in services.glob("*.py")
        if "cctv_v2" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == [], f"V1 services import V2: {offenders}"


def test_the_pipeline_flag_still_defaults_to_v1():
    from app.core.config import get_settings

    assert get_settings().cctv_pipeline == "v1"


def test_the_worker_count_is_configurable_and_defaults_to_three():
    from app.core.config import get_settings

    assert get_settings().cctv_v2_inference_workers == 3
