"""A camera keeps its newest frame and nothing else.

The rule this pins: CCTV inference answers "what is happening now", so a frame
that has been superseded is worthless and must be dropped, not queued.

It matters here more than in most systems. YOLO costs 2.2-3.4s per pass on this
hardware while the cameras deliver 12fps, so a queue would grow by roughly forty
frames for every one processed. The pipeline would fall further behind every
second it ran, and by the time a frame reached inference it would describe a
corridor that emptied ten seconds ago.

`frames_dropped` is deliberately counted and exposed rather than hidden. It is
the honest measure of how far behind inference is running, and on this hardware
it should be a large number.
"""
import threading
import time

import pytest

from app.cctv_v2.capture.grabber import CameraGrabber, FrameSnapshot


def _grabber(camera_id=57):
    return CameraGrabber(camera_id=camera_id, source="test://none")


def test_a_camera_with_no_frames_yet_returns_none():
    """The scheduler must be able to skip a camera, not wait for one."""
    assert _grabber().get_latest() is None


def test_the_latest_frame_replaces_the_previous_one():
    g = _grabber()
    g.publish("frame-1")
    g.publish("frame-2")
    g.publish("frame-3")

    snap = g.get_latest()
    assert snap.frame == "frame-3"


def test_there_is_no_queue_growth():
    """A hundred frames arrive; exactly one is retained."""
    g = _grabber()
    for i in range(100):
        g.publish(f"frame-{i}")

    assert g.get_latest().frame == "frame-99"
    # Only the newest is reachable -- repeated reads never yield an older one.
    assert {g.get_latest().frame for _ in range(5)} == {"frame-99"}


def test_sequence_increments_and_proves_frames_were_dropped():
    """Gaps in the sequence are the evidence that a backlog was discarded."""
    g = _grabber()
    first = g.publish("a")
    for i in range(9):
        g.publish(f"b{i}")
    last = g.get_latest()

    assert first.sequence == 1
    assert last.sequence == 10
    # Nine frames existed between them and none were consumed.
    assert last.sequence - first.sequence == 9


def test_dropped_frames_are_counted_not_hidden():
    g = _grabber()
    g.publish("a")          # nothing consumed it
    g.publish("b")          # -> a was dropped
    g.publish("c")          # -> b was dropped
    assert g.health().frames_dropped == 2

    g.get_latest()          # consume c
    g.publish("d")          # c was consumed, so this is not a drop
    assert g.health().frames_dropped == 2


def test_the_timestamp_is_preserved_exactly():
    """The scheduler scores on staleness; a rewritten timestamp would corrupt
    every scheduling decision and every metric derived from it."""
    g = _grabber()
    t = time.time() - 5.0
    g.publish("old", timestamp=t)

    snap = g.get_latest()
    assert snap.timestamp == t
    assert snap.staleness() >= 5.0


def test_staleness_is_never_negative():
    """A clock skew must not produce a negative score."""
    g = _grabber()
    g.publish("future", timestamp=time.time() + 10.0)
    assert g.get_latest().staleness() == 0.0


def test_a_snapshot_is_immutable():
    """It crosses a thread boundary; one that can change after being scored
    would make selection non-deterministic."""
    g = _grabber()
    g.publish("x")
    snap = g.get_latest()
    with pytest.raises(Exception):
        snap.timestamp = 0.0


def test_concurrent_publish_and_read_stay_consistent():
    """The reader must never see a torn snapshot: frame, timestamp and sequence
    always belong to the same publish."""
    g = _grabber()
    seen: list[FrameSnapshot] = []
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            i += 1
            g.publish(("frame", i))

    def reader():
        while not stop.is_set():
            s = g.get_latest()
            if s is not None:
                seen.append(s)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    time.sleep(0.25)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert seen, "reader observed nothing"
    for s in seen:
        # The frame payload carries its own index; it must match the sequence
        # published alongside it.
        assert s.frame[1] == s.sequence
    # Sequence never goes backwards for a reader.
    assert all(b.sequence >= a.sequence for a, b in zip(seen, seen[1:]))


def test_health_reports_what_the_camera_has_done():
    g = _grabber(camera_id=58)
    g.publish("a")
    g.publish("b")
    h = g.health()
    assert h.camera_id == 58
    assert h.frames_grabbed == 2
    assert h.last_frame_time > 0
