"""Face-inference admission: bounded concurrency, attendance cameras first.

Background. `face_service` guarded all detection and embedding with one
exclusive `threading.Lock`. That serialised inference process-wide and served
the queue first-come-first-served, so an entrance camera waited behind a
MONITOR camera's ~500 ms face-crop pass. A person is in front of an entrance
camera for about two seconds; someone at a desk sits still for minutes, so the
ordering was exactly backwards. The existing code even throttled monitor
cameras to a 1.5 s analysis interval to work around it.

These tests pin the two properties that make the gate correct:
  * concurrency never exceeds the configured slot count
  * high-priority callers are admitted ahead of waiting low-priority ones,
    WITHOUT starving them indefinitely

Threads only — no models, no cameras, no OpenCV.
"""
import threading
import time

import pytest
from app.services.inference_gate import (
    PriorityInferenceGate,
    is_high_priority,
    set_inference_priority,
)


def test_concurrency_never_exceeds_slots():
    """The gate's one hard safety property: don't oversubscribe the CPU."""
    gate = PriorityInferenceGate(slots=2)
    active = 0
    peak = 0
    guard = threading.Lock()

    def worker():
        nonlocal active, peak
        with gate.acquire():
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with guard:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert peak <= 2, f"admitted {peak} concurrent callers into 2 slots"
    assert gate.stats()["active"] == 0, "a slot was leaked"


def test_high_priority_is_admitted_before_waiting_low_priority():
    """The actual fix: an attendance camera must not queue behind monitors."""
    gate = PriorityInferenceGate(slots=1)
    order = []
    order_lock = threading.Lock()

    # Occupy the only slot so everything else must queue.
    blocker_release = threading.Event()
    blocker_started = threading.Event()

    def blocker():
        with gate.acquire():
            blocker_started.set()
            blocker_release.wait(timeout=5)

    def contender(label, high):
        with gate.acquire(high_priority=high), order_lock:
            order.append(label)

    b = threading.Thread(target=blocker)
    b.start()
    assert blocker_started.wait(timeout=5)

    # Queue the LOW priority callers first — they must still be overtaken.
    lows = [threading.Thread(target=contender, args=(f"low{i}", False)) for i in range(3)]
    for t in lows:
        t.start()
    time.sleep(0.15)          # ensure they are genuinely waiting

    high = threading.Thread(target=contender, args=("high", True))
    high.start()
    time.sleep(0.05)

    blocker_release.set()
    for t in [b, high, *lows]:
        t.join(timeout=10)

    assert order[0] == "high", (
        f"low-priority callers were served first: {order}. An entrance camera "
        f"would be waiting behind monitor cameras again."
    )


def test_low_priority_is_not_starved_forever():
    """Bound the preference, or a busy entrance freezes every monitor feed."""
    gate = PriorityInferenceGate(slots=1, low_priority_max_wait=0.2)
    admitted = threading.Event()

    stop = threading.Event()

    def high_priority_flood():
        while not stop.is_set():
            with gate.acquire(high_priority=True):
                time.sleep(0.01)

    def low_priority():
        with gate.acquire(high_priority=False):
            admitted.set()

    floods = [threading.Thread(target=high_priority_flood, daemon=True) for _ in range(3)]
    for t in floods:
        t.start()

    low = threading.Thread(target=low_priority)
    low.start()

    got_in = admitted.wait(timeout=5)
    stop.set()
    low.join(timeout=5)
    for t in floods:
        t.join(timeout=5)

    assert got_in, "low-priority caller starved despite the max-wait override"
    assert gate.stats()["low_starvation_overrides"] >= 1


def test_slot_is_released_when_the_body_raises():
    """An exception inside the guarded block must not leak the slot.

    Inference genuinely does raise (corrupt frame, model error), and the
    recognition loop catches and continues — a leaked slot would silently
    reduce capacity until the process was restarted.
    """
    gate = PriorityInferenceGate(slots=1)
    with pytest.raises(ValueError), gate.acquire():
        raise ValueError("boom")
    assert gate.stats()["active"] == 0

    # The gate is still usable.
    with gate.acquire():
        pass


def test_stats_track_wait_time_by_priority():
    gate = PriorityInferenceGate(slots=1)
    with gate.acquire(high_priority=True):
        pass
    with gate.acquire(high_priority=False):
        pass
    stats = gate.stats()
    assert stats["high_calls"] == 1
    assert stats["low_calls"] == 1
    assert stats["slots"] == 1
    assert "high_avg_wait_ms" in stats and "low_avg_wait_ms" in stats


def test_priority_is_per_thread():
    """Priority is ambient thread state, so one camera's setting must not leak
    into another camera's thread."""
    set_inference_priority(True)
    assert is_high_priority() is True

    seen = {}

    def other_thread():
        seen["default"] = is_high_priority()      # untouched thread -> low
        set_inference_priority(False)
        seen["after"] = is_high_priority()

    t = threading.Thread(target=other_thread)
    t.start()
    t.join(timeout=5)

    assert seen["default"] is False
    assert seen["after"] is False
    assert is_high_priority() is True, "another thread changed this thread's priority"
