"""Admission control for face inference, with priority.

Why this exists
---------------
`face_service` used a single exclusive `threading.Lock` around every detection
and embedding call, process-wide. That serialised all inference across all
cameras, and — more damaging than the lost parallelism — it made the queue
FIRST-COME-FIRST-SERVED. A MONITOR camera's per-person face-crop pass (~500 ms
of SCRFD) would take the lock, and the entrance camera would wait behind it.

That ordering is backwards. A person walks past an entrance camera in about two
seconds; a monitor camera is watching people who sit still for minutes. Delaying
the monitor camera costs a slightly staler label. Delaying the attendance camera
costs a missed check-in. The existing code even documents the symptom — monitor
cameras were given a 1.5 s analysis interval specifically to stop them
"MONOPOLISING the single global inference lock" and leaving "a person at the
entrance waiting 20-30s for a free inference slot". This gate removes the cause,
so that workaround is no longer load-bearing.

Is concurrent inference actually safe?
--------------------------------------
Yes, for this stack. Verified against the installed insightface + onnxruntime:

  * `SCRFD.detect()` / `SCRFD.forward()` and `ArcFaceONNX.get()` perform NO
    `self.<attr> =` assignment — all model state is fixed in `__init__` /
    `prepare()` / `_init_vars()`.
  * The one shared mutable is `SCRFD.center_cache`, a bounded (<100 entry) memo
    of anchor centres keyed by (h, w, stride). Writes are single `dict[k] = v`
    operations, atomic under the GIL, and the value is deterministic for a given
    key — so a concurrent double-compute is benign.
  * `onnxruntime.InferenceSession.run()` is thread-safe by design.

So the lock was not protecting correctness. It was only bounding CPU
oversubscription, which is what this gate does — while fixing the ordering.

Concurrency
-----------
Deliberately LOW by default. `bytetrack_engine` records a measurement from this
same 4-physical-core box: two concurrent inferences were ~10% WORSE than one,
because a single inference already saturates the cores. Parallelism only pays
once there are spare physical cores or a GPU, so the default mirrors that gate's
rule. Priority, not parallelism, is the win here.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def _auto_slots() -> int:
    """Concurrent inference slots. Explicit setting wins; else auto.

    Mirrors bytetrack_engine._auto_concurrency: >=8 physical cores gets 2 slots,
    anything less gets 1 (identical to the historical serial behaviour).
    """
    try:
        from app.core.config import get_settings

        configured = int(getattr(get_settings(), "face_max_concurrent_inference", 0) or 0)
    except Exception:
        configured = 0
    if configured > 0:
        return configured

    env = int(os.getenv("FACE_MAX_CONCURRENT_INFERENCE", "0") or 0)
    if env > 0:
        return env

    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or 0
    except Exception:
        physical = (os.cpu_count() or 4) // 2   # assume hyperthreading
    return 2 if physical >= 8 else 1


# How long a low-priority (MONITOR) caller may be held back before it is let
# through regardless. Without this, two continuously-busy attendance cameras
# would starve the monitor cameras completely and their boxes would freeze.
_LOW_PRIORITY_MAX_WAIT = float(os.getenv("FACE_LOW_PRIORITY_MAX_WAIT", "3.0"))


class PriorityInferenceGate:
    """Bounded-concurrency gate that admits high-priority callers first."""

    def __init__(self, slots: int, low_priority_max_wait: float = _LOW_PRIORITY_MAX_WAIT):
        self._cv = threading.Condition()
        self._slots = max(1, int(slots))
        self._low_priority_max_wait = float(low_priority_max_wait)
        self._active = 0
        self._waiting_high = 0
        # Low-priority waiters that have exceeded low_priority_max_wait. While
        # this is non-zero, HIGH priority callers yield — a temporary priority
        # inversion, and the only thing that actually breaks starvation. Merely
        # relaxing the starving waiter's own condition is not enough: the slot
        # it is waiting for gets taken by the next high-priority caller, which
        # under a continuous attendance load is always available.
        self._starving_low = 0
        # Diagnostics — how long callers spend queueing, split by priority.
        self._stats = {
            "high_calls": 0, "high_wait_ms": 0.0, "high_max_wait_ms": 0.0,
            "low_calls": 0, "low_wait_ms": 0.0, "low_max_wait_ms": 0.0,
            "low_starvation_overrides": 0,
        }

    @property
    def slots(self) -> int:
        return self._slots

    @contextmanager
    def acquire(self, high_priority: bool = False):
        """Admit one inference call, blocking until a slot is free.

        High-priority callers (IN/OUT attendance cameras) jump the queue. Low
        priority callers yield to them, but only up to
        `low_priority_max_wait` — after that they are admitted anyway, so a busy
        entrance can never freeze the monitor feeds entirely.
        """
        started = time.monotonic()
        deadline = None if high_priority else started + self._low_priority_max_wait
        counted_starving = False

        with self._cv:
            if high_priority:
                self._waiting_high += 1
            try:
                while True:
                    starving = deadline is not None and time.monotonic() >= deadline
                    if starving and not counted_starving:
                        # Announce the starvation so high-priority callers stand
                        # down until this waiter is served.
                        counted_starving = True
                        self._starving_low += 1
                        self._cv.notify_all()

                    if self._active >= self._slots:
                        can_run = False
                    elif high_priority:
                        # Yield to any low-priority caller that has waited too
                        # long. Without this a continuous attendance load takes
                        # every freed slot and the monitor feeds never update.
                        can_run = self._starving_low == 0
                    else:
                        can_run = starving or self._waiting_high == 0

                    if can_run:
                        self._active += 1
                        break
                    # Timed wait so a low-priority waiter re-evaluates its own
                    # deadline even if no notify arrives.
                    self._cv.wait(timeout=0.05)
            finally:
                if high_priority:
                    self._waiting_high -= 1
                if counted_starving:
                    self._starving_low -= 1
                    self._cv.notify_all()

            waited_ms = (time.monotonic() - started) * 1000.0
            self._record(high_priority, waited_ms, counted_starving)

        try:
            yield
        finally:
            with self._cv:
                self._active -= 1
                self._cv.notify_all()

    def _record(self, high_priority: bool, waited_ms: float, overridden: bool) -> None:
        key = "high" if high_priority else "low"
        self._stats[f"{key}_calls"] += 1
        self._stats[f"{key}_wait_ms"] += waited_ms
        if waited_ms > self._stats[f"{key}_max_wait_ms"]:
            self._stats[f"{key}_max_wait_ms"] = waited_ms
        if overridden:
            self._stats["low_starvation_overrides"] += 1

    def stats(self) -> dict:
        """Queueing statistics. `avg_wait_ms` per priority is the number to watch:
        if the high-priority average is not near zero, attendance cameras are
        still queueing and the slot count or the detector cost needs attention.
        """
        with self._cv:
            snapshot = dict(self._stats)
        for key in ("high", "low"):
            calls = snapshot[f"{key}_calls"]
            snapshot[f"{key}_avg_wait_ms"] = (
                round(snapshot[f"{key}_wait_ms"] / calls, 1) if calls else 0.0
            )
        snapshot["slots"] = self._slots
        snapshot["active"] = self._active
        return snapshot


# ---------------------------------------------------------------------------
# Per-thread priority
# ---------------------------------------------------------------------------
# Priority is ambient rather than a parameter because the call chain is deep:
# _RecognitionThread -> extract_faces_from_rgb / _face_in_person_crop ->
# recognize_face -> _build_face_result. Threading a flag through all of those
# would touch every signature for something that is a property of the CALLING
# THREAD (which camera it serves), not of any individual call.
_thread_ctx = threading.local()


def set_inference_priority(high: bool) -> None:
    """Mark the CURRENT thread's inference priority. Call once at thread start."""
    _thread_ctx.high_priority = bool(high)


def is_high_priority() -> bool:
    return getattr(_thread_ctx, "high_priority", False)


_gate: PriorityInferenceGate | None = None
_gate_lock = threading.Lock()


def get_gate() -> PriorityInferenceGate:
    """The process-wide inference gate (built lazily so settings are loaded)."""
    global _gate
    if _gate is None:
        with _gate_lock:
            if _gate is None:
                slots = _auto_slots()
                _gate = PriorityInferenceGate(slots)
                logger.info(
                    "Face inference gate: slots=%d low_priority_max_wait=%.1fs (cores=%s)",
                    slots, _LOW_PRIORITY_MAX_WAIT, os.cpu_count(),
                )
    return _gate


@contextmanager
def inference_slot():
    """Admit one inference call at this thread's priority."""
    with get_gate().acquire(high_priority=is_high_priority()):
        yield
