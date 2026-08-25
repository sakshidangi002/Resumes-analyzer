"""One capture thread per camera, holding only the newest frame.

WHY A SLOT AND NOT A QUEUE
--------------------------
CCTV inference answers "what is happening now". A queue answers "what happened,
in order" -- which is the wrong question, and on this hardware it is actively
harmful: YOLO costs 2.2-3.4s per pass while cameras deliver 12fps, so a queue
would grow by ~40 frames for every one processed and the pipeline would fall
further behind every second it ran.

So each camera keeps exactly ONE frame. A new frame overwrites the old one and
the old one is dropped, unexamined. If the scheduler was busy for nine seconds,
it comes back to the newest frame, not to a nine-second backlog.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
No detection, no tracking, no recognition, no drawing. In V1 the equivalent code
lived inside a 3,166-line module that also did all of those, which is why a slow
inference could stall frame capture and freeze the live view. Capture here does
nothing but capture, so a stalled scheduler cannot affect the stream.

LOCK DISCIPLINE
---------------
The lock is held only long enough to swap or read a reference -- never across a
decode, and never across inference. `cap.retrieve()` allocates a fresh array per
call, so handing that reference out needs no defensive copy: the grabber never
mutates a frame it has published.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Back-off between reconnect attempts. Rising, because a camera that refused
# once usually refuses again immediately, and hammering a Hikvision DVR with
# fresh RTSP sessions is how V1 kept its own lockout alive.
_RECONNECT_MIN = 2.0
_RECONNECT_MAX = 30.0


@dataclass(frozen=True)
class FrameSnapshot:
    """One frame plus the metadata the scheduler needs to reason about it.

    Frozen: it is handed to another thread, and a snapshot whose timestamp can
    change after the scheduler has scored it would make selection
    non-deterministic and the metrics untrue.
    """

    camera_id: int
    frame: object          # np.ndarray; typed loosely so tests need no numpy
    timestamp: float       # time.time() when the frame was decoded
    sequence: int          # monotonic per camera; gaps prove frames were dropped

    def staleness(self, now: Optional[float] = None) -> float:
        """Seconds since this frame was captured."""
        return max(0.0, (time.time() if now is None else now) - self.timestamp)


@dataclass(frozen=True)
class CameraHealth:
    camera_id: int
    connected: bool
    frames_grabbed: int
    frames_dropped: int    # replaced in the slot before anything consumed them
    reconnects: int
    last_frame_time: float
    last_error: Optional[str]


class CameraGrabber:
    """Keeps one camera connected and publishes only its newest frame."""

    def __init__(
        self,
        camera_id: int,
        source: str,
        open_capture: Optional[Callable[[str], object]] = None,
        read_timeout: float = 15.0,
    ) -> None:
        self.camera_id = int(camera_id)
        self.source = source
        self._open_capture = open_capture or _default_open
        self._read_timeout = read_timeout

        self._lock = threading.Lock()
        self._latest: Optional[FrameSnapshot] = None
        self._seq = 0
        self._grabbed = 0
        self._dropped = 0
        self._consumed_seq = 0
        self._reconnects = 0
        self._connected = False
        self._last_error: Optional[str] = None
        self._last_frame_time = 0.0

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"grab-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    # ── consumer API ─────────────────────────────────────────────────────────
    def get_latest(self) -> Optional[FrameSnapshot]:
        """The newest frame, or None if this camera has produced nothing yet.

        Does not clear the slot. The scheduler may legitimately look at a frame
        it decides not to process, and clearing on read would turn a scoring
        decision into a destructive one.
        """
        with self._lock:
            snap = self._latest
            if snap is not None:
                self._consumed_seq = snap.sequence
            return snap

    def health(self) -> CameraHealth:
        with self._lock:
            return CameraHealth(
                camera_id=self.camera_id,
                connected=self._connected,
                frames_grabbed=self._grabbed,
                frames_dropped=self._dropped,
                reconnects=self._reconnects,
                last_frame_time=self._last_frame_time,
                last_error=self._last_error,
            )

    # ── producer ─────────────────────────────────────────────────────────────
    def publish(self, frame: object, timestamp: Optional[float] = None) -> FrameSnapshot:
        """Put a frame in the slot, replacing whatever was there.

        Public so tests can drive a grabber without a camera. Counts a DROP when
        the frame being replaced was never handed to a consumer -- that number is
        the honest measure of how far behind inference is running, and it should
        be large on this hardware rather than hidden.
        """
        with self._lock:
            self._seq += 1
            self._grabbed += 1
            if self._latest is not None and self._latest.sequence > self._consumed_seq:
                self._dropped += 1
            snap = FrameSnapshot(
                camera_id=self.camera_id,
                frame=frame,
                timestamp=time.time() if timestamp is None else timestamp,
                sequence=self._seq,
            )
            self._latest = snap
            self._last_frame_time = snap.timestamp
            return snap

    def _run(self) -> None:
        cap = None
        delay = _RECONNECT_MIN
        while not self._stop.is_set():
            if cap is None:
                try:
                    cap = self._open_capture(self.source)
                except Exception as exc:                      # noqa: BLE001
                    cap = None
                    self._note_error(f"open failed: {exc}")
                if cap is None or not _is_open(cap):
                    cap = None
                    self._reconnects += 1
                    self._connected = False
                    self._stop.wait(delay)
                    delay = min(delay * 1.5, _RECONNECT_MAX)
                    continue
                delay = _RECONNECT_MIN
                self._connected = True
                self._last_error = None
                logger.info("grabber camera=%s connected", self.camera_id)

            try:
                ok, frame = _read(cap)
            except Exception as exc:                          # noqa: BLE001
                ok, frame = False, None
                self._note_error(f"read failed: {exc}")

            if not ok or frame is None:
                # Drop the connection rather than spin: a stream that stops
                # decoding does not usually recover in place.
                _release(cap)
                cap = None
                self._connected = False
                self._reconnects += 1
                self._stop.wait(delay)
                delay = min(delay * 1.5, _RECONNECT_MAX)
                continue

            self.publish(frame)

        if cap is not None:
            _release(cap)
        self._connected = False
        logger.info("grabber camera=%s stopped", self.camera_id)

    def _note_error(self, msg: str) -> None:
        self._last_error = msg
        logger.warning("grabber camera=%s %s", self.camera_id, msg)


# ── cv2 seams, kept behind functions so tests never need OpenCV ──────────────
def _default_open(source: str):
    import cv2

    return cv2.VideoCapture(source, cv2.CAP_FFMPEG)


def _is_open(cap) -> bool:
    return bool(cap.isOpened()) if hasattr(cap, "isOpened") else True


def _read(cap):
    # grab() advances the stream cheaply; retrieve() pays for the colour
    # conversion. Split so a camera nobody is processing still drains its
    # buffer -- that is what keeps the FFmpeg queue, and therefore latency,
    # from growing.
    if hasattr(cap, "grab") and hasattr(cap, "retrieve"):
        if not cap.grab():
            return False, None
        return cap.retrieve()
    return cap.read()


def _release(cap) -> None:
    try:
        cap.release()
    except Exception:                                          # noqa: BLE001
        logger.debug("capture release failed", exc_info=True)
