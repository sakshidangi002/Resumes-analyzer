"""Periodically ask the detector what chairs it can see, and tell the registry.

This is the only part of automatic chair counting that costs anything. It is
kept deliberately cheap and deliberately separate:

  * It runs on its OWN timer, not per analysis pass. Chairs are furniture --
    checking a few times an hour is enough to notice one being added, and
    checking every pass would put chair inference on the critical path of the
    people count, which is the thing that actually has to stay responsive.

  * It uses its OWN YOLO instance rather than borrowing a camera's. Each
    ByteTrackEngine owns a model because ultralytics keeps tracker state on the
    model's predictor; calling `predict()` on one of those would tear down the
    predictor that `track()` set up and reset that camera's ByteTrack ids. One
    extra model shared across all room cameras is the cheap, safe option --
    chair detection carries no tracker state, so sharing one is fine.

  * It never touches occupancy directly. It writes to the registry; the
    occupancy path reads from it. A sweep that fails, or a model that will not
    load, therefore degrades to "the inventory stops changing", not to "the
    room reports no chairs".
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from app.cctv_v2.config.cameras import cameras_with_role
from app.cctv_v2.pipeline import chair_registry
from app.cctv_v2.pipeline.detect import detect_chairs

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Seconds between sweeps of one camera. A minute is far more often than office
# furniture moves, and slow enough that the cost does not register next to the
# people pipeline.
SWEEP_SEC = _env_float("CCTV_CHAIR_SWEEP_SEC", 60.0)

# Confidence floor for a chair detection. Higher than the person floor on
# purpose: the registry needs SPARSE, TRUSTWORTHY evidence, not everything the
# model half-suspects. Weak detections are what produced the 1-9 spread that
# made raw chair detection unusable in the first place.
CHAIR_CONF = _env_float("CCTV_CHAIR_CONF", 0.25)

CHAIR_IMGSZ = _env_int("CCTV_CHAIR_IMGSZ", 960)

# Sweeps between writes of the inventory to disk. Not every sweep: the file only
# matters across restarts, and most sweeps change nothing.
SAVE_EVERY = _env_int("CCTV_CHAIR_SAVE_EVERY", 5)


class ChairSweeper:
    """Background thread that keeps every room camera's chair registry fed."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._model = None
        self._model_failed = False
        self._lock = threading.Lock()
        self.sweeps = 0
        self.last_error: Optional[str] = None
        self.last_sweep_at: float = 0.0

    # -- model ------------------------------------------------------------
    def _get_model(self):
        """Load the shared chair model once, and remember a failure.

        A failure is sticky: if the weights are missing there is no point
        retrying every minute for the life of the process, and the retry log
        would bury everything else.
        """
        with self._lock:
            if self._model is not None or self._model_failed:
                return self._model
            try:
                from ultralytics import YOLO

                from app.cctv_v2.config.profiles import ROOM
                from app.cctv_v2.pipeline.detect import _resolve

                path = _resolve(ROOM.model)
                if path is None:
                    raise FileNotFoundError(f"YOLO weights not found: {ROOM.model}")
                self._model = YOLO(path)
                logger.info("cctv_v2: chair sweeper loaded %s", path)
            except Exception as exc:                            # noqa: BLE001
                self._model_failed = True
                self.last_error = str(exc)
                logger.warning(
                    "cctv_v2: chair sweeper could not load a model; the chair "
                    "inventory will stay as configured. %s", exc,
                )
            return self._model

    # -- one camera -------------------------------------------------------
    @staticmethod
    def _frame_and_people(camera_id: int):
        """A private copy of the latest frame, plus current person boxes.

        Copied under V1's frame lock because the stream thread overwrites that
        array in place -- running inference on the live buffer would read a
        frame that is being rewritten underneath it.
        """
        from app.cctv_v2.pipeline.v1_bridge import _v1_worker

        worker = _v1_worker(camera_id)
        if worker is None:
            return None, []
        try:
            with worker._frame_lock:                            # type: ignore[union-attr]
                frame = getattr(worker, "_latest_frame", None)
                frame = None if frame is None else frame.copy()
                tracks = list(getattr(worker, "_latest_tracks", []) or [])
        except Exception:                                       # noqa: BLE001
            return None, []
        people = [tuple(t.box) for t in tracks if getattr(t, "box", None)]
        return frame, people

    def sweep_camera(self, camera_id: int) -> bool:
        """One sweep of one camera. True if the registry was actually fed."""
        model = self._get_model()
        if model is None:
            return False

        frame, people_px = self._frame_and_people(camera_id)
        if frame is None or not hasattr(frame, "shape"):
            return False
        height, width = frame.shape[:2]
        if not width or not height:
            return False

        boxes_px = detect_chairs(model, frame, CHAIR_IMGSZ, CHAIR_CONF, camera_id)

        # The registry works in normalised coordinates, like the configured chair
        # zones, so the inventory survives a resolution change.
        def norm(box):
            x1, y1, x2, y2 = box
            return (x1 / width, y1 / height, x2 / width, y2 / height)

        chairs = [norm(b) for b in boxes_px]
        people = [norm(b) for b in people_px if b and len(b) == 4]

        registry = chair_registry.store().get(camera_id)
        registry.sweep(chairs, person_boxes=people)
        logger.info(
            "cctv_v2: CHAIR-SWEEP camera=%s detected=%s confirmed=%s pending=%s people=%s",
            camera_id, len(chairs), registry.confirmed_count,
            registry.pending_count, len(people),
        )
        return True

    # -- loop -------------------------------------------------------------
    def _run(self) -> None:
        # Staggered by one interval so the first sweep does not land in the
        # middle of camera start-up, when there is no frame to read anyway.
        if self._stop.wait(min(SWEEP_SEC, 30.0)):
            return
        while not self._stop.is_set():
            started = time.time()
            try:
                for camera_id in cameras_with_role("room"):
                    if self._stop.is_set():
                        break
                    try:
                        self.sweep_camera(camera_id)
                    except Exception as exc:                    # noqa: BLE001
                        self.last_error = str(exc)
                        logger.exception(
                            "cctv_v2: chair sweep failed camera=%s", camera_id
                        )
                self.sweeps += 1
                self.last_sweep_at = time.time()
                if SAVE_EVERY > 0 and self.sweeps % SAVE_EVERY == 0:
                    chair_registry.store().save()
            except Exception:                                   # noqa: BLE001
                logger.exception("cctv_v2: chair sweeper loop error")
            # Measured from the START of the round, so a slow round does not
            # push every later sweep further and further apart.
            self._stop.wait(max(1.0, SWEEP_SEC - (time.time() - started)))

    def start(self) -> bool:
        if not chair_registry.auto_enabled():
            logger.info("cctv_v2: automatic chair detection disabled (CCTV_CHAIR_AUTO)")
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="cctv-chair-sweeper", daemon=True
        )
        self._thread.start()
        logger.info(
            "cctv_v2: chair sweeper started (every %.0fs, conf %.2f, cameras %s)",
            SWEEP_SEC, CHAIR_CONF, list(cameras_with_role("room")),
        )
        return True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._thread = None
        chair_registry.store().save()

    def status(self) -> dict:
        return {
            "running": bool(self._thread is not None and self._thread.is_alive()),
            "enabled": chair_registry.auto_enabled(),
            "sweeps": self.sweeps,
            "sweep_interval_sec": SWEEP_SEC,
            "last_sweep_at": self.last_sweep_at or None,
            "model_loaded": self._model is not None,
            "last_error": self.last_error,
        }


_SWEEPER = ChairSweeper()


def sweeper() -> ChairSweeper:
    return _SWEEPER


def start() -> bool:
    return _SWEEPER.start()


def stop() -> None:
    _SWEEPER.stop()
