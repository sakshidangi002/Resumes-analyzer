"""YOLO11 + ByteTrack person-tracking engine (office monitoring).

Detection and tracking are done together by Ultralytics' `model.track(...)`,
which runs ByteTrack internally and returns STABLE track IDs that survive brief
occlusion and people crossing paths — exactly what the seating area needs.

Each returned object is a `PersonTrack` (from person_tracker), so the rest of
the pipeline (face binding, identity persistence, display) is unchanged.

Requires: `pip install ultralytics` and a YOLO11 model file (e.g. `yolo11n.pt`).
If unavailable, `is_available()` is False and the caller falls back to the
OpenCV-DNN person detector + IoU tracker.
"""
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache

from app.core.config import get_settings
from app.services.person_tracker import PersonTrack

logger = logging.getLogger(__name__)
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_model():
    try:
        from ultralytics import YOLO
    except Exception:
        logger.warning("ultralytics not installed — YOLO11+ByteTrack unavailable")
        return None
    path = get_settings().yolo_person_model_path
    if not os.path.exists(path):
        # Resolve relative to the backend root (app/services/ → backend/).
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        alt = os.path.join(backend_root, path)
        if os.path.exists(alt):
            path = alt
        else:
            logger.warning(
                "YOLO11 model not found at %s or %s — ByteTrack unavailable", path, alt
            )
            return None
    try:
        model = YOLO(path)
        logger.info("YOLO11+ByteTrack loaded from %s", path)
        return model
    except Exception:
        logger.exception("Failed to load YOLO11 model")
        return None


def is_available() -> bool:
    return _load_model() is not None


class ByteTrackEngine:
    """Per-camera YOLO11+ByteTrack tracker keeping PersonTrack identity state."""

    def __init__(self, conf: float = 0.35, max_misses: int = 30):
        self.conf = conf
        self.max_misses = max_misses
        self.tracks: dict[int, PersonTrack] = {}

    def update(self, frame_bgr) -> list[PersonTrack]:
        model = _load_model()
        if model is None:
            return list(self.tracks.values())

        with _lock:
            # persist=True keeps ByteTrack state across calls; classes=[0] = person.
            results = model.track(
                frame_bgr, persist=True, classes=[0], conf=self.conf,
                tracker="bytetrack.yaml", verbose=False,
            )

        seen: set[int] = set()
        if results:
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is not None and getattr(boxes, "id", None) is not None:
                ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                for tid, box in zip(ids, xyxy):
                    seen.add(int(tid))
                    b = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    pt = self.tracks.get(int(tid))
                    if pt is None:
                        self.tracks[int(tid)] = PersonTrack(
                            track_id=int(tid), box=b, max_misses=self.max_misses
                        )
                    else:
                        pt.update_box(b)

        # Age / expire tracks ByteTrack no longer reports.
        for tid in list(self.tracks.keys()):
            if tid not in seen:
                self.tracks[tid].mark_missed()
                if self.tracks[tid].is_expired():
                    del self.tracks[tid]

        return list(self.tracks.values())

    def reset(self) -> None:
        self.tracks.clear()
