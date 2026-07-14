"""YOLO11 + ByteTrack person-tracking engine (office monitoring).

Detection and tracking are done together by Ultralytics' `model.track(...)`,
which runs ByteTrack internally and returns STABLE track IDs that survive brief
occlusion and people crossing paths — exactly what the seating area needs.

Each returned object is a `PersonTrack` (from person_tracker), so the rest of
the pipeline (face binding, identity persistence, display) is unchanged.

IMPORTANT — per-camera tracker state:
`model.track(persist=True)` stores the ByteTrack state (Kalman filters, track
IDs) on the *model's predictor*. A model shared between cameras would therefore
associate camera A's detections against camera B's tracks. Each ByteTrackEngine
consequently owns its OWN `YOLO(...)` instance, so every camera gets its own
predictor and its own tracker state. The weights file is tiny (yolo11n ≈ 6 MB),
so per-camera instances are cheap; only the inference call is serialised (below).

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
# Serialises INFERENCE so several cameras don't oversubscribe the CPU/GPU.
# It only orders the calls — tracker state is per-engine (each owns its model).
_lock = threading.Lock()


def _resolve(path: str) -> str | None:
    """Resolve a path relative to CWD, else relative to the backend root."""
    if os.path.exists(path):
        return path
    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    alt = os.path.join(backend_root, path)
    return alt if os.path.exists(alt) else None


@lru_cache(maxsize=1)
def _model_path() -> str | None:
    path = get_settings().yolo_person_model_path
    resolved = _resolve(path)
    if resolved is None:
        logger.warning("YOLO11 model not found (%s) — ByteTrack unavailable", path)
    return resolved


@lru_cache(maxsize=1)
def _tracker_cfg() -> str:
    """Tuned ByteTrack config; falls back to ultralytics' built-in bytetrack.yaml."""
    cfg = get_settings().bytetrack_config_path
    resolved = _resolve(cfg)
    if resolved is None:
        logger.warning(
            "ByteTrack config not found (%s) — using ultralytics default bytetrack.yaml", cfg
        )
        return "bytetrack.yaml"
    return resolved


def is_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        logger.warning("ultralytics not installed — YOLO11+ByteTrack unavailable")
        return False
    return _model_path() is not None


class ByteTrackEngine:
    """Per-camera YOLO11+ByteTrack tracker keeping PersonTrack identity state."""

    def __init__(
        self,
        conf: float = 0.10,
        max_misses: int = 30,
        imgsz: int | None = None,
        device: str | None = None,
        camera_id: str = "?",
        tracker_cfg: str | None = None,
    ):
        s = get_settings()
        # Low detection floor on purpose: ByteTrack's stage-2 association needs the
        # low-score boxes. Track CREATION precision is guarded by new_track_thresh
        # inside the tracker config, not by this value.
        self.conf = conf
        self.max_misses = max_misses
        self.imgsz = int(imgsz) if imgsz else int(s.yolo_person_imgsz)
        # NMS IoU — see config: 0.7 (ultralytics default) lets a second, oversized
        # box survive on the same person, producing duplicate boxes/track ids.
        self.iou = float(s.yolo_person_iou)
        dev = device if device not in (None, "") else (s.yolo_person_device or None)
        self.device = dev or None
        self.camera_id = str(camera_id)
        # A steep top-down camera needs a far more permissive tracker than a
        # well-aimed one (its people score 0.11 instead of 0.36-0.67), so the
        # caller can hand this engine its own config.
        self.tracker_cfg = (_resolve(tracker_cfg) or _tracker_cfg()) if tracker_cfg else _tracker_cfg()
        self.tracks: dict[int, PersonTrack] = {}
        self._model = None            # own model  ⇒ own predictor ⇒ own ByteTrack state
        self._last_sig: tuple | None = None   # for change-based logging (no spam)

    def _get_model(self):
        """Lazily build this camera's OWN YOLO instance (isolated tracker state)."""
        if self._model is None:
            path = _model_path()
            if path is None:
                return None
            try:
                from ultralytics import YOLO

                self._model = YOLO(path)
            except Exception:
                logger.exception("Camera %s: failed to load YOLO11 model", self.camera_id)
                return None
            logger.info(
                "Camera %s: YOLO11 loaded path=%s imgsz=%d conf=%.2f tracker=%s device=%s",
                self.camera_id, path, self.imgsz, self.conf,
                os.path.basename(self.tracker_cfg), self.device or "auto",
            )
        return self._model

    def update(self, frame_bgr) -> list[PersonTrack]:
        model = self._get_model()
        if model is None:
            return list(self.tracks.values())

        kwargs = dict(
            persist=True,          # keep THIS camera's ByteTrack state across calls
            classes=[0],           # class 0 = person (only)
            conf=self.conf,
            iou=self.iou,          # NMS — suppresses duplicate boxes on one person
            imgsz=self.imgsz,
            tracker=self.tracker_cfg,
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device

        with _lock:
            results = model.track(frame_bgr, **kwargs)

        detections = 0
        seen: set[int] = set()
        if results:
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                detections = int(len(boxes))          # after conf + class + NMS
                if getattr(boxes, "id", None) is not None:
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

        # Drop "nested" duplicates: on this ceiling view YOLO often emits a tight
        # box on a seated person AND a second, bloated box running down over their
        # chair/bag. The two barely overlap, so NMS keeps both and one person gets
        # two boxes + two ids. Whenever one box swallows another, keep the TIGHT
        # one (the bloated box is the wrong one — it is mostly furniture).
        for tid in self._nested_ids(seen):
            seen.discard(tid)
            self.tracks.pop(tid, None)

        # Age / expire tracks ByteTrack no longer reports.
        for tid in list(self.tracks.keys()):
            if tid not in seen:
                self.tracks[tid].mark_missed()
                if self.tracks[tid].is_expired():
                    del self.tracks[tid]

        # Which tracks get DRAWN.
        #
        # In this room people sit behind HIGH-BACK CHAIRS that hide them almost
        # completely, so YOLO honestly reports 0 people even though nobody has
        # moved. Publishing only what was detected THIS frame therefore made boxes
        # (and names) blink out constantly while everyone was still sitting there.
        #
        # So we also publish tracks that are merely "missed", until they expire
        # after CCTV_IDENTITY_HOLD_SEC. A seated person doesn't move, so their
        # frozen box stays correct — and any single re-detection resets the timer.
        #
        # Trade-off (deliberate): someone who actually walks out keeps a stale box
        # until their track expires. That is far less disruptive than every box
        # disappearing whenever a chair hides its occupant. Set
        # `person_publish_held=False` to go back to detected-only.
        if get_settings().person_publish_held:
            live = list(self.tracks.values())
        else:
            live = [self.tracks[tid] for tid in seen if tid in self.tracks]

        # Log only when the picture CHANGES (detections / tracks / ids) so a
        # steady scene doesn't spam the log every analysis tick.
        sig = (detections, len(live), tuple(sorted(seen)))
        if sig != self._last_sig:
            self._last_sig = sig
            logger.info(
                "YOLO camera=%s detections=%d tracks=%d ids=%s",
                self.camera_id, detections, len(live), sorted(seen),
            )

        return live

    def _nested_ids(self, seen: set) -> list:
        """Ids of bloated boxes that swallow another box (same person, twice).

        Returns the LARGER box's id when it contains >= `contain` of a smaller box
        and is at least `ratio`x its area. Two genuinely separate people never sit
        one inside the other, so this is safe on a fixed-desk room view.
        """
        s = get_settings()
        contain = float(s.person_nested_contain)
        ratio = float(s.person_nested_area_ratio)
        ids = list(seen)
        drop: list = []
        for a in ids:
            ta = self.tracks.get(a)
            if ta is None:
                continue
            for b in ids:
                if a == b:
                    continue
                tb = self.tracks.get(b)
                if tb is None:
                    continue
                area_a, area_b = _area(ta.box), _area(tb.box)
                if area_b <= 0 or area_a < area_b * ratio:
                    continue  # `a` is not substantially bigger than `b`
                inter = _intersection(ta.box, tb.box)
                if inter / area_b >= contain:   # `a` swallows `b` → `a` is bloated
                    drop.append(a)
                    break
        return drop

    def reset(self) -> None:
        self.tracks.clear()
        self._last_sig = None


def _area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _intersection(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
