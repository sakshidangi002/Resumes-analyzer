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
import time
from functools import lru_cache

from app.core.config import get_settings
from app.services.person_tracker import PersonTrack

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inference admission control
#
# This used to be a single global Lock: every camera waited for every other one,
# so with 2 monitor cameras at ~2.7 s/frame each camera only got a fresh look
# every ~5.3 s and the overlay was drawn from a frame that old.
#
# It is now a bounded SEMAPHORE. The purpose is unchanged — stop N cameras from
# oversubscribing the CPU — but N is configurable instead of hard-wired to 1:
#   yolo_max_concurrent_inference = 1  -> identical to the old behaviour
#   yolo_max_concurrent_inference = 0  -> auto (one slot per 4 cores)
# Tracker state is still per-engine (each camera owns its model), so allowing
# concurrency cannot mix up ByteTrack ids between cameras.
# ---------------------------------------------------------------------------
def _auto_concurrency() -> int:
    """Slots for simultaneous inference. Explicit setting wins; else auto.

    MEASURED (this 4-physical-core box, yolo11m @960, 2 monitor cameras):
        1 slot  -> each camera refreshes every 4.10 s
        2 slots -> each camera refreshes every 4.52 s   (10% WORSE)
    One inference already saturates the physical cores, so a second concurrent
    one just splits them and adds overhead. Parallelism only pays once there are
    spare PHYSICAL cores (or a GPU), hence the >=8 gate — anything less defaults
    to 1, which is exactly the historical serial behaviour.
    """
    s = get_settings()
    n = int(getattr(s, "yolo_max_concurrent_inference", 0) or 0)
    if n > 0:
        return n
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or 0
    except Exception:
        physical = (os.cpu_count() or 4) // 2  # assume hyperthreading
    return 2 if physical >= 8 else 1


_MAX_CONCURRENT = _auto_concurrency()
_slots = threading.BoundedSemaphore(_MAX_CONCURRENT)

# How close to the frame border counts as "at the edge" (fraction of width or
# height). A person leaves a room through an edge; a person hidden by a chair
# does not move. See the expiry logic in ByteTrackEngine.update.
_EDGE_FRACTION = float(os.getenv("CCTV_TRACK_EDGE_FRACTION", "0.12"))

# How long a no-longer-detected track is kept, expressed in ANALYSIS CYCLES
# rather than seconds.
#
# An earlier version used absolute seconds (1s at the edge) — which was shorter
# than a monitor camera's analysis interval, so a track missed even ONCE near an
# edge was deleted instantly. Detection on this view is intermittent by nature:
# a seated person flickers in and out between cycles, and people at the edges of
# the frame are exactly the ones most often missed. The result was worse recall,
# not better.
#
# Cycles are converted to seconds using the engine's OWN measured update
# interval, so this self-calibrates whatever the analysis rate or inference
# queue contention happens to be.
_EDGE_HOLD_CYCLES = float(os.getenv("CCTV_TRACK_EDGE_HOLD_CYCLES", "2"))
_INTERIOR_HOLD_CYCLES = float(os.getenv("CCTV_TRACK_INTERIOR_HOLD_CYCLES", "6"))
# Floors, so a fast camera still holds a track for a usable length of time.
_EDGE_HOLD_MIN_SEC = float(os.getenv("CCTV_TRACK_EDGE_HOLD_MIN_SEC", "2.0"))


# Cameras that ALSO run a close overlapping crop through YOLO, on top of the
# full-frame pass.
#
# MEASURED on this box (yolo11m @960, one Exit-camera frame, model warm):
#
#     full frame only          4131 ms   1 track
#     full frame + crop pass   8581 ms   1 track
#
# The second pass doubled the per-frame cost and found nothing the full-frame
# pass had not already tracked. Paying that on every feed is what put analysis
# seconds behind the live picture, which is why a plainly visible person still
# read "People: 0" on the dashboard.
#
# It can still earn its cost on a room camera where people sit behind desks and
# monitors, so the capability stays -- but opt-in per camera, not billed to all.
# Comma-separated ids, e.g. CCTV_CROP_ASSIST_CAMERAS=59,60
_CROP_ASSIST_CAMERAS = {
    c.strip() for c in os.getenv("CCTV_CROP_ASSIST_CAMERAS", "").split(",") if c.strip()
}

# Adopt a confident detection that ByteTrack has NOT yet turned into a track.
#
# ByteTrack only assigns an id once a detection has been matched across TWO
# passes; until then Ultralytics returns the box with id=None and this engine
# used to drop it. That rule assumes passes are close together. They are not:
# measured on the live Exit camera, consecutive passes were 5-42s apart, while
# a person crosses the corridor in about 2s. So somebody walking through is
# seen on exactly ONE pass, never gets an id, and the camera reports
#
#     detections=1 scores=[0.784] tracks=0 ids=[]
#
# - a plainly visible person, detected at 0.784, counted as nobody.
#
# Above this confidence the detection is adopted immediately as a provisional
# track. The bar is deliberately well clear of noise: empty corridor frames
# score 0.00-0.01, real people 0.43-0.78. Set to 0 to restore the old
# id-only behaviour.
#
# Safe for attendance: a body track cannot mark attendance on its own. That
# requires a face match clearing services/attendance_gate.
_ADOPT_UNTRACKED_MIN_CONF = float(os.getenv("CCTV_ADOPT_UNTRACKED_MIN_CONF", "0.35"))
_INTERIOR_HOLD_MIN_SEC = float(os.getenv("CCTV_TRACK_INTERIOR_HOLD_MIN_SEC", "6.0"))

# Two published boxes overlapping by at least this fraction of the SMALLER box
# are treated as the same person and merged. See _dedupe_overlapping.
_DEDUPE_OVERLAP = float(os.getenv("CCTV_TRACK_DEDUPE_OVERLAP", "0.55"))

# Cap the threads each inference session may use. Without this ONNX Runtime /
# torch take every core for EVERY concurrent session (2 x 8 threads on 8 cores),
# which thrashes and makes the parallel version SLOWER than the serial one.
def _cap_threads() -> None:
    s = get_settings()
    per = int(getattr(s, "yolo_threads_per_session", 0) or 0)
    if per <= 0:
        cores = os.cpu_count() or 4
        per = max(1, cores // max(1, _MAX_CONCURRENT))
    # Must be set before the runtime builds its thread pools (i.e. before the
    # first model load), so this runs at import time.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ORT_NUM_THREADS"):
        os.environ.setdefault(var, str(per))
    try:
        import torch  # noqa
        torch.set_num_threads(per)
    except Exception:
        logger.warning("torch.set_num_threads failed", exc_info=True)
    logger.info(
        "Person inference: max_concurrent=%d threads_per_session=%d (cores=%s)",
        _MAX_CONCURRENT, per, os.cpu_count(),
    )


_cap_threads()

# Rolling performance counters, per camera id.
_perf_lock = threading.Lock()
_perf: dict[str, dict] = {}


def get_perf_stats() -> dict:
    """Per-camera inference stats: {camera_id: {calls, avg_ms, last_ms, waits_ms}}."""
    with _perf_lock:
        return {k: dict(v) for k, v in _perf.items()}


def _record_perf(camera_id: str, wait_ms: float, infer_ms: float) -> dict:
    with _perf_lock:
        d = _perf.setdefault(
            camera_id, {"calls": 0, "total_ms": 0.0, "last_ms": 0.0, "wait_ms": 0.0}
        )
        d["calls"] += 1
        d["total_ms"] += infer_ms
        d["last_ms"] = infer_ms
        d["wait_ms"] = wait_ms
        d["avg_ms"] = d["total_ms"] / d["calls"]
        return dict(d)


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


def _static_onnx_size(path: str) -> int | None:
    """The fixed square input size of an ONNX export, or None if it is dynamic.

    Exists because a mismatch here does not degrade gracefully -- it kills the
    camera outright. models/yolo11m.onnx is exported with a STATIC
    [1,3,960,960] input, so running it at any other imgsz makes onnxruntime
    reject EVERY frame with

        INVALID_ARGUMENT : Got invalid dimensions for input: images
        index: 2 Got: 480 Expected: 960

    which surfaces only as a per-frame traceback while the camera detects
    nothing at all, indefinitely. That happened on the Exit camera: a
    YOLO_PERSON_IMGSZ tuned against the .pt build was applied to the .onnx one.
    """
    if not path.lower().endswith(".onnx"):
        return None
    try:
        import onnx

        model = onnx.load(path, load_external_data=False)
        dims = model.graph.input[0].type.tensor_type.shape.dim
        h = dims[2].dim_value if dims[2].HasField("dim_value") else 0
        w = dims[3].dim_value if dims[3].HasField("dim_value") else 0
        return int(h) if h and h == w else None
    except Exception:
        # Never let a probe stop a camera starting; the mismatch will simply
        # surface as it did before.
        logger.debug("could not read ONNX input shape for %s", path, exc_info=True)
        return None


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
        model_path: str | None = None,
    ):
        s = get_settings()
        # MONITOR cameras may run a lighter/faster model than the IN/OUT
        # attendance cameras, which must stay on the accurate one.
        self.model_path_override = model_path or None
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
        self._last_update_ts: float = 0.0
        self._cycle_sec: float = 0.0          # smoothed interval between update() calls
        # Monitor-only supplemental detections use a closer overlapping crop for
        # the far/right workstation. They are merged into the same body-track
        # table after the normal full-frame ByteTrack pass.
        self._next_aux_id: int = 1_000_000

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def _add_crop_assist_tracks(self, model, frame_bgr, seen: set[int]) -> list[float]:
        """Recover people too small/occluded for the full-frame pass.

        Opt-in per camera (see _CROP_ASSIST_CAMERAS): this runs a SECOND full
        YOLO inference, which on this hardware doubles the per-frame cost.

        The Dev-room camera puts the lower-right workstation far from the
        optical centre and behind monitors/chairs. A closer overlapping crop
        gives YOLO more pixels for that region. This is body detection only;
        face recognition still happens later in camera_service.
        """
        import cv2

        height, width = frame_bgr.shape[:2]
        # Use a broad overlapping crop for every body-tracking camera. The
        # hallway/Exit view puts people at the far end of the frame, while the
        # room view hides them behind desks; both benefit from giving YOLO more
        # pixels without changing the full-frame detector's geometry.
        crop_x1 = int(width * 0.15)
        crop_y1 = int(height * 0.10)
        crop = frame_bgr[crop_y1:height, crop_x1:width]
        if crop.size == 0:
            return []

        try:
            _slots.acquire()
            try:
                result = model.predict(
                    crop,
                    classes=[0],
                    conf=max(0.02, self.conf * 0.5),
                    iou=self.iou,
                    imgsz=self.imgsz,
                    verbose=False,
                )[0]
            finally:
                _slots.release()
        except Exception:
            logger.exception("Camera %s: supplemental body crop failed", self.camera_id)
            return []

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        scores: list[float] = []
        for score, local_box in zip(
            boxes.conf.cpu().tolist(), boxes.xyxy.cpu().tolist()
        ):
            score = float(score)
            x1, y1, x2, y2 = local_box
            box = (
                int(x1 + crop_x1), int(y1 + crop_y1),
                int(x2 + crop_x1), int(y2 + crop_y1),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            # Associate with an existing full-frame track when possible.
            matched_id = None
            best_iou = 0.0
            for tid, track in self.tracks.items():
                overlap = self._iou(box, track.box)
                if overlap > best_iou:
                    best_iou, matched_id = overlap, tid
            if matched_id is not None and best_iou >= 0.15:
                self.tracks[matched_id].update_box(box)
                seen.add(matched_id)
                scores.append(round(score, 3))
                continue

            aux_id = self._next_aux_id
            self._next_aux_id += 1
            self.tracks[aux_id] = PersonTrack(
                track_id=aux_id, box=box, max_misses=self.max_misses
            )
            seen.add(aux_id)
            scores.append(round(score, 3))
            logger.info(
                "YOLO camera=%s supplemental body track=%d score=%.3f box=%s",
                self.camera_id, aux_id, score, box,
            )
        return scores

    def _measured_cycle_sec(self) -> float:
        """Smoothed seconds between update() calls.

        Track hold windows are expressed in cycles and converted with this, so
        they stay correct whether this camera is analysing every 0.12s or every
        4s under inference-queue contention.
        """
        now = time.time()
        if self._last_update_ts:
            delta = now - self._last_update_ts
            if 0.0 < delta < 60.0:
                self._cycle_sec = (
                    delta if self._cycle_sec <= 0 else 0.7 * self._cycle_sec + 0.3 * delta
                )
        self._last_update_ts = now
        return self._cycle_sec if self._cycle_sec > 0 else 1.0

    def _get_model(self):
        """Lazily build this camera's OWN YOLO instance (isolated tracker state)."""
        if self._model is None:
            path = _resolve(self.model_path_override) if self.model_path_override else None
            if path is None:
                path = _model_path()
            if path is None:
                return None
            try:
                from ultralytics import YOLO

                self._model = YOLO(path)
            except Exception:
                logger.exception("Camera %s: failed to load YOLO11 model", self.camera_id)
                return None
            # A static-input ONNX model cannot run at any other size. Correct it
            # loudly instead of letting every frame raise INVALID_ARGUMENT and
            # the camera silently see nothing.
            required = _static_onnx_size(path)
            if required and required != self.imgsz:
                logger.error(
                    "Camera %s: %s has a STATIC %dx%d input but imgsz=%d was "
                    "requested — forcing %d. Set imgsz to %d for this model, or "
                    "point the camera at a .pt build, which accepts any size.",
                    self.camera_id, os.path.basename(path), required, required,
                    self.imgsz, required, required,
                )
                self.imgsz = required
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

        # Bounded concurrency: `_slots` admits up to N inferences at once (N=1 is
        # the historical serial behaviour). `wait` is time spent queueing behind
        # other cameras — the number that used to make the overlay stale.
        _t_wait0 = time.time()
        _slots.acquire()
        _wait_ms = (time.time() - _t_wait0) * 1000.0
        _t_inf0 = time.time()
        try:
            results = model.track(frame_bgr, **kwargs)
        finally:
            _slots.release()
        _infer_ms = (time.time() - _t_inf0) * 1000.0
        stats = _record_perf(self.camera_id, _wait_ms, _infer_ms)

        s_perf = get_settings()
        _every = int(getattr(s_perf, "perf_log_every", 20) or 0)
        if _every and stats["calls"] % _every == 0:
            logger.info(
                "PERF camera=%s infer=%.0fms avg=%.0fms wait=%.0fms "
                "cycle=%.2fs fps=%.2f concurrency=%d",
                self.camera_id, _infer_ms, stats["avg_ms"], _wait_ms,
                (_wait_ms + _infer_ms) / 1000.0,
                (1000.0 / (_wait_ms + _infer_ms)) if (_wait_ms + _infer_ms) > 0 else 0.0,
                _MAX_CONCURRENT,
            )

        detections = 0
        det_scores: list[float] = []
        seen: set[int] = set()
        if results:
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                detections = int(len(boxes))          # after conf + class + NMS
                # Raw per-detection confidences. THE diagnostic for "the room
                # has 4 people but shows 1": if the scores are there but low,
                # it is the tracker's new_track_thresh discarding them; if the
                # detections themselves are missing, no threshold will help and
                # the model or input resolution is the problem.
                try:
                    if getattr(boxes, "conf", None) is not None:
                        det_scores = [round(float(c), 3) for c in boxes.conf.cpu().tolist()]
                except Exception:
                    det_scores = []
                xyxy = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
                ids = None
                if getattr(boxes, "id", None) is not None:
                    ids = boxes.id.int().cpu().tolist()

                for idx, box in enumerate(xyxy):
                    b = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    tid = ids[idx] if (ids is not None and idx < len(ids)) else None

                    if tid is not None:
                        seen.add(int(tid))
                        pt = self.tracks.get(int(tid))
                        if pt is None:
                            self.tracks[int(tid)] = PersonTrack(
                                track_id=int(tid), box=b, max_misses=self.max_misses
                            )
                        else:
                            pt.update_box(b)
                        continue

                    # No id: ByteTrack has seen this detection once and is
                    # waiting for a second pass to confirm it. At this cadence
                    # that second pass may never see the person. Adopt it now if
                    # it is confidently a person -- see _ADOPT_UNTRACKED_MIN_CONF.
                    score = det_scores[idx] if idx < len(det_scores) else 0.0
                    if _ADOPT_UNTRACKED_MIN_CONF <= 0 or score < _ADOPT_UNTRACKED_MIN_CONF:
                        continue

                    # Do not double-count: if it overlaps a track we already
                    # hold, refresh that one instead of inventing a second.
                    matched_id, best_iou = None, 0.0
                    for known_id, known in self.tracks.items():
                        overlap = self._iou(b, known.box)
                        if overlap > best_iou:
                            best_iou, matched_id = overlap, known_id
                    if matched_id is not None and best_iou >= 0.30:
                        self.tracks[matched_id].update_box(b)
                        seen.add(matched_id)
                        continue

                    aux_id = self._next_aux_id
                    self._next_aux_id += 1
                    # NOTE max_misses is NOT what retires this track. Retention
                    # in this engine is TIME based -- see the edge/interior hold
                    # windows below -- so a provisional track lives for
                    # max(_INTERIOR_HOLD_MIN_SEC, _INTERIOR_HOLD_CYCLES * cycle)
                    # like any other. It is passed for consistency only.
                    #
                    # Consequence worth knowing on a doorway: a walker who has
                    # left keeps their box until that window elapses, so the
                    # count decays rather than dropping instantly. That hold
                    # exists for room cameras, where a seated person vanishes
                    # behind a chair back and must not blink out. Shorten it with
                    # CCTV_TRACK_INTERIOR_HOLD_CYCLES / _MIN_SEC if a lingering
                    # doorway count matters more than steady room boxes.
                    self.tracks[aux_id] = PersonTrack(
                        track_id=aux_id, box=b, max_misses=self.max_misses
                    )
                    seen.add(aux_id)
                    logger.info(
                        "YOLO camera=%s adopted untracked detection score=%.3f as "
                        "provisional track=%d (ByteTrack had not confirmed it yet)",
                        self.camera_id, score, aux_id,
                    )

        # Full-frame inference is the primary detector and is enough for almost
        # every view. A camera listed in CCTV_CROP_ASSIST_CAMERAS additionally
        # gets one overlapping close crop, for rooms where people sit behind
        # desks and monitors. It costs a SECOND full YOLO inference per frame --
        # see _CROP_ASSIST_CAMERAS for the measurement -- so it is never paid by
        # default, and never by an IN/OUT camera unless explicitly listed.
        if str(self.camera_id) in _CROP_ASSIST_CAMERAS:
            supplemental_scores = self._add_crop_assist_tracks(model, frame_bgr, seen)
            if supplemental_scores:
                detections += len(supplemental_scores)
                det_scores.extend(supplemental_scores)

        # Drop "nested" duplicates: on this ceiling view YOLO often emits a tight
        # box on a seated person AND a second, bloated box running down over their
        # chair/bag. The two barely overlap, so NMS keeps both and one person gets
        # two boxes + two ids. Whenever one box swallows another, keep the TIGHT
        # one (the bloated box is the wrong one — it is mostly furniture).
        for tid in self._nested_ids(seen):
            seen.discard(tid)
            self.tracks.pop(tid, None)

        # Age / expire tracks ByteTrack no longer reports.
        #
        # A track that stops being detected means one of two very different
        # things, and the old frame-count expiry conflated them:
        #
        #   * OCCLUDED  — a seated person hidden by a high-backed chair. They
        #     have not moved, so holding their box is correct.
        #   * DEPARTED  — they walked out. Holding their box leaves a ghost
        #     hovering over an empty room.
        #
        # Position separates the cases: you leave a room through an EDGE of the
        # frame. A track last seen against an edge is treated as departed and
        # dropped quickly; one in the interior is held.
        #
        # The hold is also capped in WALL-CLOCK seconds rather than analysis
        # frames. Frame counting was unreliable because the analysis interval
        # varies with inference-queue contention — the same 3-frame hold could
        # mean 4 seconds or 15.
        frame_h, frame_w = frame_bgr.shape[:2]
        edge_x = frame_w * _EDGE_FRACTION
        edge_y = frame_h * _EDGE_FRACTION
        now = time.time()

        # Hold windows scale with how fast this engine is actually being called,
        # so an intermittent detection is never dropped after a single miss.
        cycle = self._measured_cycle_sec()
        edge_limit = max(_EDGE_HOLD_MIN_SEC, _EDGE_HOLD_CYCLES * cycle)
        interior_limit = max(_INTERIOR_HOLD_MIN_SEC, _INTERIOR_HOLD_CYCLES * cycle)

        for tid in list(self.tracks.keys()):
            if tid in seen:
                continue
            track = self.tracks[tid]
            track.mark_missed()

            cx, cy = track.centroid()
            at_edge = (
                cx <= edge_x or cx >= frame_w - edge_x
                or cy <= edge_y or cy >= frame_h - edge_y
            )
            held_for = now - track.last_seen
            limit = edge_limit if at_edge else interior_limit

            if held_for > limit:
                logger.debug(
                    "Camera %s: dropping track %d after %.1fs (%s, limit %.1fs)",
                    self.camera_id, tid, held_for,
                    "left via frame edge" if at_edge else "occluded too long",
                    limit,
                )
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

        # ONE PERSON, ONE BOX.
        #
        # A permissive new_track_thresh (needed so seated people are detected at
        # all) makes ByteTrack spawn a fresh id rather than re-associating when a
        # detection is weak. Combined with publishing held tracks, one person
        # ends up wearing several overlapping boxes with different ids — observed
        # live as "#36" stacked with "2" and "3", and a People count of 6 in a
        # room of 4.
        #
        # It is not merely cosmetic: each duplicate track accumulates its OWN
        # embedding fusion, so the observations that should combine into one
        # confident identity are split across fragments that each stay too weak
        # to match. Deduplicating is a prerequisite for recognition working at
        # all here.
        live = self._dedupe_overlapping(live, seen)

        # Log only when the picture CHANGES (detections / tracks / ids) so a
        # steady scene doesn't spam the log every analysis tick.
        sig = (detections, len(live), tuple(sorted(seen)))
        if sig != self._last_sig:
            self._last_sig = sig
            logger.info(
                "YOLO camera=%s detections=%d scores=%s tracks=%d ids=%s "
                "new_track_thresh=%s",
                self.camera_id, detections, det_scores, len(live), sorted(seen),
                os.path.basename(self.tracker_cfg),
            )
            # Detections that will never become tracks. If this fires
            # repeatedly the tracker config is discarding real people.
            if det_scores and len(live) < detections:
                logger.info(
                    "YOLO camera=%s %d detection(s) did NOT become tracks "
                    "(lowest kept vs dropped: %s) — if these are real people, "
                    "this camera needs the permissive tracker "
                    "(add its id to CCTV_STEEP_CAMERAS)",
                    self.camera_id, detections - len(live), sorted(det_scores),
                )

        return live

    def _dedupe_overlapping(self, live: list, seen: set) -> list:
        """Collapse overlapping tracks so each person is published once.

        Overlap is measured as intersection / smaller-area rather than IoU: a
        tight box on a torso sitting inside a bloated full-body box has low IoU
        but is obviously the same person.

        When two tracks collide the survivor is chosen by evidence — detected
        this frame beats merely held, then longer-lived, then lower id (older).
        The loser's IDENTITY AND FUSED EMBEDDINGS ARE MERGED INTO THE SURVIVOR,
        which is the valuable part: ByteTrack changes a person's id constantly on
        this view, and without the merge every id change threw away both their
        name and every face observation collected so far.
        """
        if len(live) < 2:
            return live

        # Best evidence first, so earlier entries win collisions.
        ordered = sorted(
            live,
            key=lambda t: (t.track_id in seen, t.age, -t.track_id),
            reverse=True,
        )

        kept: list = []
        for track in ordered:
            duplicate_of = None
            for survivor in kept:
                if _overlap_ratio(track.box, survivor.box) >= _DEDUPE_OVERLAP:
                    duplicate_of = survivor
                    break

            if duplicate_of is None:
                kept.append(track)
                continue

            # Merge upward, then drop.
            if duplicate_of.employee_id is None and track.employee_id is not None:
                duplicate_of.employee_id = track.employee_id
                duplicate_of.employee_name = track.employee_name
                duplicate_of.employee_code = track.employee_code
                duplicate_of.matched = track.matched
                duplicate_of.confidence = track.confidence
                duplicate_of.identity_source = track.identity_source
            if track.fuser.observations > duplicate_of.fuser.observations:
                duplicate_of.fuser = track.fuser
            self.tracks.pop(track.track_id, None)

        if len(kept) != len(live):
            logger.debug(
                "Camera %s: merged %d duplicate box(es) -> %d people",
                self.camera_id, len(live) - len(kept), len(kept),
            )
        return kept

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


def _overlap_ratio(a, b) -> float:
    """Intersection over the SMALLER box's area.

    Deliberately not IoU: on this view a tight torso box often sits fully inside
    a bloated full-body box of the same person, which scores low IoU but 1.0
    here — which is the answer we want.
    """
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return (inter / smaller) if smaller > 0 else 0.0


def _area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _intersection(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
