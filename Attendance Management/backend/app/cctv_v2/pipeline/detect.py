"""YOLO person detection for the V2 pipeline. Detection only -- nothing else.

WHAT THIS DOES AND WHERE IT SITS
--------------------------------
    RTSP -> grabber -> latest-frame slot -> scheduler -> [THIS] -> detections

The scheduler has already decided which camera goes next and confirmed its
frame is fresh enough to be worth looking at. This module's whole job is to turn
that one frame into person boxes.

It does NOT track, recognise faces, judge line crossings or write attendance.
Those are later steps and must not leak in here. In V1 the equivalent code lived
inside a 3,166-line module that did all of them, which is why a change to
attendance rules could alter what the detector saw.

ONE MODEL, NOT FOUR
-------------------
There is a single shared YOLO instance for all four cameras. The role profile
supplies imgsz and conf per call, so a doorway runs at 640/0.15 and a room at
480/0.03 through the same weights.

This is a deliberate difference from V1, which gave every camera its OWN YOLO
object -- and had to, because it called `track(persist=True)` and ultralytics
keeps ByteTrack state on the model. Four instances of yolo11m is four times the
memory for one set of weights.

Step 6 does stateless `predict`, so one instance is correct here. WHEN TRACKING
ARRIVES IN STEP 7 THIS MUST BE REVISITED: sharing a model across cameras while
persisting tracker state would let camera 57's tracks follow a frame from
camera 59, which is the kind of bug that produces a plausible number nobody can
explain.

THREADING
---------
The scheduler runs ONE inference worker, so `detect` is called from one thread
at a time. That is what makes a shared model safe. If a second worker is ever
added, this needs a lock -- and the measurement that justified a single worker
(1 slot 0.58 passes/s vs 2 slots 0.56, because one YOLO pass already saturates
four physical cores) says it should not be.

PERSON CLASS ONLY
-----------------
`classes=[0]` restricts COCO inference to the person class at the source, and
the results are filtered again on the way out. Two checks for one rule because
they fail differently: the argument is an optimisation the model may ignore if
it is ever swapped for a build with a different class map, while the filter is
the guarantee.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from app.cctv_v2.capture.grabber import FrameSnapshot
from app.cctv_v2.config.cameras import profile_for, role_for

logger = logging.getLogger(__name__)

# COCO class 0. Named rather than spelled inline so a reader does not have to
# know the COCO ordering to understand the filter.
PERSON_CLASS_ID = 0

# COCO class 56. Used ONLY for setup and sanity checks, never as the per-frame
# source of truth for occupancy: measured against these two rooms it returned
# 1-9 chairs on camera 59 (about 10 visible) and 3-10 on camera 60 (6-8
# visible), depending on input size and confidence. Seats are configured in
# config/geometry.py instead. See occupancy.py.
CHAIR_CLASS_ID = 56

# How many recent latencies to keep per role for percentile reporting. Bounded
# because this runs for days; an unbounded list would be a slow leak.
_LATENCY_WINDOW = 200


@dataclass(frozen=True)
class PersonDetection:
    """One person the detector found in one frame."""

    camera_id: int
    frame_timestamp: float
    frame_sequence: int
    bbox: tuple[float, float, float, float]     # x1, y1, x2, y2 in frame pixels
    confidence: float
    class_id: int = PERSON_CLASS_ID

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class DetectionResult:
    """Everything one inference pass produced, including what it cost.

    The cost fields are not decoration. The whole point of Step 6 is to find out
    whether V2 feeds the right frames to YOLO at a workable rate, and that
    cannot be answered without per-pass latency and the age of the frame that
    was actually processed.
    """

    camera_id: int
    role: str
    frame_sequence: int
    frame_timestamp: float
    detections: tuple[PersonDetection, ...]
    inference_ms: float
    frame_age_ms: float          # how old the picture was when inference STARTED
    imgsz: int                   # what the profile asked for, recorded per pass
    conf: float

    @property
    def count(self) -> int:
        return len(self.detections)


@dataclass
class RoleTiming:
    """Latency telemetry for one role.

    Doorway and room are never pooled. They run at different input sizes against
    different scenes, so a combined average would describe no camera that
    exists.
    """

    role: str
    passes: int = 0
    frames_with_people: int = 0
    total_detections: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    min_ms: float = 0.0
    failures: int = 0
    recent_ms: list = field(default_factory=list)

    def record(self, ms: float, n_detections: int) -> None:
        self.passes += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)
        self.min_ms = ms if self.passes == 1 else min(self.min_ms, ms)
        self.total_detections += n_detections
        self.frames_with_people += int(n_detections > 0)
        self.recent_ms.append(ms)
        if len(self.recent_ms) > _LATENCY_WINDOW:
            del self.recent_ms[0]

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.passes if self.passes else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.recent_ms:
            return 0.0
        s = sorted(self.recent_ms)
        return s[min(len(s) - 1, int(len(s) * 0.95))]

    @property
    def detections_per_frame(self) -> float:
        return self.total_detections / self.passes if self.passes else 0.0

    @property
    def inference_fps(self) -> float:
        """Passes per second this role could sustain if it had the worker to
        itself.

        NOT the rate it actually gets. The scheduler shares one worker between
        four cameras by design, so real per-camera throughput is several times
        lower. Reporting only this number is how V1 came to believe a doorway
        was being analysed every 0.12s.
        """
        return 1000.0 / self.avg_ms if self.avg_ms > 0 else 0.0


class PersonDetector:
    """Turns one fresh frame into person boxes, using its camera's profile."""

    def __init__(
        self,
        model_loader: Optional[Callable[[], object]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._load = model_loader or _load_yolo
        self._clock = clock
        self._model = None
        self._load_failed = False
        self.timing: dict[str, RoleTiming] = {
            "doorway": RoleTiming("doorway"),
            "room": RoleTiming("room"),
        }

    # -- model ---------------------------------------------------------------
    def _model_or_none(self):
        """Load once, lazily. A failed load is remembered, not retried per frame.

        Retrying would turn a missing weights file into a stack trace at the
        camera frame rate, which is how V1 hid a broken model behind a wall of
        identical tracebacks while every camera silently detected nothing.
        """
        if self._model is None and not self._load_failed:
            try:
                self._model = self._load()
            except Exception:
                self._load_failed = True
                logger.exception("cctv_v2: YOLO unavailable -- detection disabled")
        return self._model

    @property
    def available(self) -> bool:
        return self._model_or_none() is not None

    # -- detection -----------------------------------------------------------
    def detect(self, camera_id: int, snapshot: FrameSnapshot) -> DetectionResult:
        """Detect people in one frame, at this camera's role settings.

        Never raises. A camera whose inference fails returns an empty result and
        is counted as a failure, because one bad frame must not stop the
        scheduler serving the other three.
        """
        role = role_for(camera_id)
        profile = profile_for(camera_id)
        started = self._clock()
        frame_age_ms = max(0.0, (started - snapshot.timestamp) * 1000.0)

        model = self._model_or_none()
        if model is None:
            return self._empty(camera_id, role, snapshot, profile, 0.0, frame_age_ms)

        try:
            raw = model.predict(
                snapshot.frame,
                imgsz=profile.input_size,
                conf=profile.predict_conf,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )
        except Exception:
            self.timing[role].failures += 1
            logger.exception(
                "cctv_v2: detection failed camera=%s seq=%s",
                camera_id, snapshot.sequence,
            )
            elapsed = (self._clock() - started) * 1000.0
            return self._empty(camera_id, role, snapshot, profile, elapsed, frame_age_ms)

        elapsed_ms = (self._clock() - started) * 1000.0
        detections = _parse(raw, camera_id, snapshot)
        self.timing[role].record(elapsed_ms, len(detections))

        logger.debug(
            "DETECT camera=%s role=%s seq=%s people=%d %.0fms imgsz=%d conf=%.2f",
            camera_id, role, snapshot.sequence, len(detections),
            elapsed_ms, profile.input_size, profile.predict_conf,
        )
        return DetectionResult(
            camera_id=camera_id,
            role=role,
            frame_sequence=snapshot.sequence,
            frame_timestamp=snapshot.timestamp,
            detections=detections,
            inference_ms=elapsed_ms,
            frame_age_ms=frame_age_ms,
            imgsz=profile.input_size,
            conf=profile.predict_conf,
        )

    @staticmethod
    def _empty(camera_id, role, snapshot, profile, ms, age_ms) -> DetectionResult:
        return DetectionResult(
            camera_id=camera_id, role=role,
            frame_sequence=snapshot.sequence, frame_timestamp=snapshot.timestamp,
            detections=(), inference_ms=ms, frame_age_ms=age_ms,
            imgsz=profile.input_size, conf=profile.predict_conf,
        )

    def summary(self) -> dict:
        """Per-ROLE timing. See RoleTiming for why the two are never pooled."""
        out = {}
        for role, t in self.timing.items():
            out[role] = {
                "passes": t.passes,
                "failures": t.failures,
                "avg_ms": round(t.avg_ms, 1),
                "p95_ms": round(t.p95_ms, 1),
                "min_ms": round(t.min_ms, 1),
                "max_ms": round(t.max_ms, 1),
                "inference_fps": round(t.inference_fps, 3),
                "detections_per_frame": round(t.detections_per_frame, 2),
                "frames_with_people": t.frames_with_people,
            }
        return out


def detect_chairs(model, frame, imgsz: int, conf: float,
                  camera_id: int) -> tuple[tuple[float, float, float, float], ...]:
    """Chair boxes for SETUP ONLY.

    Deliberately not a method on PersonDetector and not called by the pipeline.
    Chair occupancy is decided from configured zones plus person association;
    this exists so a human placing those zones has the detector's opinion in
    front of them, and so a later sanity check can ask whether the configured
    seat count is in the right neighbourhood.
    """
    try:
        raw = model.predict(frame, imgsz=imgsz, conf=conf,
                            classes=[CHAIR_CLASS_ID], verbose=False)
    except Exception:                                          # noqa: BLE001
        logger.exception("cctv_v2: chair detection failed camera=%s", camera_id)
        return ()
    out = []
    for result in _as_sequence(raw):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        try:
            xyxy, classes = boxes.xyxy.tolist(), boxes.cls.tolist()
        except Exception:                                      # noqa: BLE001
            continue
        for box, cls in zip(xyxy, classes):
            if int(cls) != CHAIR_CLASS_ID:
                continue
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2))
    return tuple(out)


def _parse(raw, camera_id: int, snapshot: FrameSnapshot) -> tuple[PersonDetection, ...]:
    """Pull person boxes out of an ultralytics Results list.

    Defensive on purpose: this is the boundary with a third-party API whose
    result shape has changed across versions, and a parse error here would
    otherwise surface as "the camera sees nobody".
    """
    out: list[PersonDetection] = []
    for result in _as_sequence(raw):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        try:
            xyxy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            classes = boxes.cls.tolist()
        except Exception:
            logger.exception("cctv_v2: unreadable YOLO boxes camera=%s", camera_id)
            continue

        for box, conf, cls in zip(xyxy, confs, classes):
            # Second person-class check. `classes=[0]` in the predict call is
            # the request; this is the guarantee. See the module docstring.
            if int(cls) != PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            if x2 <= x1 or y2 <= y1:
                continue                      # degenerate box, not a person
            out.append(PersonDetection(
                camera_id=camera_id,
                frame_timestamp=snapshot.timestamp,
                frame_sequence=snapshot.sequence,
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                class_id=PERSON_CLASS_ID,
            ))
    return tuple(out)


def _as_sequence(raw) -> Sequence:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return raw
    return (raw,)


def _load_yolo():
    """Load the shared yolo11m instance.

    The path comes from the profiles, which name the same weights for both
    roles -- only imgsz and conf differ. That assumption is checked rather than
    trusted: if the two profiles ever name different models, one shared instance
    silently becomes wrong for one of the roles.
    """
    from ultralytics import YOLO

    from app.cctv_v2.config.profiles import DOORWAY, ROOM

    if DOORWAY.model != ROOM.model:
        raise RuntimeError(
            "cctv_v2 detect assumes one shared model; profiles name "
            f"{DOORWAY.model!r} and {ROOM.model!r}. Give each role its own "
            "detector before letting these diverge."
        )

    path = _resolve(DOORWAY.model)
    if path is None:
        raise FileNotFoundError(f"YOLO weights not found: {DOORWAY.model}")
    model = YOLO(path)
    logger.info(
        "cctv_v2: YOLO loaded %s (doorway %dpx/%.2f, room %dpx/%.2f)",
        path, DOORWAY.input_size, DOORWAY.predict_conf,
        ROOM.input_size, ROOM.predict_conf,
    )
    return model


def _resolve(path: str) -> Optional[str]:
    """Resolve relative to CWD, else to the backend root."""
    if os.path.exists(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    candidate = os.path.join(backend_root, path)
    return candidate if os.path.exists(candidate) else None
