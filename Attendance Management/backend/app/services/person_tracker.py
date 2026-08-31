"""IoU-based multi-person tracker with identity binding.

Tracks whole bodies across frames (stable Track IDs). A face recognised inside
a person's box binds that employee to the person's track, and the identity is
kept while the person is tracked — even when the face is no longer visible —
until they leave the frame. No external dependencies.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.services.embedding_fusion import EmbeddingFuser

logger = logging.getLogger(__name__)


def check_line_crossing(
    track: "PersonTrack",
    orientation: str,
    line_px: float,
    entry_direction: str,
    debounce_sec: float = 1.0,
) -> bool:
    """True if this track just crossed the doorway line in the entry direction.

    Crossing is detected as the centroid moving from one side of the line to the
    other between the previous and current frame. Debounced so line jitter does
    not fire repeatedly.
    """
    if track.prev_centroid is None:
        return False
    now = time.time()
    if now - track.last_crossing_time < debounce_sec:
        return False
    px, py = track.prev_centroid
    cx, cy = track.centroid()
    if orientation == "vertical":
        crossed = (px < line_px <= cx) or (px > line_px >= cx)
        direction = "right" if cx > px else "left"
    else:  # horizontal
        crossed = (py < line_px <= cy) or (py > line_px >= cy)
        direction = "down" if cy > py else "up"
    if crossed and direction == entry_direction:
        track.last_crossing_time = now
        return True
    return False


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class PersonTrack:
    track_id: int
    box: Tuple[int, int, int, int]
    last_seen: float = field(default_factory=time.time)

    # How confident the PERSON DETECTOR was about this body, 0..1.
    #
    # A separate field from `confidence` below, which despite its name is the
    # FACE MATCH score written by `bind_identity`. Two different measurements of
    # two different things had one name between them, and the occupancy API was
    # publishing the wrong one: it reported `confidence` as the person's
    # detection score, so an unrecognised person -- which on a room camera is
    # everyone -- was published as 0.0 confidence while being detected perfectly
    # well.
    #
    # Anything that asks "how sure are we somebody is there" wants this one.
    # Anything that asks "how sure are we WHO they are" wants the other.
    detection_confidence: float = 0.0

    # Identity bound from a recognised face inside this body.
    employee_id: Optional[int] = None
    employee_name: Optional[str] = None
    employee_code: Optional[str] = None
    matched: bool = False
    # FACE MATCH score, not detection. Kept under this name because
    # `bind_identity`, `get_display_info` and the dedupe merge all use it and
    # renaming it would touch the attendance path for no functional gain.
    confidence: float = 0.0
    last_recognition_time: float = 0.0
    # HOW this identity was established: "face" (ArcFace matched a visible
    # face) or "seat" (nobody's face was visible; this is who normally sits
    # here). They are very different claims and the overlay must not present
    # them identically — a positional guess shown as a confident name is the
    # same failure mode as the mislabelling this system already suffered.
    identity_source: Optional[str] = None

    # True when this track was ADOPTED from a detection ByteTrack had not yet
    # confirmed, rather than created from a tracker-assigned id. It asserts only
    # "something person-shaped was here on this pass" and carries no motion
    # history, so it must not be coasted like a confirmed track -- see the
    # retention loop in bytetrack_engine.
    provisional: bool = False

    # Stable-confirmation + attendance state (mirrors FaceTrack).
    pending_employee_id: Optional[int] = None
    confirm_count: int = 0
    attendance_marked: bool = False
    # Recognition provenance from the last successful face match on this track
    # (score, margin, snapshot path). Captured at match time because that is the
    # only moment the frame and the match result exist together — attendance is
    # marked later, and asynchronously.
    last_evidence: Optional[dict] = None

    # Quality-weighted fusion of every face embedding seen on this body track.
    # Matters MORE here than on a face track: a seated person is in view for
    # minutes, so there are far more observations to average, and a room
    # camera's faces are the smallest and noisiest in the system.
    fuser: EmbeddingFuser = field(default_factory=EmbeddingFuser)

    age: int = 0
    consecutive_misses: int = 0
    max_misses: int = 30  # frames a body survives without detection before delete

    # Line-crossing state
    prev_centroid: Optional[Tuple[float, float]] = None
    last_crossing_time: float = 0.0
    crossed: bool = False  # has crossed the doorway line at least once

    # The most recent attendance_gate decision for this track. Held between the
    # identification stage and the attendance stage of the same analysis tick,
    # which are separated because line-crossing state has to be updated in
    # between. Cleared once acted on.
    pending_decision: object = None
    unknown_event_marked: bool = False

    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update_box(self, box: Tuple[int, int, int, int]) -> None:
        self.prev_centroid = self.centroid()  # remember where we were
        self.box = box
        self.last_seen = time.time()
        self.age += 1
        self.consecutive_misses = 0

    def mark_missed(self) -> None:
        self.consecutive_misses += 1

    def is_expired(self) -> bool:
        return self.consecutive_misses >= self.max_misses

    def add_observation(self, embedding, quality: float) -> bool:
        """Fold one face embedding into this body track's fused template.

        ``quality`` is the FaceQuality soft score in (0, 1]. Returns False when
        the fuser rejected the observation as an outlier — on a body track that
        usually means the tracker handed this box to a different person, which
        is routine when people pass each other in a doorway.
        """
        return self.fuser.add(embedding, quality)

    def fused_embedding(self):
        """Quality-weighted mean of every face seen on this person, or None."""
        return self.fuser.fused()

    @property
    def observations(self) -> int:
        return self.fuser.observations

    @property
    def consensus(self) -> float:
        """How much this track's face observations agree with each other."""
        return self.fuser.consensus()

    @property
    def best_quality(self) -> float:
        return self.fuser.best_quality

    def needs_recognition(
        self, reverify_sec: float, min_observations: int = 0
    ) -> bool:
        """Recognise when not yet identified, when still short of the evidence
        the attendance gate will demand, or periodically to re-verify.

        The middle clause exists because the throttle and the gate were set
        against different timescales and together made attendance unreachable.

        `attendance_gate` requires `observations >= profile.min_observations`
        (3 on an IN/OUT camera), and observations are only accumulated on a pass
        where this method returns True. Throttling to one attempt per
        `reverify_sec` (5s) the moment a track has ANY match meant:

            pass 1  t=0.0   unmatched -> obs=1, matched=True
            pass 2  t=2.5   throttled -> no observation
            pass 3  t=5.0   obs=2
            pass 5  t=10.0  obs=3

        i.e. ~10 seconds of continuous tracking to satisfy a gate that guards a
        doorway people cross in about two. Measured consequence: 33 attendance
        decisions all-time, exactly one allowed, the rest blocked on
        `insufficient_observations` / `unstable_identity`.

        The throttle is right for what it was written for -- an employee sitting
        at a desk does not need re-identifying on every pass. It is only wrong
        while the track still lacks the evidence that will be demanded of it,
        which is precisely the window this clause covers. Nothing about the
        quality, score, margin or agreement bars changes.

        Callers pass 0 (the default) for cameras that never mark attendance, so
        MONITOR cameras keep the cheap throttled behaviour.
        """
        if not self.matched:
            return True
        if min_observations:
            fuser = getattr(self, "fuser", None)
            accepted = int(getattr(fuser, "accepted", 0) or 0) if fuser is not None else 0
            if accepted < min_observations:
                return True
        return (time.time() - self.last_recognition_time) >= reverify_sec

    def bind_identity(
        self,
        employee_id: Optional[int],
        employee_name: Optional[str],
        employee_code: Optional[str],
        matched: bool,
        confidence: float,
        source: str = "face",
    ) -> None:
        # Only overwrite with a positive match; a failed read never erases a
        # name that was already established for this person.
        if not matched:
            self.last_recognition_time = time.time()
            return

        # A FACE match always wins. A seat guess must never overwrite an
        # identity that was actually seen — otherwise someone sitting at a
        # colleague's desk would be renamed to that colleague.
        if source == "seat" and self.identity_source == "face":
            return

        self.employee_id = employee_id
        self.employee_name = employee_name
        self.employee_code = employee_code
        self.matched = True
        self.confidence = confidence
        self.identity_source = source
        self.last_recognition_time = time.time()

    # The attendance decision moved to services/attendance_gate — see the note
    # on FaceTrack for why a consecutive-frame counter was the wrong test.

    def get_display_info(self) -> dict:
        return {
            "track_id": self.track_id,
            "box": self.box,
            "employee_name": self.employee_name or "Person",
            "employee_id": self.employee_id,
            "employee_code": self.employee_code,
            "matched": self.matched,
            "confidence": self.confidence,
            "identity_source": self.identity_source,
        }


class PersonTracker:
    """Greedy IoU tracker for people."""

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 30):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_track_id = 1

    def update(self, detections: List[dict]) -> List[PersonTrack]:
        boxes = [d["box"] for d in detections if d.get("box")]

        for track in self.tracks.values():
            track.mark_missed()

        used = set()
        # Greedy: for each track, take the best-IoU unused detection.
        for track in self.tracks.values():
            if track.is_expired():
                continue
            best_idx, best_iou = None, self.iou_threshold
            for idx, box in enumerate(boxes):
                if idx in used:
                    continue
                score = _iou(track.box, box)
                if score >= best_iou:
                    best_iou, best_idx = score, idx
            if best_idx is not None:
                track.update_box(boxes[best_idx])
                used.add(best_idx)

        for idx, box in enumerate(boxes):
            if idx not in used:
                self.tracks[self.next_track_id] = PersonTrack(
                    track_id=self.next_track_id, box=box, max_misses=self.max_misses
                )
                self.next_track_id += 1

        for tid in [t for t, tr in self.tracks.items() if tr.is_expired()]:
            del self.tracks[tid]

        return list(self.tracks.values())

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
