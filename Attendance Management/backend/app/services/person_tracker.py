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

    # Identity bound from a recognised face inside this body.
    employee_id: Optional[int] = None
    employee_name: Optional[str] = None
    employee_code: Optional[str] = None
    matched: bool = False
    confidence: float = 0.0
    last_recognition_time: float = 0.0

    # Stable-confirmation + attendance state (mirrors FaceTrack).
    pending_employee_id: Optional[int] = None
    confirm_count: int = 0
    attendance_marked: bool = False

    age: int = 0
    consecutive_misses: int = 0
    max_misses: int = 30  # frames a body survives without detection before delete

    # Line-crossing state
    prev_centroid: Optional[Tuple[float, float]] = None
    last_crossing_time: float = 0.0
    crossed: bool = False  # has crossed the doorway line at least once

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

    def needs_recognition(self, reverify_sec: float) -> bool:
        """Recognise when not yet identified, or periodically to re-verify."""
        if not self.matched:
            return True
        return (time.time() - self.last_recognition_time) >= reverify_sec

    def bind_identity(
        self,
        employee_id: Optional[int],
        employee_name: Optional[str],
        employee_code: Optional[str],
        matched: bool,
        confidence: float,
    ) -> None:
        # Only overwrite with a positive match; a failed read never erases a
        # name that was already established for this person.
        if matched:
            self.employee_id = employee_id
            self.employee_name = employee_name
            self.employee_code = employee_code
            self.matched = True
            self.confidence = confidence
        self.last_recognition_time = time.time()

    def register_identification(
        self, employee_id: Optional[int], matched: bool, confirm_frames: int
    ) -> bool:
        """Return True once the same employee is confirmed N times (attendance)."""
        if not matched or employee_id is None:
            self.pending_employee_id = None
            self.confirm_count = 0
            return False
        if self.pending_employee_id == employee_id:
            self.confirm_count += 1
        else:
            self.pending_employee_id = employee_id
            self.confirm_count = 1
        if self.confirm_count >= confirm_frames and not self.attendance_marked:
            self.attendance_marked = True
            return True
        return False

    def get_display_info(self) -> dict:
        return {
            "track_id": self.track_id,
            "box": self.box,
            "employee_name": self.employee_name or "Person",
            "employee_id": self.employee_id,
            "employee_code": self.employee_code,
            "matched": self.matched,
            "confidence": self.confidence,
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
