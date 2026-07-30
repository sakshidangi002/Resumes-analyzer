"""
Face tracking system for smooth multi-face recognition across frames.

Uses centroid-based tracking to maintain face identity across consecutive frames.
Each track maintains recognition state with cooldown to prevent flickering.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from app.services.embedding_fusion import EmbeddingFuser

logger = logging.getLogger(__name__)


@dataclass
class FaceTrack:
    """Represents a tracked face across frames."""
    track_id: int
    centroid: Tuple[float, float]  # (x, y) center of bounding box
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    last_seen: float = field(default_factory=time.time)

    # Latest detection dict (with embedding) associated with this track in the
    # current frame. Used to recognise the face WITHOUT re-running detection on
    # a crop. Set to None on frames where the track was not matched.
    face: Optional[dict] = None
    
    # Recognition state
    employee_id: Optional[int] = None
    employee_name: Optional[str] = None
    employee_code: Optional[str] = None
    matched: bool = False
    confidence: float = 0.0
    last_recognition_time: float = 0.0
    recognition_cooldown: float = 3.0  # Standard cooldown

    # Stable-confirmation state: attendance is only recorded after the SAME
    # employee has been identified on this track for `confirm_count` frames.
    # This prevents a single spurious frame from marking a false attendance.
    pending_employee_id: Optional[int] = None
    confirm_count: int = 0
    attendance_marked: bool = False  # attendance already recorded for this track
    # Recognition provenance from the last successful match (score, margin,
    # snapshot path). Captured at match time — that is the only moment the frame
    # and the match result coexist, since attendance is written asynchronously.
    last_evidence: Optional[dict] = None

    # ── Embedding fusion ────────────────────────────────────────────────────
    # Shared with PersonTrack — see services/embedding_fusion.py.
    fuser: EmbeddingFuser = field(default_factory=EmbeddingFuser)
    
    # Track lifecycle
    age: int = 0  # number of frames tracked
    consecutive_misses: int = 0
    max_misses: int = 10  # frames before an UNKNOWN track is deleted
    # A RECOGNISED track is kept much longer, so the employee's name stays
    # locked onto them even when their face turns away — until they leave view.
    identity_max_misses: int = 60
    # Smoothed motion (px per analysis frame). Used to "coast" the box while the
    # face is not visible so the label follows the person instead of freezing.
    velocity: Tuple[float, float] = (0.0, 0.0)
    expired: bool = False  # forced expiry (e.g. coasted out of frame)

    def update(self, centroid: Tuple[float, float], box: Tuple[int, int, int, int]) -> None:
        """Update track with new detection."""
        # Exponentially-smoothed velocity from consecutive detections.
        vx = centroid[0] - self.centroid[0]
        vy = centroid[1] - self.centroid[1]
        self.velocity = (0.5 * self.velocity[0] + 0.5 * vx,
                         0.5 * self.velocity[1] + 0.5 * vy)
        self.centroid = centroid
        self.box = box
        self.last_seen = time.time()
        self.age += 1
        self.consecutive_misses = 0

    def predict_forward(self, frame_shape: Optional[Tuple[int, int]] = None) -> None:
        """Advance the box along last known motion while the face is unseen.

        Keeps the label moving with the person. If the coasted centroid drifts
        out of the frame, the track is force-expired (the person has left).
        """
        vx, vy = self.velocity
        # Damp so a stale velocity can't run away across the screen.
        self.velocity = (vx * 0.9, vy * 0.9)
        cx, cy = self.centroid[0] + vx, self.centroid[1] + vy
        self.centroid = (cx, cy)
        x1, y1, x2, y2 = self.box
        self.box = (int(x1 + vx), int(y1 + vy), int(x2 + vx), int(y2 + vy))
        if frame_shape is not None:
            h, w = frame_shape[:2]
            if cx < 0 or cy < 0 or cx > w or cy > h:
                self.expired = True

    def mark_missed(self) -> None:
        """Mark that track was not detected in current frame."""
        self.consecutive_misses += 1

    def is_expired(self) -> bool:
        """Check if track should be deleted."""
        if self.expired:
            return True
        # Recognised tracks persist for identity_max_misses; unknown ones don't.
        limit = self.identity_max_misses if self.matched else self.max_misses
        return self.consecutive_misses >= limit
    
    def can_recognize(self) -> bool:
        """Check if recognition can be performed (respecting cooldown)."""
        return time.time() - self.last_recognition_time >= self.recognition_cooldown
    
    def add_observation(self, embedding, quality: float) -> None:
        """Fold one frame's embedding into this track's fused template.

        `quality` should be the face width in pixels — see EmbeddingFuser.
        """
        self.fuser.add(embedding, quality)

    def fused_embedding(self) -> Optional[np.ndarray]:
        """Quality-weighted mean embedding for this track, or None."""
        return self.fuser.fused()

    @property
    def observations(self) -> int:
        return self.fuser.observations

    @property
    def best_face_px(self) -> float:
        return self.fuser.best_quality

    def update_recognition(
        self,
        employee_id: Optional[int],
        employee_name: Optional[str],
        employee_code: Optional[str],
        matched: bool,
        confidence: float
    ) -> None:
        """Update recognition result."""
        self.employee_id = employee_id
        self.employee_name = employee_name
        self.employee_code = employee_code
        self.matched = matched
        self.confidence = confidence
        self.last_recognition_time = time.time()

    def register_identification(
        self, employee_id: Optional[int], matched: bool, confirm_frames: int
    ) -> bool:
        """Track consecutive identifications and decide when to mark attendance.

        Returns True EXACTLY ONCE per track lifetime — on the frame where the
        same employee has been confirmed `confirm_frames` times in a row and
        attendance has not yet been recorded. Any mismatch/unknown frame resets
        the counter, so a transient wrong match never reaches the threshold.
        """
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
        """Get display information for overlay rendering."""
        return {
            "track_id": self.track_id,
            "box": self.box,
            "employee_name": self.employee_name or "Unknown Person",
            "employee_id": self.employee_id,
            "employee_code": self.employee_code,
            "matched": self.matched,
            "confidence": self.confidence,
        }


class FaceTracker:
    """Multi-face tracker using centroid-based tracking."""
    
    def __init__(
        self,
        max_distance: float = 100.0,  # max pixels to consider same face
        recognition_cooldown: float = 3.0,  # seconds between recognitions
        max_misses: int = 10,  # frames before deleting an UNKNOWN track
        identity_max_misses: int = 60,  # frames a RECOGNISED track survives w/o a face
    ):
        self.max_distance = max_distance
        self.recognition_cooldown = recognition_cooldown
        self.max_misses = max_misses
        self.identity_max_misses = identity_max_misses

        self.tracks: Dict[int, FaceTrack] = {}
        self.next_track_id = 1
        self._lock = False  # Simple lock for thread safety

    def update(
        self,
        detections: List[dict],
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[FaceTrack]:
        """
        Update tracker with new face detections.

        Args:
            detections: List of face detections with 'box' key (x1, y1, x2, y2)
            frame_shape: optional (H, W) so coasted tracks can expire off-frame.

        Returns:
            List of active FaceTrack objects
        """
        if self._lock:
            logger.warning("FaceTracker: update called while locked")
            return list(self.tracks.values())
        
        # Calculate centroids for new detections
        detection_centroids = []
        detection_boxes = []
        detection_faces = []  # full detection dict (with embedding) per index
        for det in detections:
            box = det.get("box", [])
            if len(box) >= 4:
                x1, y1, x2, y2 = box[:4]
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                detection_centroids.append(centroid)
                detection_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                detection_faces.append(det.get("face"))

        # Mark all existing tracks as missed initially, and clear any stale face
        # so a track that is not matched this frame is never re-recognised from
        # an outdated embedding.
        for track in self.tracks.values():
            track.mark_missed()
            track.face = None

        # Match detections to existing tracks
        used_detection_indices = set()
        for track_id, track in self.tracks.items():
            if track.is_expired():
                continue

            best_idx = self._find_best_match(track.centroid, detection_centroids, used_detection_indices)
            if best_idx is not None:
                # Update track with new detection
                track.update(detection_centroids[best_idx], detection_boxes[best_idx])
                track.face = detection_faces[best_idx]
                used_detection_indices.add(best_idx)

        # Create new tracks for unmatched detections
        for idx in range(len(detection_centroids)):
            if idx not in used_detection_indices:
                new_track = FaceTrack(
                    track_id=self.next_track_id,
                    centroid=detection_centroids[idx],
                    box=detection_boxes[idx],
                    face=detection_faces[idx],
                    recognition_cooldown=self.recognition_cooldown,
                    max_misses=self.max_misses,
                    identity_max_misses=self.identity_max_misses,
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # Coast recognised tracks that had no face this frame: move the box along
        # last motion so the employee's name follows them while their face is
        # turned away (CPU-only identity persistence — no YOLO/tracker needed).
        for track in self.tracks.values():
            if track.consecutive_misses > 0 and track.matched and not track.is_expired():
                track.predict_forward(frame_shape)

        # Remove expired tracks
        expired_ids = [tid for tid, track in self.tracks.items() if track.is_expired()]
        for tid in expired_ids:
            del self.tracks[tid]

        return list(self.tracks.values())
    
    def _find_best_match(
        self,
        centroid: Tuple[float, float],
        detection_centroids: List[Tuple[float, float]],
        used_indices: set
    ) -> Optional[int]:
        """Find the best matching detection for a track centroid."""
        best_idx = None
        best_distance = float('inf')
        
        for idx, det_centroid in enumerate(detection_centroids):
            if idx in used_indices:
                continue
            
            distance = np.sqrt(
                (centroid[0] - det_centroid[0]) ** 2 +
                (centroid[1] - det_centroid[1]) ** 2
            )
            
            if distance < self.max_distance and distance < best_distance:
                best_distance = distance
                best_idx = idx
        
        return best_idx
    
    def get_track(self, track_id: int) -> Optional[FaceTrack]:
        """Get a specific track by ID."""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> List[FaceTrack]:
        """Get all active tracks."""
        return list(self.tracks.values())
    
    def reset(self) -> None:
        """Reset all tracks."""
        self.tracks.clear()
        self.next_track_id = 1
