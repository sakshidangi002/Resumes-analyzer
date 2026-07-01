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
    
    # Track lifecycle
    age: int = 0  # number of frames tracked
    consecutive_misses: int = 0
    max_misses: int = 10  # frames before track is deleted
    
    def update(self, centroid: Tuple[float, float], box: Tuple[int, int, int, int]) -> None:
        """Update track with new detection."""
        self.centroid = centroid
        self.box = box
        self.last_seen = time.time()
        self.age += 1
        self.consecutive_misses = 0
    
    def mark_missed(self) -> None:
        """Mark that track was not detected in current frame."""
        self.consecutive_misses += 1
    
    def is_expired(self) -> bool:
        """Check if track should be deleted."""
        return self.consecutive_misses >= self.max_misses
    
    def can_recognize(self) -> bool:
        """Check if recognition can be performed (respecting cooldown)."""
        return time.time() - self.last_recognition_time >= self.recognition_cooldown
    
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
        max_misses: int = 10,  # frames before deleting track
    ):
        self.max_distance = max_distance
        self.recognition_cooldown = recognition_cooldown
        self.max_misses = max_misses
        
        self.tracks: Dict[int, FaceTrack] = {}
        self.next_track_id = 1
        self._lock = False  # Simple lock for thread safety
    
    def update(self, detections: List[dict]) -> List[FaceTrack]:
        """
        Update tracker with new face detections.
        
        Args:
            detections: List of face detections with 'box' key (x1, y1, x2, y2)
        
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
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
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
