"""Doorway IN/OUT: turning person tracks into transit events.

    grabber -> slot -> scheduler -> YOLO -> tracker -> [THIS] -> transit events

THE ONE RULE THIS LAYER EXISTS TO PROTECT
-----------------------------------------
A person is counted whether or not anyone knows who they are.

There is no face code here, no identity, no gallery, no threshold. A transit is
created from geometry and motion alone, and it carries `identity = None` until
some later stage decides otherwise. That is not an oversight to be filled in --
it is the guarantee. In V1 recognition and counting were entangled, so a face
that could not be matched became a person who was never counted, and the office
door tally quietly under-reported everyone who walked in facing away.

An event here means "a body crossed the line". Nothing more, and nothing less.

WHAT COUNTS AS A CROSSING
-------------------------
A track's reference point must be observed on one side of the line, then on the
other. Three further conditions, each present because of a specific way this
goes wrong:

  * The track must be CONFIRMED. A one-hit track is a detection, not a person
    with a history; emitting on it would count a false positive as a human.

  * It must have travelled `min_travel` perpendicular to the line. A person
    standing ON the line has a box that jitters across it every pass, which
    without this produces a stream of alternating IN/OUT events from somebody
    who never moved.

  * `cooldown_sec` must have elapsed since that track's last event. Someone who
    steps in, hesitates and steps back out is real, but at multi-second sampling
    it is indistinguishable from box noise.

WHAT IS DELIBERATELY NOT COUNTED
--------------------------------
  * A track that disappears without crossing. No event: a person who walked out
    of frame on the side they entered did not go anywhere.
  * A track that never crosses. Movement is not a transit.
  * A stale frame. It never reaches here -- the scheduler drops it first.

THE LIMIT THAT GEOMETRY CANNOT FIX
----------------------------------
Doorway inference runs every ~5.34s (measured). A person who crosses entirely
between two passes is never seen on both sides of the line, so no crossing
exists to detect. This layer cannot recover them and does not pretend to:
`missed_by_sampling` is not a number it can report, because a person it never
saw leaves no trace to count. Establishing that number needs ground truth from
outside the pipeline.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

from app.cctv_v2.config.cameras import role_for
from app.cctv_v2.config.geometry import CrossingLine, crossing_line
from app.cctv_v2.pipeline.track import PersonTrack, TrackState

logger = logging.getLogger(__name__)

_event_ids = itertools.count(1)


class Direction(str, Enum):
    IN = "PERSON_IN"
    OUT = "PERSON_OUT"


@dataclass(frozen=True)
class TransitEvent:
    """One body crossing one line, once.

    `identity` is None by construction and stays None here. A later stage may
    attach an employee to this event; it must UPDATE this event rather than
    create another, or one person walking through becomes two transits.
    """

    event_id: int
    camera_id: int
    role: str
    track_id: int
    direction: Direction
    timestamp: float
    frame_sequence: int
    bbox: tuple[float, float, float, float]
    reference_point: tuple[float, float]
    confidence: float
    track_hits: int
    travel: float                      # perpendicular distance moved, normalised
    identity: Optional[str] = None     # ALWAYS None from this layer

    @property
    def is_in(self) -> bool:
        return self.direction is Direction.IN


@dataclass
class _TrackCrossingState:
    """What this detector remembers about one track."""

    side: Optional[str] = None          # "above" | "below"
    side_since_pos: float = 0.0         # reference coord when the side was entered
    last_event_at: float = 0.0
    events: int = 0


def _reference_point(track: PersonTrack, line: CrossingLine, w: float, h: float):
    """The point on the person that is tested against the line, normalised."""
    x1, y1, x2, y2 = track.bbox
    if line.reference == "foot":
        px, py = (x1 + x2) / 2.0, y2
    else:
        px, py = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (px / w, py / h)


class CrossingDetector:
    """Watches ONE camera's tracks and emits transits. No identity, ever."""

    def __init__(self, camera_id: int, frame_size: tuple[float, float] = (960.0, 1080.0)):
        self.camera_id = int(camera_id)
        self.role = role_for(camera_id)
        self.line = crossing_line(camera_id)
        self.frame_w, self.frame_h = frame_size

        self._state: dict[int, _TrackCrossingState] = {}
        self.events: list[TransitEvent] = []
        self.count_in = 0
        self.count_out = 0
        self.rejected_unconfirmed = 0
        self.rejected_short_travel = 0
        self.rejected_cooldown = 0

    @property
    def enabled(self) -> bool:
        return self.line is not None and self.line.enabled

    def update(self, tracks: Iterable[PersonTrack], timestamp: float,
               frame_sequence: int = 0) -> list[TransitEvent]:
        """Fold in the tracks seen on one pass; return any transits they caused."""
        if not self.enabled:
            return []

        line = self.line
        emitted: list[TransitEvent] = []

        for track in tracks:
            if track.camera_id != self.camera_id:
                raise ValueError(
                    f"camera {track.camera_id} track given to crossing detector "
                    f"{self.camera_id}; crossing state is per-camera"
                )

            rx, ry = _reference_point(track, line, self.frame_w, self.frame_h)
            coord = ry if line.orientation == "horizontal" else rx
            side = "below" if coord >= line.position else "above"

            st = self._state.setdefault(track.track_id, _TrackCrossingState())
            if st.side is None:
                st.side = side
                st.side_since_pos = coord
                continue
            if side == st.side:
                # Still on the same side. Remember the FURTHEST point reached,
                # so a person who approaches the line slowly still accumulates
                # the travel that proves they came from somewhere.
                if line.orientation == "horizontal":
                    better = coord < st.side_since_pos if side == "above" else coord > st.side_since_pos
                else:
                    better = coord < st.side_since_pos if side == "above" else coord > st.side_since_pos
                if better:
                    st.side_since_pos = coord
                continue

            # The side changed. Everything below decides whether to believe it.
            travel = abs(coord - st.side_since_pos)

            if track.state is not TrackState.CONFIRMED:
                self.rejected_unconfirmed += 1
                st.side, st.side_since_pos = side, coord
                continue
            if travel < line.min_travel:
                self.rejected_short_travel += 1
                st.side, st.side_since_pos = side, coord
                continue
            if st.last_event_at and (timestamp - st.last_event_at) < line.cooldown_sec:
                self.rejected_cooldown += 1
                st.side, st.side_since_pos = side, coord
                continue

            direction = Direction.IN if side == line.inside_side else Direction.OUT
            event = TransitEvent(
                event_id=next(_event_ids),
                camera_id=self.camera_id,
                role=self.role,
                track_id=track.track_id,
                direction=direction,
                timestamp=timestamp,
                frame_sequence=frame_sequence or track.frame_sequence,
                bbox=track.bbox,
                reference_point=(rx, ry),
                confidence=track.confidence,
                track_hits=track.hits,
                travel=travel,
                identity=None,          # never set here -- see the docstring
            )
            self.events.append(event)
            emitted.append(event)
            if direction is Direction.IN:
                self.count_in += 1
            else:
                self.count_out += 1

            st.side, st.side_since_pos = side, coord
            st.last_event_at = timestamp
            st.events += 1

            logger.info(
                "TRANSIT %s camera=%s track=%s hits=%s travel=%.3f identity=UNKNOWN",
                direction.value, self.camera_id, track.track_id,
                track.hits, travel,
            )

        return emitted

    def forget(self, track_ids: Iterable[int]) -> None:
        """Drop crossing state for tracks the tracker has expired.

        Without this the state dict grows for the life of the process. Note that
        forgetting a track does NOT retract its events -- a crossing that
        happened, happened.
        """
        for tid in list(track_ids):
            self._state.pop(tid, None)

    def summary(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "role": self.role,
            "enabled": self.enabled,
            "people_in": self.count_in,
            "people_out": self.count_out,
            # Everything this layer produces is unattributed by construction.
            # Reported explicitly so a consumer never has to infer it.
            "unknown_in": self.count_in,
            "unknown_out": self.count_out,
            "events": len(self.events),
            "rejected_unconfirmed": self.rejected_unconfirmed,
            "rejected_short_travel": self.rejected_short_travel,
            "rejected_cooldown": self.rejected_cooldown,
            "tracked_states": len(self._state),
        }


class CrossingRegistry:
    """One CrossingDetector per camera, isolated like the trackers are."""

    def __init__(self, frame_size: tuple[float, float] = (960.0, 1080.0)):
        self._detectors: dict[int, CrossingDetector] = {}
        self._frame_size = frame_size

    def get(self, camera_id: int) -> CrossingDetector:
        cid = int(camera_id)
        if cid not in self._detectors:
            self._detectors[cid] = CrossingDetector(cid, self._frame_size)
        return self._detectors[cid]

    def update(self, camera_id: int, tracks, timestamp: float,
               frame_sequence: int = 0) -> list[TransitEvent]:
        return self.get(camera_id).update(tracks, timestamp, frame_sequence)

    def summary(self) -> dict:
        return {cid: d.summary() for cid, d in sorted(self._detectors.items())}
