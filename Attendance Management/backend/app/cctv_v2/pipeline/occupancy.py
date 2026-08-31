"""Room occupancy: how many people are in the room, and which seats are taken.

    grabber -> slot -> scheduler -> YOLO -> tracker -> [THIS] -> occupancy

TWO SEPARATE QUESTIONS, DELIBERATELY NOT RECONCILED
---------------------------------------------------
    people_count       how many people the camera can currently see
    occupied_chairs    how many configured seats have somebody in them

These will disagree, and forcing them to agree would destroy information. A
person standing has no chair. A seat may be occupied by someone the detector
cannot see. Reporting `people=8, chairs_occupied=7` is not an inconsistency to
be fixed -- it is two measurements of different things, and the gap between them
is itself a signal.

PRESENT MEANS SEEN, NOT REMEMBERED
----------------------------------
`people_count` counts CONFIRMED tracks only. A track the tracker is holding for
re-association is remembered, not present, and counting it would report people
who have left. V1 did exactly that and published five people in an empty
corridor.

It also counts TRACKS, never detections. One person seen on four passes is one
person; summing detections across passes is how a still room reports a crowd.

WHY CHAIRS ARE CONFIGURED AND NOT DETECTED
------------------------------------------
Measured against these two rooms, COCO chair detection cannot enumerate seats.
Camera 59 returned between 1 and 9 chairs depending on input size and
confidence, against roughly 10 visible; camera 60 returned 3 to 10 against 6-8.
Nothing built on a seat count that moves like that can be trusted.

So a seat is a FIXED place in a fixed camera view, configured once with a stable
id, and occupancy comes from associating PEOPLE with those places. COCO chair
detections are still useful -- for placing the zones during setup, and as a
sanity check afterwards -- but never as the per-frame source of truth.

A camera with no configured seats reports zero chairs and says so. That is a
configuration gap stated plainly, not an empty room.

WHY OCCUPANCY IS SMOOTHED
-------------------------
YOLO drops a seated person for a pass routinely -- they are behind a desk, at a
high angle, half occluded. A chair that flips state on one observation is
reporting noise. State changes therefore require several consecutive
observations, and asymmetrically: a seat is slower to be declared FREE than
OCCUPIED, because a person briefly invisible is far more likely than a person
who teleported away.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

from app.cctv_v2.config.cameras import role_for, same_room_peers
from app.cctv_v2.config.geometry import ChairZone, RoomGeometry, room_geometry
from app.cctv_v2.pipeline.detect import PersonDetection
from app.cctv_v2.pipeline.track import PersonTrack, TrackState

logger = logging.getLogger(__name__)


class ChairState(str, Enum):
    OCCUPIED = "occupied"
    FREE = "free"
    UNKNOWN = "unknown"      # configured, but nothing can be said about it yet


@dataclass
class ChairStatus:
    chair_id: str
    state: ChairState = ChairState.UNKNOWN
    occupant_track_id: Optional[int] = None
    since: float = 0.0
    _pending: Optional[ChairState] = None
    _pending_count: int = 0

    @property
    def occupied(self) -> bool:
        return self.state is ChairState.OCCUPIED


@dataclass(frozen=True)
class RoomSnapshot:
    """What one room camera can say at one instant."""

    camera_id: int
    timestamp: float
    people_count: int
    active_track_ids: tuple[int, ...]
    total_chairs: int
    occupied_chairs: int
    free_chairs: int
    unknown_chairs: int
    chairs: tuple[dict, ...]
    chair_map_configured: bool

    # When any chair last CHANGED state, as opposed to when it was last looked
    # at. The two are routinely far apart -- a settled room is observed every
    # few seconds and changes nothing for minutes -- and a caller that cannot
    # tell them apart has no way to know whether a stale-looking answer is stale
    # or simply steady.
    state_updated_at: float = 0.0

    def as_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "state_updated_at": self.state_updated_at,
            "people_count": self.people_count,
            "active_track_ids": list(self.active_track_ids),
            "chairs_total": self.total_chairs,
            "chairs_occupied": self.occupied_chairs,
            "chairs_free": self.free_chairs,
            "chairs_unknown": self.unknown_chairs,
            "chair_map_configured": self.chair_map_configured,
            "chairs": list(self.chairs),
        }


def _overlap_fraction_of_seat(person_box, seat_box) -> float:
    """How much of the SEAT the person covers.

    Judged against the seat, not the person, and not IoU. A person near the
    camera has a huge box; by IoU they overlap a distant seat barely at all,
    and by fraction-of-person likewise -- yet they may be squarely in it. The
    question is "is this seat covered", so the seat is the denominator.
    """
    px1, py1, px2, py2 = person_box
    sx1, sy1, sx2, sy2 = seat_box
    ix1, iy1 = max(px1, sx1), max(py1, sy1)
    ix2, iy2 = min(px2, sx2), min(py2, sy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    seat_area = max(1e-9, (sx2 - sx1) * (sy2 - sy1))
    return (iw * ih) / seat_area


def _seat_region(track: PersonTrack, w: float, h: float):
    """The part of a person that indicates WHERE THEY ARE SITTING.

    THE WHOLE BOX, normalised. This used to be the lower half, on the reasoning
    that a person's box top drifts as YOLO includes or excludes their head while
    the part actually in the chair is the bottom.

    That reasoning describes a STANDING person, whose feet are the point of
    contact. It is wrong for a person at a desk: their box bottom is wherever
    the desk edge or the chair base cuts them off, several tens of pixels below
    the seat, so the lower half of the box is torso and desk, not lap.

    MEASURED over 36 labelled people on cameras 59 and 60, against a hand-placed
    chair map (scripts/cctv_v2_room_bench.py assoc):

        region                correct  wrong  missed
        whole_box  0.00-1.00     36      0       0     <- chosen
        upper_half 0.00-0.50     35      0       1
        torso_band 0.30-0.75     34      0       2
        lower_half 0.50-1.00     31      1       4     <- what shipped
        lower_third 0.67-1.00    14     12      10

    The trend runs the opposite way to the original reasoning: the LOWER the
    region, the worse it does, and the bottom third -- pure desk -- gets the
    wrong seat almost as often as the right one.

    Using the whole box does reintroduce the failure the halving was meant to
    prevent, a person standing in front of a seat covering it from the camera's
    point of view. That is handled where it belongs, by `max_drop_below_seat` in
    the claim loop, which is a test the lower-half crop could never perform.
    """
    x1, y1, x2, y2 = track.bbox
    return (x1 / w, y1 / h, x2 / w, y2 / h)


class RoomOccupancy:
    """People count and seat occupancy for ONE room camera."""

    def __init__(self, camera_id: int, frame_size: tuple[float, float] = (960.0, 1080.0)):
        self.camera_id = int(camera_id)
        self.role = role_for(camera_id)
        self.geometry: RoomGeometry = room_geometry(camera_id)
        self.frame_w, self.frame_h = frame_size
        self.chairs: dict[str, ChairStatus] = {
            c.chair_id: ChairStatus(chair_id=c.chair_id) for c in self.geometry.chairs
        }
        self.observations = 0

    @property
    def _zone(self) -> dict[str, tuple[float, float, float, float]]:
        """chair_id -> its box, derived from the geometry rather than cached.

        Derived on purpose. A cached copy taken in __init__ is a second source
        of truth for where the seats are, and it goes stale the moment anything
        replaces `self.geometry` -- which both the test fixtures and any future
        live re-aim do. Two chair maps that can disagree is exactly the class of
        bug this module already carries a scar from.
        """
        return {c.chair_id: c.box for c in self.geometry.chairs}

    def apply_geometry(self, geometry: RoomGeometry) -> bool:
        """Adopt a new chair map, preserving the state of seats that survive it.

        `self.chairs` has to mirror `self.geometry` exactly. `update` indexes
        `self.chairs[seat.chair_id]` for every seat in the geometry, so a seat
        the map gained without a status here is a KeyError; and the occupied /
        free / unknown counts are taken over `self.chairs`, so a seat the map
        LOST but which still had a status would keep being counted -- a chair
        removed from the room would go on reporting occupancy forever.

        Seats that exist in both maps keep their state and their smoothing
        counters, so re-aiming the map does not reset a chair somebody is
        sitting in. Genuinely new seats start UNKNOWN, which is the honest
        answer: nothing has been observed about them yet.

        Refuses an EMPTY map and reports False. An empty geometry would silently
        turn a working room into "no chairs configured", and the most likely
        source of one is a detector or config fault -- exactly when the last
        known-good map is the thing worth keeping.
        """
        if not geometry.chairs:
            return False

        self.geometry = geometry
        wanted = {c.chair_id for c in geometry.chairs}
        for chair_id in [c for c in self.chairs if c not in wanted]:
            del self.chairs[chair_id]
        for chair_id in wanted:
            if chair_id not in self.chairs:
                self.chairs[chair_id] = ChairStatus(chair_id=chair_id)
        return True

    def update(self, tracks: Iterable[PersonTrack], timestamp: float) -> RoomSnapshot:
        """Fold in the tracks currently CONFIRMED on this camera."""
        present = [t for t in tracks if t.state is TrackState.CONFIRMED]
        for t in present:
            if t.camera_id != self.camera_id:
                raise ValueError(
                    f"camera {t.camera_id} track given to room occupancy "
                    f"{self.camera_id}; occupancy state is per-camera"
                )
        self.observations += 1

        # -- seat association ------------------------------------------------
        # Each person may claim at most ONE seat: their best. Otherwise a person
        # sitting between two chairs covers both and counts twice.
        claims: dict[str, tuple[float, int]] = {}
        for track in present:
            region = _seat_region(track, self.frame_w, self.frame_h)
            best_id, best_frac = None, 0.0
            for seat in self.geometry.chairs:
                frac = _overlap_fraction_of_seat(region, seat.box)
                if frac > best_frac:
                    best_frac, best_id = frac, seat.chair_id
            if best_id is None or best_frac < self.geometry.min_overlap:
                continue
            if not self._plausibly_seated(region, best_id):
                continue
            prev = claims.get(best_id)
            # Two people overlapping one seat: the stronger claim wins, so a
            # passer-by clipping a seat cannot evict the person sitting in it.
            if prev is None or best_frac > prev[0]:
                claims[best_id] = (best_frac, track.track_id)

        for seat in self.geometry.chairs:
            status = self.chairs[seat.chair_id]
            observed = ChairState.OCCUPIED if seat.chair_id in claims else ChairState.FREE
            self._settle(status, observed, timestamp,
                         claims.get(seat.chair_id, (0.0, None))[1])

        # The most recent moment any seat actually CHANGED. `since` is stamped
        # by _settle when a state is adopted, so this is a real event time, not
        # the time of the last look.
        state_updated_at = max((c.since for c in self.chairs.values()), default=0.0)

        occupied = sum(1 for c in self.chairs.values() if c.state is ChairState.OCCUPIED)
        free = sum(1 for c in self.chairs.values() if c.state is ChairState.FREE)
        unknown = sum(1 for c in self.chairs.values() if c.state is ChairState.UNKNOWN)

        return RoomSnapshot(
            camera_id=self.camera_id,
            timestamp=timestamp,
            people_count=len(present),
            active_track_ids=tuple(sorted(t.track_id for t in present)),
            total_chairs=len(self.chairs),
            occupied_chairs=occupied,
            free_chairs=free,
            unknown_chairs=unknown,
            chairs=tuple(
                {"id": c.chair_id, "occupied": c.occupied, "state": c.state.value,
                 "occupant_track_id": c.occupant_track_id,
                 # WHERE the seat is, shipped with WHAT it is doing.
                 #
                 # An overlay has to draw the zone, and the only other way for it
                 # to know the coordinates is to keep its own copy of the chair
                 # map -- which the dashboard did, and which meant a seat added
                 # here was counted but never drawn. Geometry and state travel
                 # together so they cannot disagree.
                 "zone": list(self._zone[c.chair_id])}
                for c in sorted(self.chairs.values(), key=lambda x: x.chair_id)
            ),
            # From the geometry THIS ROOM IS USING, not from a fresh lookup in
            # the global map. The two can differ -- anything that replaces
            # `self.geometry` makes them differ -- and when they did, a snapshot
            # could report `total_chairs: 0` beside `chair_map_configured:
            # true`, which is precisely the "gap versus empty room" distinction
            # this field exists to make.
            chair_map_configured=bool(self.geometry.chairs),
            state_updated_at=state_updated_at,
        )

    def _plausibly_seated(self, region, chair_id: str) -> bool:
        """Whether this person can be IN that seat rather than in front of it.

        Overlap alone cannot tell those apart. Somebody walking down the aisle
        passes between the camera and a chair and covers it completely, which by
        any 2D measure is a perfect claim on the seat -- and on camera 59 that
        is not a corner case: the aisle runs directly in front of a row of eight
        chairs, and every one of the five standing people in the measured set
        was assigned a seat by overlap.

        What separates them is how far DOWN the person goes. A seated person
        cannot extend much below the chair they are in; a standing person in
        front of it reaches the floor, which in this projection is well below
        the chair's base. See `max_drop_below_seat` in config/geometry.py for
        the measurement that set the ceiling.
        """
        seat = self._zone.get(chair_id)
        if seat is None:
            return True
        return (region[3] - seat[3]) <= self.geometry.max_drop_below_seat

    def _settle(self, status: ChairStatus, observed: ChairState,
                timestamp: float, occupant: Optional[int]) -> None:
        """Apply a new observation, requiring agreement before changing state.

        Asymmetric on purpose: `confirm_free` is higher than `confirm_occupied`
        because a seated person vanishing for a pass is routine, while a person
        appearing in a seat they were not in is not.
        """
        if observed is status.state:
            status._pending, status._pending_count = None, 0
            if observed is ChairState.OCCUPIED and occupant is not None:
                status.occupant_track_id = occupant
            return

        if status._pending is observed:
            status._pending_count += 1
        else:
            status._pending, status._pending_count = observed, 1

        needed = (self.geometry.confirm_occupied if observed is ChairState.OCCUPIED
                  else self.geometry.confirm_free)

        # NOTHING TO PROTECT YET. A seat that has never been in any state adopts
        # FREE on its first observation instead of waiting for `confirm_free`.
        #
        # `confirm_free` exists to stop an ESTABLISHED occupied seat flipping
        # when its occupant is briefly undetected -- it is protecting a belief.
        # From UNKNOWN there is no belief and no occupant: nobody has ever been
        # seen in this chair, and the detector has just looked and found nobody.
        # Making that wait five observations was not caution, it was silence.
        #
        # It also produced a visibly wrong display. At 10-30s per observation a
        # restart left every empty chair UNKNOWN for one to two minutes, and the
        # panel read "Chairs: 6  Occupied: 2  Free: 0" -- four seats counted as
        # neither, which is a contradiction from the outside however defensible
        # it is from the inside.
        #
        # OCCUPIED still needs its confirmations from UNKNOWN, because a single
        # stray claim filling a seat is a real observed failure: two chairs were
        # claimed on exactly one pass each over six minutes, by a bloated box and
        # a passer-by.
        if status.state is ChairState.UNKNOWN and observed is ChairState.FREE:
            needed = 1
        if status._pending_count >= needed:
            status.state = observed
            status.since = timestamp
            status.occupant_track_id = occupant if observed is ChairState.OCCUPIED else None
            status._pending, status._pending_count = None, 0


class OccupancyRegistry:
    """One RoomOccupancy per room camera, isolated like the trackers are."""

    def __init__(self, frame_size: tuple[float, float] = (960.0, 1080.0)):
        self._rooms: dict[int, RoomOccupancy] = {}
        self._frame_size = frame_size
        self._latest: dict[int, RoomSnapshot] = {}

    def get(self, camera_id: int) -> RoomOccupancy:
        cid = int(camera_id)
        if cid not in self._rooms:
            self._rooms[cid] = RoomOccupancy(cid, self._frame_size)
        return self._rooms[cid]

    def update(self, camera_id: int, tracks, timestamp: float) -> RoomSnapshot:
        snap = self.get(camera_id).update(tracks, timestamp)
        self._latest[int(camera_id)] = snap
        return snap

    def snapshot(self) -> dict:
        """Per-camera occupancy. NEVER a summed room total.

        Cameras 59 and 60 are declared same-room in `cameras.py`, so their
        counts may include the same physical person twice. Adding them would
        report a number that is wrong in a direction nobody can bound, so the
        sum is not offered at all -- only the per-camera values, plus a note
        that they overlap.

        A room-level unique count needs cross-camera identity association,
        which does not exist yet. When it does, it belongs here.
        """
        out = {}
        for cid, snap in sorted(self._latest.items()):
            row = snap.as_dict()
            peers = same_room_peers(cid)
            if peers:
                row["shares_room_with"] = sorted(peers)
                row["note"] = (
                    "counts may include people also visible to "
                    f"{sorted(peers)}; do not add them"
                )
            out[cid] = row
        return out
