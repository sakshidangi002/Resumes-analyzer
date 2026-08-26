"""Where the lines and the chairs are. Per-camera geometry, kept out of the code.

WHY THIS FILE EXISTS SEPARATELY
-------------------------------
`profiles.py` holds what a ROLE does -- every doorway detects at 640/0.15. This
holds what an individual CAMERA sees, which no role can supply: camera 57's door
is at the top of frame and camera 58's corridor runs diagonally, and no shared
profile can express that.

Keeping it out of the algorithm is the point. A crossing rule with a coordinate
baked into it cannot be re-aimed without a code change, and these cameras get
re-aimed.

ALL COORDINATES ARE NORMALISED (0..1)
-------------------------------------
Fractions of frame width and height, never pixels. The same camera is processed
at 640 and displayed at 960x1080, and a pixel constant would silently mean two
different places.

WHAT WAS MEASURED, AND WHAT WAS NOT
-----------------------------------
The crossing lines below come from the live V1 configuration (0.35 horizontal on
both doorways), carried over deliberately rather than re-guessed -- that value
was tuned against these cameras and re-deriving it from one still frame would be
a downgrade.

`inside_side` was read off captured frames:

  Camera 57 is a small vestibule. The glass office door is at UPPER-CENTRE and
  the open floor is below it, so a person coming through the door moves DOWN the
  frame: outside is above the line, inside is below.

  Camera 58 watches a corridor running diagonally, with the office door at
  upper-right. The same mapping applies -- away from the camera is toward the
  door -- but this one is LESS CERTAIN than 57 and is flagged for live
  validation, because the corridor also carries through-traffic that never
  enters the office at all.

THE CHAIR ZONES ARE PLACED BY HAND
----------------------------------
COCO chair detection was measured against these two rooms and cannot enumerate
seats: camera 59 returned between 1 and 9 chairs depending on input size and
confidence, and camera 60 returned 3 to 10. And it is not merely unstable, it is
BIASED -- it can only see a chair nobody is sitting in, so the seats it finds
are precisely the seats occupancy has no use for.

So `scripts/cctv_v2_chair_setup.py` renders a labelled grid overlay and the
coordinates are read off it by a human. See the note on ROOM_GEOMETRY below for
what was mapped, how, and how accurate it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Orientation = Literal["horizontal", "vertical"]
Side = Literal["above", "below"]
RefPoint = Literal["foot", "centroid"]


@dataclass(frozen=True)
class CrossingLine:
    """A virtual line, and what crossing it in each direction MEANS."""

    position: float                  # 0..1 along the perpendicular axis
    orientation: Orientation = "horizontal"

    # Which side of the line is inside the office. Direction is derived from
    # this rather than hard-coded, because "down the frame" means enter on one
    # camera and leave on another.
    inside_side: Side = "below"

    # Which point of the person's box is tested against the line.
    #
    # "foot" (bottom-centre) by default. A seated or partly occluded person's
    # box top moves wildly as YOLO includes or excludes their head, while their
    # feet stay where they are standing. The centroid of a box whose top edge is
    # unstable crosses lines the person never crossed.
    reference: RefPoint = "foot"

    # A crossing must move at least this far, as a fraction of frame size,
    # measured perpendicular to the line. Stops a person loitering ON the line
    # from emitting an event every time the box jitters across it.
    min_travel: float = 0.06

    # After a track emits a crossing, it cannot emit another for this long.
    # Not a duplicate filter -- duplicates are already prevented by requiring a
    # side CHANGE -- but a guard against a person turning round in the doorway
    # and generating IN/OUT/IN/OUT.
    cooldown_sec: float = 3.0

    # Set False to disable crossing on a camera without deleting its geometry.
    enabled: bool = True


@dataclass(frozen=True)
class ChairZone:
    """One physical seat, as a normalised box in one camera's view.

    A chair is a FIXED place in a fixed camera view, so it is configured once
    and given a stable id. Re-detecting chairs per frame would renumber them
    whenever the detector wobbled, and "chair 3" would mean a different seat
    from one minute to the next.
    """

    chair_id: str
    box: tuple[float, float, float, float]     # x1, y1, x2, y2 normalised
    label: Optional[str] = None


@dataclass(frozen=True)
class RoomGeometry:
    """One room camera's seats, plus how strictly to judge occupancy."""

    chairs: tuple[ChairZone, ...] = ()

    # Fraction of the SEAT box that a person's occupancy region must cover
    # before that person counts as sitting in it. Judged against the seat, not
    # the person: a person standing close to the camera has a huge box that
    # would overlap a distant seat by a large absolute area while covering
    # almost none of it.
    min_overlap: float = 0.15

    # How far a person's box may extend BELOW the seat zone, as a fraction of
    # frame height, before they are judged to be in front of the chair rather
    # than in it.
    #
    # This is the only thing that separates the two in a receding ceiling view.
    # A person standing in the aisle passes between the camera and a chair and
    # covers it completely in 2D -- by overlap alone they are sitting in it. But
    # somebody sitting in a chair cannot extend far below it, while somebody
    # standing in front of it reaches the floor, which in this projection is
    # well below the chair's base.
    #
    # MEASURED over 36 labelled people on cameras 59 and 60, five of them
    # standing (scripts/cctv_v2_room_bench.py assoc):
    #
    #     max_drop   correct  wrong  missed  spurious
    #     none          36      0       0        5     <- every stander seated
    #     0.06          35      0       1        0
    #     0.08          35      0       1        0
    #     0.10          36      0       0        0
    #     0.12          36      0       0        0     <- chosen
    #     0.15          36      0       0        0
    #     0.20          36      0       0        5     <- standers return
    #
    # The window is 0.10..0.15 and 0.12 sits in the middle of it. Margins are
    # 0.02 and 0.03, which is not generous -- re-measure after a camera re-aim
    # rather than assuming it carries over.
    max_drop_below_seat: float = 0.12

    # Consecutive OBSERVATIONS before a state change is believed. Occupancy
    # flickers otherwise -- YOLO drops a seated person for one pass routinely,
    # and a chair that changes state on a single frame is noise, not signal.
    #
    # An observation is one completed analysis pass, NOT one read of the result.
    # That distinction is enforced in pipeline/v1_bridge.py and is the reason
    # these numbers mean anything; before it, the dashboard's 2s poll spent
    # three "consecutive observations" on a single pass in six seconds.
    #
    # confirm_free MEASURED, 2026-08-26. Six minutes of camera 59, 21 passes at
    # 11.7s (scripts/cctv_v2_stability.py live). For chairs with somebody
    # genuinely in them, the runs of consecutive passes where the occupant
    # produced no claim were:
    #
    #     1, 1, 1, 1, 3, 3      longest 3, median 1
    #
    # confirm_free was 3 -- exactly equal to the longest genuine gap, so a
    # seated person's normal disappearance was enough to free their chair. 5
    # clears the observed maximum with two passes to spare.
    #
    # The cost is stated rather than hidden: at 11.7s per pass a chair now takes
    # about a minute to go free after somebody really leaves. For a room display
    # that is the right trade -- a seat wrongly held for a minute is a much
    # smaller error than a seat that flickers while its occupant sits still --
    # but it IS a trade, and if the pass rate ever drops it gets worse linearly.
    #
    # The sample is six gaps in one six-minute window. Thin. Re-measure over a
    # longer window before moving it further.
    confirm_occupied: int = 2
    confirm_free: int = 5


# ---------------------------------------------------------------------------
# Doorways
# ---------------------------------------------------------------------------
CROSSING_LINES: dict[int, CrossingLine] = {
    # Vestibule. Glass door upper-centre, open floor below it, sofa to the left.
    # People entering come through the door and move DOWN the frame.
    #
    # 0.775, MEASURED. This was 0.35, inherited from V1, and it was uncrossable:
    # V1 measured a different point on the body and V2 uses the FOOT point.
    # Over 148 real boxes from this camera the unclipped foot range is
    #
    #     0.624 .. 0.998   median 0.828
    #
    # so NO foot ever reached 0.35. The camera ran 50 minutes, produced 24
    # tracks and ZERO side changes -- and that failure is silent, because a
    # doorway reporting nobody looks exactly like a quiet doorway.
    #
    # 0.775 is the position that splits the most TRACKS (3 of 5 in the measured
    # window) -- the quantity the crossing detector actually needs, counted per
    # person so one loiterer cannot outvote people walking through.
    #
    # Clipped boxes are excluded from that fit. 30% of this camera's boxes touch
    # the frame bottom, where the box edge IS the frame rather than a body part;
    # including them created a false peak that made 0.95 look ideal.
    #
    # The sample is 5 tracks in one 240s window. It is enough to rule 0.35 out
    # beyond doubt, and it is thin for choosing between 0.75 and 0.80 -- the
    # margin at 0.775 is 0.015, which is narrow. Re-fit with more traffic when
    # there is time: scripts/cctv_v2_line_fit.py.
    57: CrossingLine(position=0.775, orientation="horizontal", inside_side="below"),

    # Corridor, office door upper-right. LEFT AT 0.35 -- measured correct here.
    # Over 284 real boxes the unclipped foot range is
    #
    #     0.165 .. 0.998   median 0.304
    #
    # so 0.35 sits just past the median, with people on both sides of it. This
    # camera produced 4 real transits in a 50-minute live run while camera 57
    # produced none, which is what a well-placed line versus an unreachable one
    # looks like from the outside.
    #
    # The same number was right here and impossible there. That is why line
    # position is per-camera and must be fitted, never shared.
    #
    # STILL LESS CERTAIN in one respect: this corridor carries through-traffic
    # that never enters the office, and no line geometry can tell that apart
    # from someone coming in.
    58: CrossingLine(position=0.35, orientation="horizontal", inside_side="below"),
}


# ---------------------------------------------------------------------------
# Rooms
# ---------------------------------------------------------------------------
# Empty until a human reads coordinates off the setup overlay. See the module
# docstring for why these are not auto-generated from COCO detections.
ROOM_GEOMETRY: dict[int, RoomGeometry] = {
    # MAPPED BY HAND against the 0.05 grid overlay
    # (scripts/cctv_v2_chair_setup.py --frame ... --zones ...), replacing the
    # four zones COCO clustering had produced.
    #
    # WHY THE AUTO-DISCOVERED MAP WAS REPLACED RATHER THAN EXTENDED
    # -------------------------------------------------------------
    # Clustering COCO chair detections over 30 frames DID find four stable
    # seats, and the technique is sound -- a chair does not move, so averaging
    # removes the per-frame noise. But the four it found were all EMPTY chairs
    # in clear view, and every occupied chair was missed, because the person
    # sitting in it hides the chair from the detector.
    #
    # That is exactly backwards for occupancy: a seat map made of the seats
    # nobody uses reports "all free" no matter who is in the room. Three of the
    # four old zones (C1, C3, C4) sat on spare chairs in the middle of the
    # floor; the woman working at the right-hand desk had NO zone at all, and
    # the zone nearest her (C2) was on the empty chair beside her -- so she
    # could never occupy anything however well she was detected.
    #
    # No amount of running the discovery longer or lower fixes that, so the map
    # is now read off the picture by a human.
    #
    # HOW THE COORDINATES WERE READ
    # -----------------------------
    # From each chair's BACKREST and five-star BASE. The base is the useful
    # landmark: it is the one part of an office chair that stays visible when
    # somebody is sitting on it, so a seat can be placed correctly whether or
    # not it is occupied -- which is the whole property the COCO map lacked.
    #
    # Zones cover the seat AND the space a seated person occupies, because that
    # is what occupancy tests against (see occupancy.py).
    #
    # Ids changed from C1..C4 to R*/L* deliberately. They are not persisted
    # anywhere -- no table, no migration -- and the old ids named seats that
    # have moved, so keeping them would have made a corrected map look like the
    # old one.
    #
    # STILL APPROXIMATE. These are read off one camera's pixels by eye, to
    # about +/-0.01 normalised. Adjacent chairs in a receding row genuinely
    # overlap in image space, so a person between two seats can be attributed to
    # either. Re-check with the overlay after any camera re-aim.
    #
    # A CHAIR HAS EXACTLY ONE OWNER
    # -----------------------------
    # Cameras 59 and 60 watch the SAME room from opposite ends. That was
    # declared in cameras.py and is now demonstrated: simultaneous frames at
    # 10:19:09 and 10:19:11 (data/cctv_v2_stability/sync59_0.jpg, sync60_0.jpg)
    # show the same three people -- white shirt, dark green check, yellow top --
    # in BOTH views, in reversed order, which is what two cameras facing each
    # other along one desk produce.
    #
    # So camera 59's left-hand desk row and camera 60's row are THE SAME FIVE
    # PHYSICAL CHAIRS. Mapping them on both cameras made one chair answerable to
    # two independent state machines, and made the room's chair total look like
    # 13 + 5 = 18 when the room has 13.
    #
    # Ownership is therefore exclusive, and expressed structurally: a camera's
    # entry here lists ONLY the chairs it owns. There is no flag to get wrong
    # and no way for two cameras to claim one seat.
    #
    # Camera 60 owns the shared row because it sees it far better -- from 59 the
    # three occupants span x 0.13-0.30 and overlap each other; from 60 they span
    # x 0.20-0.60 and are cleanly separated. Camera 59 owns the right-hand row
    # because it is the only camera that can see it at all.
    #
    # Camera 59 still SEES the shared row, and those zones are kept in
    # OBSERVED_ELSEWHERE below so an overlay can draw them and say who owns
    # them -- rather than showing three people as unassigned with no explanation.
    #
    # 8 chairs: the right-hand desk row, R1 nearest the far wall to R8 nearest
    # the camera. The room's other 5 belong to camera 60.
    59: RoomGeometry(chairs=(
        ChairZone("R1", (0.392, 0.160, 0.458, 0.290), label="right row, far end"),
        ChairZone("R2", (0.370, 0.250, 0.443, 0.400), label="right row 2"),
        ChairZone("R3", (0.427, 0.282, 0.500, 0.448), label="right row 3"),
        ChairZone("R4", (0.465, 0.335, 0.545, 0.502), label="right row 4"),
        ChairZone("R5", (0.510, 0.385, 0.590, 0.578), label="right row 5"),
        ChairZone("R6", (0.563, 0.448, 0.648, 0.682), label="right row 6"),
        ChairZone("R7", (0.640, 0.528, 0.772, 0.804), label="right row 7"),
        ChairZone("R8", (0.712, 0.598, 0.852, 0.862), label="right row, nearest camera"),
    )),

    # Camera 60 OWNS the shared desk row -- the same five physical chairs camera
    # 59 can also see. See the ownership note above for the evidence that they
    # are the same chairs and why this camera is the one that owns them.
    #
    # Camera 59's right-hand row is cut off at this camera's bottom-left frame
    # edge and is deliberately NOT mapped here: a zone on a chair the camera can
    # only see a corner of would report occupancy nobody could check, and those
    # chairs already have an owner.
    # SIX, not five. S6 was missing until an operator counted the room and said
    # so, and they were right: there is a fully visible chair at the NEAR end of
    # the row, beside the ENTRANCE label, that the first two mapping passes cut
    # off. Re-counted from the five-star bases on the floor -- six of them, one
    # per zone below -- because the bases are the one landmark that stays
    # visible whether or not somebody is sitting on the chair.
    #
    # The lesson is about the METHOD, not the number. Both earlier passes worked
    # from a crop that stopped short of the frame edge, so a chair outside that
    # crop could not be found however carefully the crop was read. Map from the
    # WHOLE frame and count bases, then zoom to place the boxes.
    60: RoomGeometry(chairs=(
        ChairZone("S1", (0.205, 0.358, 0.262, 0.555), label="row far end"),
        ChairZone("S2", (0.240, 0.392, 0.308, 0.620), label="row 2"),
        ChairZone("S3", (0.294, 0.355, 0.428, 0.702), label="row 3"),
        ChairZone("S4", (0.402, 0.505, 0.508, 0.785), label="row 4"),
        ChairZone("S5", (0.505, 0.638, 0.615, 0.912), label="row 5"),
        ChairZone("S6", (0.620, 0.708, 0.735, 0.935), label="row, nearest camera"),
    )),
}


# Chairs a camera can SEE but does not own, and who does own them.
#
# Kept so an overlay can draw the seat and name its owner. Without this, camera
# 59's view of the shared desk shows three people sitting in nothing, which
# reads as a detection failure rather than as somebody else's chairs -- and
# "looks broken" is how a correct system gets tuned until it is not.
#
# These zones take NO part in occupancy. They are not counted, not claimed, and
# not settled; `RoomOccupancy` never sees them. That is deliberate: the moment a
# non-owning camera can influence a chair, two state machines share one seat.
OBSERVED_ELSEWHERE: dict[int, tuple[tuple[str, tuple[float, float, float, float]], ...]] = {
    # Camera 59's view of the row camera 60 owns. The ids are camera 59's old
    # L1..L5 boxes; they correspond to camera 60's S-row approximately, not
    # one-for-one, because the two cameras see the row from opposite ends and
    # the correspondence was never measured chair-by-chair. They are labelled
    # with the owner rather than with a specific S-id for that reason.
    59: (
        ("L1", (0.160, 0.242, 0.228, 0.360)),
        ("L2", (0.216, 0.264, 0.274, 0.396)),
        ("L3", (0.178, 0.398, 0.272, 0.578)),
        ("L4", (0.196, 0.560, 0.296, 0.708)),
        ("L5", (0.230, 0.705, 0.387, 0.999)),
    ),
}

# Which camera owns each shared row, for the overlay's label.
OBSERVED_OWNER: dict[int, int] = {59: 60}


def observed_elsewhere(camera_id: int):
    """Zones this camera sees but another camera controls. Never counted."""
    return OBSERVED_ELSEWHERE.get(int(camera_id), ())


def observed_owner(camera_id: int) -> Optional[int]:
    return OBSERVED_OWNER.get(int(camera_id))


def room_chair_total(camera_id: int) -> int:
    """Physical chairs in this camera's ROOM, counted once each.

    Not the sum of the per-camera maps -- that would double-count every chair
    two cameras can see, which is what made a 13-chair room look like 18. Each
    chair is owned by exactly one camera, so the room total is simply the sum
    over the owners.
    """
    from app.cctv_v2.config.cameras import same_room_peers

    cid = int(camera_id)
    ids = {cid} | set(same_room_peers(cid))
    return sum(len(room_geometry(c).chairs) for c in ids)


def crossing_line(camera_id: int) -> Optional[CrossingLine]:
    """The crossing line for a camera, or None if it does not count crossings.

    None rather than a default: a camera nobody configured a line for must not
    quietly acquire one at the middle of frame and start emitting transits.
    """
    return CROSSING_LINES.get(int(camera_id))


def room_geometry(camera_id: int) -> RoomGeometry:
    """A room camera's seat map. Empty geometry for an unconfigured camera,
    which reports zero chairs rather than guessing at them."""
    return ROOM_GEOMETRY.get(int(camera_id), RoomGeometry())


def has_chair_map(camera_id: int) -> bool:
    """Whether this camera has any seats configured at all.

    Callers use this to distinguish "0 of 0 chairs occupied" -- which is a
    configuration gap -- from "0 of 10 occupied", which is an empty room.
    """
    return bool(room_geometry(camera_id).chairs)
