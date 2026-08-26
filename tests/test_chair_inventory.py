"""How many chairs the room HAS is a different question from how many are taken.

    inventory   how many physical chairs exist        -> from the chair MAP
    occupancy   how many of them have somebody in them -> from person detection

Only the second may move on its own. If a wheeled chair drifts, a jacket is
draped over one, or the detector has a bad frame, `chairs_total` must not
change -- a total that wobbles makes every ratio built on it meaningless, and
"7 of 14" and "7 of 13" describe different rooms.

This is a real risk here rather than a theoretical one: COCO chair detection was
measured on these two rooms returning between 1 and 9 chairs on camera 59 and 3
to 10 on camera 60, frame to frame, on a room whose furniture never moved. Any
design that took the total from detection would inherit that.

So the inventory is CONFIGURATION. Adding a physical chair does not change the
total until somebody maps it, which is the intended behaviour and not a
limitation: a system that silently revised its own inventory would be unable to
tell "a chair was added" from "the detector had a bad second".
"""
import pytest

from app.cctv_v2.config.geometry import (
    ChairZone,
    RoomGeometry,
    has_chair_map,
    room_geometry,
)
from app.cctv_v2.pipeline.occupancy import RoomOccupancy
from app.cctv_v2.pipeline.track import PersonTrack, TrackState

W, H = 960.0, 1080.0
ROOM = 59

SEATS = (
    ChairZone("A", (0.10, 0.40, 0.25, 0.70)),
    ChairZone("B", (0.40, 0.40, 0.55, 0.70)),
    ChairZone("C", (0.70, 0.40, 0.85, 0.70)),
)


def room(chairs=SEATS):
    from app.cctv_v2.pipeline.occupancy import ChairStatus

    r = RoomOccupancy(ROOM, (W, H))
    r.geometry = RoomGeometry(chairs=chairs)
    r.chairs = {c.chair_id: ChairStatus(chair_id=c.chair_id) for c in chairs}
    return r


def sitting_in(seat: ChairZone, track_id=1):
    x1, y1, x2, y2 = seat.box
    return PersonTrack(
        camera_id=ROOM, track_id=track_id,
        bbox=(x1 * W, y1 * H, x2 * W, y2 * H),
        confidence=0.05, frame_timestamp=0.0, frame_sequence=1,
        first_seen=0.0, last_seen=0.0, hits=5, state=TrackState.CONFIRMED,
    )


SETTLE = max(RoomGeometry().confirm_free, RoomGeometry().confirm_occupied) + 1


def settle(r, tracks, n=SETTLE):
    snap = None
    for i in range(n):
        snap = r.update(tracks, float(i))
    return snap


# ---------------------------------------------------------------------------
# The total comes from the map
# ---------------------------------------------------------------------------
def test_the_total_is_the_size_of_the_configured_map():
    assert settle(room(), []).total_chairs == len(SEATS)


def test_an_empty_room_still_has_all_its_chairs():
    """The commonest way a detection-derived total would betray itself."""
    snap = settle(room(), [])
    assert snap.total_chairs == 3
    assert snap.occupied_chairs == 0
    assert snap.free_chairs == 3


def test_people_arriving_and_leaving_never_change_the_total():
    r = room()
    assert settle(r, []).total_chairs == 3
    assert settle(r, [sitting_in(SEATS[0], 1)]).total_chairs == 3
    assert settle(r, [sitting_in(SEATS[0], 1),
                      sitting_in(SEATS[1], 2)]).total_chairs == 3
    assert settle(r, []).total_chairs == 3


def test_occupied_and_free_always_account_for_every_chair():
    """The identity that makes the numbers safe to divide."""
    r = room()
    for tracks in ([], [sitting_in(SEATS[0], 1)],
                   [sitting_in(SEATS[0], 1), sitting_in(SEATS[2], 3)]):
        snap = settle(r, tracks)
        assert (snap.occupied_chairs + snap.free_chairs
                + snap.unknown_chairs) == snap.total_chairs


def test_the_pipeline_never_asks_the_detector_how_many_chairs_there_are():
    """`detect_chairs` exists for placing zones during setup. If the occupancy
    module ever imports it, the total has become a per-frame measurement of a
    thing COCO was measured returning 1-9 of, on a room with ten."""
    import inspect

    from app.cctv_v2.pipeline import occupancy

    src = inspect.getsource(occupancy)
    assert "detect_chairs" not in src
    assert "CHAIR_CLASS_ID" not in src


# ---------------------------------------------------------------------------
# Adding a physical chair
# ---------------------------------------------------------------------------
def test_adding_a_physical_chair_does_not_change_the_total_by_itself():
    """A chair carried into the room is invisible to the inventory until it is
    mapped. That is the intended behaviour: the alternative cannot distinguish
    a new chair from a noisy frame."""
    before = settle(room(), []).total_chairs
    # A chair now physically exists that the map does not know about. Nothing
    # about the running system has been told, because nothing can be.
    assert settle(room(), []).total_chairs == before == 3


def test_mapping_the_new_chair_is_what_changes_the_total():
    added = SEATS + (ChairZone("D", (0.30, 0.75, 0.45, 0.95)),)
    assert settle(room(chairs=added), []).total_chairs == 4


def test_a_newly_mapped_chair_can_be_occupied_like_any_other():
    """A seat added by hand is not second-class -- it participates in occupancy
    immediately, otherwise the map and the behaviour would disagree."""
    new = ChairZone("D", (0.30, 0.75, 0.45, 0.95))
    snap = settle(room(chairs=SEATS + (new,)), [sitting_in(new, 9)])
    assert snap.total_chairs == 4
    assert snap.occupied_chairs == 1
    assert next(c for c in snap.chairs if c["id"] == "D")["occupied"] is True


def test_removing_a_chair_from_the_map_removes_it_from_the_total():
    assert settle(room(chairs=SEATS[:2]), []).total_chairs == 2


# ---------------------------------------------------------------------------
# The live maps
# ---------------------------------------------------------------------------
def test_the_live_totals_are_whatever_the_map_says():
    """Pinned so a change to the chair map is a deliberate, visible act rather
    than something that drifts."""
    # Per-camera OWNED counts. The room has 13 chairs and neither camera owns
    # all of them -- the row both cameras see belongs to 60 alone.
    assert len(room_geometry(59).chairs) == 8
    assert len(room_geometry(60).chairs) == 6
    assert has_chair_map(59) and has_chair_map(60)


def test_every_configured_chair_id_is_unique_within_its_camera():
    """Two zones sharing an id silently become one chair in the status dict,
    so the total would count two and report one."""
    for cam in (59, 60):
        ids = [c.chair_id for c in room_geometry(cam).chairs]
        assert len(ids) == len(set(ids)), f"camera {cam} has a duplicate id"


# ---------------------------------------------------------------------------
# A chair belongs to exactly one camera
# ---------------------------------------------------------------------------
# Cameras 59 and 60 face each other along one desk. Simultaneous frames at
# 10:19:09 and 10:19:11 (data/cctv_v2_stability/sync59_0.jpg and sync60_0.jpg)
# show the same three people -- white shirt, dark green check, yellow top -- in
# BOTH views, in reversed order.
#
# So the row they both see is ONE set of physical chairs. Mapping it on both
# cameras made a single seat answerable to two independent state machines, and
# made a 13-chair room look like 18.
def test_no_chair_id_is_claimed_by_two_cameras():
    """The structural guarantee: a camera's map IS its ownership, so two
    cameras cannot both control a seat without sharing an id."""
    from app.cctv_v2.config.geometry import room_geometry

    a = {c.chair_id for c in room_geometry(59).chairs}
    b = {c.chair_id for c in room_geometry(60).chairs}
    assert a & b == set(), f"both cameras claim {sorted(a & b)}"


def test_the_shared_row_is_owned_by_the_camera_that_sees_it_better():
    """Camera 60 owns it: from 59 the three occupants span x 0.13-0.30 and
    overlap each other; from 60 they span x 0.20-0.60 and are separated."""
    from app.cctv_v2.config.geometry import (
        observed_elsewhere, observed_owner, room_geometry)

    assert len(room_geometry(60).chairs) == 6          # camera 60 owns the row
    assert observed_owner(59) == 60                    # 59 only watches it
    # Camera 59's view of the same row. FEWER zones than camera 60 owns, and
    # that is expected rather than a mismatch to fix: these are display-only,
    # and 59 cannot see the near end of the row that 60 owns. They must never be
    # used to infer how many chairs the row has.
    assert len(observed_elsewhere(59)) <= len(room_geometry(60).chairs)


def test_a_camera_only_evaluates_the_chairs_it_owns():
    """Zones another camera owns take no part in this one's occupancy -- they
    are not counted, not claimed and not settled."""
    from app.cctv_v2.config.geometry import observed_elsewhere, room_geometry

    r = RoomOccupancy(59, (W, H))
    owned = {c.chair_id for c in room_geometry(59).chairs}
    watched = {zid for zid, _ in observed_elsewhere(59)}
    assert set(r.chairs) == owned
    assert set(r.chairs) & watched == set()


def test_the_room_total_counts_each_chair_once():
    """Not the sum of the per-camera maps -- that double-counts the shared
    row, which is what made 13 chairs look like 18."""
    from app.cctv_v2.config.geometry import room_chair_total, room_geometry

    assert room_chair_total(59) == room_chair_total(60) == 14
    assert (len(room_geometry(59).chairs)
            + len(room_geometry(60).chairs)) == 14


def test_a_camera_alone_in_its_room_owns_its_whole_room():
    """The room total degenerates correctly when there is no peer."""
    from app.cctv_v2.config.geometry import room_chair_total, room_geometry

    assert room_chair_total(57) == len(room_geometry(57).chairs) == 0
