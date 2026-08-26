"""The three room-camera layers must answer three questions independently.

    1. Is there a person?          people count, from BODY tracks
    2. Who is that person?         recognition, an annotation on a track
    3. Is a chair occupied?        association of people with configured seats

The failure this file exists to prevent is any of them becoming a function of
another. Specifically:

  * A person whose face cannot be read is still a person. Recognition must never
    subtract from the count -- that is how a room full of people with their
    backs to the lens came to read "People: 0".
  * A seat is occupied whether or not anybody knows who is in it, so occupancy
    must not consult identity at all.
  * Somebody in an unmapped chair, or standing, is UNASSIGNED. Snapping them to
    the nearest seat would invent occupancy that nobody can check.

Detection RECALL -- whether the person was seen in the first place -- is a
separate question needing real frames and real weights, and lives in
test_room_recall.py.
"""
import pytest

from app.cctv_v2.config.geometry import ChairZone, RoomGeometry
from app.cctv_v2.pipeline.occupancy import ChairState, RoomOccupancy, _seat_region
from app.cctv_v2.pipeline.track import PersonTrack, TrackState

W, H = 960.0, 1080.0
ROOM = 59

# A seat in the middle of the frame, and the aisle in front of it. Injected
# rather than read from the live map so these tests describe the LOGIC and
# cannot break when somebody re-aims a camera.
SEAT = ChairZone("T1", (0.40, 0.40, 0.60, 0.70))


def room(chairs=(SEAT,), **kw):
    from app.cctv_v2.pipeline.occupancy import ChairStatus

    r = RoomOccupancy(ROOM, (W, H))
    r.geometry = RoomGeometry(chairs=chairs, **kw)
    r.chairs = {c.chair_id: ChairStatus(chair_id=c.chair_id) for c in chairs}
    return r


def person(track_id, box, state=TrackState.CONFIRMED, **kw):
    """A track from a normalised box. Carries no identity -- V2 tracks have no
    field for one, which is the point."""
    x1, y1, x2, y2 = box
    return PersonTrack(
        camera_id=ROOM, track_id=track_id,
        bbox=(x1 * W, y1 * H, x2 * W, y2 * H),
        confidence=kw.get("confidence", 0.04),
        frame_timestamp=0.0, frame_sequence=1,
        first_seen=0.0, last_seen=0.0, hits=5, state=state,
    )


# One more pass than the slowest confirmation, so a test never asserts a state
# the room has not had time to reach.
SETTLE = max(RoomGeometry().confirm_free, RoomGeometry().confirm_occupied) + 1


def settle(r, tracks, n=SETTLE):
    snap = None
    for i in range(n):
        snap = r.update(tracks, timestamp=float(i))
    return snap


# ---------------------------------------------------------------------------
# 1. Is there a person?  -- never a function of recognition
# ---------------------------------------------------------------------------
def test_a_person_with_no_face_information_still_counts():
    """A V2 track has no identity fields at all, so a face that was never seen
    cannot remove anybody. Pinned as a structural fact, not a behaviour."""
    p = person(1, (0.42, 0.35, 0.58, 0.68))
    for field in ("employee_id", "employee_name", "matched", "identity_source"):
        assert not hasattr(p, field), (
            f"PersonTrack grew {field!r}; the room count must not be able to "
            "depend on recognition"
        )
    assert room().update([p], 0.0).people_count == 1


def test_the_count_is_bodies_not_faces_however_low_the_confidence():
    """Seated staff seen from behind score 0.035-0.057 on these cameras. A body
    that weak is still a body once the tracker has confirmed it."""
    weak = person(1, (0.42, 0.35, 0.58, 0.68), confidence=0.035)
    assert room().update([weak], 0.0).people_count == 1


def test_people_and_occupied_chairs_are_counted_separately():
    """Two measurements of different things. A standing person has no chair; a
    seat may hold somebody the detector cannot see this pass."""
    r = room()
    snap = settle(r, [person(1, (0.42, 0.35, 0.58, 0.68)),
                      person(2, (0.05, 0.05, 0.15, 0.30))])
    assert snap.people_count == 2
    assert snap.occupied_chairs == 1


# ---------------------------------------------------------------------------
# 3. Is a chair occupied?  -- and only where it can be answered
# ---------------------------------------------------------------------------
def test_a_seated_person_occupies_their_seat():
    snap = settle(room(), [person(1, (0.42, 0.35, 0.58, 0.68))])
    assert snap.occupied_chairs == 1
    assert snap.chairs[0]["occupant_track_id"] == 1


def test_somebody_standing_in_front_of_a_seat_does_not_occupy_it():
    """The failure that a pure-overlap rule cannot avoid.

    Camera 59's aisle runs directly in front of a row of eight chairs, so a
    person walking down it passes between the camera and a seat and covers it
    completely. By overlap alone that is a perfect claim.

    What separates them is how far DOWN the box goes: a seated person cannot
    extend far below their chair, while somebody standing in front of it reaches
    the floor, which in this projection is well below the chair's base.
    """
    stander = person(1, (0.42, 0.30, 0.58, 0.95))     # 0.25 below the seat
    snap = settle(room(), [stander])
    assert snap.occupied_chairs == 0
    assert snap.free_chairs == 1
    assert snap.people_count == 1, "still a person, just not a seated one"


def test_the_drop_allowance_is_what_decides_it():
    """Same person, same seat, different allowance -- so the behaviour above is
    the threshold doing its job and not an accident of the box."""
    stander = person(1, (0.42, 0.30, 0.58, 0.95))
    strict = settle(room(max_drop_below_seat=0.12), [stander])
    loose = settle(room(max_drop_below_seat=0.40), [stander])
    assert strict.occupied_chairs == 0
    assert loose.occupied_chairs == 1


def test_a_person_in_no_mapped_seat_is_left_unassigned():
    """Never snapped to the nearest chair. An unmapped seat must read as
    'no mapped seat', which is a different claim from 'not sitting'."""
    snap = settle(room(), [person(1, (0.02, 0.02, 0.12, 0.25))])
    assert snap.occupied_chairs == 0
    assert snap.people_count == 1
    assert all(c["occupant_track_id"] is None for c in snap.chairs)


def test_occupancy_never_consults_identity():
    """The occupancy layer is handed tracks that have no identity, and its
    output carries none either -- a seat is occupied whether or not anybody
    knows who is in it."""
    snap = settle(room(), [person(1, (0.42, 0.35, 0.58, 0.68))])
    for chair in snap.chairs:
        assert set(chair) == {"id", "occupied", "state", "occupant_track_id",
                              "zone"}


def test_an_empty_mapped_seat_reads_free_not_unknown():
    snap = settle(room(), [])
    assert snap.chairs[0]["state"] == ChairState.FREE.value


# ---------------------------------------------------------------------------
# The association geometry itself
# ---------------------------------------------------------------------------
def test_the_whole_person_box_is_tested_against_a_seat():
    """Not the lower half, which is what shipped.

    Measured over 36 labelled people: the whole box got 36 seats right, the
    lower half 31 with 1 wrong and 4 missed. A seated person's box bottom is
    where the desk edge cuts them off, so the lower half is desk, not lap.
    """
    p = person(1, (0.20, 0.10, 0.40, 0.50))
    assert _seat_region(p, W, H) == pytest.approx((0.20, 0.10, 0.40, 0.50))


def test_two_people_cannot_both_claim_one_seat():
    r = room()
    snap = settle(r, [person(1, (0.42, 0.35, 0.58, 0.68)),
                      person(2, (0.44, 0.36, 0.56, 0.66))])
    assert snap.occupied_chairs == 1
    assert snap.people_count == 2


def test_one_person_cannot_claim_two_seats():
    seats = (ChairZone("T1", (0.40, 0.40, 0.60, 0.70)),
             ChairZone("T2", (0.55, 0.40, 0.75, 0.70)))
    snap = settle(room(chairs=seats), [person(1, (0.45, 0.35, 0.70, 0.68))])
    assert snap.occupied_chairs == 1


# ---------------------------------------------------------------------------
# Geometry travels with state
# ---------------------------------------------------------------------------
def test_each_chair_reports_where_it_is():
    """The overlay has to draw the zone. It used to keep its own copy of the
    chair map, so a seat added to the backend was counted but never drawn --
    completing the map would have made the picture less complete."""
    snap = settle(room(), [])
    assert snap.chairs[0]["zone"] == list(SEAT.box)


def test_the_zone_follows_the_geometry_the_room_is_actually_using():
    """Derived, not cached at construction. Two chair maps that can disagree is
    the bug this module already carries a scar from."""
    r = room()
    moved = ChairZone("T1", (0.10, 0.10, 0.20, 0.20))
    r.geometry = RoomGeometry(chairs=(moved,))
    assert settle(r, []).chairs[0]["zone"] == list(moved.box)
