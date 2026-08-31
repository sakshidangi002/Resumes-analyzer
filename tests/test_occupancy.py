"""Room people count and chair occupancy.

TWO MEASUREMENTS, NEVER RECONCILED. `people_count` and `occupied_chairs` answer
different questions and will disagree: a person standing has no chair, and a
seat may hold someone the detector cannot see. `people=8, chairs_occupied=7` is
two facts, not an inconsistency, and forcing them to agree destroys information.

PRESENT MEANS SEEN. Only CONFIRMED tracks count. A track the tracker is holding
for re-association is remembered, not present -- V1 conflated those and reported
five people in an empty corridor. And it counts TRACKS, never detections: one
person on four passes is one person.

CHAIRS ARE CONFIGURED, NOT DETECTED. Measured against these rooms, COCO chair
detection returned 1-9 chairs on camera 59 (about 10 visible) and 3-10 on camera
60 (6-8 visible), depending on input size and confidence. Occupancy built on a
seat count that moves like that would look authoritative and be wrong.

Chair zones are injected here rather than read from the live config, so these
tests describe the LOGIC and cannot break when somebody re-aims a camera.
"""
import pytest

from app.cctv_v2.config.geometry import ChairZone, RoomGeometry, has_chair_map
from app.cctv_v2.pipeline.occupancy import (
    ChairState,
    OccupancyRegistry,
    RoomOccupancy,
)
from app.cctv_v2.pipeline.track import PersonTrack, TrackState

W, H = 960.0, 1080.0
ROOM_A, ROOM_B = 59, 60

# Two seats side by side, each a fifth of the frame wide.
SEATS = (
    ChairZone("C1", (0.10, 0.50, 0.30, 0.80)),
    ChairZone("C2", (0.40, 0.50, 0.60, 0.80)),
)


def room(camera_id=ROOM_A, chairs=SEATS, **kw):
    """A room with injected seats, so these tests describe the LOGIC and cannot
    break when somebody re-aims a camera or fills in the live chair map."""
    from app.cctv_v2.pipeline.occupancy import ChairStatus

    r = RoomOccupancy(camera_id, (W, H))
    r.geometry = RoomGeometry(chairs=chairs, **kw)
    r.chairs = {c.chair_id: ChairStatus(chair_id=c.chair_id) for c in chairs}
    return r


def seated(camera_id, track_id, seat: ChairZone, state=TrackState.CONFIRMED):
    """A person whose LOWER HALF sits inside the given seat box."""
    x1, y1, x2, y2 = seat.box
    cx = (x1 + x2) / 2.0
    half_w = (x2 - x1) * 0.45
    # Box top well above the seat; bottom half lands in it.
    top = (y1 - (y2 - y1)) * H
    bottom = y2 * H
    return PersonTrack(
        camera_id=camera_id, track_id=track_id,
        bbox=((cx - half_w) * W, top, (cx + half_w) * W, bottom),
        confidence=0.6, frame_timestamp=0.0, frame_sequence=1,
        first_seen=0.0, last_seen=0.0, hits=5, state=state,
    )


def standing_elsewhere(camera_id, track_id):
    return PersonTrack(
        camera_id=camera_id, track_id=track_id,
        bbox=(0.80 * W, 0.10 * H, 0.95 * W, 0.45 * H),
        confidence=0.6, frame_timestamp=0.0, frame_sequence=1,
        first_seen=0.0, last_seen=0.0, hits=5, state=TrackState.CONFIRMED,
    )


# Enough passes to settle EITHER direction. Derived from the geometry rather
# than written as a number, so raising confirm_free cannot silently leave these
# tests asserting a state the room has not reached yet.
SETTLE = max(RoomGeometry().confirm_free, RoomGeometry().confirm_occupied) + 1


def settle(r, tracks, n=None, t0=0.0):
    snap = None
    for i in range(n if n is not None else SETTLE):
        snap = r.update(tracks, timestamp=t0 + i)
    return snap


# ---------------------------------------------------------------------------
# People count
# ---------------------------------------------------------------------------
def test_a_confirmed_track_counts_as_present():
    r = room()
    snap = r.update([seated(ROOM_A, 1, SEATS[0])], 0.0)
    assert snap.people_count == 1
    assert snap.active_track_ids == (1,)


def test_a_held_track_is_NOT_counted_as_present():
    """V1 published held tracks as people and showed five in an empty room."""
    r = room()
    lost = seated(ROOM_A, 1, SEATS[0], state=TrackState.LOST)
    assert r.update([lost], 0.0).people_count == 0


def test_a_tentative_track_is_not_yet_a_person():
    r = room()
    t = seated(ROOM_A, 1, SEATS[0], state=TrackState.TENTATIVE)
    assert r.update([t], 0.0).people_count == 0


def test_the_same_person_across_passes_stays_one_person():
    """Counts TRACKS, not detections. Summing detections across passes is how a
    still room reports a crowd."""
    r = room()
    person = seated(ROOM_A, 1, SEATS[0])
    for i in range(5):
        snap = r.update([person], float(i))
    assert snap.people_count == 1


def test_several_tracks_give_the_right_count():
    r = room()
    snap = r.update(
        [seated(ROOM_A, 1, SEATS[0]), seated(ROOM_A, 2, SEATS[1]),
         standing_elsewhere(ROOM_A, 3)], 0.0)
    assert snap.people_count == 3
    assert snap.active_track_ids == (1, 2, 3)


def test_an_empty_room_reports_zero():
    r = room()
    assert r.update([], 0.0).people_count == 0


def test_room_occupancy_refuses_another_camera_s_tracks():
    r = room(ROOM_A)
    with pytest.raises(ValueError, match="per-camera"):
        r.update([seated(ROOM_B, 1, SEATS[0])], 0.0)


def test_camera_59_state_never_reaches_camera_60():
    reg = OccupancyRegistry((W, H))
    reg.update(ROOM_A, [seated(ROOM_A, 1, SEATS[0])], 0.0)
    assert reg.update(ROOM_B, [], 0.0).people_count == 0
    assert reg.get(ROOM_A).observations == 1
    assert reg.get(ROOM_B).observations == 1


# ---------------------------------------------------------------------------
# Chair occupancy
# ---------------------------------------------------------------------------
def test_a_person_in_a_seat_makes_it_occupied():
    r = room()
    # Settles BOTH directions: C1 needs confirm_occupied passes to fill, C2
    # needs confirm_free passes to leave UNKNOWN. Asserting the pair after only
    # the shorter of the two would be asserting a state the room is still in the
    # middle of reaching.
    snap = settle(r, [seated(ROOM_A, 1, SEATS[0])])

    assert snap.occupied_chairs == 1
    assert snap.free_chairs == 1
    assert snap.total_chairs == 2
    c1 = [c for c in snap.chairs if c["id"] == "C1"][0]
    assert c1["occupied"] is True
    assert c1["occupant_track_id"] == 1


def test_an_empty_seat_is_free_not_occupied():
    """A chair being visible says nothing about whether anyone is in it."""
    r = room()
    snap = settle(r, [], 5)
    assert snap.occupied_chairs == 0
    assert snap.free_chairs == 2


def test_a_person_standing_away_from_every_seat_occupies_none():
    r = room()
    snap = settle(r, [standing_elsewhere(ROOM_A, 1)], 5)
    assert snap.people_count == 1
    assert snap.occupied_chairs == 0


def test_people_and_occupied_chairs_are_allowed_to_disagree():
    """One seated, one standing: 2 people, 1 chair. Not an inconsistency."""
    r = room()
    snap = settle(r, [seated(ROOM_A, 1, SEATS[0]),
                      standing_elsewhere(ROOM_A, 2)], 3)
    assert snap.people_count == 2
    assert snap.occupied_chairs == 1


def test_one_person_cannot_occupy_two_seats():
    """A person sitting between two chairs covers both. They are one person and
    they are in one seat -- their best."""
    r = room()
    between = PersonTrack(
        camera_id=ROOM_A, track_id=1,
        bbox=(0.25 * W, 0.20 * H, 0.45 * W, 0.80 * H),   # straddles C1 and C2
        confidence=0.6, frame_timestamp=0.0, frame_sequence=1,
        first_seen=0.0, last_seen=0.0, hits=5, state=TrackState.CONFIRMED,
    )
    snap = settle(r, [between], 4)
    assert snap.occupied_chairs <= 1, "one person filled two seats"


def test_two_people_in_adjacent_seats_occupy_two():
    r = room()
    snap = settle(r, [seated(ROOM_A, 1, SEATS[0]), seated(ROOM_A, 2, SEATS[1])], 3)
    assert snap.occupied_chairs == 2
    assert snap.free_chairs == 0


def test_the_chair_count_is_fixed_and_does_not_track_detections():
    """Configured seats, not detected ones. The total must not move as people
    come and go."""
    r = room()
    totals = []
    for tracks in ([], [seated(ROOM_A, 1, SEATS[0])],
                   [seated(ROOM_A, 1, SEATS[0]), seated(ROOM_A, 2, SEATS[1])], []):
        totals.append(r.update(tracks, 0.0).total_chairs)
    assert totals == [2, 2, 2, 2]


def test_chair_ids_are_stable_across_passes():
    r = room()
    ids = []
    for i in range(4):
        ids.append(tuple(c["id"] for c in r.update([], float(i)).chairs))
    assert len(set(ids)) == 1 == len({("C1", "C2")} & set(ids))


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------
def test_occupancy_does_not_flip_on_a_single_observation():
    r = room()
    first = r.update([seated(ROOM_A, 1, SEATS[0])], 0.0)
    assert first.occupied_chairs == 0, "one observation changed the state"

    second = r.update([seated(ROOM_A, 1, SEATS[0])], 1.0)
    assert second.occupied_chairs == 1


def test_a_briefly_missed_person_does_not_free_the_chair():
    """YOLO drops a seated person for a pass routinely -- behind a desk, at a
    high angle, half occluded. That must not empty the seat."""
    r = room()
    person = seated(ROOM_A, 1, SEATS[0])
    settle(r, [person], 3)

    snap = r.update([], 10.0)                  # one missed pass
    assert snap.occupied_chairs == 1, "one dropped detection emptied the seat"

    snap = r.update([person], 11.0)
    assert snap.occupied_chairs == 1


def test_a_person_who_really_leaves_does_free_the_chair():
    r = room()
    person = seated(ROOM_A, 1, SEATS[0])
    settle(r, [person], 3)
    assert r.update([], 10.0).occupied_chairs == 1

    snap = settle(r, [], 4, t0=11.0)
    assert snap.occupied_chairs == 0
    assert snap.free_chairs == 2


def test_releasing_is_slower_than_claiming():
    """Asymmetric on purpose: a seated person vanishing for a pass is routine,
    a person appearing in a seat they were not in is not."""
    r = room()
    assert r.geometry.confirm_free > r.geometry.confirm_occupied


def test_a_person_moving_between_seats_moves_the_occupancy():
    r = room()
    settle(r, [seated(ROOM_A, 1, SEATS[0])], 3)
    snap = settle(r, [seated(ROOM_A, 1, SEATS[1])], 5, t0=10.0)

    states = {c["id"]: c["occupied"] for c in snap.chairs}
    assert states["C2"] is True
    assert states["C1"] is False


def test_a_passer_by_cannot_evict_a_seated_person():
    """Two claims on one seat: the stronger wins."""
    r = room()
    sitter = seated(ROOM_A, 1, SEATS[0])
    settle(r, [sitter], 3)

    clipping = PersonTrack(
        camera_id=ROOM_A, track_id=2,
        bbox=(0.28 * W, 0.40 * H, 0.34 * W, 0.75 * H),   # barely clips C1
        confidence=0.5, frame_timestamp=0.0, frame_sequence=1,
        first_seen=0.0, last_seen=0.0, hits=5, state=TrackState.CONFIRMED,
    )
    snap = settle(r, [sitter, clipping], 3, t0=10.0)
    c1 = [c for c in snap.chairs if c["id"] == "C1"][0]
    assert c1["occupant_track_id"] == 1, "a passer-by took the seat"


# ---------------------------------------------------------------------------
# Unconfigured cameras and overlapping rooms
# ---------------------------------------------------------------------------
def test_a_camera_with_no_chair_map_says_so_rather_than_reporting_an_empty_room():
    """'0 of 0 occupied' is a configuration gap, not an empty room, and a
    consumer must be able to tell them apart.

    The empty geometry is injected, because both real room cameras now have
    maps -- a test that silently stopped exercising the empty case would be
    worse than no test. `has_chair_map` is checked separately against a camera
    id nobody has configured, which is the other half of the same guarantee.
    """
    r = RoomOccupancy(ROOM_B, (W, H))
    r.geometry = RoomGeometry(chairs=())
    r.chairs = {}
    snap = r.update([], 0.0)

    assert snap.total_chairs == 0
    assert snap.chair_map_configured is False
    assert has_chair_map(4242) is False


def test_both_room_cameras_have_a_hand_placed_chair_map():
    """Replaced the COCO-clustered map, which found 4 seats on camera 59 and
    none on 60.

    The clustered map was not merely incomplete, it was biased: COCO can only
    see a chair nobody is sitting in, so it found spare chairs and missed every
    seat in use. These are read off the setup overlay by hand instead."""
    from app.cctv_v2.config.geometry import room_geometry

    assert has_chair_map(59) is True
    assert has_chair_map(60) is True
    # 7 + 6. Cameras 59 and 60 face each other along one desk, so the row they
    # BOTH see is owned by 60 alone -- see the ownership note in geometry.py.
    # The room has 13 chairs; neither camera owns all of them.
    assert len(room_geometry(59).chairs) == 7
    assert len(room_geometry(60).chairs) == 6


def test_the_seats_people_actually_use_are_in_the_map():
    """The specific gap the old map had.

    Under the COCO map the person at camera 59's right-hand desk had no seat at
    all -- the nearest zone was on the EMPTY chair beside her -- so she could
    never occupy anything however well she was detected. R8 is that seat.

    R1 was ALSO recorded occupied through that window, and is no longer in the
    map: the operator counted the row on 2026-08-31 and said seven, not eight.
    Both things can be true. A zone does not need a chair under it to report
    OCCUPIED -- it only needs a person's box to overlap it -- so a zone drawn
    over the far end of the desk would have been "confirmed" by exactly the
    person who sits at that end. That is the failure mode this whole file
    exists to catch, and it argues for trusting the count over the observation.
    If a chair really is there, R1 goes back into geometry.py by hand."""
    from app.cctv_v2.config.geometry import room_geometry

    ids = {c.chair_id for c in room_geometry(59).chairs}
    assert "R8" in ids
    # The left-hand desk seats moved to their owner, camera 60.
    assert "S3" in {c.chair_id for c in room_geometry(60).chairs}
    assert not any(c.chair_id.startswith("L") for c in room_geometry(59).chairs)


def test_no_chair_zone_is_degenerate():
    """A zero-area zone divides by ~0 in the overlap fraction and would claim
    every person who touched it."""
    from app.cctv_v2.config.geometry import room_geometry

    for cam in (59, 60):
        for chair in room_geometry(cam).chairs:
            x1, y1, x2, y2 = chair.box
            assert x2 - x1 > 0.01, f"{cam}/{chair.chair_id} has no width"
            assert y2 - y1 > 0.01, f"{cam}/{chair.chair_id} has no height"
            assert 0.0 <= x1 < x2 <= 1.0
            assert 0.0 <= y1 < y2 <= 1.0


def test_overlapping_room_cameras_are_never_summed():
    """59 and 60 are declared same-room, so their counts may include the same
    person twice. Adding them would be wrong in a direction nobody can bound."""
    reg = OccupancyRegistry((W, H))
    reg.update(ROOM_A, [seated(ROOM_A, 1, SEATS[0])], 0.0)
    reg.update(ROOM_B, [seated(ROOM_B, 1, SEATS[0])], 0.0)

    snap = reg.snapshot()
    assert set(snap) == {ROOM_A, ROOM_B}
    assert "total_unique_room_people" not in snap
    assert "room_total" not in snap
    for cid in (ROOM_A, ROOM_B):
        assert snap[cid]["shares_room_with"]
        assert "do not add" in snap[cid]["note"]
