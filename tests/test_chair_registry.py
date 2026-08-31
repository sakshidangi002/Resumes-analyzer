"""The chair inventory must change slowly, and never lose an occupied chair.

Automatic chair detection was kept out of the live pipeline for a reason: the
raw COCO detector returns 1-9 chairs on camera 59 and 3-10 on camera 60 across
sweeps of the same unchanged room. These tests pin the behaviour that makes it
usable anyway -- evidence accumulated over sweeps, and a retirement rule that
refuses to count a miss as absence while somebody is sitting in the way.

The occlusion test is the one that matters most. Without that rule the system
would delete precisely the chairs that are in use, because a chair with a person
in it is largely invisible to a chair detector.
"""
import pytest

from app.cctv_v2.pipeline.chair_registry import (
    CANDIDATE_TTL_SWEEPS,
    PROMOTE_SWEEPS,
    RETIRE_SWEEPS,
    ChairRegistry,
    ChairRegistryStore,
)

# A seat, and a person standing squarely in front of it.
CHAIR = (0.40, 0.40, 0.50, 0.52)
PERSON_OVER_CHAIR = (0.35, 0.30, 0.56, 0.70)
# Far enough away to cover none of CHAIR.
PERSON_ELSEWHERE = (0.80, 0.10, 0.95, 0.40)


def test_seeded_registry_matches_the_hand_verified_config_map():
    """Day one must equal the measured room, not an empty room counting up."""
    assert ChairRegistry(59).confirmed_count == 7
    assert ChairRegistry(60).confirmed_count == 6


def test_geometry_keeps_the_cameras_tuned_occupancy_thresholds():
    """Only the chair LIST is learned; the thresholds were tuned per camera."""
    reg = ChairRegistry(59)
    learned = reg.geometry()
    base = reg._base
    assert learned.min_overlap == base.min_overlap
    assert learned.max_drop_below_seat == base.max_drop_below_seat
    assert learned.confirm_occupied == base.confirm_occupied
    assert learned.confirm_free == base.confirm_free


def test_a_new_chair_needs_repeated_sightings_before_it_counts():
    reg = ChairRegistry(59, seed=False)
    for sweep in range(1, PROMOTE_SWEEPS):
        reg.sweep([CHAIR])
        assert reg.confirmed_count == 0, f"admitted after only {sweep} sighting(s)"
        assert reg.pending_count == 1
    reg.sweep([CHAIR])
    assert reg.confirmed_count == 1


def test_a_single_false_positive_never_becomes_a_chair():
    """One sweep's phantom must expire, not linger as a permanent seat."""
    reg = ChairRegistry(59, seed=False)
    reg.sweep([CHAIR])
    assert reg.pending_count == 1
    for _ in range(CANDIDATE_TTL_SWEEPS):
        reg.sweep([])
    assert reg.pending_count == 0
    assert reg.confirmed_count == 0


def test_a_chair_hidden_by_a_person_is_never_retired():
    """The rule the whole design turns on.

    A chair detector cannot see a chair somebody is sitting in. If those misses
    counted as evidence of absence, every occupied chair would be deleted -- so
    an occupied room would slowly report fewer and fewer seats.
    """
    reg = ChairRegistry(59, seed=False)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([CHAIR])
    assert reg.confirmed_count == 1

    # Many sweeps where the detector never finds it, but a person is over it.
    for _ in range(RETIRE_SWEEPS * 3):
        reg.sweep([], person_boxes=[PERSON_OVER_CHAIR])

    assert reg.confirmed_count == 1, "an occupied chair was deleted"
    chair = next(iter(reg.chairs.values()))
    assert chair.visible_misses == 0, "an occluded miss must not count toward retirement"


def test_a_chair_that_is_visibly_gone_is_retired():
    reg = ChairRegistry(59, seed=False)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([CHAIR])
    assert reg.confirmed_count == 1

    for _ in range(RETIRE_SWEEPS - 1):
        reg.sweep([], person_boxes=[PERSON_ELSEWHERE])
    assert reg.confirmed_count == 1, "retired before the evidence bar was met"

    reg.sweep([], person_boxes=[PERSON_ELSEWHERE])
    assert reg.confirmed_count == 0


def test_one_sighting_resets_the_retirement_clock():
    """A chair seen again is not on its way out, however long it was missing."""
    reg = ChairRegistry(59, seed=False)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([CHAIR])
    for _ in range(RETIRE_SWEEPS - 1):
        reg.sweep([])
    reg.sweep([CHAIR])
    for _ in range(RETIRE_SWEEPS - 1):
        reg.sweep([])
    assert reg.confirmed_count == 1


def test_retiring_demands_more_evidence_than_admitting():
    """Losing a real chair is worse than gaining a phantom one."""
    assert RETIRE_SWEEPS > PROMOTE_SWEEPS


def test_a_moved_chair_follows_rather_than_becoming_a_second_chair():
    reg = ChairRegistry(59, seed=False)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([CHAIR])

    shifted = CHAIR
    for _ in range(12):
        shifted = (shifted[0] + 0.004, shifted[1], shifted[2] + 0.004, shifted[3])
        reg.sweep([shifted])

    assert reg.confirmed_count == 1, "a nudged chair was registered twice"
    box = next(iter(reg.chairs.values())).box
    assert box[0] > CHAIR[0], "the stored box did not follow the chair"


def test_ids_are_never_reused_after_a_chair_is_retired():
    """A reused id would attach a retired seat's history to a new one."""
    reg = ChairRegistry(59, seed=False)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([CHAIR])
    first_id = next(iter(reg.chairs))

    for _ in range(RETIRE_SWEEPS):
        reg.sweep([])
    assert reg.chairs == {}

    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([CHAIR])
    second_id = next(iter(reg.chairs))
    assert second_id != first_id


def test_people_elsewhere_do_not_shield_a_missing_chair():
    reg = ChairRegistry(59, seed=False)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([CHAIR])
    for _ in range(RETIRE_SWEEPS):
        reg.sweep([], person_boxes=[PERSON_ELSEWHERE])
    assert reg.confirmed_count == 0


def test_malformed_boxes_are_ignored_rather_than_crashing():
    """This is fed by a third-party detector; a bad row must not stop the sweep."""
    reg = ChairRegistry(59, seed=False)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([CHAIR, (0.1, 0.2), None])
    assert reg.confirmed_count == 1


def test_inventory_survives_a_restart(tmp_path):
    """Otherwise a newly added chair is forgotten every time the backend bounces."""
    path = tmp_path / "chairs.json"
    store = ChairRegistryStore(path=path)
    reg = store.get(59)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([(0.05, 0.05, 0.12, 0.14)])   # a chair added to the room
    expected = reg.confirmed_count
    assert expected == 8, "7 configured + 1 newly detected"
    store.save()

    reloaded = ChairRegistryStore(path=path)
    reloaded.load()
    assert reloaded.get(59).confirmed_count == expected


def test_an_unreadable_registry_file_falls_back_to_the_config_map(tmp_path):
    path = tmp_path / "chairs.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = ChairRegistryStore(path=path)
    store.load()
    assert store.get(59).confirmed_count == 7


def test_saving_without_a_path_is_a_no_op():
    ChairRegistryStore(path=None).save()


@pytest.mark.parametrize("camera_id, expected", [(59, 7), (60, 6)])
def test_confirmed_zones_are_ordered_stably(camera_id, expected):
    """The overlay draws these in order; a shuffling list would renumber badges."""
    reg = ChairRegistry(camera_id)
    ids = [z.chair_id for z in reg.confirmed_zones()]
    assert len(ids) == expected
    assert ids == sorted(ids)


# --- adopting a learned map into a live room ------------------------------
#
# `RoomOccupancy.chairs` must mirror `RoomOccupancy.geometry` exactly: `update`
# indexes the dict by every seat in the geometry, and the occupied/free/unknown
# counts are taken over the dict. A map that gains or loses a seat without the
# dict following is a KeyError or a phantom chair respectively.

from app.cctv_v2.config.geometry import ChairZone, RoomGeometry   # noqa: E402
from app.cctv_v2.pipeline.occupancy import ChairState, RoomOccupancy   # noqa: E402
from app.cctv_v2.pipeline.track import PersonTrack, TrackState   # noqa: E402

W, H = 960.0, 1080.0


def _room():
    return RoomOccupancy(camera_id=59, frame_size=(W, H))


def _geometry(*chair_ids):
    return RoomGeometry(
        chairs=tuple(
            ChairZone(cid, (0.10 + 0.1 * i, 0.40, 0.18 + 0.1 * i, 0.52))
            for i, cid in enumerate(chair_ids)
        )
    )


def test_adopting_a_map_with_a_new_chair_does_not_crash_the_next_update():
    """The KeyError this guards against would stop occupancy for the whole room."""
    room = _room()
    assert room.apply_geometry(_geometry("R1", "R2", "NEW1")) is True
    room.update([], timestamp=1.0)          # must not raise
    assert set(room.chairs) == {"R1", "R2", "NEW1"}


def test_a_newly_detected_chair_starts_unknown_not_free():
    """Nothing has been observed about it yet, and saying FREE would be a claim."""
    room = _room()
    room.apply_geometry(_geometry("R1", "NEW1"))
    assert room.chairs["NEW1"].state is ChairState.UNKNOWN


def test_a_removed_chair_stops_being_counted():
    room = _room()
    before = len(room.chairs)
    room.apply_geometry(_geometry("R1", "R2"))
    snap = room.update([], timestamp=1.0)
    assert len(room.chairs) == 2 < before
    assert snap.total_chairs == 2


def test_a_surviving_chair_keeps_its_state_across_a_map_change():
    """Re-aiming the map must not reset a chair somebody is sitting in."""
    room = _room()
    room.apply_geometry(_geometry("R1", "R2"))
    seated = PersonTrack(
        camera_id=59, track_id=1,
        bbox=(0.10 * W, 0.36 * H, 0.18 * W, 0.53 * H),
        confidence=0.6, frame_timestamp=0.0, frame_sequence=1,
        first_seen=0.0, last_seen=0.0, hits=5, state=TrackState.CONFIRMED,
    )
    for i in range(room.geometry.confirm_occupied + 1):
        room.update([seated], timestamp=float(i))
    assert room.chairs["R1"].state is ChairState.OCCUPIED

    room.apply_geometry(_geometry("R1", "R2", "NEW1"))
    assert room.chairs["R1"].state is ChairState.OCCUPIED, "an occupied seat was reset"


def test_an_empty_map_is_refused_so_a_fault_cannot_blank_the_room():
    room = _room()
    before = dict(room.chairs)
    assert room.apply_geometry(RoomGeometry(chairs=())) is False
    assert room.chairs == before


# --- the two ways a learned count inflates -------------------------------
#
# Both of these were caught in LIVE data on 2026-08-31, within five sweeps of
# switching automatic counting on. The boxes below are the real ones.

def test_a_second_box_on_one_chair_does_not_become_a_second_chair():
    """Matching is one-to-one, so a chair drawn twice leaves a box unmatched.

    Camera 59 admitted "A1" at (0.692, 0.697, 0.854, 0.999) while already
    holding R8 at (0.716, 0.618, 0.855, 0.872) -- an IoU of 0.40, plainly the
    same seat, split in two because a different detection had already claimed
    R8 that sweep.
    """
    reg = ChairRegistry(59)
    before = reg.confirmed_count
    duplicate_of_r8 = (0.692, 0.697, 0.854, 0.999)
    for _ in range(PROMOTE_SWEEPS * 3):
        reg.sweep([duplicate_of_r8])
    assert reg.confirmed_count == before, "one chair was counted twice"
    assert reg.pending_count == 0


def test_a_chair_a_peer_camera_owns_is_never_admitted():
    """Cameras 59 and 60 both see one desk row; each seat has ONE owner.

    Camera 59 admitted a chair at (0.218, 0.710, 0.374, 1.000), which is camera
    60's seat -- it sits under camera 59's own L5 observation zone. Counting it
    here makes the room total the sum of two overlapping views.
    """
    reg = ChairRegistry(59)
    before = reg.confirmed_count
    owned_by_camera_60 = (0.218, 0.710, 0.374, 1.000)
    for _ in range(PROMOTE_SWEEPS * 3):
        reg.sweep([owned_by_camera_60])
    assert reg.confirmed_count == before
    assert reg.pending_count == 0


def test_the_room_total_stays_at_fourteen_under_overlapping_detections():
    """The end-to-end property: what the operator actually sees."""
    r59, r60 = ChairRegistry(59), ChairRegistry(60)
    for _ in range(PROMOTE_SWEEPS * 3):
        # Both cameras keep seeing the shared row, from opposite ends.
        r59.sweep([(0.218, 0.710, 0.374, 1.000), (0.692, 0.697, 0.854, 0.999)])
        r60.sweep([(0.202, 0.352, 0.270, 0.571)])
    assert r59.confirmed_count + r60.confirmed_count == 13


def test_a_genuinely_new_chair_in_open_space_is_still_admitted():
    """The suppression must not make the feature inert."""
    reg = ChairRegistry(59)
    before = reg.confirmed_count
    empty_corner = (0.03, 0.03, 0.10, 0.12)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([empty_corner])
    assert reg.confirmed_count == before + 1


def test_the_detector_cannot_delete_a_chair_a_human_measured():
    """The most consequential default in the module.

    A live sweep of camera 59 finds 4-6 of the 8 chairs that are really there.
    Intermittent misses are harmless -- any sighting resets the counter -- but a
    chair the detector consistently cannot see would otherwise be retired, and
    it would be one somebody had counted by hand. The detector may extend the
    human's inventory; it does not overrule it.
    """
    reg = ChairRegistry(59)
    seeded = reg.confirmed_count
    assert seeded == 7

    # The detector never sees any of them, for far longer than the bar.
    for _ in range(RETIRE_SWEEPS * 5):
        reg.sweep([])

    assert reg.confirmed_count == seeded, "a hand-verified chair was deleted"


def test_a_chair_the_detector_added_can_still_be_retired_by_it():
    """The protection is for the human's count, not for the detector's guesses."""
    reg = ChairRegistry(59, seed=False)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([CHAIR])
    assert reg.confirmed_count == 1
    for _ in range(RETIRE_SWEEPS):
        reg.sweep([])
    assert reg.confirmed_count == 0


def test_the_room_total_counts_this_cameras_own_chairs_too():
    """Seen live: camera 59's panel read "this camera owns 12 of 9 in the room".

    `same_room_peers` returns the OTHER cameras in the room -- it subtracts the
    caller. Summing only those handed each camera its neighbour's count as the
    room total, so a camera could own more chairs than the room contained.
    """
    from app.cctv_v2.pipeline.chair_registry import ChairRegistryStore, room_confirmed_total

    store = ChairRegistryStore()
    own = store.get(59).confirmed_count
    peer = store.get(60).confirmed_count
    total = room_confirmed_total(59, store)

    assert total == own + peer == 13
    assert total >= own, "a camera cannot own more chairs than its room has"
    # And it is symmetric: both cameras describe the same room.
    assert room_confirmed_total(60, store) == total


# --- two more ways a phantom got in, both caught in live data 2026-08-31 ----

def test_a_box_clipped_by_the_frame_edge_is_not_a_new_chair():
    """Both cameras admitted exactly one phantom this way, on the bottom edge.

    Camera 59's sat at y2=0.999 and camera 60's at y2=1.000. A box the frame
    cuts off is a partial view: its size and position are truncated, so neither
    the seat nor any occupancy judged against it can be trusted.
    """
    reg59 = ChairRegistry(59)
    before = reg59.confirmed_count
    for _ in range(PROMOTE_SWEEPS * 3):
        reg59.sweep([(0.702, 0.790, 0.871, 0.999)])
    assert reg59.confirmed_count == before
    assert reg59.pending_count == 0

    reg60 = ChairRegistry(60)
    before60 = reg60.confirmed_count
    for _ in range(PROMOTE_SWEEPS * 3):
        reg60.sweep([(0.076, 0.865, 0.212, 1.000)])
    assert reg60.confirmed_count == before60


def test_an_auto_chair_that_drifts_onto_a_real_one_is_dropped():
    """`_is_duplicate` only runs at CREATION, and both boxes then move.

    Camera 59 held an auto chair at IoU 0.235 with its own R8 -- well above the
    suppression bar it had passed earlier at a lower overlap, because EMA had
    since walked the two boxes together.
    """
    reg = ChairRegistry(59, seed=False)
    # Two clearly separate chairs, both admitted legitimately. Both are on the
    # RIGHT of the frame: camera 59's left-hand zones belong to camera 60, and a
    # detection there is refused by the ownership rule, not the drift rule.
    left, right = (0.45, 0.30, 0.55, 0.48), (0.75, 0.30, 0.85, 0.48)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([left, right])
    assert reg.confirmed_count == 2

    # The right one is now consistently detected on top of the left one.
    for _ in range(30):
        reg.sweep([left, (0.46, 0.30, 0.56, 0.48)])

    assert reg.confirmed_count == 1, "two zones on one chair survived"


def test_pruning_never_removes_the_chair_a_person_placed():
    """Between a seeded chair and an auto one, the human's survives."""
    reg = ChairRegistry(59)                       # seeded R2..R8
    seeded_ids = {c.chair_id for c in reg.chairs.values() if c.seeded}
    r8 = reg.chairs["R8"].box
    # An auto chair created away from R8, then walked onto it.
    away = (0.05, 0.05, 0.15, 0.20)
    for _ in range(PROMOTE_SWEEPS):
        reg.sweep([away])
    assert reg.confirmed_count == len(seeded_ids) + 1
    for _ in range(30):
        reg.sweep([r8])

    surviving = {c.chair_id for c in reg.chairs.values()}
    assert seeded_ids <= surviving, "a hand-placed chair was pruned"
    assert reg.confirmed_count == len(seeded_ids)
