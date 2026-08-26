"""If the room does not change, the answer must not change.

The reported failure: chair states flipping between FREE and OCCUPIED about a
second apart while nobody in the room moved.

THE CAUSE WAS NOT THE THRESHOLDS. It was that the thing being counted was wrong.
`occupancy_snapshot` runs once per HTTP poll -- every 2s from the dashboard, and
again for every extra viewer -- while V1 completes an analysis pass only every
4-17s. Each poll advanced the smoothing, so "change state only after N
consecutive observations" actually meant "after N READS OF ONE observation".
Measured: one pass was read about six times, so `confirm_free = 3` was satisfied
roughly six seconds after a single noisy pass rather than after three
independent ones. No threshold could have fixed that, because the thresholds
were being spent on duplicates.

So the rule these tests defend is: AN OBSERVATION IS AN ANALYSIS PASS. Reading
the result again is not evidence, and must change nothing at all.
"""
import pytest

from app.cctv_v2.config.geometry import ChairZone, RoomGeometry
from app.cctv_v2.pipeline import v1_bridge
from app.cctv_v2.pipeline.occupancy import OccupancyRegistry, RoomOccupancy
from app.cctv_v2.pipeline.track import PersonTrack, TrackState

CAMERA = 59
SEAT = ChairZone("T1", (0.40, 0.40, 0.60, 0.70))


class _Lock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _State:
    """V1 stamps this once per completed analysis pass, under the same lock that
    publishes the tracks -- which is what makes it an observation identity."""

    def __init__(self, updated_at=1000.0):
        self.updated_at = updated_at


class _V1Track:
    def __init__(self, track_id, box):
        self.track_id, self.box = track_id, box
        self.confidence, self.last_seen = 0.5, 1000.0


class _Worker:
    def __init__(self, tracks, updated_at=1000.0):
        self._latest_tracks = list(tracks)
        self._latest_frame = None
        self._frame_lock = _Lock()
        self.state = _State(updated_at)

    def new_pass(self, tracks):
        """What V1 does at the end of an analysis pass."""
        self._latest_tracks = list(tracks)
        self.state.updated_at += 1.0


def seated_box():
    x1, y1, x2, y2 = SEAT.box
    return (x1 * 960, y1 * 1080, x2 * 960, y2 * 1080)


@pytest.fixture
def bridge(monkeypatch):
    """A clean bridge with a fake V1 worker. Module state is per-process, so it
    has to be reset or one test's history leaks into the next."""
    worker = _Worker([_V1Track(1, seated_box())])
    monkeypatch.setattr(v1_bridge, "_v1_worker", lambda cid: worker)
    monkeypatch.setattr(v1_bridge, "_registry", OccupancyRegistry())
    monkeypatch.setattr(v1_bridge, "_last_observed", {})
    monkeypatch.setattr(v1_bridge, "_last_payload", {})
    return worker


def state_of(camera=CAMERA, chair="R8"):
    snap = v1_bridge.occupancy_snapshot(camera)
    return next(c for c in snap["chairs"] if c["id"] == chair)["state"]


def settle_into_seat(worker, chair_zone_box, passes=3):
    """Put a person in a real mapped seat and run enough passes to confirm.

    Each pass is FOLLOWED BY A READ, because the bridge folds an observation in
    when it is asked for one. That is a real property of the design and worth
    stating: a pass nobody ever reads is never counted. It is safe -- the state
    machine sees a subsample of real passes rather than duplicates of one -- but
    it does mean a very slow poller accumulates evidence more slowly than the
    camera produces it.
    """
    for _ in range(passes):
        worker.new_pass([_V1Track(1, chair_zone_box)])
        v1_bridge.occupancy_snapshot(CAMERA)


# ---------------------------------------------------------------------------
# A poll is not an observation
# ---------------------------------------------------------------------------
def test_polling_does_not_advance_the_state_machine(bridge):
    """The whole bug, in one test.

    Repeated reads of ONE analysis pass must leave the state exactly where the
    pass left it -- not creep toward a change because the dashboard is chatty.
    """
    from app.cctv_v2.config.geometry import room_geometry

    seat = room_geometry(CAMERA).chairs[-1]
    box = tuple(v * s for v, s in zip(seat.box, (960, 1080, 960, 1080)))
    settle_into_seat(bridge, box, passes=3)
    assert state_of(chair=seat.chair_id) == "occupied"

    # The person vanishes from V1's list, but V1 has NOT completed a new pass,
    # so this is the same observation being re-read.
    bridge._latest_tracks = []
    for _ in range(20):
        assert state_of(chair=seat.chair_id) == "occupied"


def test_a_re_read_is_reported_as_a_re_read(bridge):
    """A caller must be able to tell a fresh answer from a repeated one, or a
    stale reading looks exactly like a current one."""
    first = v1_bridge.occupancy_snapshot(CAMERA)
    second = v1_bridge.occupancy_snapshot(CAMERA)
    assert first["from_new_observation"] is True
    assert second["from_new_observation"] is False


def test_a_new_pass_is_reported_as_new(bridge):
    v1_bridge.occupancy_snapshot(CAMERA)
    bridge.new_pass(bridge._latest_tracks)
    assert v1_bridge.occupancy_snapshot(CAMERA)["from_new_observation"] is True


def test_the_number_of_viewers_cannot_change_the_answer(bridge):
    """Two dashboards open used to drive the state machine twice as fast, so
    occupancy depended on who was watching."""
    from app.cctv_v2.config.geometry import room_geometry

    seat = room_geometry(CAMERA).chairs[-1]
    box = tuple(v * s for v, s in zip(seat.box, (960, 1080, 960, 1080)))

    settle_into_seat(bridge, box, passes=3)

    # Two viewers: one polls once per pass, the other ten times per pass.
    for _ in range(RoomGeometry().confirm_free - 1):
        bridge.new_pass([])
        for _ in range(10):
            v1_bridge.occupancy_snapshot(CAMERA)
    # Still under confirm_free PASSES, however many polls happened.
    assert state_of(chair=seat.chair_id) == "occupied"


def test_a_real_departure_still_frees_the_seat(bridge):
    """The fix must not turn the display into a freeze. Enough genuine passes
    showing nobody must still free the chair."""
    from app.cctv_v2.config.geometry import room_geometry

    seat = room_geometry(CAMERA).chairs[-1]
    box = tuple(v * s for v, s in zip(seat.box, (960, 1080, 960, 1080)))
    settle_into_seat(bridge, box, passes=3)

    for _ in range(RoomGeometry().confirm_free + 1):
        bridge.new_pass([])
        v1_bridge.occupancy_snapshot(CAMERA)
    assert state_of(chair=seat.chair_id) == "free"


def test_the_answer_carries_how_old_its_observation_is(bridge):
    snap = v1_bridge.occupancy_snapshot(CAMERA)
    assert "observation_age_sec" in snap


# ---------------------------------------------------------------------------
# The smoothing itself, in observations
# ---------------------------------------------------------------------------
def _room(chairs=(SEAT,), **kw):
    from app.cctv_v2.pipeline.occupancy import ChairStatus

    r = RoomOccupancy(CAMERA, (960.0, 1080.0))
    r.geometry = RoomGeometry(chairs=chairs, **kw)
    r.chairs = {c.chair_id: ChairStatus(chair_id=c.chair_id) for c in chairs}
    return r


def _person(box=(0.42, 0.42, 0.58, 0.68)):
    x1, y1, x2, y2 = box
    return PersonTrack(
        camera_id=CAMERA, track_id=1,
        bbox=(x1 * 960, y1 * 1080, x2 * 960, y2 * 1080),
        confidence=0.05, frame_timestamp=0.0, frame_sequence=1,
        first_seen=0.0, last_seen=0.0, hits=5, state=TrackState.CONFIRMED,
    )


def test_one_missed_pass_does_not_free_a_seat():
    """The commonest real event on these cameras: YOLO drops a seated person for
    a pass. Measured runs of consecutive misses on a genuinely occupied chair
    were 1, 1, 1, 1, 3, 3."""
    r = _room()
    for i in range(4):
        r.update([_person()], float(i))
    assert r.update([], 5.0).chairs[0]["occupied"] is True


def test_a_seat_survives_the_longest_gap_that_was_actually_measured():
    """confirm_free was 3 and the longest genuine gap measured was ALSO 3, so a
    seated person's normal disappearance freed their chair. The threshold has to
    clear the observed maximum, not merely equal it."""
    r = _room()
    for i in range(4):
        r.update([_person()], float(i))
    for i in range(3):                       # the worst gap seen in 6 minutes
        snap = r.update([], 10.0 + i)
    assert snap.chairs[0]["occupied"] is True


def test_one_stray_pass_does_not_fill_a_seat():
    """A bloated box or somebody walking past clips a seat for one pass. Two
    such passes were observed in six minutes, on chairs nobody sat in."""
    r = _room()
    for i in range(6):
        r.update([], float(i))
    assert r.update([_person()], 10.0).chairs[0]["occupied"] is False


def test_freeing_is_slower_than_filling():
    """Asymmetric on purpose: a person briefly invisible is far more likely than
    a person who teleported into a chair."""
    g = RoomGeometry()
    assert g.confirm_free > g.confirm_occupied


# ---------------------------------------------------------------------------
# UNDECIDED IS NOT FREE, AND MUST NOT LINGER
# ---------------------------------------------------------------------------
# Reported from the live feed: the panel read "Chairs: 6  Occupied: 2  Free: 0"
# while four seats on the picture were labelled FREE. Both halves were wrong in
# their own way -- the four seats were UNKNOWN, so the counter was right not to
# call them free, and the overlay was wrong to label them free.
#
# The cause was that UNKNOWN -> FREE waited for `confirm_free`. That threshold
# protects an ESTABLISHED occupied seat from flickering when its occupant is
# briefly undetected. From UNKNOWN there is nothing to protect: nobody has ever
# been seen in the chair and the detector has just looked and found nobody.
def test_an_untouched_seat_reads_free_on_the_first_observation():
    r = _room()
    snap = r.update([], 0.0)
    assert snap.chairs[0]["state"] == "free"
    assert snap.free_chairs == 1
    assert snap.unknown_chairs == 0


def test_the_totals_always_account_for_every_chair():
    """occupied + free + unknown == total, at EVERY step. A seat counted as
    none of the three is what made the panel look broken."""
    seats = (ChairZone("A", (0.10, 0.40, 0.25, 0.70)),
             ChairZone("B", (0.40, 0.40, 0.60, 0.70)),
             ChairZone("C", (0.70, 0.40, 0.85, 0.70)))
    r = _room(chairs=seats)
    for i in range(8):
        snap = r.update([_person()] if i % 3 else [], float(i))
        assert (snap.occupied_chairs + snap.free_chairs
                + snap.unknown_chairs) == snap.total_chairs == 3


def test_a_room_with_people_reports_the_other_seats_free_immediately():
    """The exact reported scene: two of six seats taken, four empty. The four
    must read FREE, not sit undecided while the panel says 'Free: 0'."""
    seats = tuple(ChairZone(f"S{i}", (0.10 + i * 0.13, 0.40, 0.20 + i * 0.13, 0.70))
                  for i in range(6))
    r = _room(chairs=seats)
    sitter = _person(box=(0.10, 0.42, 0.20, 0.68))
    snap = r.update([sitter], 0.0)

    # Five empty seats: free at once, nothing to protect.
    assert snap.free_chairs == 5, "empty seats must be free from the first look"
    # The claimed seat is the ONE still undecided, because filling a seat needs
    # confirmation. That asymmetry is the point, and the totals still add up.
    assert snap.occupied_chairs == 0
    assert snap.unknown_chairs == 1
    assert snap.free_chairs + snap.unknown_chairs == snap.total_chairs

    snap = r.update([sitter], 1.0)
    assert (snap.occupied_chairs, snap.free_chairs, snap.unknown_chairs) == (1, 5, 0)


def test_a_stray_claim_still_cannot_fill_a_seat_from_unknown():
    """Only the FREE direction is fast. Filling a seat still needs
    confirmation, because single-pass claims were observed on chairs nobody
    was sitting in."""
    r = _room()
    snap = r.update([_person()], 0.0)
    assert snap.occupied_chairs == 0
    assert snap.chairs[0]["state"] == "unknown"


def test_an_occupied_seat_still_takes_the_full_wait_to_free():
    """The protection that matters is untouched: an ESTABLISHED occupied seat
    still needs confirm_free consecutive empty observations."""
    r = _room()
    for i in range(3):
        r.update([_person()], float(i))
    assert r.chairs["T1"].state.value == "occupied"

    for i in range(RoomGeometry().confirm_free - 1):
        snap = r.update([], 10.0 + i)
    assert snap.chairs[0]["occupied"] is True, "freed too early"
    snap = r.update([], 100.0)
    assert snap.chairs[0]["state"] == "free"
