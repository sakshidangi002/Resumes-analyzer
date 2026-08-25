"""The scheduler must CHOOSE, and no camera may be starved.

WHAT THIS REPLACES. V1 ran four recognition threads that slept for their
profile's interval and then queued on a `BoundedSemaphore(1)`. Nothing decided
anything -- the winner was whichever thread the OS woke, and Python's semaphore
promises no fairness. Measured on the real system, that produced starvation
rather than scheduling:

    one camera -> 17 inference passes
    the other three -> 1 pass each

and a doorway asking for 0.12s actually received a pass every 4.5-9s.

Raising the semaphore was measured and does not help (1 slot 0.58/s, 2 slots
0.56/s): a single YOLO pass already saturates four physical cores. The fix is
not more slots, it is an explicit choice.

THE RULE UNDER TEST: score = role_priority x staleness, with a hard deadline
(`max_starvation_sec`) that overrides the score for any camera left too long.

Every test drives an injected clock. Real time would make fairness assertions
flaky, and a scheduler whose decisions cannot be reproduced cannot be trusted
when its behaviour is later questioned.
"""
import pytest

from app.cctv_v2.capture.grabber import CameraGrabber
from app.cctv_v2.scheduler.loop import InferenceScheduler


class Clock:
    """Manual clock. Nothing here depends on wall time."""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


DOORWAYS = (57, 58)
ROOMS = (59, 60)
ALL = DOORWAYS + ROOMS


def build(clock, cameras=ALL, max_starvation=20.0, inference_cost=0.0):
    """A scheduler over grabbers with no real cameras behind them."""
    grabbers = {cid: CameraGrabber(cid, f"test://{cid}") for cid in cameras}
    processed = []

    def process(camera_id, snap):
        processed.append((camera_id, snap.sequence))
        clock.advance(inference_cost)

    sched = InferenceScheduler(
        grabbers.values(), process=process,
        max_starvation_sec=max_starvation, clock=clock,
    )
    return sched, grabbers, processed


def publish_all(grabbers, clock, cameras=None):
    for cid in (cameras or grabbers):
        grabbers[cid].publish(f"f{cid}", timestamp=clock.t)


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------
def test_doorway_wins_when_all_cameras_are_equally_stale():
    """Equal staleness -> priority decides. A doorway transit lasts ~2s; a
    seated person in a room is still there next pass."""
    clock = Clock()
    sched, grabbers, _ = build(clock)
    publish_all(grabbers, clock)
    clock.advance(1.0)

    assert sched.select_next().camera_id in DOORWAYS


def test_a_long_waiting_room_camera_beats_a_recently_served_doorway():
    """Priority must WEIGHT the decision, not dominate it.

    NOTE ON THE SPEC. The brief's worked example expressed this in terms of
    FRAME age ("doorway stale 0.1s vs room stale 1.5s -> room wins"). That
    cannot be implemented as written: all four cameras run at 12fps, so every
    camera's newest frame is ~0.08s old at every instant and frame age never
    separates anybody. Measured, scoring on it gave camera 57 134 selections and
    camera 58 22 - two cameras of identical priority.

    The intent behind the example is fairness: a camera nobody has looked at for
    a while should get a turn. That is time since SERVICE, which is what is
    scored. It also reads the right way round on information value - a 1.5s-old
    frame describes the past, so it is less useful to process, not more urgent.

    Here a room camera has waited 4s while the doorways were served 1s ago:
    1 x 4.0 = 4.0 beats 3 x 1.0 = 3.0.
    """
    clock = Clock()
    sched, grabbers, _ = build(clock)
    publish_all(grabbers, clock)

    # Doorways were served recently; the rooms have been waiting.
    clock.advance(4.0)
    for cid in DOORWAYS:
        sched.stats[cid].last_served = clock.t - 1.0
    for cid in ROOMS:
        sched.stats[cid].last_served = clock.t - 4.0
    publish_all(grabbers, clock)

    assert sched.select_next().camera_id in ROOMS


def test_a_doorway_that_has_waited_longer_still_wins():
    """The second worked example, in service terms: doorway 5s vs room 1s."""
    clock = Clock()
    sched, grabbers, _ = build(clock)
    publish_all(grabbers, clock)
    clock.advance(5.0)
    for cid in DOORWAYS:
        sched.stats[cid].last_served = clock.t - 5.0
    for cid in ROOMS:
        sched.stats[cid].last_served = clock.t - 1.0
    publish_all(grabbers, clock)

    assert sched.select_next().camera_id in DOORWAYS


def test_a_stale_doorway_beats_a_fresher_room():
    """The second worked example: doorway 5s vs room 1s -> doorway."""
    clock = Clock()
    sched, grabbers, _ = build(clock)
    publish_all(grabbers, clock, DOORWAYS)
    clock.advance(4.0)
    publish_all(grabbers, clock, ROOMS)
    clock.advance(1.0)

    assert sched.select_next().camera_id in DOORWAYS


# ---------------------------------------------------------------------------
# Starvation
# ---------------------------------------------------------------------------
def test_no_camera_can_monopolise_the_worker():
    """The V1 failure, asserted directly: 17/1/1/1 must not recur."""
    clock = Clock()
    sched, grabbers, _ = build(clock, inference_cost=0.5)

    for _ in range(60):
        publish_all(grabbers, clock)     # every camera always has a fresh frame
        sched.run_once()
        clock.advance(0.1)

    counts = {cid: s.selections for cid, s in sched.stats.items()}
    assert all(n > 0 for n in counts.values()), f"a camera was never served: {counts}"
    assert max(counts.values()) <= 4 * min(counts.values()), (
        f"service was not remotely fair: {counts}"
    )


def test_the_starvation_deadline_overrides_the_score():
    """A room camera held below the doorways still gets served, by deadline."""
    clock = Clock()
    sched, grabbers, _ = build(clock, max_starvation=5.0)
    publish_all(grabbers, clock)

    served = set()
    for _ in range(40):
        # Doorways are refreshed constantly, rooms are not touched again.
        grabbers[57].publish("fresh", timestamp=clock.t)
        grabbers[58].publish("fresh", timestamp=clock.t)
        sel = sched.run_once()
        if sel:
            served.add(sel.camera_id)
        clock.advance(1.0)

    assert served >= set(ROOMS), f"room cameras starved: {served}"
    assert any(s.starvation_selections > 0 for cid, s in sched.stats.items() if cid in ROOMS)


def test_every_camera_is_eventually_served():
    clock = Clock()
    sched, grabbers, _ = build(clock)
    for _ in range(40):
        publish_all(grabbers, clock)
        sched.run_once()
        clock.advance(0.3)
    assert all(sched.stats[cid].selections > 0 for cid in ALL)


# ---------------------------------------------------------------------------
# Latest-frame semantics
# ---------------------------------------------------------------------------
def test_the_newest_frame_is_processed_not_the_oldest():
    """Frames 100-103 arrive while the worker is busy; it must take 103."""
    clock = Clock()
    sched, grabbers, processed = build(clock, cameras=(57,))
    for i in range(4):
        grabbers[57].publish(f"frame-{i}", timestamp=clock.t)
    clock.advance(1.0)

    sched.run_once()
    assert processed == [(57, 4)]          # the 4th publish, not the 1st


def test_a_backlog_is_never_worked_through():
    """After a long stall, the scheduler resumes at the present, not the past."""
    clock = Clock()
    sched, grabbers, processed = build(clock, cameras=(57,))
    for i in range(50):
        grabbers[57].publish(f"f{i}", timestamp=clock.t)
    clock.advance(9.0)                     # a nine-second inference stall

    sched.run_once()
    sched.run_once()
    assert len(processed) == 2
    assert all(seq == 50 for _, seq in processed), processed


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------
def test_a_camera_with_no_frames_does_not_block_the_others():
    """One dead camera must not stop the other three."""
    clock = Clock()
    sched, grabbers, _ = build(clock)
    publish_all(grabbers, clock, (57, 59, 60))     # 58 never produces anything
    clock.advance(1.0)

    for _ in range(12):
        sched.run_once()
        clock.advance(0.5)

    assert sched.stats[58].selections == 0
    assert all(sched.stats[cid].selections > 0 for cid in (57, 59, 60))


def test_a_camera_that_starts_late_is_picked_up():
    """Reconnect: a camera that begins producing mid-run joins normally."""
    clock = Clock()
    sched, grabbers, _ = build(clock)
    publish_all(grabbers, clock, (57, 59, 60))
    for _ in range(6):
        sched.run_once()
        clock.advance(0.5)
    assert sched.stats[58].selections == 0

    grabbers[58].publish("late", timestamp=clock.t)
    clock.advance(3.0)
    for _ in range(6):
        sched.run_once()
        clock.advance(0.5)

    assert sched.stats[58].selections > 0


def test_a_failing_process_callback_does_not_stop_the_scheduler():
    """One camera's processing error must not take the worker down."""
    clock = Clock()
    grabbers = {cid: CameraGrabber(cid, "t") for cid in ALL}
    calls = []

    def process(camera_id, snap):
        calls.append(camera_id)
        if camera_id == 57:
            raise RuntimeError("detector blew up")

    sched = InferenceScheduler(grabbers.values(), process=process, clock=clock)
    for cid in ALL:
        grabbers[cid].publish("f", timestamp=clock.t)
    clock.advance(1.0)

    for _ in range(20):
        sched.run_once()
        publish_all(grabbers, clock)
        clock.advance(0.4)

    assert 57 in calls                       # it kept being scheduled
    assert set(calls) == set(ALL)            # and everyone else still ran


def test_an_idle_system_selects_nothing():
    clock = Clock()
    sched, _, _ = build(clock)
    assert sched.select_next() is None
    assert sched.run_once() is None


# ---------------------------------------------------------------------------
# Determinism + the synthetic fairness simulation
# ---------------------------------------------------------------------------
def test_selection_is_deterministic():
    """Identical inputs must give an identical choice, or the scheduler cannot
    be tested and cannot be explained after the fact."""
    picks = set()
    for _ in range(5):
        clock = Clock()
        sched, grabbers, _ = build(clock)
        publish_all(grabbers, clock)
        clock.advance(1.0)
        picks.add(sched.select_next().camera_id)
    assert len(picks) == 1


def test_synthetic_fairness_simulation():
    """200 selections at a realistic 2.5s inference cost, 12fps cameras.

    Reports service counts; the V1 shape (17/1/1/1) must not reappear.
    """
    clock = Clock()
    sched, grabbers, _ = build(clock, inference_cost=2.5)

    for _ in range(200):
        publish_all(grabbers, clock)       # every camera always has a fresh frame
        sched.run_once()
        clock.advance(0.083)               # ~12fps between passes

    counts = {cid: sched.stats[cid].selections for cid in ALL}
    total = sum(counts.values())
    assert total == 200
    assert all(n > 0 for n in counts.values()), counts

    doorway = sum(counts[c] for c in DOORWAYS)
    room = sum(counts[c] for c in ROOMS)
    assert doorway > room, f"doorways should out-serve rooms: {counts}"
    # ...but rooms must get a real share, not a token one.
    assert room >= total * 0.15, f"rooms under-served: {counts}"
    # The two cameras within a role are treated alike.
    assert abs(counts[57] - counts[58]) <= max(3, 0.35 * max(counts[57], counts[58]))
    assert abs(counts[59] - counts[60]) <= max(3, 0.35 * max(counts[59], counts[60]))


def test_summary_reports_actual_not_requested_cadence():
    """The point of the rebuild: V1 slept for 0.12s then queued on a lock, which
    made the target look satisfied while the real cadence was 4.5-9s."""
    clock = Clock()
    sched, grabbers, _ = build(clock, inference_cost=2.5)
    sched._started_at = clock.t
    for _ in range(20):
        publish_all(grabbers, clock)
        sched.run_once()
        clock.advance(0.1)

    s = sched.summary()
    d = s["cameras"][57]
    assert d["requested_interval"] == 0.12
    assert d["actual_interval"] > d["requested_interval"], (
        "actual cadence must be reported honestly, not as the target"
    )
    assert s["total_selections"] == 20
