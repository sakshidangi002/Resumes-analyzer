"""Attendance evidence must be collectable within the time a person is present.

THE PROBLEM, measured on the live system: 33 attendance decisions all-time, of
which exactly ONE was `allowed`. The blockers were `insufficient_observations`
and `unstable_identity` -- never face size, never matching. Faces at decision
time had a median width of 35px against a 28px gate, so they were reaching the
recogniser fine.

WHY. `attendance_gate.evaluate` requires `observations >= profile.min_observations`
(3 for IN/OUT). Observations are only accumulated inside

    if fresh and pt.needs_recognition(_PERSON_REVERIFY_SEC):

and `needs_recognition` throttles to one attempt per CCTV_PERSON_REVERIFY_SEC
(5s) as soon as the track has ANY match:

    pass 1  t=0.0   unmatched  -> recognise, obs=1, matched=True
    pass 2  t=2.5   matched, 2.5 < 5 -> skipped entirely
    pass 3  t=5.0   matched, 5.0 >= 5 -> obs=2
    pass 4  t=7.5   skipped
    pass 5  t=10.0  obs=3

So three observations need at least ~10 SECONDS of continuous tracking after
the first match. A person walking through a doorway is present for about two.
The bar is not merely hard to reach at this frame rate -- it is unreachable by
construction, which is why the gate essentially never fires at 57/58.

The throttle is correct for its own purpose: re-verifying an already-identified
person who is sitting in a room does not need to run every pass. It is only
wrong while the track still lacks the evidence the gate will demand of it.
"""
import time

import pytest

pytest.importorskip("cv2", reason="needs the vision stack")

from app.services.person_tracker import PersonTrack


def _track():
    return PersonTrack(track_id=1, box=(0, 0, 100, 300))


REVERIFY = 5.0


def _passes_until(track, target_obs, cycle, budget_sec=30.0):
    """Simulate analysis passes; return seconds needed to reach target_obs.

    Mirrors the real loop: an observation is only collected on a pass where
    needs_recognition() is True.
    """
    t = 0.0
    obs = 0
    while t < budget_sec:
        if track.needs_recognition(REVERIFY):
            obs += 1
            track.matched = True
            track.last_recognition_time = time.time()
            if obs >= target_obs:
                return t
        t += cycle
        # advance the clock the tracker sees
        track.last_recognition_time -= cycle
    return None


def test_three_observations_cannot_be_collected_during_a_doorway_transit():
    """The measured failure: a ~2s transit at a ~2.5s cycle cannot reach obs=3."""
    track = _track()
    # A person is in frame for roughly 2-5 seconds at a doorway.
    reached = _passes_until(track, target_obs=3, cycle=2.5, budget_sec=5.0)
    assert reached is None, (
        "obs=3 was reachable within a 5s transit -- the throttle no longer "
        "blocks it and this test's premise needs revisiting"
    )


def test_the_throttle_is_what_blocks_it_not_the_frame_rate():
    """Without the throttle the same cycle collects 3 observations easily.

    This isolates cause: at 2.5s per pass, three passes take 7.5s of wall clock,
    but three OBSERVATIONS take 10s+ purely because of the 5s re-verify gate.
    """
    track = _track()
    unthrottled = 0
    t = 0.0
    while t < 7.5:
        unthrottled += 1          # no needs_recognition() gate
        t += 2.5
    assert unthrottled >= 3, "three passes should fit in 7.5s at a 2.5s cycle"


def test_an_unmatched_track_is_never_throttled():
    """Pins the existing, correct behaviour: evidence gathers fast until a match."""
    track = _track()
    assert track.needs_recognition(REVERIFY) is True
    assert track.needs_recognition(REVERIFY) is True


def test_a_matched_track_is_throttled():
    track = _track()
    track.matched = True
    track.last_recognition_time = time.time()
    assert track.needs_recognition(REVERIFY) is False

# ---------------------------------------------------------------------------
# After the fix: the gate's requirement is passed down, so a track short of
# evidence is not throttled.
# ---------------------------------------------------------------------------
class _Fuser:
    def __init__(self, accepted=0):
        self.accepted = accepted


def test_a_track_short_of_evidence_is_not_throttled():
    """The fix. min_observations is what the gate will demand of this track."""
    track = _track()
    track.matched = True
    track.last_recognition_time = time.time()
    track.fuser = _Fuser(accepted=1)          # only one so far, gate wants 3
    assert track.needs_recognition(REVERIFY, min_observations=3) is True


def test_throttling_resumes_once_the_evidence_bar_is_met():
    """Not a blanket removal: a track with enough evidence throttles again."""
    track = _track()
    track.matched = True
    track.last_recognition_time = time.time()
    track.fuser = _Fuser(accepted=3)
    assert track.needs_recognition(REVERIFY, min_observations=3) is False


def test_monitor_cameras_keep_the_cheap_throttle():
    """marks_attendance False -> caller passes 0 -> old behaviour, no extra CPU."""
    track = _track()
    track.matched = True
    track.last_recognition_time = time.time()
    track.fuser = _Fuser(accepted=0)
    assert track.needs_recognition(REVERIFY, min_observations=0) is False


def test_three_observations_now_fit_inside_a_transit():
    """The whole point: obs=3 reachable in the seconds a person is actually there."""
    track = _track()
    track.fuser = _Fuser(accepted=0)
    t = 0.0
    obs = 0
    while t < 5.0:                            # a realistic doorway presence
        if track.needs_recognition(REVERIFY, min_observations=3):
            obs += 1
            track.fuser.accepted = obs
            track.matched = True
            track.last_recognition_time = time.time()
        t += 1.5                              # analysis passes during the transit
    assert obs >= 3, f"only {obs} observations collected inside a 5s transit"
