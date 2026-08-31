"""Doorway IN/OUT counting. A person is counted whether or not anyone knows who.

THE RULE THIS FILE GUARDS. There is no identity in the crossing layer, and these
tests assert its ABSENCE as hard as they assert the counting. In V1 recognition
and counting were entangled, so a face that could not be matched became a person
who was never counted, and the door tally quietly under-reported everyone who
walked through facing away.

Every event this layer produces carries `identity=None`. That is the guarantee,
not a gap to be filled in later.

WHAT A CROSSING HAS TO SATISFY. Being on one side and then the other is not
enough on its own -- each extra condition here exists because of a specific way
the naive version misbehaves:

    unconfirmed track   a one-hit track is a detection, not a person
    short travel        a person standing ON the line jitters across it
    cooldown            a hesitation in the doorway is not four transits

Every test drives explicit timestamps and hand-built tracks. Nothing depends on
wall-clock timing or on a real camera.
"""
import pytest

from app.cctv_v2.config.geometry import CrossingLine, crossing_line
from app.cctv_v2.pipeline.crossing import (
    CrossingDetector,
    CrossingRegistry,
    Direction,
)
from app.cctv_v2.pipeline.track import PersonTrack, TrackState

W, H = 960.0, 1080.0
ENTRANCE, EXIT = 57, 58


# A FIXED line for the behaviour tests below.
#
# Those tests describe how crossing WORKS -- direction, jitter, cooldown,
# duplicates -- and must not move when a camera is re-aimed. They therefore use
# this synthetic line rather than the live config. The tests that deliberately
# check the real configuration are grouped at the end of the file and say so.
TEST_LINE = CrossingLine(position=0.35, orientation="horizontal",
                         inside_side="below")


def detector(camera_id=ENTRANCE):
    d = CrossingDetector(camera_id, (W, H))
    object.__setattr__(d, "line", TEST_LINE)
    return d


def track(camera_id=ENTRANCE, track_id=1, y_norm=0.2, x_norm=0.5,
          state=TrackState.CONFIRMED, hits=4, conf=0.8):
    """A track whose FOOT point sits at the given normalised height."""
    foot_y = y_norm * H
    cx = x_norm * W
    return PersonTrack(
        camera_id=camera_id, track_id=track_id,
        bbox=(cx - 45, foot_y - 400, cx + 45, foot_y),
        confidence=conf, frame_timestamp=0.0, frame_sequence=1,
        first_seen=0.0, last_seen=0.0, hits=hits, state=state,
    )


def walk(detector, ys, camera_id=ENTRANCE, track_id=1, start_t=0.0, step=1.0,
         state=TrackState.CONFIRMED, hits=4):
    """Walk one track through a sequence of normalised heights."""
    events = []
    for i, y in enumerate(ys):
        t = start_t + i * step
        events += detector.update(
            [track(camera_id, track_id, y_norm=y, state=state, hits=hits)],
            timestamp=t, frame_sequence=i + 1,
        )
    return events


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
def test_crossing_toward_the_inside_is_ONE_in_event():
    d = detector(ENTRANCE)
    events = walk(d, [0.10, 0.20, 0.50, 0.70])      # above -> below, inside=below

    assert len(events) == 1
    assert events[0].direction is Direction.IN
    assert d.count_in == 1 and d.count_out == 0


def test_crossing_toward_the_outside_is_ONE_out_event():
    d = detector(ENTRANCE)
    events = walk(d, [0.70, 0.50, 0.20, 0.10])

    assert len(events) == 1
    assert events[0].direction is Direction.OUT
    assert d.count_out == 1 and d.count_in == 0


def test_one_physical_crossing_produces_exactly_one_event():
    """Continuing to walk after crossing must not keep emitting."""
    d = detector(ENTRANCE)
    events = walk(d, [0.05, 0.15, 0.45, 0.60, 0.75, 0.90, 0.95])

    assert len(events) == 1, f"one walk produced {len(events)} events"


def test_the_event_carries_what_a_consumer_needs():
    d = detector(ENTRANCE)
    ev = walk(d, [0.10, 0.70])[0]

    assert ev.camera_id == ENTRANCE
    assert ev.role == "doorway"
    assert ev.track_id == 1
    assert ev.timestamp == 1.0
    assert ev.frame_sequence == 2
    assert ev.confidence == pytest.approx(0.8)
    assert ev.track_hits == 4
    assert ev.travel > 0
    assert len(ev.bbox) == 4


def test_event_ids_are_unique():
    d = detector(ENTRANCE)
    walk(d, [0.10, 0.70], track_id=1)
    walk(d, [0.10, 0.70], track_id=2, start_t=100.0)

    ids = [e.event_id for e in d.events]
    assert len(ids) == len(set(ids)) == 2


# ---------------------------------------------------------------------------
# Identity independence -- the most important rule here
# ---------------------------------------------------------------------------
def test_every_event_is_unattributed_by_construction():
    d = detector(ENTRANCE)
    ev = walk(d, [0.10, 0.70])[0]
    assert ev.identity is None


def test_an_unknown_person_is_counted_exactly_like_a_known_one():
    """There is no 'known' at this layer. Nothing about a track can make it
    countable or not -- only its geometry."""
    d = detector(ENTRANCE)
    walk(d, [0.10, 0.70], track_id=1)
    walk(d, [0.10, 0.70], track_id=2, start_t=50.0)

    assert d.count_in == 2
    assert all(e.identity is None for e in d.events)


def test_the_crossing_layer_holds_no_recognition_machinery():
    """A guard against a later step wiring identity in here rather than into the
    stage above it, which is how counting and recognition became entangled the
    last time."""
    import app.cctv_v2.pipeline.crossing as mod

    source = open(mod.__file__, encoding="utf-8").read().lower()
    for forbidden in ("adaface", "scrfd", "insightface", "embedding",
                      "gallery", "face_service", "recognit"):
        assert f"import {forbidden}" not in source
    assert not hasattr(mod, "recognise")
    assert not hasattr(mod, "identify")


def test_the_summary_reports_everything_as_unknown():
    d = detector(ENTRANCE)
    walk(d, [0.10, 0.70], track_id=1)
    walk(d, [0.70, 0.10], track_id=2, start_t=50.0)

    s = d.summary()
    assert s["people_in"] == s["unknown_in"] == 1
    assert s["people_out"] == s["unknown_out"] == 1


# ---------------------------------------------------------------------------
# What must NOT be counted
# ---------------------------------------------------------------------------
def test_a_stationary_person_never_crosses():
    d = detector(ENTRANCE)
    assert walk(d, [0.70] * 8) == []
    assert d.count_in == 0 and d.count_out == 0


def test_moving_on_one_side_of_the_line_is_not_a_crossing():
    d = detector(ENTRANCE)
    assert walk(d, [0.50, 0.60, 0.75, 0.90, 0.60, 0.50]) == []


def test_a_person_jittering_on_the_line_emits_nothing():
    """The failure this guards: a person standing on the line has a box that
    crosses it every pass, and without the travel rule that is an endless
    stream of alternating IN and OUT."""
    d = detector(ENTRANCE)
    events = walk(d, [0.34, 0.36, 0.34, 0.36, 0.34, 0.36, 0.34])

    assert events == []
    assert d.rejected_short_travel > 0


def test_an_unconfirmed_track_cannot_create_a_transit():
    """A one-hit track is a detection, not a person with a history."""
    d = detector(ENTRANCE)
    events = walk(d, [0.10, 0.70], state=TrackState.TENTATIVE, hits=1)

    assert events == []
    assert d.rejected_unconfirmed == 1


def test_a_track_that_simply_disappears_creates_nothing():
    d = detector(ENTRANCE)
    walk(d, [0.10, 0.15, 0.20])          # never reaches the line
    d.update([], timestamp=99.0)         # gone
    assert d.events == []


def test_a_hesitation_in_the_doorway_is_not_four_transits():
    """In and straight back out within the cooldown. Real, but at multi-second
    sampling indistinguishable from box noise -- so one event, not four."""
    d = detector(ENTRANCE)
    events = walk(d, [0.10, 0.70, 0.10, 0.70], step=0.5)

    assert len(events) == 1
    assert d.rejected_cooldown >= 1


def test_a_genuine_return_after_the_cooldown_IS_counted():
    d = detector(ENTRANCE)
    events = walk(d, [0.10, 0.70, 0.10], step=10.0)

    assert len(events) == 2
    assert events[0].direction is Direction.IN
    assert events[1].direction is Direction.OUT


# ---------------------------------------------------------------------------
# Several people, several cameras
# ---------------------------------------------------------------------------
def test_two_tracks_produce_two_people():
    d = detector(ENTRANCE)
    d.update([track(track_id=1, y_norm=0.10), track(track_id=2, y_norm=0.12)], 0.0)
    events = d.update(
        [track(track_id=1, y_norm=0.70), track(track_id=2, y_norm=0.72)], 1.0)

    assert len(events) == 2
    assert {e.track_id for e in events} == {1, 2}
    assert d.count_in == 2


def test_a_detector_refuses_another_camera_s_tracks():
    d = detector(ENTRANCE)
    with pytest.raises(ValueError, match="per-camera"):
        d.update([track(camera_id=EXIT)], timestamp=0.0)


def test_camera_57_crossing_state_never_reaches_camera_58():
    reg = CrossingRegistry((W, H))
    for c in (ENTRANCE, EXIT):
        object.__setattr__(reg.get(c), "line", TEST_LINE)
    reg.update(ENTRANCE, [track(ENTRANCE, 1, 0.10)], 0.0)
    reg.update(ENTRANCE, [track(ENTRANCE, 1, 0.70)], 1.0)

    assert reg.get(ENTRANCE).count_in == 1
    assert reg.get(EXIT).count_in == 0
    assert reg.get(EXIT).events == []


def test_the_same_track_id_on_two_cameras_is_two_people():
    """Track ids are unique per camera, never globally. Anything treating them
    as an identity across cameras is wrong."""
    reg = CrossingRegistry((W, H))
    for cam in (ENTRANCE, EXIT):
        object.__setattr__(reg.get(cam), "line", TEST_LINE)
    for cam in (ENTRANCE, EXIT):
        reg.update(cam, [track(cam, 1, 0.10)], 0.0)
        reg.update(cam, [track(cam, 1, 0.70)], 1.0)

    assert reg.get(ENTRANCE).count_in == 1
    assert reg.get(EXIT).count_in == 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_the_line_comes_from_config_not_from_the_algorithm():
    """A coordinate baked into the rule cannot be re-aimed without a code
    change, and these cameras get re-aimed."""
    import app.cctv_v2.pipeline.crossing as mod

    source = open(mod.__file__, encoding="utf-8").read()
    for coord in ("0.35", "0.775"):
        assert coord not in source, (
            f"crossing coordinate {coord} leaked into the algorithm")
    assert crossing_line(ENTRANCE) is not None
    assert crossing_line(EXIT) is not None


def test_a_room_camera_has_no_line_and_emits_nothing():
    """Room cameras are not a doorway mechanism. An unconfigured camera must not
    quietly acquire a line at mid-frame and start emitting transits."""
    assert crossing_line(59) is None
    d = CrossingDetector(59, (W, H))       # real config: no line at all
    assert d.enabled is False
    assert d.update([track(59, 1, 0.10)], 0.0) == []
    assert d.update([track(59, 1, 0.90)], 1.0) == []


def test_which_side_is_inside_is_configurable():
    """The same downward motion means enter on one camera and leave on another,
    so direction cannot be hard-coded to an axis."""
    d = detector(ENTRANCE)
    object.__setattr__(d, "line",
                       CrossingLine(position=0.35, inside_side="above"))
    events = walk(d, [0.10, 0.70])

    assert events[0].direction is Direction.OUT, "inside_side was ignored"


def test_the_foot_point_is_used_not_the_box_centre():
    """A seated or occluded person's box TOP moves as YOLO includes or excludes
    their head; their feet stay where they are standing. A centroid on an
    unstable box crosses lines the person never crossed."""
    d = detector(ENTRANCE)
    assert d.line.reference == "foot"

    # Feet below the line, centroid above it. Foot semantics say "inside".
    tall = PersonTrack(
        camera_id=ENTRANCE, track_id=1,
        bbox=(400, 0.05 * H, 500, 0.60 * H),      # centroid ~0.32, foot 0.60
        confidence=0.9, frame_timestamp=0.0, frame_sequence=1,
        first_seen=0.0, last_seen=0.0, hits=4, state=TrackState.CONFIRMED,
    )
    d.update([track(track_id=1, y_norm=0.10)], 0.0)
    events = d.update([tall], 1.0)
    assert len(events) == 1 and events[0].direction is Direction.IN


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
def test_forgetting_expired_tracks_does_not_retract_their_events():
    """A crossing that happened, happened. Forgetting the track only frees the
    state that would otherwise grow for the life of the process."""
    d = detector(ENTRANCE)
    walk(d, [0.10, 0.70])
    assert d.count_in == 1

    d.forget([1])
    assert d.summary()["tracked_states"] == 0
    assert d.count_in == 1
    assert len(d.events) == 1


# ---------------------------------------------------------------------------
# The line must sit where feet actually are
#
# Camera 57 ran for 50 minutes producing 24 tracks and ZERO side changes. The
# line was at 0.35, inherited from V1 -- but V1 measured a different point on
# the body, and V2 uses the FOOT point. Measured over 148 real boxes:
#
#     camera 57 foot range  0.624 .. 0.998   median 0.828
#     camera 58 foot range  0.165 .. 0.998   median 0.304
#
# 0.35 is OUTSIDE camera 57's range: no foot ever reached it, so no crossing was
# possible and the failure was silent -- a camera reporting zero people looks
# exactly like a quiet corridor. On camera 58 the same 0.35 sits near the
# median, which is why that camera produced transits throughout.
#
# One number, correct on one camera and impossible on the other. These tests
# encode the invariant that would have caught it on day one.
# ---------------------------------------------------------------------------
MEASURED_FOOT_RANGE = {
    # camera: (min, max, median) over unclipped boxes from real footage
    57: (0.624, 0.998, 0.828),
    58: (0.165, 0.998, 0.304),
}


@pytest.mark.parametrize("camera_id", [57, 58])
def test_the_configured_line_lies_within_the_measured_foot_range(camera_id):
    """The invariant that was missing. A line outside the range of foot points
    a camera actually produces can never be crossed by anybody."""
    line = crossing_line(camera_id)
    lo, hi, _ = MEASURED_FOOT_RANGE[camera_id]
    assert line is not None
    assert lo <= line.position <= hi, (
        f"camera {camera_id}: line at {line.position} is outside the measured "
        f"foot range {lo}..{hi} -- no foot point can ever reach it"
    )


@pytest.mark.parametrize("camera_id", [57, 58])
def test_the_line_has_margin_on_both_sides(camera_id):
    """Not merely inside the range, but with room to approach from each side.
    A line pinned against the edge of the range is technically crossable and
    practically never crossed."""
    line = crossing_line(camera_id)
    lo, hi, _ = MEASURED_FOOT_RANGE[camera_id]
    span = hi - lo
    assert line.position - lo >= 0.10 * span, "line hugs the near edge"
    assert hi - line.position >= 0.10 * span, "line hugs the far edge"


def test_camera_57_old_line_could_not_produce_a_crossing():
    """Reproduces the original failure directly: feed the tracker foot points
    from camera 57's measured range and confirm the OLD line yields nothing."""
    d = CrossingDetector(57, (W, H))
    object.__setattr__(d, "line", CrossingLine(position=0.35, inside_side="below"))

    lo, hi, _ = MEASURED_FOOT_RANGE[57]
    events = walk(d, [lo, 0.70, 0.80, hi, 0.80, lo], camera_id=57)
    assert events == [], "0.35 produced a crossing it could not physically produce"
    assert d.count_in == 0 and d.count_out == 0


def test_camera_57_current_line_does_produce_crossings():
    """The same walk, at the configured line, must be counted."""
    d = CrossingDetector(57, (W, H))
    lo, hi, _ = MEASURED_FOOT_RANGE[57]
    pos = d.line.position

    below = min(hi, pos + 0.10)
    above = max(lo, pos - 0.10)
    events = walk(d, [above, above, below, below], camera_id=57)
    assert len(events) == 1, f"a walk across {pos} produced {len(events)} events"
    assert events[0].direction is Direction.IN


def test_camera_57_a_person_who_stays_on_one_side_is_not_counted():
    """Camera 57 is a vestibule where people sit and linger. Movement within
    the waiting area must not be mistaken for passing through the door."""
    d = CrossingDetector(57, (W, H))
    pos = d.line.position
    loiter = [min(0.99, pos + 0.05), min(0.99, pos + 0.15),
              min(0.99, pos + 0.08), min(0.99, pos + 0.20)]
    assert walk(d, loiter, camera_id=57) == []


def test_a_clipped_box_is_not_treated_as_a_foot_position():
    """30% of camera 57's boxes touch the frame bottom -- the person's feet are
    out of shot and the box edge is the frame, not a body part. Those points
    pile up into a false peak that made a line at 0.95 look ideal until they
    were excluded."""
    lo, hi, _ = MEASURED_FOOT_RANGE[57]
    assert crossing_line(57).position < 0.95, (
        "the line sits in the clipped-box peak, not in real foot positions"
    )

