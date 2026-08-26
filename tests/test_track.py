"""Per-camera person tracking: stable ids, and state that never crosses cameras.

TWO RULES ARE PINNED HERE AND BOTH HAVE ALREADY FAILED IN PRODUCTION.

CAMERA ISOLATION. `model.track(persist=True)` keeps tracker state on the model,
and V2 shares one YOLO instance across three workers. If tracking went through
the model, camera 57's tracks would follow a camera 59 frame. Tracking is
therefore keyed on camera id and the tracker REFUSES another camera's
detections rather than silently corrupting both.

ASSOCIATION ACROSS SECONDS, NOT MILLISECONDS. Textbook ByteTrack assumes ~33ms
between frames, where a person barely moves and IoU is high. Measured live, the
doorways are served every 5.34 SECONDS -- a walker covers ~7.5m and consecutive
boxes have IoU of exactly 0. Pure-IoU association mints a new id every pass:

    pass 0 id=1   pass 1 id=2   pass 2 id=3 ...

The count looks right, one track per pass, while identity churns completely.
That is what broke attendance in V1: evidence accumulates ON a track, so a fresh
id every pass resets the observation counter forever.

Every test drives explicit timestamps. Nothing here depends on wall-clock
timing, because a tracker whose behaviour cannot be reproduced cannot be trusted
when its output is later questioned.
"""
import pytest

from app.cctv_v2.pipeline.detect import DetectionResult, PersonDetection
from app.cctv_v2.pipeline.track import (
    MATCH_DIST_MIN_WIDTHS,
    CameraTracker,
    PersonTrack,
    TrackerRegistry,
    TrackState,
)

DOORWAY, ROOM = 57, 59


def det(x1, y1, x2, y2, conf=0.8, camera_id=DOORWAY, ts=0.0, seq=1):
    return PersonDetection(
        camera_id=camera_id, frame_timestamp=ts, frame_sequence=seq,
        bbox=(float(x1), float(y1), float(x2), float(y2)), confidence=conf,
    )


def result(camera_id, detections, ts, seq):
    return DetectionResult(
        camera_id=camera_id, role="doorway" if camera_id in (57, 58) else "room",
        frame_sequence=seq, frame_timestamp=ts, detections=tuple(detections),
        inference_ms=1.0, frame_age_ms=10.0, imgsz=640, conf=0.15,
    )


def feed(tracker, boxes, ts, seq, camera_id=None):
    cid = camera_id if camera_id is not None else tracker.camera_id
    dets = [det(*b, camera_id=cid, ts=ts, seq=seq) for b in boxes]
    return tracker.update(result(cid, dets, ts, seq))


# ---------------------------------------------------------------------------
# One person, one track
# ---------------------------------------------------------------------------
def test_one_person_keeps_one_track_id_across_passes():
    t = CameraTracker(DOORWAY)
    ids = []
    for i, x in enumerate([100, 140, 180, 220, 260]):
        got = feed(t, [(x, 300, x + 80, 700)], ts=i * 0.5, seq=i + 1)
        ids.append(got[0].track_id)

    assert len(set(ids)) == 1, f"id churned across passes: {ids}"
    assert t.total_created == 1


def test_a_walker_keeps_one_id_at_the_REAL_sampling_interval():
    """The measured case: 5.34s between passes, boxes with ZERO overlap.

    This is the test the whole design exists for. With pure-IoU association it
    fails with five different ids -- one per pass.
    """
    t = CameraTracker(DOORWAY)
    ids = []
    x = 100.0
    for i in range(5):
        got = feed(t, [(x, 300, x + 90, 700)], ts=i * 5.34, seq=i + 1)
        ids.append(got[0].track_id)
        x += 260.0                    # far beyond its own width: IoU is 0

    assert len(set(ids)) == 1, (
        f"a walker got {len(set(ids))} ids across 5 passes at 5.34s: {ids}"
    )


def test_boxes_that_do_not_overlap_at_all_still_associate():
    t = CameraTracker(DOORWAY)
    a = feed(t, [(100, 300, 180, 700)], ts=0.0, seq=1)[0]
    b = feed(t, [(400, 300, 480, 700)], ts=2.0, seq=2)[0]   # no overlap

    from app.cctv_v2.pipeline.track import _iou
    assert _iou((100, 300, 180, 700), (400, 300, 480, 700)) == 0.0
    assert a.track_id == b.track_id


def test_a_track_is_confirmed_after_min_hits():
    t = CameraTracker(DOORWAY)
    first = feed(t, [(100, 300, 180, 700)], ts=0.0, seq=1)[0]
    assert first.state is TrackState.TENTATIVE

    second = feed(t, [(120, 300, 200, 700)], ts=0.5, seq=2)[0]
    assert second.state is TrackState.CONFIRMED
    assert t.active_tracks()[0].track_id == first.track_id


# ---------------------------------------------------------------------------
# Several people
# ---------------------------------------------------------------------------
def test_two_people_get_two_independent_tracks():
    t = CameraTracker(DOORWAY)
    got = feed(t, [(100, 300, 180, 700), (600, 300, 680, 700)], ts=0.0, seq=1)
    assert len({g.track_id for g in got}) == 2

    got = feed(t, [(130, 300, 210, 700), (630, 300, 710, 700)], ts=0.5, seq=2)
    assert len({g.track_id for g in got}) == 2
    assert t.total_created == 2, "a third track appeared for two people"


def test_two_people_far_apart_do_not_collapse_into_one_track():
    """V1 shipped this bug: distance scaled by max(width, height) gave a 600px
    reach for a 100px-wide person and merged two people 600px apart."""
    t = CameraTracker(DOORWAY)
    feed(t, [(100, 200, 200, 900)], ts=0.0, seq=1)          # tall, narrow
    got = feed(t, [(100, 200, 200, 900), (700, 200, 800, 900)], ts=0.4, seq=2)

    assert len({g.track_id for g in got}) == 2, "two people collapsed into one"


def test_one_detection_cannot_claim_a_track_another_already_took():
    t = CameraTracker(DOORWAY)
    feed(t, [(100, 300, 180, 700)], ts=0.0, seq=1)
    got = feed(t, [(110, 300, 190, 700), (150, 300, 230, 700)], ts=0.4, seq=2)

    assert len({g.track_id for g in got}) == 2
    assert t.total_created == 2


def test_people_crossing_keep_separate_ids():
    """Two people walking past each other in opposite directions."""
    t = CameraTracker(DOORWAY)
    left, right = 100.0, 700.0
    ids_left, ids_right = [], []
    for i in range(6):
        got = feed(t, [(left, 300, left + 80, 700), (right, 300, right + 80, 700)],
                   ts=i * 0.5, seq=i + 1)
        got = sorted(got, key=lambda g: g.bbox[0])
        ids_left.append(got[0].track_id)
        ids_right.append(got[-1].track_id)
        left += 100
        right -= 100

    assert len(set(ids_left)) <= 2, f"left walker churned: {ids_left}"
    assert len(set(ids_right)) <= 2, f"right walker churned: {ids_right}"


# ---------------------------------------------------------------------------
# Loss, occlusion, expiry
# ---------------------------------------------------------------------------
def test_a_missed_detection_does_not_destroy_the_track():
    t = CameraTracker(DOORWAY)
    first = feed(t, [(100, 300, 180, 700)], ts=0.0, seq=1)[0]
    feed(t, [(140, 300, 220, 700)], ts=0.5, seq=2)

    feed(t, [], ts=1.0, seq=3)                 # occluded: nothing detected
    assert t.active_tracks() == []
    assert [x.track_id for x in t.lost_tracks()] == [first.track_id]

    again = feed(t, [(220, 300, 300, 700)], ts=1.5, seq=4)[0]
    assert again.track_id == first.track_id, "the person came back as a new id"
    assert t.total_reassociated == 1


def test_a_held_track_is_remembered_but_NOT_reported_as_present():
    """V1 published held tracks as people: 55% of passes reported more people
    than the detector found, including an empty corridor showing five."""
    t = CameraTracker(DOORWAY)
    feed(t, [(100, 300, 180, 700)], ts=0.0, seq=1)
    feed(t, [(140, 300, 220, 700)], ts=0.5, seq=2)
    assert len(t.active_tracks()) == 1

    got = feed(t, [], ts=1.0, seq=3)
    assert got == [], "an undetected person was reported on this pass"
    assert t.active_tracks() == [], "an empty corridor still reports a person"
    assert len(t.lost_tracks()) == 1, "the track was forgotten, not held"


def test_a_track_expires_on_ELAPSED_TIME_not_missed_passes():
    """The pipeline skips source frames deliberately, so 'three misses' means
    0.1s on one camera and 20s on another. V1 counted cycles and, under load,
    that scaled the wrong way -- the busier the box, the longer the dead
    lingered."""
    t = CameraTracker(DOORWAY)
    feed(t, [(100, 300, 180, 700)], ts=0.0, seq=1)
    feed(t, [(140, 300, 220, 700)], ts=0.5, seq=2)

    feed(t, [], ts=5.0, seq=3)                 # 4.5s gap, one miss
    assert len(t.lost_tracks()) == 1, "expired too early"

    feed(t, [], ts=100.0, seq=4)               # far past the 12s doorway limit
    assert t.lost_tracks() == []
    assert t.all_tracks() == []
    assert t.total_removed == 1


def test_the_room_holds_tracks_far_longer_than_the_doorway():
    """A seated person is occluded for much longer than a walker, and a room
    track that dies gets re-counted as a new person when they lean back."""
    doorway = CameraTracker(DOORWAY)
    room = CameraTracker(ROOM)
    assert room.max_age_sec > doorway.max_age_sec

    for t, cid in ((doorway, DOORWAY), (room, ROOM)):
        feed(t, [(100, 300, 180, 700)], ts=0.0, seq=1)
        feed(t, [(100, 300, 180, 700)], ts=0.5, seq=2)
        feed(t, [], ts=20.0, seq=3)

    assert doorway.all_tracks() == [], "doorway held a 20s-old track"
    assert len(room.lost_tracks()) == 1, "room dropped a seated person at 20s"


def test_a_returning_person_after_expiry_is_a_new_person():
    """Deterministic by design. Once a track has aged out this layer has no
    basis to claim it is the same human -- that would be a recognition claim,
    and recognition does not exist here."""
    t = CameraTracker(DOORWAY)
    first = feed(t, [(100, 300, 180, 700)], ts=0.0, seq=1)[0]
    feed(t, [], ts=200.0, seq=2)
    second = feed(t, [(100, 300, 180, 700)], ts=201.0, seq=3)[0]

    assert second.track_id != first.track_id


# ---------------------------------------------------------------------------
# Camera isolation -- non-negotiable
# ---------------------------------------------------------------------------
def test_a_tracker_refuses_another_camera_s_detections():
    t = CameraTracker(57)
    with pytest.raises(ValueError, match="per-camera"):
        t.update(result(58, [det(100, 300, 180, 700, camera_id=58)], 0.0, 1))


def test_camera_57_state_never_reaches_camera_58():
    reg = TrackerRegistry()
    for i in range(4):
        reg.update(result(57, [det(100 + i * 40, 300, 180 + i * 40, 700,
                                   camera_id=57, ts=i * 0.5, seq=i + 1)],
                          i * 0.5, i + 1))
    assert len(reg.get(57).active_tracks()) == 1
    assert reg.get(58).all_tracks() == [], "camera 58 acquired 57's tracks"
    assert reg.get(58).total_created == 0


def test_camera_59_state_never_reaches_camera_60():
    reg = TrackerRegistry()
    for i in range(4):
        reg.update(result(59, [det(100, 300, 180, 700, camera_id=59,
                                   ts=i * 0.5, seq=i + 1)], i * 0.5, i + 1))
    assert reg.get(60).all_tracks() == []


def test_every_camera_gets_its_own_tracker_object():
    reg = TrackerRegistry()
    trackers = [reg.get(c) for c in (57, 58, 59, 60)]
    assert len({id(t) for t in trackers}) == 4
    assert reg.cameras() == (57, 58, 59, 60)


def test_track_ids_are_unique_within_a_camera():
    t = CameraTracker(DOORWAY)
    seen = set()
    for i in range(10):
        got = feed(t, [(100 + i * 300, 300, 180 + i * 300, 700)],
                   ts=i * 30.0, seq=i + 1)      # far apart in time AND space
        for g in got:
            assert g.track_id not in seen, "a track id was reused"
            seen.add(g.track_id)


def test_ids_are_not_reused_after_a_reset():
    """A downstream step keys evidence on track_id; 'track 3' meaning two people
    in one shift surfaces as an unexplainable record, not an error."""
    t = CameraTracker(DOORWAY)
    first = feed(t, [(100, 300, 180, 700)], ts=0.0, seq=1)[0]
    t.reset()
    second = feed(t, [(100, 300, 180, 700)], ts=1.0, seq=1)[0]

    assert second.track_id != first.track_id


def test_two_cameras_may_share_a_track_id_without_meaning_anything():
    """Ids are unique per camera, not globally. Anything comparing them across
    cameras is wrong, and this records that."""
    reg = TrackerRegistry()
    a = reg.update(result(57, [det(100, 300, 180, 700, camera_id=57)], 0.0, 1))[0]
    b = reg.update(result(59, [det(100, 300, 180, 700, camera_id=59)], 0.0, 1))[0]
    assert a.track_id == b.track_id == 1
    assert a.camera_id != b.camera_id


# ---------------------------------------------------------------------------
# Frame provenance and ordering
# ---------------------------------------------------------------------------
def test_frame_timestamp_and_sequence_are_preserved():
    t = CameraTracker(DOORWAY)
    feed(t, [(100, 300, 180, 700)], ts=1234.5, seq=77)
    got = t.all_tracks()[0]
    assert got.frame_timestamp == 1234.5
    assert got.frame_sequence == 77


def test_an_out_of_order_pass_is_refused():
    """Would rewind velocity and ageing. The scheduler always serves the newest
    frame, so this guards a caller bug rather than an expected condition."""
    t = CameraTracker(DOORWAY)
    feed(t, [(100, 300, 180, 700)], ts=10.0, seq=10)
    got = feed(t, [(500, 300, 580, 700)], ts=5.0, seq=5)

    assert got == []
    assert t.rejected_out_of_order == 1
    assert t.total_created == 1


def test_a_backlog_is_never_worked_through():
    """The tracker sees only what the scheduler chose to process -- one frame in
    roughly forty. Sequence gaps are normal and must not be treated as loss."""
    t = CameraTracker(DOORWAY)
    a = feed(t, [(100, 300, 180, 700)], ts=0.0, seq=1)[0]
    b = feed(t, [(160, 300, 240, 700)], ts=0.5, seq=64)[0]      # 63-frame gap
    assert a.track_id == b.track_id
    assert b.frame_sequence == 64


# ---------------------------------------------------------------------------
# Independence from recognition
# ---------------------------------------------------------------------------
def test_tracking_works_with_no_face_information_whatsoever():
    """COUNT THE PERSON FIRST. A back-facing person with no usable face is a
    person, and nothing in this layer may consult identity."""
    t = CameraTracker(DOORWAY)
    for i in range(4):
        feed(t, [(100 + i * 50, 300, 180 + i * 50, 700)], ts=i * 0.5, seq=i + 1)

    track = t.active_tracks()[0]
    assert track.hits == 4
    assert not hasattr(track, "employee_id")
    assert not hasattr(track, "identity")


def test_low_confidence_detections_still_get_tracked():
    """ByteTrack's second stage. A marginal box is often a real person who is
    partly occluded -- discarding it is how people vanish mid-transit."""
    t = CameraTracker(DOORWAY)
    first = feed(t, [(100, 300, 180, 700, 0.80)], ts=0.0, seq=1)[0]
    weak = feed(t, [(140, 300, 220, 700, 0.16)], ts=0.5, seq=2)[0]

    assert weak.track_id == first.track_id
    assert weak.confidence == pytest.approx(0.16)


def test_confident_detections_are_matched_before_marginal_ones():
    t = CameraTracker(DOORWAY)
    feed(t, [(100, 300, 180, 700, 0.9)], ts=0.0, seq=1)
    got = feed(t, [(115, 300, 195, 700, 0.05), (105, 300, 185, 700, 0.9)],
               ts=0.4, seq=2)

    strong = [g for g in got if g.confidence > 0.5][0]
    assert strong.track_id == 1, "the confident box lost its own track"


# ---------------------------------------------------------------------------
# Prediction bounds
# ---------------------------------------------------------------------------
def test_a_long_lost_track_cannot_predict_across_the_frame():
    """Uncapped, a 30s-old track predicts halfway across the county and matches
    anything that happens to be there."""
    t = CameraTracker(ROOM)
    feed(t, [(100, 300, 200, 700)], ts=0.0, seq=1)
    feed(t, [(200, 300, 300, 700)], ts=1.0, seq=2)     # 100 px/s rightwards

    track = t.all_tracks()[0]
    predicted = track.predict(track.last_seen + 25.0)   # would be +2500px
    drift = predicted[0] - track.bbox[0]
    assert drift <= 4.0 * track.width + 1e-6, f"predicted {drift:.0f}px away"


def test_prediction_is_the_reason_a_fast_walker_stays_one_track():
    t = CameraTracker(DOORWAY)
    ids = []
    x = 50.0
    for i in range(4):
        ids.append(feed(t, [(x, 300, x + 70, 700)], ts=i * 5.34, seq=i + 1)[0].track_id)
        x += 200.0

    assert len(set(ids)) == 1, f"fast walker churned: {ids}"


# ---------------------------------------------------------------------------
# Registry lifecycle
# ---------------------------------------------------------------------------
def test_resetting_one_camera_leaves_the_others_alone():
    reg = TrackerRegistry()
    for cid in (57, 58):
        for i in range(3):
            reg.update(result(cid, [det(100, 300, 180, 700, camera_id=cid,
                                        ts=i * 0.5, seq=i + 1)], i * 0.5, i + 1))
    assert len(reg.get(57).active_tracks()) == 1
    assert len(reg.get(58).active_tracks()) == 1

    reg.reset(57)
    assert reg.get(57).all_tracks() == []
    assert len(reg.get(58).active_tracks()) == 1, "resetting 57 disturbed 58"


def test_a_dead_camera_does_not_disturb_the_other_trackers():
    reg = TrackerRegistry()
    for i in range(6):
        ts = i * 0.5
        reg.update(result(57, [det(100 + i * 40, 300, 180 + i * 40, 700,
                                   camera_id=57, ts=ts, seq=i + 1)], ts, i + 1))
        if i < 2:                     # 59 stops producing after two passes
            reg.update(result(59, [det(100, 300, 180, 700, camera_id=59,
                                       ts=ts, seq=i + 1)], ts, i + 1))

    assert len(reg.get(57).active_tracks()) == 1
    assert len(reg.get(59).active_tracks()) == 1, "59's seated person was dropped"


def test_stats_report_per_camera_without_pooling():
    reg = TrackerRegistry()
    reg.update(result(57, [det(100, 300, 180, 700, camera_id=57)], 0.0, 1))
    reg.update(result(59, [det(100, 300, 180, 700, camera_id=59)], 0.0, 1))

    s = reg.stats()
    assert set(s) == {57, 59}
    assert s[57]["role"] == "doorway" and s[59]["role"] == "room"
    assert s[57]["max_age_sec"] != s[59]["max_age_sec"]


# ---------------------------------------------------------------------------
# KNOWN LIMITATION -- pinned deliberately, not endorsed
#
# The reach that lets a walker keep one id across a 5.34s gap is the same reach
# that can merge two DIFFERENT people. At multi-second sampling there is
# genuinely no evidence separating "the same person walked on" from "somebody
# else appeared" -- the intervening seconds were never looked at.
#
# These tests exist so the trade-off is visible in the suite instead of being
# discovered later in an attendance report. They will need updating if the
# sampling interval drops, which is the actual fix.
# ---------------------------------------------------------------------------
def test_at_long_gaps_two_different_people_CAN_merge():
    """The cost of the wide reach. Not a bug to fix here -- a consequence of
    sampling every 5.34s, removable only by sampling faster."""
    t = CameraTracker(DOORWAY)
    feed(t, [(100, 300, 190, 700)], ts=0.0, seq=1)          # person A, left
    feed(t, [(100, 300, 190, 700)], ts=0.5, seq=2)          # A stands still

    # A leaves unseen; B appears 500px away 5.3s later. Nothing observed the gap.
    got = feed(t, [(600, 300, 690, 700)], ts=5.8, seq=3)

    assert len(got) == 1
    assert got[0].track_id == 1, (
        "behaviour changed -- if this now creates a new track, the reach was "
        "tightened and the walker tests should be re-checked"
    )


def test_the_merge_risk_is_bounded_by_the_reach_ceiling():
    """8 box-widths, not unlimited. A detection beyond it starts a new track
    however long the gap has been."""
    t = CameraTracker(DOORWAY)
    feed(t, [(100, 300, 190, 700)], ts=0.0, seq=1)
    feed(t, [(100, 300, 190, 700)], ts=0.5, seq=2)

    far = feed(t, [(1500, 300, 1590, 700)], ts=30.0, seq=3)   # >8 widths away
    assert far[0].track_id != 1, "an arbitrarily distant box joined the track"


def test_a_stationary_person_is_never_at_risk_of_the_wide_reach():
    """The reach grows with time SINCE LAST SEEN. A person detected on every
    pass keeps a 2.5-width reach, so the risk applies only after a miss."""
    t = CameraTracker(ROOM)
    for i in range(5):
        feed(t, [(400, 300, 500, 700)], ts=i * 4.9, seq=i + 1)

    track = t.all_tracks()[0]
    assert t.total_created == 1
    assert CameraTracker._reach(track, det(400, 300, 500, 700), track.last_seen)         == pytest.approx(MATCH_DIST_MIN_WIDTHS * 100.0)



# ---------------------------------------------------------------------------
# One person is published ONCE, however many boxes they wear
# ---------------------------------------------------------------------------
# The permissive creation threshold that makes a seated person detectable at all
# also makes YOLO emit a tight torso box AND a bloated body-plus-chair box for
# the same person. Both became tracks, so the room over-reported: observed live
# on camera 59, two people at the left desk wearing four boxes between them and
# a count of 5 in a room containing 4.
def test_a_tight_box_inside_a_bloated_one_is_one_person():
    """The case IoU cannot see. A torso box fully inside a body-and-chair box
    scores near zero by IoU, which is why overlap is measured against the
    SMALLER box instead."""
    t = CameraTracker(ROOM)
    live = feed(t, [(100, 100, 200, 500), (110, 150, 190, 300)], 0.0, 1)
    assert len(live) == 1


def test_the_survivor_is_kept_not_the_pair_deleted():
    """A rule that DROPPED the bloated box instead was measured costing two real
    people their only detection, because one large box can be the only evidence
    of a person standing behind another. Merging keeps somebody; dropping does
    not."""
    t = CameraTracker(ROOM)
    live = feed(t, [(100, 100, 200, 500), (110, 150, 190, 300)], 0.0, 1)
    assert len(t.all_tracks()) == 1
    assert live[0].bbox in {(100.0, 100.0, 200.0, 500.0),
                            (110.0, 150.0, 190.0, 300.0)}


def test_two_people_side_by_side_are_still_two():
    """The rule must not merge a crowd. These overlap slightly, as people at
    adjacent desks do, but neither is inside the other."""
    t = CameraTracker(ROOM)
    live = feed(t, [(100, 100, 200, 500), (190, 100, 290, 500)], 0.0, 1)
    assert len(live) == 2


def test_the_established_track_is_the_one_that_survives():
    """Evidence order: confirmed beats tentative, then more hits, then the older
    id. Otherwise a duplicate box could evict the track that has been following
    the person for minutes."""
    t = CameraTracker(ROOM)
    for i in range(3):
        feed(t, [(100, 100, 200, 500)], float(i), i + 1)
    established = t.active_tracks()[0].track_id

    live = feed(t, [(100, 100, 200, 500), (105, 140, 195, 320)], 3.0, 4)
    assert len(live) == 1
    assert live[0].track_id == established


def test_deduplication_is_counted_so_it_can_be_seen():
    """A silent merge is indistinguishable from a detector that never
    duplicated, and the two want very different responses."""
    t = CameraTracker(ROOM)
    feed(t, [(100, 100, 200, 500), (110, 150, 190, 300)], 0.0, 1)
    assert t.stats()["deduped"] == 1


def test_a_lone_box_is_never_touched():
    t = CameraTracker(ROOM)
    assert len(feed(t, [(100, 100, 200, 500)], 0.0, 1)) == 1
    assert CameraTracker(ROOM).stats()["deduped"] == 0


def test_one_box_spanning_two_people_loses_to_their_own_boxes():
    """The case that is NOT a duplicate.

    Observed live on camera 59: a single box covered both people at the
    left-hand desk and, being the better-established track, absorbed their
    individual boxes. The count improved and the occupancy got worse -- two
    seats that had been correctly OCCUPIED went free, because the surviving box
    reached the floor and failed the seated check.
    """
    t = CameraTracker(ROOM)
    # Two people, plus one box swallowing both.
    live = feed(t, [(100, 100, 300, 700),
                    (110, 120, 190, 380),
                    (210, 300, 290, 660)], 0.0, 1)
    assert len(live) == 2
    widths = sorted(round(p.bbox[2] - p.bbox[0]) for p in live)
    assert widths == [80, 80], "the two tight boxes should be what survives"


def test_a_container_with_only_one_child_is_still_a_duplicate():
    """The gate is TWO children. With one, the pair really is a person seen
    twice, and dropping the larger box there was measured costing two people
    their only detection."""
    t = CameraTracker(ROOM)
    live = feed(t, [(100, 100, 300, 700), (110, 120, 190, 380)], 0.0, 1)
    assert len(live) == 1
