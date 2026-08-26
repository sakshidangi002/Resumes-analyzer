"""The two CCTV profiles: doorway and room.

FOUR CAMERAS, TWO PROFILES
--------------------------
There are four physical cameras but only two ways of behaving, so there are two
profiles here and not four configurations. A camera declares its ROLE (see
cameras.py) and inherits everything else. Adding a fifth camera is a one-line
entry in the role map, not a fifth copy of twenty numbers that will drift.

The split is not cosmetic. The two profiles answer different questions and have
opposite failure costs:

  DOORWAY (57, 58)  Their matches become attendance rows. A false accept writes
                    the wrong person's payroll data and may go unnoticed for
                    weeks; a miss is recoverable by hand. So: strict everything,
                    and real evidence before anything is written.

  ROOM (59, 60)     Hard-blocked from attendance. Their worst failure is a wrong
                    name on a live overlay for a few seconds, while a MISSED
                    person defeats the camera's only purpose. So: relaxed
                    everything, optimised for coverage.

EVERY NUMBER HERE IS MEASURED, NOT CHOSEN
-----------------------------------------
These values came out of benchmarking against real frames from these cameras,
and several of them are counter-intuitive:

  * input_size 640 for doorways but 480 for rooms. Bigger is NOT better. At 480
    a walking person scored 0.148 on the Exit camera - under the 0.20 track
    threshold, so they vanished - while at 640 the same people scored 0.475+.
    In the rooms the opposite held: 480 gave better worst-case scores than 960
    (0.418 vs 0.290) AND correct counts where 960 turned 2 people into 4, at
    half the cost.

  * predict_conf 0.03 for rooms. Seated staff seen from behind score 0.035-0.057
    because legs are hidden by the chair and the head/limb silhouette a COCO
    detector keys on is absent. The two genuinely empty doorway cameras produced
    zero detections at every threshold down to 0.02, so this buys coverage and
    not noise.

They are FROZEN for the V2 rebuild. V2 is an architecture and scheduling change;
if the CV parameters move at the same time, a V1/V2 comparison cannot attribute
any difference to either. Proposed changes belong in a separate review, after
the comparison.

2026-08-26: THE ROOM NUMBERS ABOVE ARE NOW WRONG, AND HAVE BEEN CORRECTED
------------------------------------------------------------------------
Room detection moved from 480/0.03 to 960/0.015 with new_track_thresh 0.02.
The claim in this docstring that 480 gave better worst-case scores than 960 was
measured per FRAME, against whichever frames were to hand, and against no fixed
ground truth. Re-measured against 16 labelled frames containing 41 hand-boxed
people (scripts/cctv_v2_room_bench.py), scored per PERSON:

    camera 59, four people   480     960     1280
      LEFT-SEATED               0.30    0.40    0.70
      TOP-DESK                  0.00    0.20    0.20   <- INVISIBLE at 480
      RIGHT-SEATED              0.30    0.40    0.70
      STANDING                  1.00    1.00    1.00
    camera 60, one person       0.17    0.67    0.83
    cost per pass               1.9s    3.8s    9.8s

At 480 one of the four people in that room produces no detection at any
confidence, on any frame. 960 rather than 1280 because all four cameras share
one inference slot on this box and the doorways -- which mark attendance --
queue behind the rooms.

This does NOT break the V1/V2 comparison the freeze was protecting: V1 and V2
were changed together, to identical values, in the same commit. What would break
it is one of them moving alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["doorway", "room"]


@dataclass(frozen=True)
class Profile:
    """Everything a camera's behaviour depends on, in one immutable object.

    Frozen because these are read from several threads (the scheduler, the
    inference worker, the overlay renderer) and a value that can be mutated
    mid-pass is a source of decisions that disagree with the log line explaining
    them.
    """

    role: Role

    # ── Detection ────────────────────────────────────────────────────────────
    model: str
    input_size: int
    predict_conf: float

    # ── Tracking ─────────────────────────────────────────────────────────────
    tracker_cfg: str
    new_track_thresh: float

    # ── Scheduling ───────────────────────────────────────────────────────────
    # What this camera would LIKE. The scheduler treats it as a target, never a
    # promise: measured YOLO cost on this CPU is 2.2s (room@480) to 3.4s
    # (doorway@640), so a doorway asking for 0.12s will not get it. V1 slept for
    # the interval and then queued on a lock, which produced a 4.5-9s effective
    # cycle and hid the fact. V2 records requested vs actual separately.
    analysis_interval_target: float

    # The oldest frame this camera is willing to have processed. A VALIDITY
    # check, not a scheduling weight -- see the scheduler's docstring for why
    # those must stay separate.
    #
    # Measured, like everything else here. Live, a healthy camera's frame is
    # 0.03-0.04s old when the scheduler picks it up, with a worst case of 0.111s
    # across 100 selections. So both cutoffs sit far above normal operation and
    # only trigger on a stream that has actually stopped.
    #
    # They differ because a stale frame costs the two roles differently. A
    # doorway answers "is somebody crossing right now", and a person crosses in
    # about two seconds -- a one-second-old frame is already half a crossing out
    # of date, and inference on it can only produce a false answer about the
    # present. A room answers "who is sitting here", and a seated person is
    # still there five seconds later, so the same picture retains most of its
    # value. Hence 1.0s and 5.0s.
    #
    # Found by killing camera 59 mid-run: its slot kept the last frame, the
    # scheduler kept choosing it, and seven inference passes went to a picture
    # that aged to 71 seconds.
    max_frame_age: float

    # ── Tracking ─────────────────────────────────────────────────────────────
    # ADDITIVE for Step 7. No existing measured value was changed.
    #
    # track_max_age_sec  how long a track is REMEMBERED after it stops being
    #                    detected. Remembered, not reported -- a held track is
    #                    kept only so the same person can be re-associated to it,
    #                    and is never counted as present. V1 conflated those and
    #                    an empty corridor reported five people.
    #
    #                    It must exceed the sampling interval or every track dies
    #                    between passes. Measured live: doorways are served every
    #                    5.34s and rooms every 4.87s.
    #
    #                    Doorway 12s -- a little over two passes. A doorway
    #                    transit lasted a median 13.5s (camera 58, lunch hour),
    #                    so two consecutive misses still belong to one crossing.
    #                    Room 30s -- seated people are occluded for far longer,
    #                    and a room track that dies gets re-counted as a new
    #                    person the moment they lean back into view.
    #
    # track_min_hits     detections before a track is CONFIRMED rather than
    #                    TENTATIVE. 2 for both: at multi-second sampling a third
    #                    hit can cost 15s, and a doorway transit may only ever
    #                    yield two (median 2.3 detections per transit, measured).
    #
    # track_high_conf    the ByteTrack high/low split. Set above each role's
    #                    detection floor so the first association pass uses boxes
    #                    that are actually reliable, and the leftovers get a
    #                    second chance rather than being discarded.
    # Whether the scheduler should prefer this camera when its picture is
    # MOVING. Doorways only.
    #
    # Measured: 80% of camera 57's passes and 66% of camera 58's saw nobody at
    # all, while a frame-difference costs ~2ms against a YOLO pass at ~4.9s. So
    # most doorway inference was spent confirming an empty corridor, and the
    # cost of noticing that cheaply first is negligible.
    #
    # ROOMS ARE DELIBERATELY NOT GATED. A room answers "who is sitting here",
    # and a seated person barely moves -- gating on motion would let occupancy
    # decay exactly when the room is calm, which is most of the time. A doorway
    # asks "is somebody passing through", and somebody passing through moves by
    # definition.
    #
    # This is a WEIGHTING, never a hard skip: an idle camera's score is damped,
    # not zeroed, and the starvation deadline still guarantees it a turn. A
    # camera cannot go blind because nothing happened to move in front of it.
    motion_gated: bool

    track_max_age_sec: float
    track_min_hits: int
    track_high_conf: float

    # ── Recognition ──────────────────────────────────────────────────────────
    match_threshold: float
    match_margin: float
    min_face_px: float
    max_yaw: float

    # ── Evidence required before an identity is believed ──────────────────────
    observations_required: int
    quality_required: float
    consensus_required: float

    # ── Policy ───────────────────────────────────────────────────────────────
    marks_attendance: bool
    line_crossing: bool
    line_position: float | None
    show_today_total: bool

    # ── Scheduler priority ───────────────────────────────────────────────────
    # Doorways rank higher because a person crosses in ~2s and is gone, while a
    # seated person in a room is still there on the next pass. This weights the
    # choice; it does not make it exclusive - see scheduler starvation rules.
    priority: int


DOORWAY = Profile(
    role="doorway",
    model="models/yolo11m.pt",
    input_size=640,
    predict_conf=0.15,
    tracker_cfg="models/bytetrack_person.yaml",
    new_track_thresh=0.20,
    analysis_interval_target=0.12,
    max_frame_age=1.0,
    motion_gated=True,
    track_max_age_sec=12.0,
    track_min_hits=2,
    track_high_conf=0.35,
    match_threshold=0.45,
    match_margin=0.18,
    min_face_px=28.0,
    max_yaw=40.0,
    observations_required=3,
    quality_required=0.28,
    consensus_required=0.55,
    marks_attendance=True,
    line_crossing=True,
    line_position=0.35,
    show_today_total=True,
    priority=3,
)

ROOM = Profile(
    role="room",
    model="models/yolo11m.pt",
    # 960/0.015, was 480/0.03. See the note at the end of this docstring block
    # and the matching entries in .env -- V1 and V2 are changed TOGETHER and to
    # the same values, which is what keeps them comparable.
    input_size=960,
    predict_conf=0.015,
    tracker_cfg="models/bytetrack_person_lowconf.yaml",
    new_track_thresh=0.02,
    analysis_interval_target=1.5,
    max_frame_age=5.0,
    motion_gated=False,
    track_max_age_sec=30.0,
    track_min_hits=2,
    track_high_conf=0.02,
    match_threshold=0.42,
    match_margin=0.10,
    min_face_px=16.0,
    max_yaw=75.0,
    observations_required=2,
    quality_required=0.15,
    consensus_required=0.40,
    marks_attendance=False,
    line_crossing=False,
    line_position=None,
    show_today_total=False,
    priority=1,
)

PROFILES: dict[Role, Profile] = {"doorway": DOORWAY, "room": ROOM}


def get_profile(role: Role) -> Profile:
    """The profile for a role. Raises on an unknown role rather than guessing.

    Defaulting here would be dangerous in one specific direction: an unknown
    role silently resolving to DOORWAY would hand attendance-writing behaviour
    to a camera nobody vetted for it.
    """
    try:
        return PROFILES[role]
    except KeyError:
        raise ValueError(
            f"unknown camera role {role!r}; expected one of {sorted(PROFILES)}"
        ) from None
