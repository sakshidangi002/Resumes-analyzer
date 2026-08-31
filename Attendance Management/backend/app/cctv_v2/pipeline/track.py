"""Per-camera person tracking. One tracker per camera, state never shared.

WHERE THIS SITS
---------------
    grabber -> slot -> scheduler -> YOLO (shared) -> [THIS] -> person tracks

It turns a stream of independent detections into persistent identities WITHIN
one camera. It does not recognise faces, judge crossings or write attendance --
those are later steps and must not leak in here.

WHY NOT ULTRALYTICS' BYTETRACK
------------------------------
`model.track(persist=True)` keeps tracker state ON THE MODEL OBJECT. V2 shares
one YOLO instance across three inference workers, so using it would mean camera
57's tracks following a camera 59 frame -- the exact contamination that is
forbidden. Rather than un-share the model (measured: separate instances are the
only reason the pool is faster than one worker), tracking is done here, keyed on
camera id, from plain detections.

That inversion is the whole architectural point:

    ONE detector, shared, stateless.
    N trackers, isolated, stateful, one per camera.

THE HARD PART: THIS IS NOT VIDEO
--------------------------------
Textbook ByteTrack assumes consecutive video frames ~33ms apart, where a person
barely moves and IoU between frames is high. That assumption is FALSE here and
the difference is not marginal.

Measured live: doorway cameras are served every 5.34 SECONDS. A person walking
at ~1.4 m/s covers roughly 7.5 metres between passes. Consecutive boxes of the
same person do not overlap at all -- IoU is exactly 0. Pure-IoU association
would mint a fresh track id on every single pass:

    pass 0 id=1    pass 1 id=2    pass 2 id=3 ...

The COUNT would look plausible -- one track per pass -- while identity churned
completely. V1 hit precisely this, and it is why attendance failed there:
evidence accumulates ON a track, so a new id every pass reset the observation
counter forever and no evidence threshold could ever be reached.

So association here is deliberately NOT IoU-only:

  1. Predict where each track should be now, from its velocity and the ELAPSED
     TIME since it was last seen. At 5s gaps this moves the box most of the way
     to the person.
  2. Match on IoU against the PREDICTED box.
  3. Fall back to centroid distance scaled by the track's box WIDTH.

Width, not `max(width, height)`. A standing person's box is roughly 3x taller
than wide, so scaling by height gave a 600px reach for a 100px-wide person and
merged two people standing 600px apart into one track. Width is the right scale
for horizontal displacement, which is how people cross a doorway.

REMEMBERED IS NOT PRESENT
-------------------------
A track that stops being detected goes LOST. It is kept so the same person can
be re-associated to it, and it is NOT reported as present. V1 conflated these
and published held tracks as people: 55% of passes reported more people than the
detector found, including an empty corridor showing five.

`update()` therefore returns only tracks that got a detection on THIS pass.
`lost_tracks()` exposes the remembered ones separately, for anyone who needs
them.

SAFETY
------
This associates BODIES, never identities. A wrong merge cannot mislabel anybody,
because identity does not exist at this layer at all -- it arrives later, and
the evidence rules that consume it are unchanged.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

from app.cctv_v2.config.cameras import profile_for, role_for
from app.cctv_v2.pipeline.detect import DetectionResult, PersonDetection

logger = logging.getLogger(__name__)

# How far a detection may sit from a track's PREDICTED position and still be
# judged the same person, in multiples of the track's box width.
#
# V1 used a FIXED 2.5 widths, tuned when passes were ~2.5s apart. At the
# measured 5.34s doorway interval that is too tight: a walker outruns it before
# any velocity has been observed (the first association always has velocity 0,
# so prediction contributes nothing), and every pass mints a new id.
#
# So the reach grows with the ELAPSED TIME since the track was last seen,
# because a person covers more ground in more time. A person's box width is
# roughly shoulder width, ~0.5m, so walking at ~1.5m/s is ~3 widths/second.
#
#     reach = width x clamp(3.0 x dt, 2.5, 8.0)
#
#     dt=0.5s -> 2.5 widths (floor, V1's value)
#     dt=2.0s -> 6.0 widths
#     dt=5.3s -> 8.0 widths (ceiling)
#
# THE CEILING IS A HONEST ADMISSION, NOT A TUNING CHOICE. At 5.34s sampling a
# reach of 8 widths spans most of the frame, so two DIFFERENT people can be
# merged into one track -- see the test that pins this. That ambiguity is
# forced by the sampling rate and cannot be engineered away here: at multi-
# second gaps there is genuinely no evidence distinguishing "the same person
# walked on" from "someone else appeared". The ceiling bounds the damage; only
# a shorter sampling interval removes the cause.
MATCH_DIST_MIN_WIDTHS = 2.5
MATCH_DIST_MAX_WIDTHS = 8.0
MATCH_SPEED_WIDTHS_PER_SEC = 3.0

# Minimum IoU (against the predicted box) to accept a match outright.
MATCH_IOU = 0.30

# Two boxes overlapping this much OF THE SMALLER ONE are the same person seen
# twice, and are collapsed to one track. Matches V1's CCTV_TRACK_DEDUPE_OVERLAP
# rather than inventing a second number for the same decision; measured
# insensitive between 0.55 and 0.90 on the labelled frames.
DEDUPE_OVERLAP = 0.55

# Velocity is smoothed rather than taken from the last pair alone: at multi-
# second sampling a single noisy box would otherwise fling the prediction far
# from the person. 0.5 keeps it responsive to a real direction change while
# damping one bad measurement.
VELOCITY_SMOOTHING = 0.5

# A predicted box is never flung further than this many box-widths, however
# stale the track. Without a cap, a 30s-old room track predicts halfway across
# the county and matches anything.
MAX_PREDICT_WIDTHS = 4.0


class TrackState(str, Enum):
    TENTATIVE = "tentative"   # seen, not yet confirmed by `track_min_hits`
    CONFIRMED = "confirmed"   # believed to be a real person
    LOST = "lost"             # not detected recently; remembered, NOT present
    REMOVED = "removed"       # aged out


@dataclass
class PersonTrack:
    """One person, as this camera has come to understand them."""

    camera_id: int
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    frame_timestamp: float
    frame_sequence: int

    first_seen: float
    last_seen: float                       # timestamp of the last DETECTION
    hits: int = 1
    misses: int = 0
    state: TrackState = TrackState.TENTATIVE
    velocity: tuple[float, float] = (0.0, 0.0)     # pixels per second
    history: list = field(default_factory=list)    # (timestamp, centroid)

    @property
    def width(self) -> float:
        return max(1.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def age_sec(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def time_since_seen(self, now: float) -> float:
        return max(0.0, now - self.last_seen)

    def predict(self, now: float) -> tuple[float, float, float, float]:
        """Where this person probably is now, given how they were moving.

        The whole reason association survives multi-second gaps. Displacement is
        capped at `MAX_PREDICT_WIDTHS` box-widths so a long-lost track cannot
        predict its way across the frame and swallow an unrelated detection.
        """
        dt = self.time_since_seen(now)
        if dt <= 0 or self.velocity == (0.0, 0.0):
            return self.bbox
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        cap = MAX_PREDICT_WIDTHS * self.width
        mag = math.hypot(dx, dy)
        if mag > cap and mag > 0:
            dx, dy = dx * cap / mag, dy * cap / mag
        x1, y1, x2, y2 = self.bbox
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _overlap_ratio(a, b) -> float:
    """Intersection over the SMALLER box's area.

    Deliberately not IoU. A tight torso box sitting fully inside a bloated
    body-and-chair box of the same person scores near zero by IoU and 1.0 here.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min(max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1),
                  max(0.0, bx2 - bx1) * max(0.0, by2 - by1))
    return (inter / smaller) if smaller > 0 else 0.0


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centre(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class CameraTracker:
    """Tracks people for ONE camera. Never sees another camera's detections."""

    def __init__(self, camera_id: int, clock=None) -> None:
        self.camera_id = int(camera_id)
        self.role = role_for(camera_id)
        profile = profile_for(camera_id)
        self.max_age_sec = profile.track_max_age_sec
        self.min_hits = profile.track_min_hits
        self.high_conf = profile.track_high_conf

        self._tracks: dict[int, PersonTrack] = {}
        self._next_id = 1
        self._last_sequence = -1
        self._clock = clock

        # Counters, for measuring stability rather than asserting it.
        self.total_created = 0
        self.total_removed = 0
        self.total_reassociated = 0     # LOST -> matched again
        self.total_deduped = 0          # merged as the same person seen twice
        self.rejected_stale = 0
        self.rejected_out_of_order = 0

        # Hits each track had reached when it was expired. Kept because the
        # live tracks alone cannot answer "is identity churning?" -- a churning
        # camera expires its 1-hit tracks continuously and the survivors are a
        # biased sample of the ones that happened to be alive at the end.
        self.hits_at_removal: list[int] = []

    # -- public API ----------------------------------------------------------
    def update(self, result: DetectionResult) -> list[PersonTrack]:
        """Fold one detection pass in; return the tracks seen on THIS pass.

        Tracks that were not detected are NOT returned -- they are remembered
        for re-association only. See the module docstring: reporting held tracks
        as present is how V1 came to show five people in an empty corridor.
        """
        if result.camera_id != self.camera_id:
            raise ValueError(
                f"camera {result.camera_id} detections given to tracker "
                f"{self.camera_id}; tracker state is per-camera and must never "
                f"be crossed"
            )
        return self._update(
            result.detections, result.frame_timestamp, result.frame_sequence,
        )

    def _update(self, detections, timestamp: float, sequence: int) -> list[PersonTrack]:
        # A pass older than one already folded in would rewind velocity and
        # ageing. The scheduler always serves the newest frame, so this is a
        # guard against a caller bug, not an expected condition.
        if sequence <= self._last_sequence:
            self.rejected_out_of_order += 1
            logger.warning(
                "tracker camera=%s ignoring out-of-order pass seq=%s (last %s)",
                self.camera_id, sequence, self._last_sequence,
            )
            return []
        self._last_sequence = sequence

        self._expire(timestamp)

        dets = list(detections)
        candidates = [t for t in self._tracks.values() if t.state is not TrackState.REMOVED]
        # Detections are addressed by INDEX throughout. Two people can produce
        # byte-identical boxes on different passes, and PersonDetection is a
        # frozen dataclass compared by value, so identity-by-equality would
        # quietly merge them.
        matched: dict[int, int] = {}            # track_id -> detection index
        claimed: set[int] = set()
        used_dets: set[int] = set()

        # ByteTrack's two-stage idea: trust the confident boxes first, then give
        # the marginal ones a second chance against whatever is left, instead of
        # discarding them.
        order = sorted(
            range(len(dets)),
            key=lambda i: (dets[i].confidence < self.high_conf, -dets[i].confidence),
        )
        for i in order:
            tid = self._best_match(dets[i], candidates, claimed, timestamp)
            if tid is not None:
                claimed.add(tid)
                matched[tid] = i
                used_dets.add(i)

        seen_now: list[PersonTrack] = []
        for tid, di in matched.items():
            track = self._tracks[tid]
            if track.state is TrackState.LOST:
                self.total_reassociated += 1
            self._absorb(track, dets[di], timestamp, sequence)
            seen_now.append(track)

        for i, det in enumerate(dets):
            if i not in used_dets:
                seen_now.append(self._create(det, timestamp, sequence))

        # Anything not matched this pass drifts toward LOST.
        for track in candidates:
            if track.track_id not in matched:
                track.misses += 1
                if track.state is not TrackState.LOST:
                    track.state = TrackState.LOST

        self._dedupe_overlapping(seen_now)
        return sorted(seen_now, key=lambda t: t.track_id)

    def _dedupe_overlapping(self, seen_now: list) -> None:
        """Collapse tracks that are the same person seen twice.

        A permissive creation threshold is what makes a seated person detectable
        at all, and the same permissiveness makes YOLO emit a tight torso box
        AND a bloated body-plus-chair box for one person. Both become tracks, so
        the room over-reports. Observed live on camera 59 after the threshold
        change: two people at the left-hand desk wearing four boxes between
        them, and a count of 5 in a room containing 4.

        Overlap is intersection over the SMALLER box, not IoU. A tight box
        sitting fully inside a bloated one scores near zero by IoU and 1.0 here,
        and 1.0 is the answer that is true.

        MEASURED over the 16 labelled frames, on the detections this config
        produces (scripts/cctv_v2_room_bench.py data):

            threshold   duplicates merged   people left with no box   surplus
              0.55            7                      1                   0
              0.90            7                      1                   0
              off             0                      0                   4

        Insensitive between 0.55 and 0.90, so this matches V1's 0.55 rather than
        inventing a second number. The single loss is a bloated box spanning two
        people: merging into it keeps one of them, which is why the survivor is
        MERGED INTO rather than deleted -- an overlap rule that dropped the
        bloated box instead costs two people, and was measured doing so (see
        person_nested_contain in core/config.py).
        """
        if len(seen_now) < 2:
            return

        # A box that swallows TWO OR MORE distinct boxes is not a duplicate of
        # any of them -- it is one detection spanning several people, and it must
        # lose to its children rather than absorb them.
        #
        # Observed live on camera 59: one box covered both people at the
        # left-hand desk, outranked their individual boxes on evidence, and the
        # merge below collapsed the pair into a single track. The count improved
        # and the occupancy got worse -- two seats that had been correctly
        # OCCUPIED went free, because the surviving box reached the floor and
        # failed the seated check.
        #
        # Gated on TWO children, never one. With one child the pair really is a
        # person seen twice, and dropping the larger box there is the rule that
        # was measured costing two people their only detection (see
        # person_nested_contain in core/config.py).
        bloated = set()
        for track in seen_now:
            children = [o for o in seen_now
                        if o is not track
                        and _area(track.bbox) > _area(o.bbox)
                        and _overlap_ratio(track.bbox, o.bbox) >= DEDUPE_OVERLAP]
            distinct = [c for c in children
                        if not any(d is not c
                                   and _overlap_ratio(c.bbox, d.bbox) >= DEDUPE_OVERLAP
                                   for d in children)]
            if len(distinct) >= 2:
                bloated.add(track.track_id)

        # Best evidence first, so the more established track survives -- but a
        # box identified as spanning several people goes last, where it will be
        # merged away rather than doing the merging.
        ordered = sorted(
            seen_now,
            key=lambda t: (t.track_id not in bloated,
                           t.state is TrackState.CONFIRMED, t.hits, -t.track_id),
            reverse=True,
        )
        kept: list = []
        for track in ordered:
            if any(_overlap_ratio(track.bbox, k.bbox) >= DEDUPE_OVERLAP
                   for k in kept):
                self._tracks.pop(track.track_id, None)
                seen_now.remove(track)
                self.total_deduped += 1
                continue
            kept.append(track)

    def active_tracks(self) -> list[PersonTrack]:
        """Tracks currently believed to be a real, present person."""
        return sorted(
            (t for t in self._tracks.values() if t.state is TrackState.CONFIRMED),
            key=lambda t: t.track_id,
        )

    def lost_tracks(self) -> list[PersonTrack]:
        """Remembered but NOT present. Exposed so nobody has to guess."""
        return sorted(
            (t for t in self._tracks.values() if t.state is TrackState.LOST),
            key=lambda t: t.track_id,
        )

    def all_tracks(self) -> list[PersonTrack]:
        return sorted(self._tracks.values(), key=lambda t: t.track_id)

    def reset(self) -> None:
        """Forget everything. Track ids are NOT reused -- see `_create`."""
        self._tracks.clear()
        self._last_sequence = -1

    # -- internals -----------------------------------------------------------
    def _best_match(self, det, candidates, claimed, now) -> Optional[int]:
        """The track this detection most likely belongs to, or None.

        Tracks already claimed on this pass are excluded. Without that, two
        detections a few hundred pixels apart both match the same nearest track
        and two people collapse into one -- a bug V1 shipped and had to fix.
        """
        best_id, best_iou = None, 0.0
        for track in candidates:
            if track.track_id in claimed:
                continue
            overlap = _iou(det.bbox, track.predict(now))
            if overlap > best_iou:
                best_iou, best_id = overlap, track.track_id
        if best_id is not None and best_iou >= MATCH_IOU:
            return best_id

        # IoU failed. At multi-second sampling that is the NORMAL case for a
        # walker, not an exception -- fall back to distance from the predicted
        # position, scaled by the track's own width.
        best_id, best_dist = None, float("inf")
        dcx, dcy = _centre(det.bbox)
        for track in candidates:
            if track.track_id in claimed:
                continue
            pcx, pcy = _centre(track.predict(now))
            dist = math.hypot(dcx - pcx, dcy - pcy)
            reach = self._reach(track, det, now)
            if dist <= reach and dist < best_dist:
                best_dist, best_id = dist, track.track_id
        return best_id

    @staticmethod
    def _reach(track, det, now) -> float:
        """How far this track's person could plausibly have travelled by now.

        Scaled by box WIDTH, never max(width, height): a standing person's box
        is ~3x taller than wide, so height gave a 600px reach for a 100px-wide
        person and merged two people standing 600px apart into one track.
        """
        dt = track.time_since_seen(now)
        widths = min(
            MATCH_DIST_MAX_WIDTHS,
            max(MATCH_DIST_MIN_WIDTHS, MATCH_SPEED_WIDTHS_PER_SEC * dt),
        )
        return widths * max(track.width, det.bbox[2] - det.bbox[0])

    def _absorb(self, track, det, timestamp, sequence) -> None:
        dt = max(1e-6, timestamp - track.last_seen)
        ocx, ocy = track.centroid
        ncx, ncy = _centre(det.bbox)
        vx, vy = (ncx - ocx) / dt, (ncy - ocy) / dt
        if track.velocity == (0.0, 0.0):
            track.velocity = (vx, vy)
        else:
            s = VELOCITY_SMOOTHING
            track.velocity = (
                s * vx + (1 - s) * track.velocity[0],
                s * vy + (1 - s) * track.velocity[1],
            )

        track.bbox = det.bbox
        track.confidence = det.confidence
        track.frame_timestamp = timestamp
        track.frame_sequence = sequence
        track.last_seen = timestamp
        track.hits += 1
        track.misses = 0
        track.history.append((timestamp, (ncx, ncy)))
        if track.hits >= self.min_hits:
            track.state = TrackState.CONFIRMED
        else:
            track.state = TrackState.TENTATIVE

    def _create(self, det, timestamp, sequence) -> PersonTrack:
        """A new person. Ids are never reused, even after a reset.

        Reuse would make a downstream consumer's `track_id` ambiguous across
        time -- a later step keys evidence on it, and "track 3" meaning two
        different people in one shift is a bug that surfaces as an unexplainable
        attendance record rather than an error.
        """
        tid = self._next_id
        self._next_id += 1
        self.total_created += 1
        cx, cy = _centre(det.bbox)
        track = PersonTrack(
            camera_id=self.camera_id,
            track_id=tid,
            bbox=det.bbox,
            confidence=det.confidence,
            frame_timestamp=timestamp,
            frame_sequence=sequence,
            first_seen=timestamp,
            last_seen=timestamp,
            hits=1,
            state=TrackState.CONFIRMED if self.min_hits <= 1 else TrackState.TENTATIVE,
            history=[(timestamp, (cx, cy))],
        )
        self._tracks[tid] = track
        return track

    def _expire(self, now: float) -> None:
        """Drop tracks nobody has seen for `track_max_age_sec`.

        Elapsed TIME, never a miss count. The pipeline deliberately skips source
        frames, so "3 missed frames" means 0.1s on one camera and 20s on
        another. V1 expressed holds in cycles and, under load, that scaled the
        wrong way: the busier the machine, the longer dead people lingered.
        """
        dead = [
            tid for tid, t in self._tracks.items()
            if t.time_since_seen(now) > self.max_age_sec
        ]
        for tid in dead:
            track = self._tracks[tid]
            track.state = TrackState.REMOVED
            self.hits_at_removal.append(track.hits)
            del self._tracks[tid]
            self.total_removed += 1

    def hit_histogram(self) -> list[int]:
        """Hits for EVERY track this camera has had, live and expired.

        The honest basis for a churn judgement. A camera whose tracks all reach
        1 hit and die is churning even if the two still alive look fine.
        """
        return self.hits_at_removal + [t.hits for t in self._tracks.values()]

    def stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "role": self.role,
            "active": len(self.active_tracks()),
            "lost": len(self.lost_tracks()),
            "created": self.total_created,
            "removed": self.total_removed,
            "deduped": self.total_deduped,
            "single_hit_tracks": sum(1 for h in self.hit_histogram() if h == 1),
            "reassociated": self.total_reassociated,
            "rejected_stale": self.rejected_stale,
            "rejected_out_of_order": self.rejected_out_of_order,
            "max_age_sec": self.max_age_sec,
        }


class TrackerRegistry:
    """One tracker per camera. The isolation guarantee lives here.

    A registry rather than a tracker passed around, so there is exactly one
    place where a camera id maps to tracker state and no caller can accidentally
    hand camera 57's detections to camera 59's tracker -- `update` checks, and
    raises rather than silently corrupting both.
    """

    def __init__(self, clock=None) -> None:
        self._trackers: dict[int, CameraTracker] = {}
        self._clock = clock

    def get(self, camera_id: int) -> CameraTracker:
        cid = int(camera_id)
        if cid not in self._trackers:
            self._trackers[cid] = CameraTracker(cid, clock=self._clock)
            logger.info("tracker created for camera %s", cid)
        return self._trackers[cid]

    def update(self, result: DetectionResult) -> list[PersonTrack]:
        return self.get(result.camera_id).update(result)

    def reset(self, camera_id: int) -> None:
        """Forget one camera's tracks. Used on reconnect, when the scene may
        have changed entirely while the stream was down."""
        if int(camera_id) in self._trackers:
            self._trackers[int(camera_id)].reset()
            logger.info("tracker reset for camera %s", camera_id)

    def cameras(self) -> tuple[int, ...]:
        return tuple(sorted(self._trackers))

    def stats(self) -> dict:
        return {cid: t.stats() for cid, t in sorted(self._trackers.items())}
