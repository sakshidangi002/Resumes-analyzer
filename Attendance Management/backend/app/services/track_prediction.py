"""Display-only bounding-box extrapolation.

WHY THIS EXISTS
---------------
The detector is not blind; it is late. Measured on camera 58 (Exit) from a
screen recording lined up against hrms.log:

    video 26s  overlay 10:44:12  woman mid-corridor          People: 0, no box
    video 29s  overlay 10:44:15  woman close, fully visible  People: 0, no box
    video 31s  overlay 10:44:17  woman about to exit         People: 0, no box
    video 32s  overlay 10:44:18  corridor EMPTY              People: 1 + box
    video 34s  overlay 10:44:20  corridor EMPTY              box still drawn

and the log for that moment:

    10:44:17.597  adopted untracked detection score=0.842 height=520
                  as provisional track=1000011

So YOLO saw her perfectly (0.842, 520px tall) on a frame captured around
10:44:16, and the box reached the screen at 10:44:18 -- by which time she had
walked out. A gate camera analyses every ~3.9s (camera 58 median, from the PERF
counters) while a person is in shot for ~7s, so between passes the box is frozen
at a position that is already 1-4 seconds stale.

This module does NOT make the detector faster. It draws each track at where it
is *estimated to be now*, using the velocity implied by its own recent measured
positions, and snaps back to truth the moment a real detection arrives.

SCOPE -- read this before extending it
--------------------------------------
This is a DISPLAY concern and nothing else. It is deliberately a separate module
with no imports from the detection/attendance stack, and it never mutates a
PersonTrack: it hands the renderer shallow COPIES carrying a predicted `box`.

    PersonTrack.box        REAL, measured. Crossing lines, attendance, identity
                           and the database all keep reading this. Untouched.
    copy.box               PREDICTED. Drawn on the preview JPEG. Nothing else
                           ever sees it.

If you ever find yourself wanting the predicted position for a crossing test or
an attendance decision, stop: that is how a person gets marked present for
walking through a line they never reached.
"""
from __future__ import annotations

import copy
import logging
import os
import time
from collections import deque
from typing import Any, Deque, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("%s is not a number; using %s", name, default)
        return default


# Master switch. Off restores the previous behaviour exactly (boxes frozen at
# the last measured position) with no other change.
ENABLED = os.getenv("CCTV_BOX_PREDICT", "true").lower() in {"1", "true", "yes"}

# Longest a box may be extrapolated past its last MEASUREMENT.
#
# Past this the box reverts to the measured position and simply sits there —
# it is NOT hidden. Hiding is the retention logic's job (bytetrack_engine's
# hold windows), not this module's, and quietly dropping boxes here would make
# a seated person on a MONITOR camera flicker out while still plainly tracked.
# See HIDE_STALE_SEC for the opt-in if you do want them gone.
#
# 1.0s covers most of a gate's ~3.9s pass interval without letting a box march
# far on a stale heading. Beyond roughly a second a walker's velocity estimate
# is guesswork: they can stop, turn, or be occluded.
MAX_EXTRAPOLATION_SEC = _f("CCTV_BOX_PREDICT_MAX_SEC", 1.0)

# MONITOR cameras get a shorter leash. Seated people shuffle, lean and turn far
# more than they translate, so a confident velocity is more often wrong there,
# and the payoff is small because they are not crossing the frame anyway.
MAX_EXTRAPOLATION_SEC_MONITOR = _f("CCTV_BOX_PREDICT_MAX_SEC_MONITOR", 0.5)

# Optional and OFF by default: hide a box whose measurement is older than this
# many seconds. 0 disables. This exists because "box sitting on an empty
# corridor" is a real complaint, but switching it on changes WHICH boxes are
# displayed, not merely where they are drawn — a different thing from this
# module's job — so it is a deliberate opt-in rather than a default.
HIDE_STALE_SEC = _f("CCTV_BOX_PREDICT_HIDE_STALE_SEC", 0.0)

# Velocity sanity ceiling, as a fraction of FRAME WIDTH per second. A person
# cannot cross one and a half frame widths in a second; anything claiming to is
# an association error (two people swapped, or an id reused), and extrapolating
# it would fling a box across the picture.
MAX_SPEED_FRAC = _f("CCTV_BOX_PREDICT_MAX_SPEED_FRAC", 1.5)

# Below this speed (fraction of frame width per second) a track is treated as
# STATIONARY and not moved at all. Without a deadband, detector jitter of a few
# pixels between passes turns into a permanently drifting box on someone who is
# standing still — the most visible possible failure on a MONITOR camera.
MIN_SPEED_FRAC = _f("CCTV_BOX_PREDICT_MIN_SPEED_FRAC", 0.01)

# How hard the drawn position chases its target each display frame, 0..1.
# 1.0 = snap (lowest latency, visible jump when a correction lands).
# 0.5 = converge in ~2-3 frames at 8 FPS, i.e. within ~250-375ms.
# Deliberately high: the brief is low latency over smoothness, and heavy
# smoothing would reintroduce exactly the lag this module exists to remove.
BLEND = min(1.0, max(0.05, _f("CCTV_BOX_PREDICT_BLEND", 0.5)))

# Velocity smoothing across consecutive measurements, 0..1 (weight of the NEW
# estimate). Keeps a single noisy pass from throwing the heading, while still
# turning within a couple of detections when somebody genuinely changes
# direction.
VEL_SMOOTH = min(1.0, max(0.05, _f("CCTV_BOX_PREDICT_VEL_SMOOTH", 0.6)))

# Positions kept per track for the velocity fit. Three points over the newest
# and oldest span is steadier than differencing the last two, and costs nothing.
HISTORY = 3

# A track id absent for longer than this is considered a NEW person if it comes
# back. Ids are recycled (and the adopt path mints 1000000+ ids freshly), so
# without this a reused id would inherit a stranger's velocity.
STALE_STATE_SEC = _f("CCTV_BOX_PREDICT_STATE_TTL_SEC", 5.0)

# Ceiling on the measurement lag the display will compensate for. Beyond this
# the value is not believable (a clock step, or a camera that just restarted)
# and compensating it would fling every box across the frame.
MAX_MEASUREMENT_LAG_SEC = _f("CCTV_BOX_PREDICT_MAX_LAG_SEC", 5.0)

# Rate limit for the periodic summary line. Per camera, not per track.
LOG_EVERY_SEC = _f("CCTV_BOX_PREDICT_LOG_EVERY_SEC", 30.0)


class _TrackState:
    """Prediction state for one track id. Display-only; never persisted."""

    __slots__ = (
        "history", "anchor_box", "anchor_t", "anchor_lag", "vx", "vy",
        "last_seen_val", "last_present_t", "drawn_cx", "drawn_cy", "predicting",
    )

    def __init__(self) -> None:
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=HISTORY)
        self.anchor_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.anchor_t: float = 0.0
        self.anchor_lag: float = 0.0
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.last_seen_val: Optional[float] = None
        self.last_present_t: float = 0.0
        self.drawn_cx: Optional[float] = None
        self.drawn_cy: Optional[float] = None
        self.predicting: bool = False


class BoxPredictor:
    """Per-camera display-side box extrapolator.

    One instance per camera, owned by the display thread, so there is no locking
    and no cross-camera state. `predict()` is pure arithmetic over at most a
    handful of tracks: no model, no frame data, no allocation beyond a shallow
    copy of the tracks it actually moves.
    """

    def __init__(self, camera_id: Any, is_monitor: bool = False) -> None:
        self.camera_id = camera_id
        self.is_monitor = bool(is_monitor)
        self.max_sec = (
            MAX_EXTRAPOLATION_SEC_MONITOR if is_monitor else MAX_EXTRAPOLATION_SEC
        )
        self._states: dict[Any, _TrackState] = {}
        self._last_log_t = 0.0

    # -- internals ---------------------------------------------------------
    def _observe(self, st: _TrackState, box, seen_val, now: float,
                 lag: float = 0.0) -> None:
        """Fold one NEW measurement into the state and re-fit velocity.

        `lag` is how old the measurement ALREADY IS on arrival: the gap between
        the capture of the frame it was measured on and now. Everything is
        anchored at the capture instant (`now - lag`) rather than at the moment
        this thread noticed it, which is what makes the drawn box line up with
        the live picture instead of trailing the pipeline by a fixed offset.
        It also makes the velocity fit correct when the lag varies pass to pass.
        """
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        # The velocity fit uses ARRIVAL times, not capture times.
        #
        # Capture time = arrival - lag, and the lag is neither stable nor always
        # believable: measured live it swings 1.5s-5.0s pass to pass and is
        # clamped at MAX_MEASUREMENT_LAG_SEC. Subtracting a jittery lag from a
        # clean clock INVERTS the ordering of consecutive captures, which
        # collapses dt to the 1e-3 floor and silently yields vx=vy=0 -- observed
        # in the live TRACK_PREDICT lines before this was changed.
        #
        # Arrival times are monotonic and spaced by the pass period, which is
        # the same spacing the captures have on average. The lag still matters,
        # but only for the projection HORIZON, where it is applied once and
        # clamped -- never differenced.
        st.history.append((now, cx, cy))

        vx = vy = 0.0
        if len(st.history) >= 2:
            t0, x0, y0 = st.history[0]
            t1, x1, y1 = st.history[-1]
            dt = t1 - t0
            # Guard a zero/negative dt (clock oddity, duplicate observation) and
            # an absurdly long one, where the person's path in between is
            # unknowable and a straight-line fit is meaningless.
            if 1e-3 < dt < 10.0:
                vx = (x1 - x0) / dt
                vy = (y1 - y0) / dt

        # Smooth toward the new estimate rather than replacing it outright.
        st.vx = VEL_SMOOTH * vx + (1.0 - VEL_SMOOTH) * st.vx
        st.vy = VEL_SMOOTH * vy + (1.0 - VEL_SMOOTH) * st.vy
        st.anchor_box = tuple(int(v) for v in box[:4])  # type: ignore[assignment]
        # Anchored at ARRIVAL, with the lag kept alongside it. The two are
        # budgeted differently -- see the horizon calculation in _predict.
        st.anchor_t = now
        st.anchor_lag = lag
        st.last_seen_val = seen_val

    def _clamp_velocity(self, st: _TrackState, frame_w: int) -> None:
        if frame_w <= 0:
            return
        ceiling = MAX_SPEED_FRAC * frame_w
        floor = MIN_SPEED_FRAC * frame_w
        speed = (st.vx * st.vx + st.vy * st.vy) ** 0.5
        if speed > ceiling and speed > 0:
            scale = ceiling / speed
            st.vx *= scale
            st.vy *= scale
        elif speed < floor:
            # Standing still: park the box rather than let jitter walk it.
            st.vx = 0.0
            st.vy = 0.0

    # -- public ------------------------------------------------------------
    def predict(self, tracks: Iterable[Any], frame_shape,
                measured_at: float = 0.0) -> list:
        """Return draw-ready tracks, extrapolated where it is safe to do so.

        `measured_at` is the wall-clock CAPTURE time of the frame these boxes
        were measured on (worker._latest_tracks_frame_ts). Pass 0 to disable
        latency compensation and extrapolate only from arrival.

        Never raises: on any internal failure the ORIGINAL tracks are returned,
        so a bug here can degrade the overlay but can never stop the display
        thread or blank the preview.
        """
        tracks = list(tracks or [])
        if not ENABLED or not tracks:
            return tracks
        try:
            return self._predict(tracks, frame_shape, measured_at)
        except Exception:  # noqa: BLE001
            logger.debug(
                "camera %s: box prediction failed, drawing measured boxes",
                self.camera_id, exc_info=True,
            )
            return tracks

    def _predict(self, tracks: list, frame_shape, measured_at: float = 0.0) -> list:
        try:
            frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        except Exception:  # noqa: BLE001
            frame_h = frame_w = 0

        now = time.monotonic()          # monotonic ONLY: wall clock can step

        # How stale the incoming measurement already is. The capture timestamp
        # is wall-clock (worker._latest_frame_ts), so this DURATION is computed
        # wall-to-wall and only then applied on the monotonic scale — a clock
        # step can therefore distort one frame's offset but never corrupt the
        # monotonic anchors. Clamped for exactly that reason.
        lag = 0.0
        if measured_at:
            lag = time.time() - measured_at
            if not (0.0 <= lag <= MAX_MEASUREMENT_LAG_SEC):
                lag = 0.0 if lag < 0 else MAX_MEASUREMENT_LAG_SEC
        out: list = []
        alive: set = set()
        n_predicted = 0
        sample = None

        for tr in tracks:
            tid = getattr(tr, "track_id", None)
            box = getattr(tr, "box", None)
            if tid is None or not box or len(box) < 4:
                out.append(tr)
                continue
            alive.add(tid)

            st = self._states.get(tid)
            if st is None or (now - st.last_present_t) > STALE_STATE_SEC:
                # First sighting, or an id recycled onto a different person.
                st = _TrackState()
                self._states[tid] = st
                st.anchor_box = tuple(int(v) for v in box[:4])  # type: ignore
                st.anchor_t = now
                st.anchor_lag = lag
                st.last_seen_val = getattr(tr, "last_seen", None)
                # ARRIVAL time, matching _observe -- seeding this one with the
                # capture time instead made the first dt a whole lag too long
                # and halved every velocity.
                st.history.append(
                    (now, (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                )
            st.last_present_t = now

            # A NEW measurement is signalled by PersonTrack.last_seen changing.
            # update_box() always restamps it, so this is exact and needs no
            # comparison of coordinates.
            seen_val = getattr(tr, "last_seen", None)
            if seen_val is not None and seen_val != st.last_seen_val:
                self._observe(st, box, seen_val, now, lag)
            self._clamp_velocity(st, frame_w)

            # TWO different things, deliberately budgeted apart:
            #
            #   anchor_lag   KNOWN. The frame this box was measured on was
            #                captured this long ago (~1.8s on camera 58). The
            #                person demonstrably kept moving during it. Not a
            #                guess, so it is always compensated.
            #   since_arrival SPECULATIVE. Time since the measurement reached
            #                us, during which we have no evidence at all. This
            #                is what MAX_EXTRAPOLATION_SEC bounds.
            #
            # Capping the SUM instead (the obvious first cut) makes the feature
            # silently inert: with a 1.8s lag every box is already past a 1.0s
            # cap the moment it arrives, so nothing is ever predicted.
            since_arrival = now - st.anchor_t
            age = st.anchor_lag + since_arrival
            ax1, ay1, ax2, ay2 = st.anchor_box
            bw, bh = ax2 - ax1, ay2 - ay1
            acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0

            if HIDE_STALE_SEC > 0 and age > HIDE_STALE_SEC:
                # Opt-in only: drop the box entirely rather than draw a person
                # who has not been measured for a long time.
                continue

            moving = (st.vx or st.vy) and age > 0
            if moving and since_arrival <= self.max_sec:
                target_cx = acx + st.vx * age
                target_cy = acy + st.vy * age
                st.predicting = True
            else:
                # Past the cap (or stationary) the box reverts to the MEASURED
                # position: once the measurement is this old the heading is no
                # longer trustworthy either. It is not hidden -- see
                # HIDE_STALE_SEC.
                target_cx, target_cy = acx, acy
                if st.predicting and since_arrival > self.max_sec:
                    st.predicting = False
                    logger.debug(
                        "camera %s track %s: extrapolation capped at %.2fs",
                        self.camera_id, tid, self.max_sec,
                    )

            # Chase the target so a correction lands as a short glide, not a jump.
            if st.drawn_cx is None or st.drawn_cy is None:
                st.drawn_cx, st.drawn_cy = target_cx, target_cy
            else:
                st.drawn_cx += (target_cx - st.drawn_cx) * BLEND
                st.drawn_cy += (target_cy - st.drawn_cy) * BLEND

            nx1 = int(round(st.drawn_cx - bw / 2.0))
            ny1 = int(round(st.drawn_cy - bh / 2.0))
            nx2, ny2 = nx1 + bw, ny1 + bh

            # Keep the box inside the picture, preserving its size by SHIFTING
            # rather than cropping -- a clipped box reads as the person having
            # shrunk, which is worse than a slightly wrong position.
            if frame_w > 0 and frame_h > 0:
                if nx1 < 0:
                    nx1, nx2 = 0, min(frame_w, bw)
                if nx2 > frame_w:
                    nx2, nx1 = frame_w, max(0, frame_w - bw)
                if ny1 < 0:
                    ny1, ny2 = 0, min(frame_h, bh)
                if ny2 > frame_h:
                    ny2, ny1 = frame_h, max(0, frame_h - bh)

            new_box = (nx1, ny1, nx2, ny2)
            if new_box == tuple(int(v) for v in box[:4]):
                out.append(tr)                     # nothing moved: no copy
                continue

            # Shallow copy: the renderer reads `box` via get_display_info(), and
            # every identity field it also reads stays shared with the original.
            # The REAL track object is never touched.
            drawn = copy.copy(tr)
            drawn.box = new_box
            out.append(drawn)
            n_predicted += 1
            if sample is None:
                sample = (tid, age, st.vx, st.vy)

        # Drop state for ids that have gone away, so this cannot grow unbounded
        # on a camera that churns provisional ids.
        if len(self._states) > len(alive):
            for tid in [k for k in self._states
                        if k not in alive
                        and (now - self._states[k].last_present_t) > STALE_STATE_SEC]:
                self._states.pop(tid, None)

        if n_predicted and (now - self._last_log_t) >= LOG_EVERY_SEC:
            self._last_log_t = now
            tid, age, vx, vy = sample  # type: ignore[misc]
            logger.info(
                "TRACK_PREDICT camera=%s track=%s age_ms=%.0f lag_ms=%.0f "
                "vx=%.1f vy=%.1f predicted=true moved=%d/%d cap=%.2fs",
                self.camera_id, tid, age * 1000.0, lag * 1000.0, vx, vy,
                n_predicted, len(tracks), self.max_sec,
            )
        return out
