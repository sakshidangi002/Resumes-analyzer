"""A bounded pool of inference workers that decides, explicitly, which camera
goes next.

WHAT THIS REPLACES
------------------
V1 ran four recognition threads that each slept for their profile's analysis
interval and then queued on a shared `threading.BoundedSemaphore(1)`. Nothing in
that arrangement decided anything: the winner was whichever thread the OS
happened to wake, and Python's semaphore makes no fairness promise.

Measured on this box, that produced starvation rather than scheduling -- one
camera took 17 inference passes in a window where the other three took one
each -- and a doorway asking for a 0.12s interval actually got 4.5-9s.

So the choice is explicit, deterministic and logged.

WHY A POOL, AND WHY EACH WORKER OWNS ITS MODEL
----------------------------------------------
This ran on ONE worker until a benchmark showed why it should not. Measured on
this box (4 physical / 8 logical cores):

    workers   passes/s   vs 1     latency   cores busy
       1        0.403    1.00x      2.48s   1.73 of 8
       2        0.587    1.46x      3.40s   2.35 of 8
       3        0.733    1.82x      3.99s   2.83 of 8
       4        0.811    2.01x      4.81s   3.21 of 8
    2 (SHARED)  0.280    0.93x      6.88s   1.35 of 8

Read the last row against the second. Two workers sharing ONE ultralytics model
are SLOWER than one worker, and use fewer cores while doing it. Two workers with
their own instances are 1.46x faster. The earlier conclusion that "one YOLO pass
already saturates four cores" was drawn from a measurement of exactly that
shared-instance contention, and it was wrong -- YOLO uses 1.73 of 8.

Hence `process_factory`: it is called ONCE PER WORKER THREAD, and each worker
keeps whatever it returns for its lifetime. A caller that hands one shared
detector to every worker gets the 0.280 row.

THREE, NOT FOUR. Throughput still rises at 4, but per-pass latency rises with it
(2.48s -> 4.81s) and a doorway cares about latency: a person is in shot for a
few seconds and then gone. `cctv_v2_inference_workers` makes this benchmarkable
without a code change.

THERE IS NO QUEUE
-----------------
Workers PULL. A worker that is free asks for a camera and is handed that
camera's NEWEST frame at that instant; work is never pushed at a worker, and no
frame ever waits in line.

This is not a stylistic choice. A queue of frames would mean a worker starting
a 3.4s inference on a picture that was newest when it was enqueued and is stale
by the time it runs -- exactly the failure the frame-age cutoff exists to
prevent, reintroduced one layer up. If every worker is busy, the cameras go on
overwriting their slots and the next free worker picks up whatever is current.
Nothing accumulates.

A camera already being processed is excluded from selection, so three workers
cannot all pile onto camera 57.

THE RULE
--------
    score(camera) = role_priority x seconds_since_this_camera_was_last_served

The waiting term is time since SERVICE, not time since the frame was captured.
That distinction was found by a test and matters completely.

Frame age cannot schedule these cameras. They all run at 12fps, so every
camera's newest frame is ~0.08s old at every instant, always. Scoring on frame
age leaves all four effectively tied forever and the tie-break decides
everything: measured, camera 57 took 134 selections and camera 58 took 22, two
cameras of identical priority.

Time since service is the quantity that actually grows while a camera waits, so
it separates equals and self-corrects. Frame age is still recorded per selection
-- it is the right measure of how out-of-date the processed picture was -- but it
does not drive the choice.

Doorways carry a higher priority because a person crosses one in about two
seconds and is then gone, while a seated person in a room is still there on the
next pass.

Against the two cases that define the intended behaviour (priorities 3 and 1):

    doorway waited 0.1s -> 0.3    room waited 1.5s -> 1.5   room wins
    doorway waited 5.0s -> 15.0   room waited 1.0s -> 1.0   doorway wins

WHY THIS FAIRNESS MECHANISM
---------------------------
Of the obvious candidates -- age bonus, minimum service interval, round-robin
fallback, weighted fair queueing -- this is the least machinery for the
guarantee needed, because priority x staleness is ALREADY self-correcting:
staleness grows without bound, so a neglected room camera's score must
eventually overtake a doorway's. Round-robin would ignore urgency; weighted fair
queueing needs virtual-time bookkeeping to express what one multiplication
already says.

It has one weakness, and it is bounded rather than absent: a room camera needs
`priority_ratio` times the staleness of a doorway to win, so under sustained
doorway load it waits about 3x longer. `max_starvation_sec` therefore acts as a
hard deadline -- any camera unserved for that long is selected outright,
whatever the scores say. The score handles the normal case; the deadline turns
"eventually" into a number.

Ties break on camera id so a given set of inputs always yields the same choice.
A scheduler that reorders between runs cannot be tested, and cannot be trusted
when its behaviour is later questioned.

TWO AGES, TWO JOBS
------------------
Two different clocks are read here and conflating them has already caused one
bug each way:

    SERVICE AGE  time since this camera was last SERVED.
                 Answers "how long has it been waiting?"  -> drives the score.

    FRAME AGE    time since the frame in the slot was CAPTURED.
                 Answers "is this picture still true?"    -> drives eligibility.

Frame age must never become the score: all four cameras run at 12fps, so every
newest frame is ~0.08s old at every instant, which leaves the cameras
permanently tied and hands the decision to the tie-break -- measured, 134
selections to camera 57 against 22 for camera 58 at equal priority.

Service age must never become the eligibility test either: a camera that has
waited a long time is not thereby showing a valid picture. Camera 59 was killed
mid-run and kept getting selected on service age alone while the frame it was
offering aged to 71 seconds.

So a camera is first asked whether its picture is still true, and only the ones
that pass are ranked on how long they have waited.

SCOPE
-----
Selection, dispatch and measurement only. `process` is a callback: detection,
tracking, recognition and attendance are later steps and must not leak in here.

In particular the pool is deliberately ignorant of tracking. A worker may
process camera 57 then 59 then 58, which is safe precisely because YOLO
`predict` is stateless. When tracking arrives, TRACKER STATE MUST BELONG TO THE
CAMERA AND NEVER TO THE WORKER -- one ByteTrack instance per camera, looked up
by camera id, or a worker's previous camera will contaminate its next one.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from app.cctv_v2.capture.grabber import CameraGrabber, FrameSnapshot
from app.cctv_v2.config.cameras import profile_for, role_for

logger = logging.getLogger(__name__)

# Any camera unserved for this long is selected regardless of score. Turns the
# self-correcting-score argument into a bounded guarantee.
DEFAULT_MAX_STARVATION_SEC = 20.0

# How often to repeat the "still stale" line for a camera that stays dead.
_STALE_LOG_EVERY_SEC = 30.0

# A motion-gated camera whose picture is not moving has its score multiplied by
# this. Damped, NEVER zeroed: a camera must not be able to go blind because
# nothing happened to move in front of it, and the starvation deadline still
# applies unchanged.
#
# Measured: 80% of camera 57's passes and 66% of camera 58's saw nobody, while a
# frame-difference costs ~2ms against a ~4.9s YOLO pass. Damping idle doorways
# concentrates the pool on the camera that actually has somebody in front of it,
# which is what shortens the interval DURING a crossing -- the only interval
# that matters for catching one.
# 1.0 = DISABLED. Measured, and it did not work.
#
# The idea was sound: 80% of camera 57's passes and 66% of camera 58's saw
# nobody, and a frame-difference costs ~2ms against a ~4.9s YOLO pass, so
# ranking down an empty doorway should have concentrated the pool where somebody
# actually was.
#
# It regressed. A doorway has priority 3 and a room has priority 1, so any
# factor below 0.334 drops an IDLE DOORWAY BELOW A ROOM and the rooms absorb the
# freed capacity. Measured live at 0.25, the doorway interval went from 6.5s to
# 11.0s -- the opposite of the intent. Simulated across factors:
#
#     factor   idle doorway   active doorway
#      1.00        4.79s          4.79s
#      0.75        4.79s          4.79s      (no effect: 3*0.75 still > 1)
#      0.50        6.29s          3.97s      (idle doorways regress)
#      0.25        6.29s          3.97s
#
# There is no setting that helps the active doorway without hurting the idle
# ones, because the only place the capacity can come from is the other
# doorway or the rooms. The mechanism is kept, tested and off; making it pay
# needs cheaper doorway inference, not a different weight.
IDLE_SCORE_FACTOR = 1.0

# Motion above this fraction of changed pixels counts as "something is
# happening". Set an order of magnitude below what a person produces: a person
# crossing a doorway moved 11% of camera 58's pixels, an empty corridor 0.13%.
MOTION_THRESHOLD = 0.004

# Damping stops applying once a camera has gone this long unserved.
#
# Without it an idle doorway falls back on the 20s starvation deadline, which is
# WORSE than the 6.5s it gets today -- a 20s blind window on the camera whose
# whole job is catching a 2-second event. This bounds the cost of the motion
# hint being wrong: if a person is standing still at the doorway, or motion
# scoring is misbehaving, the camera is still looked at every 8s regardless.
MOTION_DAMP_MAX_SEC = 8.0

# Sleep when no camera has a frame. Short enough to pick one up promptly,
# long enough not to spin a core doing nothing.
_IDLE_SLEEP = 0.05


@dataclass(frozen=True)
class Selection:
    """One scheduling decision and what it cost. The audit trail for fairness."""

    camera_id: int
    role: str
    frame_sequence: int
    frame_timestamp: float
    selection_timestamp: float
    staleness_at_selection: float
    score: float
    reason: str                    # "score" | "starvation"
    inference_start: float = 0.0
    inference_end: float = 0.0
    worker: int = 0

    @property
    def inference_duration(self) -> float:
        return max(0.0, self.inference_end - self.inference_start)


@dataclass
class CameraStats:
    camera_id: int
    role: str
    selections: int = 0
    starvation_selections: int = 0
    stale_skips: int = 0           # times passed over for offering a dead picture
    idle_damped: int = 0           # times ranked down for showing no motion
    total_staleness: float = 0.0
    max_staleness: float = 0.0
    total_inference: float = 0.0
    last_served: float = field(default_factory=time.time)

    @property
    def avg_staleness(self) -> float:
        return self.total_staleness / self.selections if self.selections else 0.0

    @property
    def avg_inference(self) -> float:
        return self.total_inference / self.selections if self.selections else 0.0


class InferenceScheduler:
    """Owns the inference workers and decides which camera each one gets."""

    def __init__(
        self,
        grabbers: Iterable[CameraGrabber],
        process: Optional[Callable[[int, FrameSnapshot], None]] = None,
        max_starvation_sec: float = DEFAULT_MAX_STARVATION_SEC,
        clock: Callable[[], float] = time.time,
        workers: int = 1,
        process_factory: Optional[Callable[[], Callable[[int, FrameSnapshot], None]]] = None,
    ) -> None:
        """
        `process_factory` is called ONCE PER WORKER and the result is that
        worker's alone. Use it whenever the callback owns an expensive stateful
        resource -- a YOLO model, above all. `process` is the single-callback
        form: correct for one worker, and the measured anti-pattern for several
        (see the module docstring's shared-instance row).
        """
        if process is None and process_factory is None:
            raise ValueError("give either process or process_factory")

        self._grabbers = {g.camera_id: g for g in grabbers}
        self._process = process
        self._process_factory = process_factory
        self._max_starvation = float(max_starvation_sec)
        self._clock = clock
        self._worker_count = max(1, int(workers))

        if self._worker_count > 1 and process_factory is None:
            logger.warning(
                "scheduler: %d workers sharing ONE process callback. If it owns "
                "a YOLO model this is SLOWER than a single worker (measured "
                "0.280 vs 0.403 passes/s) -- pass process_factory instead.",
                self._worker_count,
            )

        self.stats: dict[int, CameraStats] = {}
        for cid in self._grabbers:
            role = role_for(cid)
            self.stats[cid] = CameraStats(camera_id=cid, role=role, last_served=clock())

        # Cameras currently being processed by some worker. Excluded from
        # selection so N workers cannot all pile onto the same camera.
        self._in_flight: set[int] = set()

        # Per-camera stale-logging state. A dead camera is re-examined every
        # _IDLE_SLEEP, so logging each skip would write ~20 lines/second and
        # bury the event that matters. Instead: one line when it goes stale, one
        # every _STALE_LOG_EVERY_SEC while it stays that way, one when it comes
        # back.
        self._stale_since: dict[int, float] = {}
        self._stale_logged_at: dict[int, float] = {}

        self.history: list[Selection] = []
        self._lock = threading.Lock()          # guards selection, stats, history
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._started_at = 0.0

    @property
    def worker_count(self) -> int:
        return self._worker_count

    # -- selection -----------------------------------------------------------
    def select_next(self) -> Optional[Selection]:
        """Choose the next camera. Runs no inference.

        Separated from the run loop so fairness can be tested deterministically
        with an injected clock and no threads.

        Not pure: it records stale skips and their log state, because "camera 59
        was passed over 40 times for offering a dead frame" is the only evidence
        that would explain an otherwise silent gap in that camera's coverage.
        Returning None leaves the caller to idle -- see `_run` -- so a fleet
        where every camera is stale waits rather than spinning.

        Callers holding `self._lock` get in-flight exclusion; `select_next` is
        also safe to call directly in tests, where nothing is in flight.
        """
        now = self._clock()
        candidates: list[tuple[float, int, FrameSnapshot, str]] = []

        for cid, grabber in sorted(self._grabbers.items()):
            if cid in self._in_flight:
                # Another worker is already looking at this camera. Giving it to
                # a second worker would spend two inferences on one scene while
                # another camera waits.
                continue

            snap = grabber.get_latest()
            if snap is None:
                # A camera with no frame is skipped, never waited for. One dead
                # camera must not stop the other three.
                continue

            frame_age = max(0.0, now - snap.timestamp)
            since_served = max(0.0, now - self.stats[cid].last_served)

            # Is the picture still true? Asked BEFORE anything else, including
            # the starvation deadline: a camera that has waited a long time has
            # earned a turn, but not the right to have a dead frame processed.
            # Letting starvation override this would reintroduce exactly the
            # behaviour being fixed, and on the camera least likely to recover.
            max_age = profile_for(cid).max_frame_age
            if frame_age > max_age:
                self.stats[cid].stale_skips += 1
                self._log_stale(cid, frame_age, max_age)
                continue

            self._clear_stale(cid)

            if since_served >= self._max_starvation:
                # Hard deadline. Ranked above every scored candidate by using a
                # score that cannot be reached normally, and ordered among
                # themselves by how long they have waited.
                candidates.append((float("inf") - 1.0 / (1.0 + since_served),
                                   cid, snap, "starvation"))
                continue

            profile = profile_for(cid)
            score = profile.priority * since_served

            # Damp a doorway that is showing an empty corridor. Rooms are never
            # damped -- a seated person barely moves, and gating them would let
            # occupancy decay exactly when the room is calm.
            if (profile.motion_gated
                    and getattr(snap, "motion", 0.0) < MOTION_THRESHOLD
                    and since_served < MOTION_DAMP_MAX_SEC):
                score *= IDLE_SCORE_FACTOR
                self.stats[cid].idle_damped += 1

            candidates.append((score, cid, snap, "score"))

        if not candidates:
            return None

        # Highest score wins; ties break on the LOWEST camera id so the choice
        # is reproducible.
        score, cid, snap, reason = max(candidates, key=lambda c: (c[0], -c[1]))
        return Selection(
            camera_id=cid,
            role=self.stats[cid].role,
            frame_sequence=snap.sequence,
            frame_timestamp=snap.timestamp,
            selection_timestamp=now,
            staleness_at_selection=max(0.0, now - snap.timestamp),   # frame age, reported not scored
            score=score,
            reason=reason,
        )

    def _log_stale(self, camera_id: int, frame_age: float, max_age: float) -> None:
        """Record that a camera was passed over, without flooding the log."""
        now = self._clock()
        first = camera_id not in self._stale_since
        if first:
            self._stale_since[camera_id] = now
        last = self._stale_logged_at.get(camera_id, 0.0)
        if first or (now - last) >= _STALE_LOG_EVERY_SEC:
            self._stale_logged_at[camera_id] = now
            logger.warning(
                "SCHED-STALE camera=%s role=%s frame_age=%.1fs > max %.1fs - "
                "skipped, stale for %.0fs (stream stopped?)",
                camera_id, self.stats[camera_id].role, frame_age, max_age,
                now - self._stale_since[camera_id],
            )

    def _clear_stale(self, camera_id: int) -> None:
        """A fresh frame arrived. Recovery is automatic and needs no reset.

        The camera re-enters normal scheduling with its existing `last_served`
        untouched, so the time it spent stale counts as waiting and it is picked
        up promptly rather than having to earn its turn again from zero.
        """
        if camera_id in self._stale_since:
            logger.info(
                "SCHED-FRESH camera=%s recovered after %.0fs stale; "
                "eligible again", camera_id, self._clock() - self._stale_since[camera_id],
            )
            self._stale_since.pop(camera_id, None)
            self._stale_logged_at.pop(camera_id, None)

    # -- dispatch ------------------------------------------------------------
    def _claim(self) -> Optional[tuple[Selection, FrameSnapshot]]:
        """Reserve a camera and take its NEWEST frame, atomically.

        The frame is re-read here rather than carried over from `select_next`,
        so a worker always starts on what is current at the moment it actually
        begins -- never on what was current when the decision was made. That
        re-read can only ever yield a fresher frame, so it cannot smuggle a
        stale one past the eligibility check.
        """
        with self._lock:
            sel = self.select_next()
            if sel is None:
                return None
            snap = self._grabbers[sel.camera_id].get_latest()
            if snap is None:
                return None
            self._in_flight.add(sel.camera_id)
            return sel, snap

    def _release(self, camera_id: int) -> None:
        with self._lock:
            self._in_flight.discard(camera_id)

    def run_once(self, process=None, worker: int = 0) -> Optional[Selection]:
        """Select one camera, process its newest frame, record what it cost."""
        claimed = self._claim()
        if claimed is None:
            return None
        sel, snap = claimed

        callback = process or self._process
        start = self._clock()
        try:
            if callback is not None:
                callback(sel.camera_id, snap)
        except Exception:                                       # noqa: BLE001
            # One camera's processing failure must not stop the scheduler; the
            # other three are still waiting for their turn. Nor may it leave the
            # camera reserved -- hence the finally below.
            logger.exception(
                "scheduler: processing failed camera=%s seq=%s worker=%s",
                sel.camera_id, snap.sequence, worker,
            )
        finally:
            self._release(sel.camera_id)
        end = self._clock()

        done = Selection(
            camera_id=sel.camera_id, role=sel.role,
            frame_sequence=snap.sequence, frame_timestamp=snap.timestamp,
            selection_timestamp=sel.selection_timestamp,
            staleness_at_selection=sel.staleness_at_selection,
            score=sel.score, reason=sel.reason,
            inference_start=start, inference_end=end, worker=worker,
        )
        self._record(done)
        return done

    def _record(self, sel: Selection) -> None:
        with self._lock:
            st = self.stats[sel.camera_id]
            st.selections += 1
            st.starvation_selections += int(sel.reason == "starvation")
            st.total_staleness += sel.staleness_at_selection
            st.max_staleness = max(st.max_staleness, sel.staleness_at_selection)
            st.total_inference += sel.inference_duration
            st.last_served = sel.inference_end or self._clock()
            self.history.append(sel)

        logger.info(
            "SCHED camera=%s role=%s seq=%s stale=%.2fs score=%.2f via=%s "
            "infer=%.2fs worker=%s",
            sel.camera_id, sel.role, sel.frame_sequence,
            sel.staleness_at_selection, sel.score, sel.reason,
            sel.inference_duration, sel.worker,
        )

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._threads and any(t.is_alive() for t in self._threads):
            return
        self._stop.clear()
        self._started_at = self._clock()
        self._threads = [
            threading.Thread(target=self._run, args=(i,),
                             name=f"cctv-infer-{i}", daemon=True)
            for i in range(self._worker_count)
        ]
        for t in self._threads:
            t.start()
        logger.info(
            "CCTV v2 scheduler started: %d camera(s), %d inference worker(s) "
            "with independent models, priority x staleness, starvation "
            "deadline %.0fs",
            len(self._grabbers), self._worker_count, self._max_starvation,
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout)

    def _run(self, worker: int = 0) -> None:
        """One worker: build its OWN callback, then pull work until stopped.

        The callback is built here, inside the thread, so a factory that loads a
        model produces one model per worker rather than one shared between them.
        A worker whose callback cannot be built stops rather than spinning --
        the others carry on.
        """
        try:
            callback = self._process_factory() if self._process_factory else self._process
        except Exception:                                       # noqa: BLE001
            logger.exception(
                "scheduler: worker %s could not build its processor; "
                "this worker is disabled, the others continue", worker,
            )
            return

        while not self._stop.is_set():
            if self.run_once(process=callback, worker=worker) is None:
                self._stop.wait(_IDLE_SLEEP)

    # -- metrics -------------------------------------------------------------
    def summary(self) -> dict:
        """What the hardware actually did -- not what the profiles asked for.

        `requested_interval` is the profile's target and `actual_interval` is
        measured. Reporting both is the point: V1 slept for the target and then
        queued on a lock, which made a 0.12s request look satisfied while the
        real cadence was 4.5-9s.
        """
        elapsed = max(1e-9, self._clock() - self._started_at) if self._started_at else 0.0
        total = sum(s.selections for s in self.stats.values())
        cams = {}
        for cid, s in sorted(self.stats.items()):
            cams[cid] = {
                "role": s.role,
                "selections": s.selections,
                "starvation_selections": s.starvation_selections,
                "stale_skips": s.stale_skips,
                "idle_damped": s.idle_damped,
                "motion_gated": profile_for(cid).motion_gated,
                "max_frame_age": profile_for(cid).max_frame_age,
                "avg_staleness": round(s.avg_staleness, 3),
                "max_staleness": round(s.max_staleness, 3),
                "avg_inference": round(s.avg_inference, 3),
                "requested_interval": profile_for(cid).analysis_interval_target,
                "actual_interval": round(elapsed / s.selections, 3) if s.selections else None,
            }
        return {
            "elapsed_sec": round(elapsed, 2),
            "workers": self._worker_count,
            "total_selections": total,
            "selections_per_sec": round(total / elapsed, 3) if elapsed else 0.0,
            "cameras": cams,
        }
