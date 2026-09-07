"""In-process counters for the CCTV pipeline's OUTCOMES.

Why this exists
---------------
Every number in the September 2026 production audit was obtained by grepping
110 MB of log text. That is not an operational capability: it cannot be alerted
on, it cannot be graphed, and it cannot answer a question at 9am on a Tuesday
when somebody says "it didn't mark me in".

The pipeline already measures its own TIMING well (bytetrack_engine._perf, the
two inference gates' stats(), CameraRuntimeState). What it has never counted is
what it DECIDED:

  * how many gate evaluations ended in each reason,
  * how big the faces actually were,
  * how many attendance writes succeeded, retried, or were lost.

Those three are the difference between "the cameras are running" and "the
cameras are working", and all three previously existed only as log lines.

Scope
-----
Counters and histograms only. This module deliberately does NOT:

  * own timing — `bytetrack_engine` already does, next to the code being timed;
  * persist anything — a restart resets to zero, which is correct for a
    process-local gauge and keeps the write path free of I/O;
  * sample or aggregate on the hot path — the endpoint does that on demand.

Design notes
------------
Counts are keyed by (camera_id, reason) so a single bad camera is visible
rather than averaged away — the audit's central finding was that per-camera
behaviour diverges sharply, and a global total hides exactly that.

Bounded by construction: the key space is (cameras x reasons), both small and
fixed. Nothing here grows with traffic, so it cannot become the leak it is
meant to help find.

Not persisted, not exported anywhere on the hot path — see
`app/api/routes/metrics.py` for the read side.
"""
from __future__ import annotations

import threading
from collections import defaultdict

# Face-size buckets, in pixels of detected face HEIGHT.
#
# The edges are the system's own thresholds, not round numbers:
#   28  — _ATTENDANCE_DEFAULTS.limits.min_face_px, the quality floor
#   70  — good_face_px, where the quality score stops being penalised
#   112 — ArcFace's native input; anything below this is upscaled interpolation
# Measured median at the time of writing was 33 px, i.e. one pixel above the
# floor and well under half of "good". The whole point of this histogram is to
# make that visible without grepping, and to show T-10 (re-aim the cameras)
# moving it.
_FACE_PX_EDGES = (28.0, 40.0, 55.0, 70.0, 112.0)
_FACE_PX_LABELS = ("<28", "28-40", "40-55", "55-70", "70-112", ">=112")


def _bucket_for(value: float) -> str:
    for index, edge in enumerate(_FACE_PX_EDGES):
        if value < edge:
            return _FACE_PX_LABELS[index]
    return _FACE_PX_LABELS[-1]


# Person-box HEIGHT buckets, in pixels.
#
# Edges are this system's own numbers, not round ones:
#   30  — _MIN_PERSON_PX, the size floor that drops a box outright
#   46  — the smallest REAL person in the 41 hand-labelled examples the floor
#         was chosen from, so 30-46 is the band where the floor is closest to
#         rejecting somebody genuine
#   80  — roughly a person at the far door on the doorway cameras
# The first bucket can only ever be populated by the size-floor rejection path;
# if it stays empty, the floor is costing nothing.
_BOX_PX_EDGES = (30.0, 46.0, 80.0, 150.0, 300.0)
_BOX_PX_LABELS = ("<30", "30-46", "46-80", "80-150", "150-300", ">=300")

# Where in the frame the detection sat, as a fraction of frame height.
#
# On a corridor camera this is a distance proxy: the far door is at the top of
# the frame and the near floor at the bottom, so a funnel split by this band
# answers "are we losing the DISTANT ones?" — which is the actual question and
# was previously unanswerable, because nothing recorded where a lost detection
# had been.
_POS_LABELS = ("y0.0-0.2", "y0.2-0.4", "y0.4-0.6", "y0.6-0.8", "y0.8-1.0")

# What happened to one raw YOLO detection. These are exhaustive and mutually
# exclusive: every box the detector emits lands in exactly one, so the counts
# sum to the number of detections and the funnel actually balances.
FUNNEL_OUTCOMES = (
    "tracked_by_id",        # ByteTrack had already confirmed it
    "adopted",              # no id, but confident enough to adopt as provisional
    "dropped_size_floor",   # box shorter than _MIN_PERSON_PX
    "dropped_adopt_bar",    # no id and score below the adopt threshold
)


def _box_bucket(value: float) -> str:
    for index, edge in enumerate(_BOX_PX_EDGES):
        if value < edge:
            return _BOX_PX_LABELS[index]
    return _BOX_PX_LABELS[-1]


def _pos_bucket(fraction: float) -> str:
    index = int(max(0.0, min(0.999, fraction)) * 5)
    return _POS_LABELS[index]


class PipelineMetrics:
    """Thread-safe outcome counters. One instance per process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (camera_id, reason) -> count
        self._decisions: dict[tuple[str, str], int] = defaultdict(int)
        # (camera_id, bucket) -> count
        self._face_px: dict[tuple[str, str], int] = defaultdict(int)
        # (camera_id, action) -> count
        self._attendance: dict[tuple[str, str], int] = defaultdict(int)
        # --- detection funnel (Fix 1) --------------------------------------
        # (camera_id, outcome) -> count
        self._funnel: dict[tuple[str, str], int] = defaultdict(int)
        # (camera_id, outcome, box-height bucket) -> count
        self._funnel_px: dict[tuple[str, str, str], int] = defaultdict(int)
        # (camera_id, outcome, frame-position bucket) -> count
        self._funnel_pos: dict[tuple[str, str, str], int] = defaultdict(int)

    # -- write side (called from the pipeline) -------------------------------
    def record_decision(
        self, camera_id, reason: str, allowed: bool, face_px: float | None = None
    ) -> None:
        """One attendance-gate outcome.

        Called from `camera_service._log_decision`, which already de-duplicates
        by (track, reason, allowed) — so this counts DISTINCT OUTCOMES, not
        re-evaluations. That is the intended unit: a track that sits in
        `already_marked` for two hundred passes is one outcome, not two hundred,
        and counting passes would bury every other reason under it.
        """
        camera = str(camera_id)
        key = (camera, str(reason or ("allowed" if allowed else "unknown")))
        with self._lock:
            self._decisions[key] += 1
            if face_px is not None:
                try:
                    self._face_px[(camera, _bucket_for(float(face_px)))] += 1
                except (TypeError, ValueError):
                    pass

    def record_attendance_write(self, camera_id, action: str) -> None:
        """One attendance write attempt's outcome.

        `action` is whatever `mark_cctv_attendance` returned, plus the synthetic
        terminal states this module cares about most: `lost` (retries exhausted
        — a payroll event that needs manual entry) and `dropped` (queue
        overflow). Both were previously visible only as an ERROR line nobody
        was watching.
        """
        with self._lock:
            self._attendance[(str(camera_id), str(action or "unknown"))] += 1

    def record_detection(
        self,
        camera_id,
        outcome: str,
        box_height_px: float | None = None,
        centroid_y_frac: float | None = None,
    ) -> None:
        """One raw YOLO detection and what became of it.

        This is the measurement the September investigation could not make. Two
        of the three ways a person is lost before tracking were bare `continue`
        statements with no log and no counter, so "the detector never saw them"
        and "we threw them away" were indistinguishable from outside — and they
        need opposite fixes.

        `outcome` must be one of FUNNEL_OUTCOMES; they are exhaustive and
        mutually exclusive, so the counts sum to the detection total and the
        funnel balances. `box_height_px` and `centroid_y_frac` are optional so
        a caller that cannot compute them still records the outcome.

        Records only. Nothing here changes what the pipeline does with the
        detection.
        """
        camera = str(camera_id)
        key = str(outcome or "unknown")
        with self._lock:
            self._funnel[(camera, key)] += 1
            if box_height_px is not None:
                try:
                    self._funnel_px[(camera, key, _box_bucket(float(box_height_px)))] += 1
                except (TypeError, ValueError):
                    pass
            if centroid_y_frac is not None:
                try:
                    self._funnel_pos[(camera, key, _pos_bucket(float(centroid_y_frac)))] += 1
                except (TypeError, ValueError):
                    pass

    # -- read side (called from the metrics endpoint) ------------------------
    def snapshot(self) -> dict:
        """Current counts, grouped per camera. Cheap; safe to call anytime."""
        with self._lock:
            decisions = dict(self._decisions)
            face_px = dict(self._face_px)
            attendance = dict(self._attendance)
            funnel = dict(self._funnel)
            funnel_px = dict(self._funnel_px)
            funnel_pos = dict(self._funnel_pos)

        return {
            "decisions": _group(decisions),
            "decisions_total": _totals(decisions),
            "face_px_histogram": _group(face_px, order=_FACE_PX_LABELS),
            "face_px_total": _totals(face_px, order=_FACE_PX_LABELS),
            "attendance_writes": _group(attendance),
            "attendance_writes_total": _totals(attendance),
            # Detection funnel: what happened to every box YOLO emitted.
            # `detection_funnel_total.detections` is the sum of the outcomes, so
            # a mismatch means a code path is not reporting itself.
            "detection_funnel": _group(funnel, order=FUNNEL_OUTCOMES),
            "detection_funnel_total": _funnel_totals(funnel),
            "detection_box_px": _group3(funnel_px, order=_BOX_PX_LABELS),
            "detection_position": _group3(funnel_pos, order=_POS_LABELS),
        }

    def reset(self) -> None:
        """Testing only. Never called by the application."""
        with self._lock:
            self._decisions.clear()
            self._face_px.clear()
            self._attendance.clear()
            self._funnel.clear()
            self._funnel_px.clear()
            self._funnel_pos.clear()


def _group(counts: dict[tuple[str, str], int], order: tuple[str, ...] = ()) -> dict:
    """{(camera, key): n} -> {camera: {key: n}}, keys in `order` first."""
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for (camera, key), value in counts.items():
        grouped[camera][key] = value
    return {
        camera: _ordered(entries, order)
        for camera, entries in sorted(grouped.items())
    }


def _totals(counts: dict[tuple[str, str], int], order: tuple[str, ...] = ()) -> dict:
    """Same counts, summed across cameras."""
    totals: dict[str, int] = defaultdict(int)
    for (_camera, key), value in counts.items():
        totals[key] += value
    return _ordered(totals, order)


def _group3(counts: dict[tuple[str, str, str], int], order: tuple[str, ...] = ()) -> dict:
    """{(camera, outcome, bucket): n} -> {camera: {outcome: {bucket: n}}}."""
    grouped: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for (camera, outcome, bucket), value in counts.items():
        grouped[camera][outcome][bucket] = value
    return {
        camera: {
            outcome: _ordered(buckets, order)
            for outcome, buckets in sorted(outcomes.items())
        }
        for camera, outcomes in sorted(grouped.items())
    }


def _funnel_totals(counts: dict[tuple[str, str], int]) -> dict:
    """Outcome totals across cameras, plus the detection count they sum to.

    `detections` is derived rather than counted separately on purpose: if it
    ever disagrees with the sum of the outcomes, a code path is dropping a
    detection without reporting itself, and that is precisely the class of bug
    this funnel exists to expose.
    """
    totals = _totals(counts, order=FUNNEL_OUTCOMES)
    detections = sum(totals.values())
    dropped = sum(
        value for key, value in totals.items() if key.startswith("dropped_")
    )
    return {
        "detections": detections,
        **totals,
        "dropped_total": dropped,
        "dropped_pct": round(100.0 * dropped / detections, 1) if detections else 0.0,
    }


def _ordered(entries: dict[str, int], order: tuple[str, ...]) -> dict:
    """Histogram buckets read in bucket order; everything else alphabetically.

    A histogram whose buckets arrive in hash order is unreadable, and reading
    these by eye in a JSON response is the primary use.
    """
    if not order:
        return dict(sorted(entries.items()))
    ranked = sorted(
        entries.items(),
        key=lambda kv: (order.index(kv[0]) if kv[0] in order else len(order), kv[0]),
    )
    return dict(ranked)


# Process-wide singleton, mirroring camera_service.camera_manager.
metrics = PipelineMetrics()
