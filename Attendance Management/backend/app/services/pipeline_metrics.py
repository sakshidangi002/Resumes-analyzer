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

    # -- read side (called from the metrics endpoint) ------------------------
    def snapshot(self) -> dict:
        """Current counts, grouped per camera. Cheap; safe to call anytime."""
        with self._lock:
            decisions = dict(self._decisions)
            face_px = dict(self._face_px)
            attendance = dict(self._attendance)

        return {
            "decisions": _group(decisions),
            "decisions_total": _totals(decisions),
            "face_px_histogram": _group(face_px, order=_FACE_PX_LABELS),
            "face_px_total": _totals(face_px, order=_FACE_PX_LABELS),
            "attendance_writes": _group(attendance),
            "attendance_writes_total": _totals(attendance),
        }

    def reset(self) -> None:
        """Testing only. Never called by the application."""
        with self._lock:
            self._decisions.clear()
            self._face_px.clear()
            self._attendance.clear()


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
