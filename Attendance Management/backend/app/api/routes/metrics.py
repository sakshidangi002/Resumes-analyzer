"""Operational metrics for the CCTV pipeline.

Every quantitative claim in the September 2026 production audit came from
grepping 110 MB of rotating log files. That is not something anyone can alert
on, graph, or answer a question with while a person is standing at the gate
saying they were not marked in. This endpoint exposes the same numbers as JSON.

It is a READ of state the pipeline already keeps. It starts no work, takes no
frame lock, touches no database and runs no inference — deliberately, because
the thing being measured is a system with no spare CPU, and a metrics endpoint
that competes with the cameras would distort what it reports.

Admin-only: the payload describes camera behaviour, employee recognition rates
and infrastructure timing.

Deliberately NOT Prometheus. Adding a client library and an exposition format
would be a new monitoring dependency for four cameras on one host, and the
shape below stays flat enough to export later if that ever changes.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends

from app.api.deps import require_roles
from app.models import User

logger = logging.getLogger(__name__)

router = APIRouter()

_STARTED_AT = time.time()


def _safe(label: str, fn, default):
    """Call a metrics source; never let one broken source empty the response.

    A metrics endpoint that 500s because one counter is unavailable tells the
    operator nothing about the other nine.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001
        logger.debug("metrics: %s unavailable", label, exc_info=True)
        return default


@router.get("/metrics", tags=["metrics"])
def get_metrics(current_user: User = Depends(require_roles(["Admin"]))):
    """Pipeline metrics: inference timing, gate contention, decisions, writes.

    What to look at first, in order:

    * ``cameras.<id>.wait_share_pct`` — share of each camera's cycle spent
      QUEUEING rather than inferring. Measured at 54-76% in the audit. Anything
      above ~40% means cameras are starving each other, not working.
    * ``cameras.<id>.cycle_p90_ms`` — a person crosses a doorway in about 2,000
      ms. A p90 above that means crossings are being missed between passes, and
      no recognition tuning can recover them.
    * ``pipeline.decisions_total`` — where attendance decisions actually die.
      ``insufficient_observations`` dominating is a scheduling problem, not a
      recognition one.
    * ``pipeline.face_px_histogram`` — ArcFace wants 112 px. A distribution
      piled up below 40 px is an optics problem no threshold can fix.
    * ``pipeline.attendance_writes_total.lost`` — every count is a payroll
      event that needs manual entry. Should be zero.
    """
    from app.services import bytetrack_engine
    from app.services import pipeline_metrics
    from app.services.camera_service import camera_manager
    from app.services.inference_gate import get_gate

    return {
        "uptime_sec": round(time.time() - _STARTED_AT, 1),
        # Per-camera inference timing and percentiles. `cycle` is wait +
        # inference for one pass: how long the camera took to be served once.
        "cameras": _safe("perf", bytetrack_engine.get_perf_stats, {}),
        # Admission control. `high_avg_wait_ms` near zero is the goal — a
        # doorway camera should not be queueing behind a room camera.
        "gates": {
            "yolo": _safe("yolo gate", bytetrack_engine.gate_stats, {}),
            "face": _safe("face gate", lambda: get_gate().stats(), {}),
        },
        # Outcome counters (app/services/pipeline_metrics.py).
        "pipeline": _safe("pipeline counters", pipeline_metrics.metrics.snapshot, {}),
        # Stream-level state, reused verbatim from the camera manager.
        "streams": _safe("camera stats", camera_manager.get_stats, {}),
        # Liveness/staleness, shared with /health (T-01).
        "health": _safe("health snapshot", camera_manager.health_snapshot, {}),
    }
