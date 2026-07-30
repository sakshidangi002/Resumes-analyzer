"""Who is currently inside the building.

Used to shrink the face-matching candidate pool. With ~50 enrolled employees,
matching every face against all 50 is a 50-way problem all day. But at an IN
camera, anyone already checked in is not a plausible candidate; at an OUT
camera, anyone who never checked in is not either. By mid-morning that turns a
50-way problem into a 15-way one.

That matters more than it sounds. The matcher accepts a result only when the
best candidate beats the runner-up by `min_match_margin` — and on a low-
resolution face the runner-up is often another employee at a near-identical
score. Removing candidates who *cannot* be the person removes exactly those
spurious runners-up, so the margin gate starts doing its job.

Cached with a short TTL because the recognition loop asks constantly and the
answer changes only when somebody walks through a door.
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Short enough that a check-in is reflected almost immediately, long enough that
# a busy entrance does not re-query per frame.
_TTL_SECONDS = float(os.getenv("PRESENCE_CACHE_TTL", "15"))

_lock = threading.Lock()
_cached: set[int] | None = None
_cached_at: float = 0.0
_cached_day = None


def _load_present_ids() -> set[int]:
    """Employee ids whose LAST event today is a work-start (i.e. inside)."""
    from app.core.datetime_utils import get_ist_now
    from app.db.session import SessionLocal
    from app.models import AttendanceEvent
    from app.services.attendance_event_service import business_date, current_state

    day = business_date(get_ist_now())
    with SessionLocal() as db:
        rows = (
            db.query(
                AttendanceEvent.employee_id,
                AttendanceEvent.event_type,
            )
            .filter(AttendanceEvent.attendance_date == day)
            .order_by(
                AttendanceEvent.employee_id.asc(),
                AttendanceEvent.event_time.asc(),
                AttendanceEvent.id.asc(),
            )
            .all()
        )

    # Ordered ascending, so the last row per employee wins. A day's events are
    # small (tens to low hundreds), so resolving in Python avoids a
    # Postgres-specific DISTINCT ON.
    last_type: dict[int, str] = {}
    for employee_id, event_type in rows:
        last_type[int(employee_id)] = event_type

    return {
        emp_id for emp_id, event_type in last_type.items()
        if current_state(event_type) == "WORKING"
    }


def get_present_employee_ids() -> set[int]:
    """Employees currently inside. Returns a COPY; never raises.

    On any failure returns an empty set, which callers must treat as "unknown"
    and fall back to the full candidate pool — never as "nobody is inside",
    which would silently block every OUT-camera match.
    """
    global _cached, _cached_at, _cached_day
    from app.core.datetime_utils import get_ist_now
    from app.services.attendance_event_service import business_date

    now = time.time()
    try:
        today = business_date(get_ist_now())
    except Exception:
        return set()

    with _lock:
        fresh = (
            _cached is not None
            and _cached_day == today
            and (now - _cached_at) < _TTL_SECONDS
        )
        if fresh:
            return set(_cached)

    try:
        present = _load_present_ids()
    except Exception:
        logger.exception("presence lookup failed — falling back to all candidates")
        return set()

    with _lock:
        _cached, _cached_at, _cached_day = present, now, today
    return set(present)


def invalidate() -> None:
    """Drop the cache. Call after writing an attendance event so the next
    recognition sees the new state without waiting out the TTL."""
    global _cached, _cached_at
    with _lock:
        _cached, _cached_at = None, 0.0
