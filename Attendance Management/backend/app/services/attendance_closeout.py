"""Close attendance days that the OUT camera never closed.

Cameras miss departures routinely — back of the head, tailgating, a steep
mounting angle. When that happens the day is left with a work-start as its last
event, and `calculate_intervals_from_events` keeps accruing an open interval
from that event until midnight. The day then freezes with `sign_out_time = NULL`
and an inflated `total_work_hours` that nobody ever corrects.

Nothing in the application closed those days: the only scheduled job was the DSR
reminder.

Design decisions worth keeping:

* Close at the LAST SEEN time, never at the cutoff. Crediting a full day to
  someone the camera stopped seeing at 14:00 silently inflates payroll. Capping
  at last-seen under-reports instead — the safe direction, and visibly wrong, so
  HR notices and corrects it.
* Mark the synthetic event `source="AUTO_CLOSE"` and the record
  `source="CORRECTION"`. Every row this job writes is then identifiable and
  reversible with a single DELETE, and the day surfaces in the UI as needing
  review rather than masquerading as a clean camera read.
* Idempotent. Re-running never double-closes: a day whose last event is already
  a work-end, or is our own AUTO_CLOSE event, is skipped.
* Dry-run by default is NOT the behaviour here, but `run_closeout(commit=False)`
  reports what it would do — use it for the first week in production.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta

from app.core.config import get_settings
from app.core.datetime_utils import get_ist_now
from app.db.session import SessionLocal
from app.models import AttendanceEvent
from app.services.attendance_event_service import (
    business_date,
    current_state,
    recalculate_attendance_summary,
    to_naive_ist,
)

logger = logging.getLogger(__name__)

CLOSEOUT_SOURCE = "AUTO_CLOSE"


def find_open_days(db, target_day: date) -> list[tuple[int, AttendanceEvent]]:
    """Employees whose `target_day` is still open, with their last event."""
    employee_ids = [
        row[0]
        for row in db.query(AttendanceEvent.employee_id)
        .filter(AttendanceEvent.attendance_date == target_day)
        .distinct()
        .all()
    ]

    open_days: list[tuple[int, AttendanceEvent]] = []
    for employee_id in employee_ids:
        last = (
            db.query(AttendanceEvent)
            .filter(
                AttendanceEvent.employee_id == employee_id,
                AttendanceEvent.attendance_date == target_day,
            )
            .order_by(AttendanceEvent.event_time.desc(), AttendanceEvent.id.desc())
            .first()
        )
        if last is None:
            continue
        if current_state(last.event_type) != "WORKING":
            continue                      # already closed
        if last.source == CLOSEOUT_SOURCE:
            continue                      # our own event — idempotency guard
        open_days.append((employee_id, last))
    return open_days


def _close_at(last_event: AttendanceEvent, target_day: date, cutoff_hour: int) -> datetime:
    """When to record the synthetic departure.

    The last time the employee was actually seen, capped at the cutoff. Never
    the cutoff itself when the camera stopped seeing them earlier — that would
    invent hours they did not work.
    """
    return min(
        to_naive_ist(last_event.event_time),
        datetime.combine(target_day, time(cutoff_hour, 0)),
    )


def run_closeout(now: datetime | None = None, commit: bool = True) -> int:
    """Close every still-open attendance day for the previous business day.

    Returns the number of days closed (or, with commit=False, the number that
    WOULD be closed). Safe to run repeatedly.
    """
    settings = get_settings()
    cutoff_hour = int(getattr(settings, "attendance_closeout_hour", 23))
    now = to_naive_ist(now or get_ist_now())

    # The previous business day. Running at 02:30 with a 05:00 day-start means
    # "now" is still inside the previous business day, so step back a full day
    # from that — never close a shift that is legitimately still running.
    target_day = business_date(now) - timedelta(days=1)

    closed = 0
    with SessionLocal() as db:
        for employee_id, last_event in find_open_days(db, target_day):
            close_at = _close_at(last_event, target_day, cutoff_hour)

            if not commit:
                logger.info(
                    "AUTO-CLOSE (dry run) employee_id=%s day=%s would_close_at=%s",
                    employee_id, target_day, close_at.isoformat(),
                )
                closed += 1
                continue

            try:
                db.add(
                    AttendanceEvent(
                        employee_id=employee_id,
                        attendance_record_id=last_event.attendance_record_id,
                        attendance_date=target_day,
                        event_time=close_at,
                        event_type="OUT",
                        source=CLOSEOUT_SOURCE,
                        camera_id=None,
                    )
                )
                db.flush()
                record = recalculate_attendance_summary(db, employee_id, target_day)
                # Surfaces in the UI as an edited/needs-review day rather than a
                # clean camera read.
                record.source = "CORRECTION"
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(
                    "AUTO-CLOSE FAILED employee_id=%s day=%s", employee_id, target_day
                )
                continue

            closed += 1
            logger.warning(
                "AUTO-CLOSE employee_id=%s day=%s closed_at=%s "
                "(OUT camera missed the departure — needs HR review)",
                employee_id, target_day, close_at.isoformat(),
            )

    if closed:
        logger.warning(
            "Attendance closeout: %d day(s) %s for %s",
            closed, "closed" if commit else "would be closed", target_day,
        )
    else:
        logger.info("Attendance closeout: nothing open for %s", target_day)
    return closed
