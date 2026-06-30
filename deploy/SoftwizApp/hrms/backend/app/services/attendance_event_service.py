"""Biometric attendance events: IN/OUT toggling, cooldown, daily summary recalculation."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal

from sqlalchemy.orm import Session

from app.core.datetime_utils import get_ist_now
from app.models import AttendanceEvent, AttendanceRecord
from app.services.attendance_service import (
    apply_status_from_hours,
    calculate_work_hours,
    get_company_config,
    get_or_create_attendance,
)

EVENT_COOLDOWN_SECONDS = 60


def _day_bounds(d: date) -> tuple[datetime, datetime]:
    start = datetime.combine(d, time.min)
    end = datetime.combine(d, time.max)
    return start, end


def get_events_for_day(db: Session, employee_id: int, d: date) -> list[AttendanceEvent]:
    start, end = _day_bounds(d)
    return (
        db.query(AttendanceEvent)
        .filter(
            AttendanceEvent.employee_id == employee_id,
            AttendanceEvent.event_time >= start,
            AttendanceEvent.event_time <= end,
        )
        .order_by(AttendanceEvent.event_time.asc(), AttendanceEvent.id.asc())
        .all()
    )


def get_latest_event_for_day(db: Session, employee_id: int, d: date) -> AttendanceEvent | None:
    start, end = _day_bounds(d)
    return (
        db.query(AttendanceEvent)
        .filter(
            AttendanceEvent.employee_id == employee_id,
            AttendanceEvent.event_time >= start,
            AttendanceEvent.event_time <= end,
        )
        .order_by(AttendanceEvent.event_time.desc(), AttendanceEvent.id.desc())
        .first()
    )


def is_within_event_cooldown(
    db: Session,
    employee_id: int,
    now_dt: datetime,
    cooldown_seconds: int = EVENT_COOLDOWN_SECONDS,
) -> bool:
    """Return True only if the most recent event for THIS day is within the cooldown window.

    Scoping to the current day prevents a late-night event from blocking
    the employee's first check-in of the following morning.
    """
    day_start, day_end = _day_bounds(now_dt.date())
    latest = (
        db.query(AttendanceEvent)
        .filter(
            AttendanceEvent.employee_id == employee_id,
            AttendanceEvent.event_time >= day_start,
            AttendanceEvent.event_time <= day_end,
        )
        .order_by(AttendanceEvent.event_time.desc(), AttendanceEvent.id.desc())
        .first()
    )
    if latest is None:
        return False
    return (now_dt - latest.event_time).total_seconds() < cooldown_seconds


def determine_next_event_type(last_event: AttendanceEvent | None) -> str:
    if last_event is None:
        return "IN"
    if last_event.event_type == "IN":
        return "OUT"
    return "IN"


def calculate_intervals_from_events(
    events: list[AttendanceEvent],
) -> tuple[Decimal | None, Decimal | None, time | None, time | None]:
    """Return total work hours, break hours, first IN time, last OUT time."""
    if not events:
        return None, None, None, None

    sorted_events = sorted(events, key=lambda e: (e.event_time, e.id))
    in_events = [e for e in sorted_events if e.event_type == "IN"]
    out_events = [e for e in sorted_events if e.event_type == "OUT"]

    first_in = in_events[0].event_time.time() if in_events else None
    last_out = out_events[-1].event_time.time() if out_events else None

    total_work_seconds = 0
    total_break_seconds = 0
    for i in range(len(sorted_events) - 1):
        cur = sorted_events[i]
        nxt = sorted_events[i + 1]
        delta = int((nxt.event_time - cur.event_time).total_seconds())
        if delta <= 0:
            continue
        if cur.event_type == "IN" and nxt.event_type == "OUT":
            total_work_seconds += delta
        elif cur.event_type == "OUT" and nxt.event_type == "IN":
            total_break_seconds += delta

    work_hours = Decimal(round(total_work_seconds / 3600, 2)) if total_work_seconds else Decimal("0")
    break_hours = Decimal(round(total_break_seconds / 3600, 2)) if total_break_seconds else Decimal("0")
    return work_hours, break_hours, first_in, last_out


def _apply_late_and_early(db: Session, rec: AttendanceRecord) -> None:
    config = get_company_config(db)
    grace_min = config.grace_time_minutes if config else 15
    standard_start = time(9, 0)
    standard_end = time(18, 0)

    if rec.sign_in_time:
        t = datetime.combine(rec.date, rec.sign_in_time)
        s = datetime.combine(rec.date, standard_start)
        rec.is_late = (t - s).total_seconds() > grace_min * 60
    else:
        rec.is_late = False

    if rec.sign_out_time:
        t = datetime.combine(rec.date, rec.sign_out_time)
        s = datetime.combine(rec.date, standard_end)
        rec.is_early_exit = t < s
    else:
        rec.is_early_exit = False


def recalculate_attendance_summary(db: Session, employee_id: int, d: date) -> AttendanceRecord:
    """Rebuild daily attendance record from events (or legacy sign-in/out).

    Guarantee: sign_in_time is always the EARLIEST IN event of the day
    (first_in from calculate_intervals_from_events). It is never overwritten
    by later detections because it is derived from all events sorted
    ascending — the oldest IN always wins.

    sign_out_time is always the LATEST OUT event of the day (last_out),
    so it is updated each time a new OUT event is added.

    Total work hours = sum of all IN→OUT pair durations.
    Total break hours = sum of all OUT→IN pair durations.
    """
    rec = get_or_create_attendance(db, employee_id, d)
    events = get_events_for_day(db, employee_id, d)

    if events:
        work_h, break_h, first_in, last_out = calculate_intervals_from_events(events)
        # first_in = earliest IN event time — NEVER overwritten by later events.
        # last_out = latest OUT event time — always updated when new OUT arrives.
        rec.sign_in_time = first_in
        rec.sign_out_time = last_out
        rec.total_work_hours = work_h if work_h and work_h > 0 else None
        rec.total_break_hours = break_h if break_h and break_h > 0 else Decimal("0")
        rec.source = "AUTO" if all(e.source == "AUTO" for e in events) else rec.source
        for event in events:
            if event.attendance_record_id != rec.id:
                event.attendance_record_id = rec.id
    else:
        rec.total_work_hours = calculate_work_hours(rec.sign_in_time, rec.sign_out_time)
        rec.total_break_hours = None

    apply_status_from_hours(db, rec)
    _apply_late_and_early(db, rec)
    db.flush()
    return rec


def validate_event_time(event_time: datetime) -> None:
    now = get_ist_now()
    if event_time.date() > date.today():
        raise ValueError("Cannot record attendance for future dates")
    if event_time > now + timedelta(seconds=5):
        raise ValueError("Cannot record future event time")


def add_attendance_event(
    db: Session,
    employee_id: int,
    event_time: datetime | None = None,
    event_type: str | None = None,
    *,
    source: str = "AUTO",
    camera_id: str | None = None,
    skip_cooldown: bool = False,
    cooldown_seconds: int = EVENT_COOLDOWN_SECONDS,
) -> tuple[AttendanceEvent | None, AttendanceRecord, str]:
    """
    Create an attendance event and refresh the daily summary.

    Returns (event, attendance_record, action) where action is the event_type created
    or 'cooldown' when ignored.
    """
    now_dt = (event_time or get_ist_now()).replace(microsecond=0)
    validate_event_time(now_dt)

    if not skip_cooldown and is_within_event_cooldown(db, employee_id, now_dt, cooldown_seconds):
        rec = get_or_create_attendance(db, employee_id, now_dt.date())
        db.refresh(rec)
        return None, rec, "cooldown"

    d = now_dt.date()
    if event_type is None:
        last_event = get_latest_event_for_day(db, employee_id, d)
        event_type = determine_next_event_type(last_event)

    event_type = event_type.upper()
    if event_type not in {"IN", "OUT"}:
        raise ValueError("event_type must be IN or OUT")

    rec = get_or_create_attendance(db, employee_id, d)
    event = AttendanceEvent(
        employee_id=employee_id,
        attendance_record_id=rec.id,
        event_time=now_dt,
        event_type=event_type,
        source=source,
        camera_id=camera_id,
    )
    db.add(event)
    db.flush()

    updated = recalculate_attendance_summary(db, employee_id, d)
    db.commit()
    db.refresh(event)
    db.refresh(updated)
    return event, updated, event_type


def record_face_attendance(
    db: Session,
    employee_id: int,
    now_dt: datetime | None = None,
    camera_id: str | None = None,
) -> tuple[AttendanceEvent | None, AttendanceRecord, str]:
    """Face recognition entry point with 60s duplicate protection."""
    now_dt = (now_dt or get_ist_now()).replace(microsecond=0)
    try:
        result = add_attendance_event(
            db,
            employee_id,
            event_time=now_dt,
            source="AUTO",
            camera_id=camera_id,
        )
        event, rec, action = result
        if action == "cooldown":
            return None, rec, "cooldown"
        return event, rec, action
    except ValueError:
        raise
