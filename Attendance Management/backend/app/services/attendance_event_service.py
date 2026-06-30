"""Biometric attendance events: IN/OUT toggling, cooldown, daily summary recalculation."""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
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

logger = logging.getLogger(__name__)

EVENT_COOLDOWN_SECONDS = 25

def to_naive_ist(dt: datetime) -> datetime:
    """Normalise any datetime (aware or naive) to timezone-naive local IST datetime.

    This ensures that it is stored exactly as-is without database-side offset conversions.
    """
    if dt.tzinfo is not None:
        try:
            import zoneinfo
            IST = zoneinfo.ZoneInfo("Asia/Kolkata")
        except Exception:
            IST = timezone(timedelta(hours=5, minutes=30))
        return dt.astimezone(IST).replace(tzinfo=None)
    return dt

_WORK_START_EVENTS = {"IN", "BREAK_IN"}
_WORK_END_EVENTS = {"OUT", "BREAK_OUT"}


def _normalize_event_type(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().upper().replace("-", "_").replace(" ", "_")
    if cleaned in {"CHECK_IN", "ENTRY", "ARRIVAL"}:
        return "IN"
    if cleaned in {"CHECK_OUT", "EXIT", "DEPARTURE"}:
        return "OUT"
    if cleaned in {"BREAKIN", "BREAK_IN"}:
        return "BREAK_IN"
    if cleaned in {"BREAKOUT", "BREAK_OUT"}:
        return "BREAK_OUT"
    if cleaned in {"IN", "OUT"}:
        return cleaned
    return cleaned or None


def _day_bounds(d: date) -> tuple[datetime, datetime]:
    start = datetime.combine(d, time.min)
    end = datetime.combine(d, time.max)
    return start, end


def get_events_for_day(db: Session, employee_id: int, d: date) -> list[AttendanceEvent]:
    return (
        db.query(AttendanceEvent)
        .filter(
            AttendanceEvent.employee_id == employee_id,
            AttendanceEvent.attendance_date == d,
        )
        .order_by(AttendanceEvent.event_time.asc(), AttendanceEvent.id.asc())
        .all()
    )


def get_latest_event_for_day(db: Session, employee_id: int, d: date) -> AttendanceEvent | None:
    return (
        db.query(AttendanceEvent)
        .filter(
            AttendanceEvent.employee_id == employee_id,
            AttendanceEvent.attendance_date == d,
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
    now_naive = to_naive_ist(now_dt)
    today = now_naive.date()
    latest = (
        db.query(AttendanceEvent)
        .filter(
            AttendanceEvent.employee_id == employee_id,
            AttendanceEvent.attendance_date == today,
        )
        .order_by(AttendanceEvent.event_time.desc(), AttendanceEvent.id.desc())
        .first()
    )
    if latest is None:
        return False
    latest_naive = to_naive_ist(latest.event_time)
    return (now_naive - latest_naive).total_seconds() < cooldown_seconds


def determine_next_event_type(last_event: AttendanceEvent | None) -> str:
    if last_event is None:
        return "IN"
    last_type = _normalize_event_type(last_event.event_type) or last_event.event_type
    if last_type in _WORK_START_EVENTS:
        return "OUT"
    return "IN"


def calculate_intervals_from_events(
    events: list[AttendanceEvent],
) -> tuple[Decimal | None, Decimal | None, time | None, time | None]:
    """Return total work hours, break hours, first IN time, last OUT time."""
    if not events:
        return None, None, None, None

    sorted_events = sorted(events, key=lambda e: (to_naive_ist(e.event_time), e.id))
    in_events = [e for e in sorted_events if _normalize_event_type(e.event_type) in _WORK_START_EVENTS]
    out_events = [e for e in sorted_events if _normalize_event_type(e.event_type) in _WORK_END_EVENTS]

    first_in = to_naive_ist(in_events[0].event_time).time() if in_events else None
    last_out = to_naive_ist(out_events[-1].event_time).time() if out_events else None

    total_work_seconds = 0
    total_break_seconds = 0
    for i in range(len(sorted_events) - 1):
        cur = sorted_events[i]
        nxt = sorted_events[i + 1]
        cur_t = to_naive_ist(cur.event_time)
        nxt_t = to_naive_ist(nxt.event_time)
        delta = int((nxt_t - cur_t).total_seconds())
        if delta <= 0:
            continue
        cur_type = _normalize_event_type(cur.event_type)
        nxt_type = _normalize_event_type(nxt.event_type)
        if cur_type in _WORK_START_EVENTS and nxt_type in _WORK_END_EVENTS:
            total_work_seconds += delta
        elif cur_type in _WORK_END_EVENTS and nxt_type in _WORK_START_EVENTS:
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
    """Validate that event_time is not in the future.

    Handles both timezone-aware and timezone-naive event_time values safely
    by converting everything to naive local IST datetimes.
    """
    now = get_ist_now()  # naive IST
    event_time_naive = to_naive_ist(event_time)

    if event_time_naive.date() > date.today():
        raise ValueError("Cannot record attendance for future dates")
    if event_time_naive > now + timedelta(seconds=5):
        raise ValueError(
            f"Cannot record future event time: event={event_time_naive.isoformat()} now={now.isoformat()}"
        )


def add_attendance_event(
    db: Session,
    employee_id: int,
    event_time: datetime | None = None,
    event_type: str | None = None,
    *,
    source: str = "AUTO",
    camera_id: str | None = None,
    camera_purpose: str | None = None,  # "IN" | "OUT" – forces event_type when set
    skip_cooldown: bool = False,
    cooldown_seconds: int = EVENT_COOLDOWN_SECONDS,
) -> tuple[AttendanceEvent | None, AttendanceRecord, str]:
    """
    Create an attendance event and refresh the daily summary.

    When camera_purpose is "IN" or "OUT" it OVERRIDES the auto-toggle logic,
    ensuring a Check-In camera always records IN and a Check-Out camera OUT.

    Returns (event, attendance_record, action) where action is the event_type
    created or 'cooldown' when ignored.
    """
    now_dt = to_naive_ist(event_time or get_ist_now()).replace(microsecond=0)
    validate_event_time(now_dt)

    if not skip_cooldown and is_within_event_cooldown(db, employee_id, now_dt, cooldown_seconds):
        rec = get_or_create_attendance(db, employee_id, now_dt.date())
        db.refresh(rec)
        logger.info(
            "attendance_event COOLDOWN employee_id=%s within_%ds_window attendance_date=%s",
            employee_id, cooldown_seconds, now_dt.date(),
        )
        return None, rec, "cooldown"

    d = now_dt.date()

    # Resolve event_type priority:
    #   1. Explicit caller override  (event_type param)
    #   2. Camera purpose            (camera_purpose param)
    #   3. Auto-toggle from last event
    if event_type is not None:
        resolved_type = event_type
    else:
        resolved_type = camera_purpose

    resolved_type = _normalize_event_type(resolved_type)
    if resolved_type is None:
        last_event = get_latest_event_for_day(db, employee_id, d)
        resolved_type = determine_next_event_type(last_event)

    resolved_type = _normalize_event_type(resolved_type) or ""
    if resolved_type not in {"IN", "OUT", "BREAK_IN", "BREAK_OUT"}:
        raise ValueError("event_type must be IN, OUT, BREAK_IN, or BREAK_OUT")

    rec = get_or_create_attendance(db, employee_id, d)
    event = AttendanceEvent(
        employee_id=employee_id,
        attendance_record_id=rec.id,
        attendance_date=d,          # NOT NULL column — must be set explicitly
        event_time=now_dt,
        event_type=resolved_type,
        source=source,
        camera_id=camera_id,
    )
    db.add(event)
    db.flush()
    logger.info(
        "attendance_event INSERT employee_id=%s event_type=%s event_time=%s camera_id=%s",
        employee_id, resolved_type, now_dt.isoformat(), camera_id,
    )

    updated = recalculate_attendance_summary(db, employee_id, d)
    db.commit()
    db.refresh(event)
    db.refresh(updated)
    logger.info(
        "attendance_event COMMITTED employee_id=%s event_id=%s sign_in=%s sign_out=%s "
        "work_hours=%s status=%s",
        employee_id, event.id,
        updated.sign_in_time, updated.sign_out_time,
        updated.total_work_hours, updated.status,
    )
    return event, updated, resolved_type


def record_face_attendance(
    db: Session,
    employee_id: int,
    now_dt: datetime | None = None,
    camera_id: str | None = None,
    camera_purpose: str | None = None,
    event_type: str | None = None,
) -> tuple[AttendanceEvent | None, AttendanceRecord, str]:
    """Face recognition entry point with 60s duplicate protection.

    camera_purpose ("IN"|"OUT") forces the event direction when provided,
    overriding the normal auto-toggle behaviour.
    """
    now_dt = (now_dt or get_ist_now()).replace(microsecond=0)
    try:
        result = add_attendance_event(
            db,
            employee_id,
            event_time=now_dt,
            event_type=event_type,
            source="AUTO",
            camera_id=camera_id,
            camera_purpose=camera_purpose,
        )
        event, rec, action = result
        if action == "cooldown":
            return None, rec, "cooldown"
        return event, rec, action
    except ValueError:
        raise
