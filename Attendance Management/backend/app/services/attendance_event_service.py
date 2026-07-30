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


def business_date(dt: datetime, day_start_hour: int | None = None) -> date:
    """The attendance day an event belongs to.

    A plain calendar date is the wrong key for anything that runs past
    midnight. An exit at 00:30 landed on a fresh day where the employee had no
    check-in, so resolve_camera_event() saw state=ABSENT and (with
    attendance_checkin_on_missing_in) turned that EXIT into a CHECK_IN for the
    new day — while the real day was never closed.

    Anchoring to a day-start hour keeps the whole shift on the day it began.

    day_start_hour=0 reproduces the old behaviour exactly, which is the safe
    value to deploy with before switching it on (see config).
    """
    if day_start_hour is None:
        from app.core.config import get_settings

        day_start_hour = int(getattr(get_settings(), "attendance_day_start_hour", 0) or 0)
    naive = to_naive_ist(dt)
    if day_start_hour and naive.hour < day_start_hour:
        return (naive - timedelta(days=1)).date()
    return naive.date()


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


def _event_direction(event_type: str | None) -> str | None:
    """Classify an event or camera purpose as 'entry' or 'exit' (or None).

    entry = IN / BREAK_IN  (check-in side)
    exit  = OUT / BREAK_OUT (check-out side)
    """
    n = _normalize_event_type(event_type)
    if n in _WORK_START_EVENTS:
        return "entry"
    if n in _WORK_END_EVENTS:
        return "exit"
    return None


def is_within_event_cooldown(
    db: Session,
    employee_id: int,
    now_dt: datetime,
    cooldown_seconds: int = EVENT_COOLDOWN_SECONDS,
    direction: str | None = None,
) -> bool:
    """Return True only if a DUPLICATE recent event should suppress this one.

    Scoping to the current day prevents a late-night event from blocking
    the employee's first check-in of the following morning.

    Directional: when ``direction`` ('entry'/'exit') is given, the cooldown only
    suppresses a repeat of the SAME direction. An OPPOSITE transition (a check-in
    shortly after a check-out, or a return from a quick <cooldown break) is a
    real state change and is never dropped — dropping it would strand the
    employee on the wrong side (shown outside while actually back inside).
    """
    now_naive = to_naive_ist(now_dt)
    # Business day, not calendar day: at 00:10 the employee's previous events
    # are still filed under yesterday, and scoping the cooldown to the calendar
    # date would make every post-midnight event look like the first of a new day.
    today = business_date(now_naive)
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
    if (now_naive - latest_naive).total_seconds() >= cooldown_seconds:
        return False
    # Within the time window. Only suppress when the incoming event is the SAME
    # direction as the last one (a true duplicate). Opposite transitions pass.
    if direction is not None:
        latest_dir = _event_direction(latest.event_type)
        if latest_dir is not None and latest_dir != direction:
            return False
    return True


def determine_next_event_type(last_event: AttendanceEvent | None) -> str:
    if last_event is None:
        return "IN"
    last_type = _normalize_event_type(last_event.event_type) or last_event.event_type
    if last_type in _WORK_START_EVENTS:
        return "OUT"
    return "IN"


# ── Attendance state machine ────────────────────────────────────────────────
# State derived from the last event of the day:
#   ABSENT   – no events yet
#   WORKING  – last event is a work-start (CHECK_IN / BREAK_IN) → employee inside
#   AWAY     – last event is a work-end  (BREAK_OUT / CHECK_OUT) → employee out
def current_state(last_type: str | None) -> str:
    if last_type is None:
        return "ABSENT"
    n = _normalize_event_type(last_type)
    if n in _WORK_START_EVENTS:
        return "WORKING"
    if n in _WORK_END_EVENTS:
        return "AWAY"
    return "ABSENT"


def resolve_camera_event(
    camera_type: str, last_type: str | None, allow_missing_in: bool
) -> tuple[str | None, str | None]:
    """Decide the event type for a camera recognition, or reject it.

    Returns (event_type, reject_reason). event_type is one of
    CHECK_IN / BREAK_IN / BREAK_OUT (the final BREAK_OUT of the day is surfaced
    as the check-out in the summary). reject_reason is set (and event_type None)
    for an invalid transition.

        IN  camera:  ABSENT  → CHECK_IN
                     AWAY    → BREAK_IN         (returning from a break)
                     WORKING → reject (duplicate check-in / already inside)
        OUT camera:  WORKING → BREAK_OUT        (a departure; last one = checkout)
                     ABSENT  → CHECK_IN if allow_missing_in else reject
                     AWAY    → reject (duplicate break-out / already outside)
    """
    state = current_state(last_type)
    cam = (camera_type or "IN").upper()

    if cam == "IN":
        if state == "ABSENT":
            return "CHECK_IN", None
        if state == "AWAY":
            return "BREAK_IN", None
        return None, "duplicate_check_in_already_working"

    # OUT / check-out camera
    if state == "WORKING":
        return "BREAK_OUT", None
    if state == "ABSENT":
        if allow_missing_in:
            return "CHECK_IN", None  # entrance was missed — record the check-in
        return None, "check_out_without_check_in"
    return None, "duplicate_out_already_away"


def calculate_intervals_from_events(
    events: list[AttendanceEvent],
    cutoff_time: time = time(23, 59),
) -> tuple[Decimal | None, Decimal | None, time | None, time | None, list[dict]]:
    """Return total work hours, break hours, first IN time, last OUT time, and timeline data.
    
    Timeline data includes all event pairs with durations for IN/OUT and break periods.
    """
    if not events:
        return None, None, None, None, []

    sorted_events = sorted(events, key=lambda e: (to_naive_ist(e.event_time), e.id))
    in_events = [e for e in sorted_events if _normalize_event_type(e.event_type) in _WORK_START_EVENTS]
    out_events = [e for e in sorted_events if _normalize_event_type(e.event_type) in _WORK_END_EVENTS]

    first_in = to_naive_ist(in_events[0].event_time).time() if in_events else None
    
    # Determine final check-out based on cutoff time
    last_out = None
    if out_events:
        last_out_event = out_events[-1]
        last_out_time = to_naive_ist(last_out_event.event_time)
        # Check if there's an IN event after the last OUT before cutoff
        has_later_in = any(
            to_naive_ist(e.event_time) > last_out_time and 
            to_naive_ist(e.event_time).time() <= cutoff_time
            for e in in_events if to_naive_ist(e.event_time) > last_out_time
        )
        if not has_later_in:
            last_out = last_out_time.time()

    total_work_seconds = 0
    total_break_seconds = 0
    timeline = []
    
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
            timeline.append({
                "start_time": cur_t.isoformat(),
                "end_time": nxt_t.isoformat(),
                "type": "work",
                "duration_seconds": delta,
                "duration_formatted": format_duration(delta),
                "event_type": cur_type,
            })
        elif cur_type in _WORK_END_EVENTS and nxt_type in _WORK_START_EVENTS:
            total_break_seconds += delta
            timeline.append({
                "start_time": cur_t.isoformat(),
                "end_time": nxt_t.isoformat(),
                "type": "break",
                "duration_seconds": delta,
                "duration_formatted": format_duration(delta),
                "event_type": cur_type,
            })

    # Live open interval: if the employee is CURRENTLY working (last event is a
    # work-start) on TODAY, count the time from that event until now, so the
    # displayed working hours reflect reality instead of freezing at the last
    # break-out. (Past days close naturally at the final check-out.)
    last_ev = sorted_events[-1]
    if _normalize_event_type(last_ev.event_type) in _WORK_START_EVENTS:
        last_t = to_naive_ist(last_ev.event_time)
        now = get_ist_now()
        # Business day on both sides: a night shift that began yesterday evening
        # is still "today's" open interval at 00:30, and a calendar comparison
        # would stop counting it at midnight.
        if business_date(last_t) == business_date(now):
            open_secs = int((now - last_t).total_seconds())
            if open_secs > 0:
                total_work_seconds += open_secs

    work_hours = Decimal(round(total_work_seconds / 3600, 2)) if total_work_seconds else Decimal("0")
    break_hours = Decimal(round(total_break_seconds / 3600, 2)) if total_break_seconds else Decimal("0")
    return work_hours, break_hours, first_in, last_out, timeline


def format_duration(seconds: int) -> str:
    """Format seconds into human-readable duration (e.g., '1h 30m', '45m')."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def count_attendance_events(
    events: list[AttendanceEvent], has_final_checkout: bool
) -> dict:
    """Count check-in/out events and break events for a single day.

    The day's boundaries — the FIRST check-in and the FINAL check-out — are NOT
    counted as breaks. Everything in between is a break:
      * break-out  = every work-end (OUT / BREAK_OUT) except the final check-out
      * break-in   = every work-start (IN / BREAK_IN) except the first check-in

    ``has_final_checkout`` tells us whether the day already has a closing
    check-out (i.e. the last event is not a work-start). When the employee is
    still working (no final check-out yet), no work-end is excluded, so every
    OUT so far is a genuine break-out. This mirrors how the daily summary derives
    sign_out_time, keeping the counts consistent with the rest of the record.

    Example (completed day)::

        09:00 IN  12:30 OUT  12:45 IN  15:30 OUT  15:45 IN  18:00 OUT
        -> check_in=3 check_out=3 break_in=2 break_out=2

    Returns a dict with check_in_count, check_out_count, break_in_count,
    break_out_count (all ints).
    """
    in_events = [e for e in events if _normalize_event_type(e.event_type) in _WORK_START_EVENTS]
    out_events = [e for e in events if _normalize_event_type(e.event_type) in _WORK_END_EVENTS]

    check_in_count = len(in_events)
    check_out_count = len(out_events)
    # Exclude the first check-in from break-ins; exclude the final check-out from
    # break-outs only when that closing check-out actually exists.
    break_in_count = max(0, check_in_count - 1)
    break_out_count = max(0, check_out_count - (1 if has_final_checkout else 0))

    return {
        "check_in_count": check_in_count,
        "check_out_count": check_out_count,
        "break_in_count": break_in_count,
        "break_out_count": break_out_count,
    }


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

    work_h = None
    if events:
        work_h, break_h, first_in, last_out, _timeline = calculate_intervals_from_events(events)
        # first_in = earliest IN event time — NEVER overwritten by later events.
        # last_out = latest OUT event time — always updated when new OUT arrives.
        # A field HR pinned manually is left alone: the camera may have missed
        # the employee, which is exactly why HR typed the value in.
        if not rec.sign_in_manual:
            rec.sign_in_time = first_in
        if not rec.sign_out_manual:
            rec.sign_out_time = last_out
        if not rec.break_manual:
            rec.total_break_hours = break_h if break_h and break_h > 0 else Decimal("0")
        for event in events:
            if event.attendance_record_id != rec.id:
                event.attendance_record_id = rec.id
    elif not rec.break_manual:
        rec.total_break_hours = None

    manual = rec.sign_in_manual or rec.sign_out_manual or rec.break_manual
    if manual or not events:
        # The IN→OUT pair sum only knows the camera's times, so it cannot be used
        # once HR has overridden anything. Work from the effective span instead,
        # net of the break, so Working Hours never includes break time.
        span = calculate_work_hours(rec.sign_in_time, rec.sign_out_time)
        if span is None:
            # Still checked in (no sign-out yet): fall back to the live
            # event-derived total rather than blanking the day's hours.
            rec.total_work_hours = work_h if (work_h and work_h > 0) else None
        else:
            net = span - (rec.total_break_hours or Decimal("0"))
            rec.total_work_hours = net if net > 0 else Decimal("0")
        if manual:
            rec.source = "ADMIN"
    else:
        rec.total_work_hours = work_h if work_h and work_h > 0 else None
        rec.source = "AUTO" if all(e.source == "AUTO" for e in events) else rec.source

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

    # Compare IST against IST. This used to compare an IST-derived date against
    # date.today() (the SERVER's local date), so on a UTC-hosted server every
    # event between 00:00 and 05:30 IST was rejected as a "future date".
    if business_date(event_time_naive) > business_date(now):
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
    evidence: dict | None = None,
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

    # Direction of THIS event (entry/exit), from an explicit type or the camera
    # purpose, so the cooldown only suppresses same-direction duplicates and
    # never drops a legitimate opposite transition.
    incoming_direction = _event_direction(event_type if event_type is not None else camera_purpose)

    if not skip_cooldown and is_within_event_cooldown(
        db, employee_id, now_dt, cooldown_seconds, direction=incoming_direction
    ):
        rec = get_or_create_attendance(db, employee_id, business_date(now_dt))
        db.refresh(rec)
        logger.info(
            "attendance_event COOLDOWN employee_id=%s within_%ds_window attendance_date=%s",
            employee_id, cooldown_seconds, business_date(now_dt),
        )
        return None, rec, "cooldown"

    # The day this event is filed under. NOT now_dt.date(): a shift running past
    # midnight must stay on the day it started, or the exit becomes the next
    # day's check-in. See business_date().
    d = business_date(now_dt)

    # Resolve event_type priority:
    #   1. Explicit caller override  (event_type param)
    #   2. Camera purpose            (camera_purpose param)
    #   3. Auto-toggle from last event
    from_camera_purpose = event_type is None and camera_purpose is not None
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

    # ── Attendance state machine (dedicated IN / OUT cameras) ───────────────
    # Classify the camera recognition into CHECK_IN / BREAK_IN / BREAK_OUT based
    # on the employee's current state, and REJECT invalid transitions (with the
    # exact reason logged). Manual/explicit events (event_type given) bypass it.
    if from_camera_purpose and resolved_type in {"IN", "OUT"}:
        last_event = get_latest_event_for_day(db, employee_id, d)
        last_type = last_event.event_type if last_event else None
        camera_type = "IN" if resolved_type == "IN" else "OUT"

        from app.core.config import get_settings
        allow_missing = get_settings().attendance_checkin_on_missing_in

        new_type, reject = resolve_camera_event(camera_type, last_type, allow_missing)
        state = current_state(last_type)
        if reject:
            rec = get_or_create_attendance(db, employee_id, d)
            db.refresh(rec)
            logger.info(
                "attendance_event REJECTED employee_id=%s camera=%s state=%s last=%s reason=%s",
                employee_id, camera_type, state, last_type, reject,
            )
            return None, rec, reject
        logger.info(
            "attendance_event STATE employee_id=%s camera=%s state=%s last=%s -> event=%s",
            employee_id, camera_type, state, last_type, new_type,
        )
        resolved_type = new_type

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
    # Why the camera believed this was that person. Absent for manual and
    # auto-close events, which is exactly what NULL should mean here.
    if evidence:
        event.match_score = evidence.get("match_score")
        event.match_margin = evidence.get("match_margin")
        event.track_id = evidence.get("track_id")
        event.snapshot_path = evidence.get("snapshot_path")
    db.add(event)
    db.flush()
    logger.info(
        "attendance_event INSERT employee_id=%s event_type=%s event_time=%s camera_id=%s",
        employee_id, resolved_type, now_dt.isoformat(), camera_id,
    )

    updated = recalculate_attendance_summary(db, employee_id, d)
    db.commit()

    # The presence set just changed; drop its cache so the next recognition
    # narrows against current state instead of waiting out the TTL.
    try:
        from app.services.presence_cache import invalidate as _invalidate_presence

        _invalidate_presence()
    except Exception:  # pragma: no cover - cache is an optimisation only
        logger.debug("presence cache invalidation failed", exc_info=True)

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
    evidence: dict | None = None,
) -> tuple[AttendanceEvent | None, AttendanceRecord, str]:
    """Face recognition entry point with 60s duplicate protection.

    camera_purpose ("IN"|"OUT") forces the event direction when provided,
    overriding the normal auto-toggle behaviour.

    ``evidence`` carries the recognition provenance (score, margin, track id,
    face snapshot) onto the stored event so a disputed record can be reviewed.
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
            evidence=evidence,
        )
        event, rec, action = result
        if action == "cooldown":
            return None, rec, "cooldown"
        return event, rec, action
    except ValueError:
        raise
