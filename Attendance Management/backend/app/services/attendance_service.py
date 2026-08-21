"""Attendance: sign-in/out, work hours, status, grace time, weekly off, holiday."""
import calendar as _calendar
from datetime import date, time, datetime, timedelta
from decimal import Decimal
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.models import AttendanceRecord, CompanyConfig, Holiday

#: Attendance-classification thresholds, in hours.
#:
#: Only these three are fixed. The half-day boundary is deliberately NOT here:
#: it is derived per employee as `expected_working_hours / 2`, because shifts
#: differ and a single number is wrong for everyone not on that shift.
GRACE_HOURS = 0.25          # 15 minutes
SHORT_LEAVE_HOURS = 2.0     # missed up to this => short leave, not half day
DEFAULT_EXPECTED_HOURS = 9.0  # used only when the employee record has none


def get_company_config(db: Session):
    return db.query(CompanyConfig).first()


def _is_weekly_off(emp_date: date, weekly_off_days: str | None) -> bool:
    """
    Determine if a given date is a weekly off.

    Company works Monday to Friday; Saturday and Sunday are week-off by default.
    weekly_off_days format: "SAT,SUN" etc.
    """
    # Default weekly off: Saturday + Sunday (Mon-Fri working week)
    days_cfg = weekly_off_days or "SAT,SUN"
    # weekday: Monday=0, Sunday=6
    day_num = emp_date.weekday()
    day_names = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    day_str = day_names[day_num]
    return day_str in days_cfg.upper().replace(" ", "").split(",")


def _is_holiday(db: Session, emp_date: date) -> bool:
    return db.query(Holiday).filter(Holiday.date == emp_date).first() is not None


def _holiday_dates_between(db: Session, start: date, end: date) -> set[date]:
    """All holiday dates in a range, in ONE query.

    Use this instead of calling `_is_holiday` from a per-day loop: that issues a
    SELECT for every single day, and the database is remote (~29 ms per round
    trip -- see db/session.py), so a month-long loop costs about a second of
    pure latency before any real work happens.
    """
    return {
        row[0]
        for row in db.query(Holiday.date)
        .filter(Holiday.date >= start, Holiday.date <= end)
        .all()
    }


def apply_weekly_off_and_holiday(db: Session, record: AttendanceRecord) -> None:
    config = get_company_config(db)
    record.is_weekly_off = _is_weekly_off(record.date, config.weekly_off_days if config else None)
    record.is_holiday = _is_holiday(db, record.date)
    # Only apply default WO/Holiday status if no work time is recorded yet
    if record.sign_in_time is None and record.total_work_hours is None:
        if record.is_weekly_off:
            record.status = "WEEKLY_OFF"
        elif record.is_holiday:
            record.status = "HOLIDAY"


def _find_attendance(db: Session, employee_id: int, d: date) -> AttendanceRecord | None:
    return db.query(AttendanceRecord).filter(
        AttendanceRecord.employee_id == employee_id,
        AttendanceRecord.date == d,
    ).first()


def get_or_create_attendance(db: Session, employee_id: int, d: date) -> AttendanceRecord:
    rec = _find_attendance(db, employee_id, d)
    if rec:
        return rec
    config = get_company_config(db)
    is_wo = _is_weekly_off(d, config.weekly_off_days if config else None)
    is_hol = _is_holiday(db, d)
    if is_wo:
        status = "WEEKLY_OFF"
    elif is_hol:
        status = "HOLIDAY"
    else:
        status = "ABSENT"
    rec = AttendanceRecord(
        employee_id=employee_id,
        date=d,
        status=status,
        is_weekly_off=is_wo,
        is_holiday=is_hol,
        source="AUTO",
    )
    # Concurrent camera workers can reach this point for the same employee/day
    # at once. The uq_attendance_employee_date constraint makes the loser fail;
    # roll back only the failed INSERT (savepoint) and take the winner's row.
    try:
        with db.begin_nested():
            db.add(rec)
            db.flush()
    except IntegrityError:
        db.expire_all()
        existing = _find_attendance(db, employee_id, d)
        if existing is None:
            raise
        return existing
    return rec


def calculate_work_hours(sign_in_t: time | None, sign_out_t: time | None) -> Decimal | None:
    if not sign_in_t or not sign_out_t:
        return None
    a = datetime.combine(date.today(), sign_in_t)
    b = datetime.combine(date.today(), sign_out_t)
    if b <= a:
        # Treat sign_out as same-day evening if it's before sign_in (e.g. 06:30 meant as 6:30 PM)
        if sign_out_t.hour < 12:
            sign_out_t = time(
                sign_out_t.hour + 12,
                sign_out_t.minute,
                sign_out_t.second,
                getattr(sign_out_t, "microsecond", 0),
            )
            b = datetime.combine(date.today(), sign_out_t)
        if b <= a:
            return None
    delta = b - a
    return Decimal(round(delta.total_seconds() / 3600, 2))


def apply_status_from_hours(db: Session, rec: AttendanceRecord) -> None:
    """
    Classify attendance status based on total_work_hours for working days.

    Thresholds are derived from the EMPLOYEE's own `expected_working_hours`, not
    from a fixed number of hours, because shifts differ: half a day is 4h for an
    8h shift, 3.5h for a 7h shift and 3h for a 6h shift.

    - missed <= 15 min grace          => PRESENT (full day)
    - missed <= SHORT_LEAVE_HOURS     => SHORT (short leave)
    - worked >= expected / 2          => HALF_DAY
    - worked > 0 but under half a day => HALF_DAY (they did show up)
    - worked == 0                     => ABSENT
    - Time In only (no Clock Out yet) => PRESENT so employee is not shown Absent
    - PAID_LEAVE / WEEKLY_OFF / HOLIDAY are HR/system-set and never overwritten by hours.
    """
    # The CURRENT day is still in progress — do NOT finalize to Half Day / Short
    # from partial hours. An employee who has shown up is PRESENT until the day
    # is over; the real classification is applied when viewing a past day.
    from app.core.datetime_utils import get_ist_now
    # Imported lazily: attendance_event_service imports THIS module, so a
    # module-level import here would be circular.
    from app.services.attendance_event_service import business_date

    # An HR edit is authoritative. Keep the selected status even when a later
    # camera event triggers a summary recalculation for the same day.
    if rec.source == "ADMIN":
        return

    # Business day, not calendar day. At 00:30 the in-progress day is still
    # yesterday's record, and a calendar comparison would finalise a night
    # shift's status to Half Day while the employee is still working.
    if rec.date >= business_date(get_ist_now()):
        if rec.sign_in_time is not None:
            rec.status = "PRESENT"
        elif rec.is_weekly_off:
            rec.status = "WEEKLY_OFF"
        elif rec.is_holiday:
            rec.status = "HOLIDAY"
        return

    # If we have hours worked, we classify based on hours even on Week Offs/Holidays.
    # If no hours worked, we use the system status (WO/Holiday/Absent).
    if rec.total_work_hours is not None:
        from app.models.employee import Employee
        emp = db.query(Employee).filter(Employee.id == rec.employee_id).first()
        expected = float(emp.expected_working_hours or DEFAULT_EXPECTED_HOURS)

        hours = float(rec.total_work_hours)
        missed = expected - hours
        # Half a day is defined against this employee's own shift length. The
        # previous fixed 4.5 was only correct for a 9-hour shift and silently
        # wrong for every other one.
        half_day_hours = expected / 2

        if missed <= GRACE_HOURS:
            rec.status = "PRESENT"
        elif missed <= SHORT_LEAVE_HOURS:
            rec.status = "SHORT"
        elif hours >= half_day_hours:
            rec.status = "HALF_DAY"
        else:
            # Below half a day: still HALF_DAY if they attended at all, so a
            # short attendance is never recorded as a full absence.
            rec.status = "ABSENT" if hours == 0 else "HALF_DAY"
    else:
        if rec.sign_in_time is not None:
            # Clocked in but not out yet: show as Present
            rec.status = "PRESENT"
        elif rec.is_weekly_off:
            rec.status = "WEEKLY_OFF"
        elif rec.is_holiday:
            rec.status = "HOLIDAY"

def monthly_attendance_summary(db: Session, employee_id: int, month: int, year: int) -> dict:
    """Aggregate an employee's attendance for a calendar month.

    Reuses the existing attendance records plus the `_is_holiday` / `_is_weekly_off`
    helpers — no attendance status is recalculated here. Days are bucketed as:

    - holiday      : company holiday (from the Holiday table)
    - weekly_off   : configured weekly off (e.g. SAT/SUN)
    - present      : PRESENT or SHORT attendance record
    - half_day     : HALF_DAY record
    - leave        : ON_LEAVE / PAID_LEAVE record
    - absent       : ABSENT record, or an elapsed working day with no record

    Working days = calendar days − holidays − weekly offs. Present/leave/absent
    are only counted for elapsed days (up to today) on/after the joining date, so
    the current month is not penalised for days that haven't happened yet.

    Attendance % = (present + ½·half_day) / (present + half_day + absent) — i.e.
    approved leave is excused and never counts against the employee.
    """
    from app.core.datetime_utils import get_ist_now
    from app.models.employee import Employee

    if not (1 <= month <= 12):
        raise ValueError("month must be between 1 and 12")

    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    config = get_company_config(db)
    weekly_off_days = config.weekly_off_days if config else None

    total_days = _calendar.monthrange(year, month)[1]
    month_start = date(year, month, 1)
    month_end = date(year, month, total_days)
    today = get_ist_now().date()
    doj = emp.date_of_joining if emp else None

    records = {
        r.date: r
        for r in db.query(AttendanceRecord).filter(
            AttendanceRecord.employee_id == employee_id,
            AttendanceRecord.date >= month_start,
            AttendanceRecord.date <= month_end,
        ).all()
    }

    # One query for the month's holidays, matching how `records` above is
    # already batched. The per-day `_is_holiday()` call this replaces ran a
    # SELECT for each of the ~31 days on every call to this summary.
    holiday_dates = _holiday_dates_between(db, month_start, month_end)

    present = half_day = leave = absent = holiday = weekly_off = 0

    d = month_start
    while d <= month_end:
        rec = records.get(d)
        status = rec.status if rec else None

        if d in holiday_dates:
            holiday += 1
        elif _is_weekly_off(d, weekly_off_days):
            weekly_off += 1
        else:
            # Working day — classify by the attendance record.
            elapsed = d <= today and (doj is None or d >= doj)
            if status in ("PRESENT", "SHORT"):
                present += 1
            elif status == "HALF_DAY":
                half_day += 1
            elif status in ("ON_LEAVE", "PAID_LEAVE"):
                leave += 1
            elif status == "ABSENT":
                absent += 1
            elif status in ("HOLIDAY", "WEEKLY_OFF"):
                # Record disagrees with the calendar; trust the record's off-day.
                pass
            elif elapsed:
                # Elapsed working day with no record at all → absent.
                absent += 1
        d += timedelta(days=1)

    working_days = total_days - holiday - weekly_off
    graded = present + half_day + absent  # days that required attendance and weren't excused
    attendance_pct = round((present + 0.5 * half_day) / graded * 100, 1) if graded > 0 else 0.0

    return {
        "month": month,
        "year": year,
        "total_calendar_days": total_days,
        "working_days": working_days,
        "present": present,
        "half_day": half_day,
        "leave": leave,
        "absent": absent,
        "holiday": holiday,
        "weekly_off": weekly_off,
        "attendance_percentage": attendance_pct,
    }


def sign_in(db: Session, employee_id: int, d: date, sign_in_time: time) -> AttendanceRecord:
    rec = get_or_create_attendance(db, employee_id, d)
    rec.sign_in_time = sign_in_time
    rec.sign_out_time = None
    rec.total_work_hours = None
    rec.source = "SELF"
    config = get_company_config(db)
    grace_min = config.grace_time_minutes if config else 15
    # Assume standard start 09:00 for "late" check; can be configurable later
    standard_start = time(9, 0)
    t = datetime.combine(d, sign_in_time)
    s = datetime.combine(d, standard_start)
    rec.is_late = (t - s).total_seconds() > grace_min * 60
    # PRESENT is correct at sign-in: there is no sign-out yet, so no worked
    # hours to classify. The real classification happens in
    # `apply_status_from_hours`, which owns the half-day rule and derives it
    # from the employee's own expected hours.
    rec.status = "PRESENT"
    db.commit()
    db.refresh(rec)
    return rec


def sign_out(db: Session, employee_id: int, d: date, sign_out_time: time) -> AttendanceRecord:
    rec = get_or_create_attendance(db, employee_id, d)
    rec.sign_out_time = sign_out_time
    rec.total_work_hours = calculate_work_hours(rec.sign_in_time, sign_out_time)
    apply_status_from_hours(db, rec)
    # Early exit: e.g. before 18:00
    standard_end = time(18, 0)
    t = datetime.combine(d, sign_out_time)
    s = datetime.combine(d, standard_end)
    rec.is_early_exit = t < s
    db.commit()
    db.refresh(rec)
    return rec
