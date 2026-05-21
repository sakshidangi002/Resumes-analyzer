"""Attendance: sign-in/out, work hours, status, grace time, weekly off, holiday."""
from datetime import date, time, datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from app.models import AttendanceRecord, CompanyConfig, Holiday


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


def get_or_create_attendance(db: Session, employee_id: int, d: date) -> AttendanceRecord:
    rec = db.query(AttendanceRecord).filter(
        AttendanceRecord.employee_id == employee_id,
        AttendanceRecord.date == d,
    ).first()
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
    db.add(rec)
    db.flush()
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

    - >= 8.5 hours  => PRESENT (full day)
    - 7 to < 8.5    => SHORT (short leave)
    - < 7           => HALF_DAY
    - Time In only (no Clock Out yet) => PRESENT so employee is not shown Absent
    - PAID_LEAVE / WEEKLY_OFF / HOLIDAY are HR/system-set and never overwritten by hours.
    """
    # If we have hours worked, we classify based on hours even on Week Offs/Holidays.
    # If no hours worked, we use the system status (WO/Holiday/Absent).
    if rec.total_work_hours is not None:
        from app.models.employee import Employee
        emp = db.query(Employee).filter(Employee.id == rec.employee_id).first()
        expected = float(emp.expected_working_hours or 9.0)
        
        hours = float(rec.total_work_hours)
        missed = expected - hours
        
        if missed <= 0.25: # 15 min grace
            rec.status = "PRESENT"
        elif missed <= 2.0: # Short Leave (up to 2h missed)
            rec.status = "SHORT"
        elif missed <= 4.5: # Half Day (up to 4.5h missed)
            rec.status = "HALF_DAY"
        else: # More than 4.5h missed (even if worked a little)
            rec.status = "ABSENT" if hours == 0 else "HALF_DAY" # Show as Half Day if they at least showed up
    else:
        if rec.sign_in_time is not None:
            # Clocked in but not out yet: show as Present
            rec.status = "PRESENT"
        elif rec.is_weekly_off:
            rec.status = "WEEKLY_OFF"
        elif rec.is_holiday:
            rec.status = "HOLIDAY"

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
    from datetime import datetime as dt
    t = datetime.combine(d, sign_in_time)
    s = datetime.combine(d, standard_start)
    rec.is_late = (t - s).total_seconds() > grace_min * 60
    half_day_hours = config.half_day_threshold_hours if config else 4
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
    from datetime import datetime as dt
    t = datetime.combine(d, sign_out_time)
    s = datetime.combine(d, standard_end)
    rec.is_early_exit = t < s
    db.commit()
    db.refresh(rec)
    return rec
