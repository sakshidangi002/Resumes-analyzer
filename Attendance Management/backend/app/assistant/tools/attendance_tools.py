"""Attendance tools — wrappers over `app.services.attendance_service`.

Each returns a plain JSON-safe dict so graph state stays small and loggable.
None of them re-derive attendance status: `monthly_attendance_summary` already
encodes the holiday / weekly-off / half-day rules and the "don't penalise days
that haven't happened yet" behaviour, and duplicating that here would guarantee
the chatbot eventually disagrees with the Attendance page.
"""
from __future__ import annotations

import calendar as _calendar
from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from app.models.attendance import AttendanceRecord
from app.models.employee import Employee, EmploymentStatus
from app.services.attendance_service import monthly_attendance_summary


def monthly_summary(db: Session, *, employee_id: int, month: int, year: int) -> dict[str, Any]:
    """Bucketed attendance for one calendar month.

    Straight delegation — the returned dict is the service's own contract
    (present / absent / leave / half_day / holiday / weekly_off / percentage).
    """
    return monthly_attendance_summary(db, employee_id, month, year)


def employee_period_context(
    db: Session, *, employee_id: int, month: int, year: int
) -> dict[str, Any]:
    """Facts needed to judge whether the requested period is even meaningful.

    Read independently of the summary (different tables, no ordering dependency),
    which is why the attendance node fetches the two in parallel.
    """
    emp = (
        db.query(Employee.date_of_joining, Employee.employment_status)
        .filter(Employee.id == employee_id)
        .first()
    )
    if emp is None:
        return {"employee_found": False}

    doj, status = emp[0], emp[1]
    month_end = date(year, month, _calendar.monthrange(year, month)[1])
    return {
        "employee_found": True,
        "date_of_joining": doj.isoformat() if doj else None,
        "is_active": status == EmploymentStatus.ACTIVE.value,
        # True when the employee had not joined by the last day of the period —
        # the summary would report a month of "absent" days that never applied.
        "joined_after_period": bool(doj and doj > month_end),
    }


def late_mark_count(db: Session, *, employee_id: int, month: int, year: int) -> dict[str, Any]:
    """How many days in the month were flagged late.

    `is_late` is set by the attendance pipeline; this only counts it.
    """
    total_days = _calendar.monthrange(year, month)[1]
    rows = (
        db.query(AttendanceRecord.is_late)
        .filter(
            AttendanceRecord.employee_id == employee_id,
            AttendanceRecord.date >= date(year, month, 1),
            AttendanceRecord.date <= date(year, month, total_days),
        )
        .all()
    )
    return {"late_days": sum(1 for (flag,) in rows if flag)}


#: Registry consumed by the attendance node and by the docs generator. Keeping
#: it explicit means a new tool is a one-line, reviewable addition.
ATTENDANCE_TOOLS = {
    "monthly_summary": monthly_summary,
    "employee_period_context": employee_period_context,
    "late_mark_count": late_mark_count,
}
