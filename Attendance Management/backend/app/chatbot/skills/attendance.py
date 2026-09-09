"""Attendance, for one employee or for the whole company.

Per-employee answers delegate to `monthly_attendance_summary`, which already
encodes the holiday, weekly-off and half-day rules and the "don't penalise days
that haven't happened yet" behaviour. Re-deriving any of that here would
guarantee the chatbot eventually disagrees with the Attendance page.

The company-wide answers are aggregates the service does not provide, so they
are computed here — but only as counts of what is *recorded*, never as a
company attendance percentage. A single percentage would have to decide how to
weight part-months, new joiners and days that have not elapsed, and whatever it
chose would quietly contradict the per-employee figures.
"""
from __future__ import annotations

from sqlalchemy import func

from app.chatbot.formatting import capitalize_first, join_names, plural, sentence
from app.chatbot.period import in_words
from app.chatbot.registry import SkillContext, SkillResult, skill
from app.models.attendance import AttendanceRecord
from app.models.employee import Employee
from app.services.attendance_service import monthly_attendance_summary

_ACTIVE = "Active"


def _active_headcount(db) -> int:
    return int(
        db.query(func.count(Employee.id))
        .filter(Employee.employment_status == _ACTIVE)
        .scalar()
        or 0
    )


@skill(
    name="attendance.employee_month",
    topic="attendance",
    summary="One employee's attendance for a month: present, absent, leave, late, %.",
    keywords={
        "attendance": 3.0, "absent": 3.0, "present": 2.5, "half day": 2.5,
        "half-day": 2.5, "working days": 2.0, "attendance percentage": 3.5,
        "days did i work": 3.0, "how many days": 2.0, "leave days": -1.0,
        "punch": 1.5, "check in": 1.5, "check-in": 1.5,
    },
    about_employee=True,
    examples=("How many days was Priya absent last month?",
              "What is my attendance for August?"),
)
def employee_month(ctx: SkillContext) -> SkillResult | None:
    employee = ctx.employee
    if employee is None:
        return None

    period = ctx.period
    summary = monthly_attendance_summary(ctx.db, employee.id, period.month, period.year)
    if not summary:
        return None

    label = period.label if not period.assumed else "this month"
    parts = [
        f"{capitalize_first(ctx.possessive())} attendance {in_words(label)}: "
        f"present {summary['present']} of {summary['working_days']} working day(s)."
    ]
    if summary.get("absent"):
        parts.append(f"Absent {summary['absent']}.")
    if summary.get("leave"):
        parts.append(f"On approved leave {summary['leave']}.")
    if summary.get("half_day"):
        parts.append(f"Half days {summary['half_day']}.")

    late = int(
        ctx.db.query(func.count(AttendanceRecord.id))
        .filter(
            AttendanceRecord.employee_id == employee.id,
            AttendanceRecord.date >= period.bounds()[0],
            AttendanceRecord.date <= period.bounds()[1],
            AttendanceRecord.is_late.is_(True),
        )
        .scalar()
        or 0
    )
    if late:
        parts.append(f"Late marks {late}.")
    parts.append(f"Attendance {summary['attendance_percentage']}%.")
    if period.assumed:
        parts.append("(I assumed the current month.)")

    return SkillResult(
        text=sentence(parts),
        data={"summary": summary, "late_days": late},
        sources=("attendance_records", "monthly_attendance_summary"),
    )


@skill(
    name="attendance.employee_day",
    topic="attendance",
    summary="Whether one employee was in on a particular day, and their timings.",
    keywords={
        "today": 2.0, "yesterday": 2.0, "sign in": 2.5, "sign out": 2.5,
        "what time": 3.0, "come in": 2.0, "in today": 3.0, "checked in": 3.0,
        "working today": 3.0, "at work": 2.5,
    },
    about_employee=True,
    examples=("Did Priya come in today?", "What time did Amit sign in?"),
)
def employee_day(ctx: SkillContext) -> SkillResult | None:
    employee = ctx.employee
    if employee is None:
        return None

    when = ctx.period.on()
    record = (
        ctx.db.query(AttendanceRecord)
        .filter(AttendanceRecord.employee_id == employee.id, AttendanceRecord.date == when)
        .first()
    )
    label = ctx.period.label if ctx.period.is_day else when.strftime("%d %b %Y")

    if record is None:
        return SkillResult(
            text=f"There is no attendance record for {ctx.subject()} on {label}.",
            data={"date": when.isoformat(), "found": False},
            sources=("attendance_records",),
        )

    parts = [f"{capitalize_first(ctx.subject())} — {label}: {record.status.replace('_', ' ').lower()}."]
    if record.sign_in_time:
        parts.append(f"Signed in {record.sign_in_time.strftime('%H:%M')}.")
    if record.sign_out_time:
        parts.append(f"Signed out {record.sign_out_time.strftime('%H:%M')}.")
    if record.total_work_hours:
        parts.append(f"Worked {record.total_work_hours} hour(s).")
    if record.is_late:
        parts.append("Marked late.")

    return SkillResult(
        text=sentence(parts),
        data={
            "date": when.isoformat(), "found": True, "status": record.status,
            "sign_in": record.sign_in_time.strftime("%H:%M") if record.sign_in_time else None,
            "sign_out": record.sign_out_time.strftime("%H:%M") if record.sign_out_time else None,
            "is_late": bool(record.is_late),
        },
        sources=("attendance_records",),
    )


@skill(
    name="attendance.company_day",
    topic="attendance",
    summary="Company roll-call for a day: present, absent, on leave, not yet marked.",
    keywords={
        "how many": 2.0, "absent": 2.5, "present": 2.0, "in the office": 3.0,
        "in office": 3.0, "attendance": 2.0, "who is absent": 4.0,
        "turned up": 3.0, "late": 2.0, "everyone": 2.0, "all employees": 2.0,
        "company": 1.5, "today": 1.5,
    },
    company_wide=True,
    examples=("How many people are absent today?", "Who is in the office today?"),
)
def company_day(ctx: SkillContext) -> SkillResult | None:
    when = ctx.period.on()
    rows = (
        ctx.db.query(AttendanceRecord.status, func.count(AttendanceRecord.id))
        .join(Employee, Employee.id == AttendanceRecord.employee_id)
        .filter(AttendanceRecord.date == when, Employee.employment_status == _ACTIVE)
        .group_by(AttendanceRecord.status)
        .all()
    )
    by_status = {(s or "UNKNOWN").upper(): int(c) for s, c in rows}
    recorded = sum(by_status.values())
    total = _active_headcount(ctx.db)

    late = int(
        ctx.db.query(func.count(AttendanceRecord.id))
        .join(Employee, Employee.id == AttendanceRecord.employee_id)
        .filter(
            AttendanceRecord.date == when,
            AttendanceRecord.is_late.is_(True),
            Employee.employment_status == _ACTIVE,
        )
        .scalar()
        or 0
    )

    label = ctx.period.label if ctx.period.is_day else when.strftime("%d %b %Y")
    present = by_status.get("PRESENT", 0)
    absent = by_status.get("ABSENT", 0)
    parts = [f"{capitalize_first(label)}: {present} of {total} active employee(s) marked present."]
    if absent:
        parts.append(f"Absent {absent}.")
    if by_status.get("ON_LEAVE"):
        parts.append(f"On leave {by_status['ON_LEAVE']}.")
    if by_status.get("HALF_DAY"):
        parts.append(f"Half day {by_status['HALF_DAY']}.")
    if late:
        parts.append(f"Late {late}.")
    # Stated separately, never folded into "absent": nobody has marked these
    # people anything, and calling them absent would contradict the Attendance page.
    not_marked = max(0, total - recorded)
    if not_marked:
        parts.append(f"Not yet marked {not_marked}.")

    # Names, when the question asked "who" rather than "how many".
    names: list[str] = []
    if "who" in ctx.low and absent:
        rows = (
            ctx.db.query(Employee.first_name, Employee.last_name)
            .join(AttendanceRecord, AttendanceRecord.employee_id == Employee.id)
            .filter(
                AttendanceRecord.date == when,
                AttendanceRecord.status == "ABSENT",
                Employee.employment_status == _ACTIVE,
            )
            .order_by(Employee.first_name)
            .all()
        )
        names = [" ".join(p for p in r if p).strip() for r in rows]
        if names:
            parts.append(f"Absent: {join_names(names)}.")

    return SkillResult(
        text=sentence(parts),
        data={
            "date": when.isoformat(), "total_active": total, "recorded": recorded,
            "present": present, "absent": absent,
            "on_leave": by_status.get("ON_LEAVE", 0),
            "half_day": by_status.get("HALF_DAY", 0),
            "late": late, "not_marked": not_marked, "absent_names": names[:20],
        },
        sources=("attendance_records", "employees"),
    )


@skill(
    name="attendance.company_month",
    topic="attendance",
    summary="Company attendance totals recorded across a month.",
    keywords={
        "attendance": 2.0, "this month": 2.0, "last month": 2.5,
        "monthly": 2.5, "for the month": 3.0, "overall": 2.0, "summary": 2.0,
        "company": 2.0, "total attendance": 3.5,
    },
    company_wide=True,
    examples=("Company attendance for August", "Overall attendance last month"),
)
def company_month(ctx: SkillContext) -> SkillResult | None:
    period = ctx.period
    if period.is_day:
        return None  # a day question belongs to company_day

    start, end = period.bounds()
    rows = (
        ctx.db.query(AttendanceRecord.status, func.count(AttendanceRecord.id))
        .join(Employee, Employee.id == AttendanceRecord.employee_id)
        .filter(
            AttendanceRecord.date >= start,
            AttendanceRecord.date <= end,
            Employee.employment_status == _ACTIVE,
        )
        .group_by(AttendanceRecord.status)
        .all()
    )
    by_status = {(s or "UNKNOWN").upper(): int(c) for s, c in rows}
    if not by_status:
        return None

    total = _active_headcount(ctx.db)
    text = (
        f"Across {total} active employee(s) {in_words(period.label)}, the HRMS "
        f"recorded {by_status.get('PRESENT', 0)} present day(s), "
        f"{by_status.get('ABSENT', 0)} absent, "
        f"{by_status.get('ON_LEAVE', 0)} on leave and "
        f"{by_status.get('HALF_DAY', 0)} half day(s)."
    )
    return SkillResult(
        text=text,
        data={"period": period.label, "total_active": total, "by_status": by_status},
        sources=("attendance_records", "employees"),
    )
