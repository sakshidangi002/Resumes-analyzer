"""Per-employee breakdowns — "one by one", "employee wise", "for each employee".

The company skills answer with a total; these answer with the rows behind it.
"What did the payroll total?" and "payroll for each employee" are genuinely
different questions, and answering the second with the first is the kind of
almost-right reply that makes a chatbot useless.

All three are `company_wide`: a breakdown across staff is an Admin/HR view. They
share a vocabulary of collective markers ("each", "per employee", "one by one"),
kept in `_BREAKDOWN` so the three stay in step and a fourth is cheap to add.
"""
from __future__ import annotations

from sqlalchemy import and_, func, or_

from app.chatbot.formatting import bullet_list, capitalize_first, days, inr, to_float
from app.chatbot.period import MONTH_NAMES, in_words
from app.chatbot.registry import SkillContext, SkillResult, skill
from app.models.attendance import AttendanceRecord
from app.models.employee import Employee
from app.models.leave import LeaveRequest, LeaveType
from app.models.payroll import Payslip, PayrollPeriod

_ACTIVE = "Active"

#: How many rows a chat bubble can usefully hold before it stops being an answer.
_ROW_LIMIT = 25

#: A breakdown skill answers ONLY when the question explicitly asks for the rows.
#: Without this gate each one competes on its own domain word — "show leave"
#: scored higher for leave-per-employee than for a leave balance, because
#: "leave" appears in both — and the breakdowns quietly swallowed their whole
#: domain. Registered as `requires=`, so the skill does not compete at all
#: unless one of these is present.
COLLECTIVE = (
    "each employee", "per employee", "every employee", "employee wise",
    "employeewise", "each person", "person wise", "name wise", "one by one",
    "by each", "of each", "each of them", "individually", "breakdown",
    "break down", "for everyone", "all employees", "every employee",
    "everyone", "everybody", "highest", "lowest", "top earner", "highest paid",
    "who took", "who has taken", "ranked", "sorted",
)

#: Phrases that ask for the rows rather than the total. Weighted high because
#: they are unambiguous: nobody says "for each employee" and wants one number.
_BREAKDOWN = {
    "each employee": 5.0, "per employee": 5.0, "every employee": 5.0,
    "employee wise": 5.0, "employeewise": 5.0, "each person": 4.5,
    "one by one": 4.5, "by each": 4.5, "of each": 4.0, "individually": 4.0,
    "breakdown": 4.0, "break down": 4.0, "for everyone": 4.0,
    "each of them": 4.0, "person wise": 4.5, "name wise": 4.5,
    "who took": 3.0, "list": 1.5, "all employees": 2.0,
}


def _name(first: str | None, last: str | None) -> str:
    return " ".join(p for p in (first, last) if p).strip()


@skill(
    name="breakdown.leave",
    topic="breakdown",
    summary="Leave days taken by each employee over a period.",
    keywords={
        **_BREAKDOWN,
        "leave": 3.0, "leaves": 3.0, "leave used": 4.5, "leave taken": 4.5,
        "days off": 3.0,
        # Other domains asking for a breakdown are a different skill.
        "payroll": -3.0, "payslip": -3.5, "salary": -3.0, "net pay": -3.5,
        "attendance": -2.5, "absent": -2.5, "present": -2.0,
    },
    company_wide=True,
    requires=COLLECTIVE,
    examples=("Leave used by each employee last month",
              "Employee wise leave for August"),
)
def leave_breakdown(ctx: SkillContext) -> SkillResult | None:
    start, end = ctx.period.bounds()

    rows = (
        ctx.db.query(
            Employee.first_name, Employee.last_name,
            LeaveRequest.start_date, LeaveRequest.end_date,
            LeaveRequest.is_half_day, LeaveType.code,
        )
        .join(LeaveRequest, LeaveRequest.employee_id == Employee.id)
        .join(LeaveType, LeaveType.id == LeaveRequest.leave_type_id)
        .filter(
            LeaveRequest.status == "APPROVED",
            LeaveRequest.start_date <= end,
            LeaveRequest.end_date >= start,
            Employee.employment_status == _ACTIVE,
        )
        .all()
    )
    if not rows:
        return SkillResult(
            text=f"No approved leave was taken {in_words(ctx.period.label)}.",
            data={"period": ctx.period.label, "count": 0},
            sources=("leave_requests", "employees"),
        )

    # Only the part of each request that falls inside the period counts — a
    # request spanning a month boundary must not be reported twice.
    totals: dict[str, float] = {}
    types: dict[str, set] = {}
    for first, last, req_start, req_end, is_half, code in rows:
        who = _name(first, last)
        overlap = (min(req_end, end) - max(req_start, start)).days + 1
        taken = 0.5 if (is_half and overlap == 1) else float(overlap)
        totals[who] = totals.get(who, 0.0) + taken
        types.setdefault(who, set()).add(code)

    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    lines = [
        f"{who} — {days(total)} day(s) ({', '.join(sorted(types[who]))})"
        for who, total in ordered
    ]
    grand = sum(totals.values())

    return SkillResult(
        text=(
            f"Leave taken {in_words(ctx.period.label)} — {len(ordered)} employee(s), "
            f"{days(grand)} day(s) in total:\n" + bullet_list(lines, limit=_ROW_LIMIT)
        ),
        data={
            "period": ctx.period.label,
            "total_days": grand,
            "employees": [{"name": w, "days": d} for w, d in ordered[:_ROW_LIMIT]],
        },
        sources=("leave_requests", "employees"),
    )


def _period_filter(period):
    """Which payroll months this period covers.

    Payroll is stored per month, not per date, so a cumulative range is
    "every period up to and including this one" rather than a date BETWEEN.
    """
    if period.cumulative:
        return or_(
            PayrollPeriod.year < period.year,
            and_(PayrollPeriod.year == period.year, PayrollPeriod.month <= period.month),
        )
    return and_(PayrollPeriod.month == period.month, PayrollPeriod.year == period.year)


@skill(
    name="breakdown.payroll",
    topic="breakdown",
    summary="Net pay for each employee in a payroll period.",
    keywords={
        **_BREAKDOWN,
        "payroll": 3.0, "salary": 3.0, "payslip": 3.5, "net pay": 3.5,
        "paid to": 3.5, "paid each": 4.5, "salary of each": 5.0,
        "leave": -3.0, "attendance": -2.5, "absent": -2.5,
    },
    company_wide=True,
    salary=True,
    requires=COLLECTIVE,
    examples=("Payroll for each employee last month",
              "How much was paid to each employee?"),
)
def payroll_breakdown(ctx: SkillContext) -> SkillResult | None:
    period = ctx.period

    def fetch(criterion):
        # Summed per employee: over a cumulative range one person has many
        # payslips, and listing each one is not what "per employee" asked for.
        return (
            ctx.db.query(
                Employee.first_name,
                Employee.last_name,
                func.sum(Payslip.net_salary),
                func.count(Payslip.id),
            )
            .join(Payslip, Payslip.employee_id == Employee.id)
            .join(PayrollPeriod, PayrollPeriod.id == Payslip.payroll_period_id)
            .filter(criterion)
            .group_by(Employee.id, Employee.first_name, Employee.last_name)
            .order_by(func.sum(Payslip.net_salary).desc())
            .all()
        )

    rows = fetch(_period_filter(period))
    # The label the person used, so "till 31 August" is not answered with the
    # words "to date" — the figure would be right and the range misreported.
    scope = period.label if period.cumulative else period.month_label
    fell_back = None

    if not rows and not period.cumulative:
        # The asked-about month has not been run. Answering "none" and stopping
        # is technically true and useless — offer the most recent run instead,
        # the same way the single-payslip skill does.
        latest = (
            ctx.db.query(PayrollPeriod.month, PayrollPeriod.year)
            .join(Payslip, Payslip.payroll_period_id == PayrollPeriod.id)
            .order_by(PayrollPeriod.year.desc(), PayrollPeriod.month.desc())
            .first()
        )
        if latest:
            rows = fetch(
                and_(PayrollPeriod.month == latest[0], PayrollPeriod.year == latest[1])
            )
            fell_back = f"{MONTH_NAMES[latest[0] - 1]} {latest[1]}"
            scope = fell_back

    if not rows:
        return SkillResult(
            text=f"No payslips have been generated {'' if period.cumulative else 'for '}"
                 f"{scope} yet.",
            data={"period": scope, "count": 0},
            sources=("payslips", "payroll_periods"),
        )

    lines = []
    total = 0.0
    for first, last, net, slips in rows:
        total += to_float(net)
        line = f"{_name(first, last)} — {inr(net)}"
        if int(slips or 0) > 1:
            line += f" ({int(slips)} payslips)"
        lines.append(line)

    heading = (
        f"Total payroll paid {scope} — {len(rows)} employee(s), {inr(total)}:"
        if period.cumulative
        else f"{scope} payroll, employee by employee — "
             f"{len(rows)} payslip(s), {inr(total)} in total:"
    )
    if fell_back:
        heading = (
            f"No payroll has been run for {period.month_label} yet. "
            f"The most recent is {fell_back} — {len(rows)} payslip(s), "
            f"{inr(total)} in total:"
        )

    return SkillResult(
        text=heading + "\n" + bullet_list(lines, limit=_ROW_LIMIT),
        data={
            "period": scope,
            "cumulative": period.cumulative,
            "total_net": total,
            "employees": [
                {"name": _name(f, l), "net_salary": to_float(n)}
                for f, l, n, _c in rows[:_ROW_LIMIT]
            ],
        },
        sources=("payslips", "payroll_periods", "employees"),
    )


@skill(
    name="breakdown.attendance",
    topic="breakdown",
    summary="Present and absent days for each employee over a period.",
    keywords={
        **_BREAKDOWN,
        "attendance": 3.5, "absent": 3.0, "present": 2.5, "late": 2.5,
        "leave": -2.5, "payroll": -3.0, "payslip": -3.5, "salary": -3.0,
    },
    company_wide=True,
    requires=COLLECTIVE,
    examples=("Attendance for each employee last month",
              "Employee wise absent days"),
)
def attendance_breakdown(ctx: SkillContext) -> SkillResult | None:
    start, end = ctx.period.bounds()

    rows = (
        ctx.db.query(
            Employee.first_name, Employee.last_name,
            AttendanceRecord.status, func.count(AttendanceRecord.id),
        )
        .join(AttendanceRecord, AttendanceRecord.employee_id == Employee.id)
        .filter(
            AttendanceRecord.date >= start,
            AttendanceRecord.date <= end,
            Employee.employment_status == _ACTIVE,
        )
        .group_by(Employee.first_name, Employee.last_name, AttendanceRecord.status)
        .all()
    )
    if not rows:
        return SkillResult(
            text=f"No attendance was recorded {in_words(ctx.period.label)}.",
            data={"period": ctx.period.label, "count": 0},
            sources=("attendance_records", "employees"),
        )

    per_person: dict[str, dict[str, int]] = {}
    for first, last, status, count in rows:
        who = _name(first, last)
        per_person.setdefault(who, {})[(status or "UNKNOWN").upper()] = int(count)

    ordered = sorted(
        per_person.items(), key=lambda kv: kv[1].get("ABSENT", 0), reverse=True
    )
    lines = []
    for who, counts in ordered:
        bits = [f"present {counts.get('PRESENT', 0)}"]
        if counts.get("ABSENT"):
            bits.append(f"absent {counts['ABSENT']}")
        if counts.get("ON_LEAVE"):
            bits.append(f"leave {counts['ON_LEAVE']}")
        if counts.get("HALF_DAY"):
            bits.append(f"half {counts['HALF_DAY']}")
        lines.append(f"{who} — {', '.join(bits)}")

    return SkillResult(
        text=(
            f"Attendance {in_words(ctx.period.label)}, employee by employee "
            f"({len(ordered)} employee(s)):\n" + bullet_list(lines, limit=_ROW_LIMIT)
        ),
        data={
            "period": ctx.period.label,
            "employees": [
                {"name": w, **{k.lower(): v for k, v in c.items()}}
                for w, c in ordered[:_ROW_LIMIT]
            ],
        },
        sources=("attendance_records", "employees"),
    )
