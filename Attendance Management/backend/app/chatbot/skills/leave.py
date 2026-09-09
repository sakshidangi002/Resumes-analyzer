"""Leave: balances, an employee's requests, who is off, and what needs approving.

Balances come from `paid_leave_summary`, which owns the monthly-accrual rule and
the paid/unpaid (LWP) split. Nothing here recomputes a balance.

Nothing here *writes*, either. `apply_leave_request` and `approve_leave_request`
live in the same service and are deliberately unreachable: applying for or
approving leave through a chat message would bypass the workflow the Leave page
enforces, and a 1.5B model's reading of "book me off next Friday" is not a sound
basis for a database write.
"""
from __future__ import annotations

from app.chatbot.formatting import capitalize_first, days, join_names, sentence, to_float
from app.chatbot.period import in_words
from app.chatbot.registry import SkillContext, SkillResult, skill
from app.models.employee import Employee
from app.models.leave import LeaveRequest, LeaveType
from app.services.leave_service import get_current_financial_year, paid_leave_summary

_ACTIVE = "Active"


@skill(
    name="leave.balance",
    topic="leave",
    summary="Paid-leave balance for the financial year: earned, used, remaining.",
    keywords={
        "leave balance": 4.5, "balance": 2.5, "leaves left": 4.0, "leave left": 4.0,
        "how many leaves": 4.0, "leave": 2.0, "casual leave": 3.0, "sick leave": 3.0,
        "paid leave": 3.5, "remaining": 2.5, "earned": 2.0, "entitlement": 3.0,
        "lop": 2.0, "loss of pay": 3.0, "unpaid": 2.0,
    },
    about_employee=True,
    examples=("What is my leave balance?", "How many leaves does Priya have left?"),
)
def balance(ctx: SkillContext) -> SkillResult | None:
    employee = ctx.employee
    if employee is None:
        return None

    fy = get_current_financial_year(ctx.db)
    if fy is None:
        return None
    summary = paid_leave_summary(ctx.db, employee.id, fy)

    remaining = to_float(summary.get("remaining"))
    earned = to_float(summary.get("earned"))
    used = to_float(summary.get("used_paid"))
    annual = to_float(summary.get("annual_days"))
    unpaid = to_float(summary.get("unpaid_used"))

    parts = [
        f"{capitalize_first(ctx.possessive())} paid-leave balance for FY {fy.name} is "
        f"{days(remaining)} day(s) — {days(earned)} earned so far, {days(used)} used."
    ]
    if annual:
        parts.append(f"Annual entitlement {days(annual)} day(s).")
    if unpaid:
        parts.append(f"Unpaid (LOP) days this year: {days(unpaid)}.")

    return SkillResult(
        text=sentence(parts),
        data={
            "financial_year": fy.name, "remaining": remaining, "earned": earned,
            "used_paid": used, "annual_days": annual, "unpaid_used": unpaid,
        },
        sources=("leave_allocations", "paid_leave_summary"),
    )


@skill(
    name="leave.requests",
    topic="leave",
    summary="One employee's leave requests and their statuses.",
    keywords={
        # No bare "requests": the matcher already tolerates the plural, so
        # declaring both made "leave requests" score twice and outrank the
        # pending-approvals skill on the word "pending".
        "leave request": 4.0, "applied": 3.0, "leave history": 4.0,
        "leave taken": 3.5, "took leave": 3.5, "on leave": 2.0,
        "approved": 2.5, "rejected": 2.5, "cancelled": 2.5,
    },
    about_employee=True,
    examples=("Has Priya applied for any leave?", "Show my leave requests"),
)
def requests(ctx: SkillContext) -> SkillResult | None:
    employee = ctx.employee
    if employee is None:
        return None

    fy = get_current_financial_year(ctx.db)
    if fy is None:
        return None

    rows = (
        ctx.db.query(
            LeaveRequest.status, LeaveRequest.start_date, LeaveRequest.end_date,
            LeaveType.code,
        )
        .join(LeaveType, LeaveType.id == LeaveRequest.leave_type_id)
        .filter(
            LeaveRequest.employee_id == employee.id,
            LeaveRequest.start_date <= fy.end_date,
            LeaveRequest.end_date >= fy.start_date,
        )
        .order_by(LeaveRequest.start_date.desc())
        .all()
    )
    if not rows:
        return SkillResult(
            text=f"{capitalize_first(ctx.subject())} has no leave requests in FY {fy.name}.",
            data={"financial_year": fy.name, "count": 0},
            sources=("leave_requests",),
        )

    by_status: dict[str, int] = {}
    for status, _start, _end, _code in rows:
        key = (status or "UNKNOWN").upper()
        by_status[key] = by_status.get(key, 0) + 1

    breakdown = ", ".join(f"{count} {status.lower()}" for status, count in by_status.items())
    latest = rows[0]
    parts = [
        f"{capitalize_first(ctx.subject())} has {len(rows)} leave request(s) in "
        f"FY {fy.name}: {breakdown}."
    ]
    parts.append(
        f"Most recent: {latest[3]} from {latest[1]:%d %b} to {latest[2]:%d %b}, "
        f"{(latest[0] or '').lower()}."
    )
    return SkillResult(
        text=sentence(parts),
        data={"financial_year": fy.name, "count": len(rows), "by_status": by_status},
        sources=("leave_requests",),
    )


@skill(
    name="leave.who_is_off",
    topic="leave",
    summary="Who is on approved leave on a given day.",
    keywords={
        "who is on leave": 5.0, "who are on leave": 5.0, "on leave today": 4.5,
        "off today": 4.0, "who is off": 4.5, "on leave": 2.5, "away": 2.0,
        "how many on leave": 4.0, "anyone on leave": 4.0,
    },
    company_wide=True,
    examples=("Who is on leave today?", "How many people are on leave?"),
)
def who_is_off(ctx: SkillContext) -> SkillResult | None:
    when = ctx.period.on()
    rows = (
        ctx.db.query(Employee.first_name, Employee.last_name, LeaveType.code)
        .join(LeaveRequest, LeaveRequest.employee_id == Employee.id)
        .join(LeaveType, LeaveType.id == LeaveRequest.leave_type_id)
        .filter(
            LeaveRequest.status == "APPROVED",
            LeaveRequest.start_date <= when,
            LeaveRequest.end_date >= when,
            Employee.employment_status == _ACTIVE,
        )
        .order_by(Employee.first_name)
        .all()
    )
    label = ctx.period.label if ctx.period.is_day else when.strftime("%d %b %Y")
    names = [" ".join(p for p in (r[0], r[1]) if p).strip() for r in rows]

    if not names:
        text = f"Nobody is on approved leave {in_words(label)}."
    else:
        text = (
            f"{len(names)} employee(s) are on approved leave {in_words(label)}: "
            f"{join_names(names)}."
        )
    return SkillResult(
        text=text,
        data={"date": when.isoformat(), "count": len(names), "names": names[:20]},
        sources=("leave_requests", "employees"),
    )


@skill(
    name="leave.pending_approvals",
    topic="leave",
    summary="Leave requests waiting for approval, and how long they have waited.",
    keywords={
        "pending": 4.0, "awaiting": 4.0, "waiting": 3.5, "to approve": 4.0,
        "approval": 3.5, "need approval": 4.5,
        "not approved": 3.0, "outstanding": 3.0,
        # "Pending advances" is a payroll question that happens to say "pending".
        "advance": -5.0, "payslip": -4.0, "salary": -3.0,
    },
    company_wide=True,
    examples=("How many leave requests are pending?",
              "What is waiting for my approval?"),
)
def pending_approvals(ctx: SkillContext) -> SkillResult | None:
    when = ctx.period.on()
    rows = (
        ctx.db.query(
            Employee.first_name, Employee.last_name,
            LeaveRequest.applied_at, LeaveRequest.start_date, LeaveType.code,
        )
        .join(LeaveRequest, LeaveRequest.employee_id == Employee.id)
        .join(LeaveType, LeaveType.id == LeaveRequest.leave_type_id)
        .filter(LeaveRequest.status == "PENDING")
        .order_by(LeaveRequest.applied_at.asc())
        .all()
    )
    if not rows:
        return SkillResult(
            text="There are no leave requests waiting for approval.",
            data={"count": 0},
            sources=("leave_requests",),
        )

    names = [" ".join(p for p in (r[0], r[1]) if p).strip() for r in rows]
    oldest = rows[0][2]
    oldest_days = None
    if oldest is not None:
        applied = oldest.date() if hasattr(oldest, "date") else oldest
        oldest_days = max(0, (when - applied).days)

    parts = [f"{len(rows)} leave request(s) are waiting for approval."]
    if oldest_days is not None:
        # The number that matters operationally: a backlog of three is fine, a
        # request sitting for eleven days is not.
        parts.append(f"The oldest has waited {oldest_days} day(s).")
    parts.append(f"Waiting on: {join_names(names)}.")

    return SkillResult(
        text=sentence(parts),
        data={"count": len(rows), "oldest_days": oldest_days, "names": names[:20]},
        sources=("leave_requests", "employees"),
    )
