"""Headcount and movement: how many people, where they sit, who came and went.

These are the questions with no per-employee reading at all, so they are all
`company_wide` and Admin/HR only.
"""
from __future__ import annotations

from sqlalchemy import func

from app.chatbot.formatting import bullet_list, capitalize_first, join_names, sentence
from app.chatbot.period import in_words
from app.chatbot.registry import SkillContext, SkillResult, skill
from app.models.employee import Department, Designation, Employee

_ACTIVE = "Active"


@skill(
    name="workforce.headcount",
    topic="workforce",
    summary="Active headcount, split by department.",
    keywords={
        "headcount": 5.0, "head count": 5.0, "employee count": 4.5,
        "how many employees": 4.0, "how many people work": 4.5,
        "total employees": 4.0, "total staff": 4.0, "staff strength": 4.5,
        "team size": 3.5, "by department": 3.0, "department wise": 3.0,
        "how many staff": 4.0, "size of the company": 4.0,
        # Domain markers that mean this is a different question entirely.
        "absent": -3.5, "present": -2.5, "attendance": -3.0, "on leave": -3.5,
        "leave balance": -3.5, "payslip": -3.5, "net pay": -3.5,
    },
    company_wide=True,
    examples=("How many employees do we have?", "Headcount by department"),
)
def headcount(ctx: SkillContext) -> SkillResult | None:
    total = int(
        ctx.db.query(func.count(Employee.id))
        .filter(Employee.employment_status == _ACTIVE)
        .scalar()
        or 0
    )
    by_dept = (
        ctx.db.query(Department.name, func.count(Employee.id))
        .join(Employee, Employee.department_id == Department.id)
        .filter(Employee.employment_status == _ACTIVE)
        .group_by(Department.name)
        .order_by(func.count(Employee.id).desc())
        .all()
    )
    # Employees with no department set would vanish from that group-by, so they
    # are counted separately rather than silently dropped from the total.
    unassigned = int(
        ctx.db.query(func.count(Employee.id))
        .filter(Employee.employment_status == _ACTIVE, Employee.department_id.is_(None))
        .scalar()
        or 0
    )
    by_status = dict(
        ctx.db.query(Employee.employment_status, func.count(Employee.id))
        .group_by(Employee.employment_status)
        .all()
    )

    parts = [f"There are {total} active employee(s)."]
    if by_dept:
        listed = ", ".join(f"{name} {count}" for name, count in by_dept[:6])
        more = len(by_dept) - 6
        parts.append(f"By department: {listed}" + (f" and {more} more." if more > 0 else "."))
    if unassigned:
        parts.append(f"{unassigned} have no department set.")
    inactive = sum(int(c) for s, c in by_status.items() if s != _ACTIVE)
    if inactive:
        parts.append(f"{inactive} former employee(s) on record.")

    return SkillResult(
        text=sentence(parts),
        data={
            "total_active": total,
            "by_department": [{"department": n, "count": int(c)} for n, c in by_dept],
            "unassigned_department": unassigned,
            "by_status": {str(k): int(v) for k, v in by_status.items()},
        },
        sources=("employees", "departments"),
    )


@skill(
    name="workforce.movement",
    topic="workforce",
    summary="Who joined and who left in a month.",
    keywords={
        "joined": 3.5, "joiners": 4.5, "new joiners": 5.0, "new hires": 4.5,
        "left the company": 4.5, "left": 3.0, "people left": 4.5,
        "resigned": 3.5, "attrition": 4.5,
        "exits": 4.0, "leavers": 4.5, "who joined": 4.5, "onboarded": 3.0,
        "turnover": 4.0,
    },
    company_wide=True,
    examples=("Who joined this month?", "How many people left last month?"),
)
def movement(ctx: SkillContext) -> SkillResult | None:
    start, end = ctx.period.bounds()

    joiners = (
        ctx.db.query(Employee.first_name, Employee.last_name, Employee.date_of_joining)
        .filter(Employee.date_of_joining >= start, Employee.date_of_joining <= end)
        .order_by(Employee.date_of_joining)
        .all()
    )
    exits = (
        ctx.db.query(Employee.first_name, Employee.last_name, Employee.date_of_leaving)
        .filter(
            Employee.date_of_leaving.isnot(None),
            Employee.date_of_leaving >= start,
            Employee.date_of_leaving <= end,
        )
        .order_by(Employee.date_of_leaving)
        .all()
    )
    joiner_names = [" ".join(p for p in (r[0], r[1]) if p).strip() for r in joiners]
    exit_names = [" ".join(p for p in (r[0], r[1]) if p).strip() for r in exits]

    if not joiner_names and not exit_names:
        return SkillResult(
            text=f"Nobody joined or left {in_words(ctx.period.label)}.",
            data={"joiners": 0, "exits": 0},
            sources=("employees",),
        )

    parts = [
        f"{capitalize_first(in_words(ctx.period.label))}: {len(joiner_names)} joined, "
        f"{len(exit_names)} left."
    ]
    if joiner_names:
        parts.append(f"Joined: {join_names(joiner_names)}.")
    if exit_names:
        parts.append(f"Left: {join_names(exit_names)}.")

    return SkillResult(
        text=sentence(parts),
        data={
            "period": ctx.period.label,
            "joiners": len(joiner_names), "exits": len(exit_names),
            "joiner_names": joiner_names[:20], "exit_names": exit_names[:20],
        },
        sources=("employees",),
    )


@skill(
    name="workforce.structure",
    topic="workforce",
    summary="The list of departments and designations in use.",
    keywords={
        "departments": 4.0, "what departments": 4.5, "designations": 4.5,
        "job titles": 4.0, "roles in the company": 4.0, "org structure": 4.5,
        "organisation structure": 4.5, "organization structure": 4.5,
        "which departments": 4.5,
    },
    company_wide=True,
    examples=("What departments do we have?", "List all designations"),
)
def structure(ctx: SkillContext) -> SkillResult | None:
    wants_designations = any(
        word in ctx.low for word in ("designation", "job title", "title", "role")
    )

    if wants_designations:
        rows = (
            ctx.db.query(Designation.title, func.count(Employee.id))
            .outerjoin(Employee, Employee.designation_id == Designation.id)
            .group_by(Designation.title)
            .order_by(Designation.title)
            .all()
        )
        if not rows:
            return None
        lines = [f"{title} — {int(count)} employee(s)" for title, count in rows]
        return SkillResult(
            text=f"{len(rows)} designation(s):\n" + bullet_list(lines, limit=20),
            data={"designations": [{"title": t, "count": int(c)} for t, c in rows]},
            sources=("designations",),
        )

    rows = (
        ctx.db.query(Department.name, func.count(Employee.id))
        .outerjoin(Employee, Employee.department_id == Department.id)
        .group_by(Department.name)
        .order_by(Department.name)
        .all()
    )
    if not rows:
        return None
    lines = [f"{name} — {int(count)} employee(s)" for name, count in rows]
    return SkillResult(
        text=f"{len(rows)} department(s):\n" + bullet_list(lines, limit=20),
        data={"departments": [{"name": n, "count": int(c)} for n, c in rows]},
        sources=("departments",),
    )
