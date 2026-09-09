"""Employee directory questions: who someone is, how to reach them, who reports
to whom, and who works where.

Contact details are returned because the actor can already see them on the
Employees page — the chatbot is not a wider door than the UI, and `Actor`
already restricted which rows got this far.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import func

from app.chatbot import resolver
from app.chatbot.formatting import bullet_list, capitalize_first, join_names, pretty_date, sentence
from app.chatbot.registry import SkillContext, SkillResult, skill
from app.core.datetime_utils import get_ist_now
from app.models.employee import Department, Designation, Employee

_SOURCES = ("employees",)


def _tenure(joined: date | None) -> str:
    if not joined:
        return ""
    today = get_ist_now().date()
    months = (today.year - joined.year) * 12 + (today.month - joined.month)
    if today.day < joined.day:
        months -= 1
    months = max(0, months)
    years, rest = divmod(months, 12)
    if years and rest:
        return f"{years} year(s) {rest} month(s)"
    if years:
        return f"{years} year(s)"
    return f"{rest} month(s)"


@skill(
    name="people.profile",
    topic="people",
    summary="An employee's role, department, manager, joining date and status.",
    keywords={
        "who is": 3.0, "profile": 3.0, "detail": 2.0,
        "designation": 3.0, "role": 2.0, "job title": 3.0, "department": 2.0,
        "information": 1.5, "tell me about": 2.5, "works as": 2.5,
        "position": 2.0, "employee code": 2.5,
        # A bare "about" matched "tell me ABOUT last month leave" and answered
        # with the asker's own profile, so it is gone. The domain markers are
        # negative for the same reason: a profile is never the answer to a
        # question that names a domain.
        "leave": -3.0, "payroll": -3.0, "payslip": -3.5, "salary": -3.0,
        "attendance": -3.0, "absent": -3.0, "each employee": -4.0,
        "per employee": -4.0, "pay": -2.5, "ctc": -3.5, "holiday": -3.0,
    },
    about_employee=True,
    examples=("Who is Priya Sharma?", "What is Priya's designation?"),
)
def profile(ctx: SkillContext) -> SkillResult | None:
    employee = ctx.employee
    if employee is None:
        return None

    who = employee.full_name if ctx.employee_named else "You"
    verb = "is" if ctx.employee_named else "are"
    parts = [f"{who} {verb} {employee.designation or 'an employee'}"]
    if employee.department:
        parts[0] += f" in {employee.department}"
    parts[0] += f" ({employee.employee_code})."

    if employee.date_of_joining:
        tenure = _tenure(employee.date_of_joining)
        parts.append(
            f"Joined {pretty_date(employee.date_of_joining)}"
            + (f" — {tenure} with the company." if tenure else ".")
        )
    if employee.manager_name:
        parts.append(f"Reports to {employee.manager_name}.")
    if employee.employment_status and employee.employment_status != "Active":
        parts.append(f"Employment status: {employee.employment_status}.")

    return SkillResult(
        text=sentence(parts),
        data={
            "employee_code": employee.employee_code,
            "designation": employee.designation,
            "department": employee.department,
            "manager": employee.manager_name,
            "date_of_joining": (
                employee.date_of_joining.isoformat() if employee.date_of_joining else None
            ),
            "employment_status": employee.employment_status,
        },
        sources=_SOURCES,
    )


@skill(
    name="people.contact",
    topic="people",
    summary="An employee's official email and phone number.",
    keywords={
        "email": 3.5, "e-mail": 3.5, "phone": 3.5, "mobile": 3.0, "number": 1.5,
        "contact": 3.5, "reach": 2.0, "call": 1.5,
    },
    about_employee=True,
    examples=("What is Priya's email?", "Give me Amit's contact details"),
)
def contact(ctx: SkillContext) -> SkillResult | None:
    employee = ctx.employee
    if employee is None:
        return None
    if not employee.official_email and not employee.phone:
        return None

    bits = []
    if employee.official_email:
        bits.append(f"email {employee.official_email}")
    if employee.phone:
        bits.append(f"phone {employee.phone}")
    return SkillResult(
        text=f"{capitalize_first(ctx.possessive())} contact — {', '.join(bits)}.",
        data={"email": employee.official_email, "phone": employee.phone},
        sources=_SOURCES,
    )


@skill(
    name="people.reporting",
    topic="people",
    summary="Reporting line: an employee's manager, and who reports to them.",
    keywords={
        "manager": 3.5, "reports to": 4.0, "reporting": 3.0, "reportees": 4.0,
        "direct reports": 4.0, "team": 2.0, "reports": 2.0, "supervisor": 3.0,
        "who works under": 4.0,
    },
    about_employee=True,
    examples=("Who reports to Priya?", "Who is Amit's manager?"),
)
def reporting(ctx: SkillContext) -> SkillResult | None:
    employee = ctx.employee
    if employee is None:
        return None

    rows = (
        ctx.db.query(Employee.first_name, Employee.last_name)
        .filter(
            Employee.reporting_manager_id == employee.id,
            Employee.employment_status == "Active",
        )
        .order_by(Employee.first_name)
        .all()
    )
    names = [" ".join(p for p in row if p).strip() for row in rows]

    parts = []
    if employee.manager_name:
        parts.append(f"{capitalize_first(ctx.possessive())} manager is {employee.manager_name}.")
    else:
        parts.append(f"{capitalize_first(ctx.subject())} has no reporting manager set.")
    if names:
        parts.append(f"{len(names)} person(s) report in: {join_names(names)}.")
    else:
        parts.append("Nobody reports in.")

    return SkillResult(
        text=sentence(parts),
        data={"manager": employee.manager_name, "reportee_count": len(names),
              "reportees": names[:20]},
        sources=_SOURCES,
    )


@skill(
    name="people.directory",
    topic="people",
    summary="List employees, optionally filtered by department or designation.",
    keywords={
        "list": 3.0, "who works": 3.5, "everyone in": 3.5, "all employees": 3.0,
        "show me the": 1.5, "employees in": 3.5, "people in": 3.0,
        "members of": 2.5, "who is in": 3.0, "names of": 2.5,
    },
    examples=("Who works in Engineering?", "List employees in Sales"),
)
def directory(ctx: SkillContext) -> SkillResult | None:
    department = resolver.find_department(ctx.db, ctx.question)
    people = resolver.search(ctx.db, ctx.actor, department=department, limit=60)
    if not people:
        return None

    where = f" in {department}" if department else ""
    lines = [
        f"{p.full_name} — {p.designation or 'no designation'} ({p.employee_code})"
        for p in people
    ]
    return SkillResult(
        text=f"{len(people)} active employee(s){where}:\n" + bullet_list(lines, limit=15),
        data={
            "department": department,
            "count": len(people),
            "employees": [
                {"name": p.full_name, "code": p.employee_code,
                 "designation": p.designation, "department": p.department}
                for p in people[:30]
            ],
        },
        sources=_SOURCES,
    )


@skill(
    name="people.tenure",
    topic="people",
    summary="How long someone has been with the company, and their work anniversary.",
    keywords={
        "how long": 3.5, "tenure": 4.0, "anniversary": 4.0, "years of service": 4.0,
        "date of joining": 3.5, "joining date": 3.5, "when did": 2.5,
        "been with": 3.0, "service": 1.5,
    },
    about_employee=True,
    examples=("How long has Priya worked here?", "When did Amit join?"),
)
def tenure(ctx: SkillContext) -> SkillResult | None:
    employee = ctx.employee
    if employee is None or not employee.date_of_joining:
        return None

    joined = employee.date_of_joining
    length = _tenure(joined)
    today = get_ist_now().date()
    next_anniversary = date(
        today.year + (1 if (today.month, today.day) > (joined.month, joined.day) else 0),
        joined.month, joined.day,
    ) if (joined.month, joined.day) != (2, 29) else None

    parts = [
        f"{capitalize_first(ctx.subject())} joined on {pretty_date(joined)}"
        + (f" — {length} of service." if length else ".")
    ]
    if next_anniversary:
        parts.append(f"Next work anniversary: {pretty_date(next_anniversary)}.")

    return SkillResult(
        text=sentence(parts),
        data={"date_of_joining": joined.isoformat(), "tenure": length,
              "next_anniversary": next_anniversary.isoformat() if next_anniversary else None},
        sources=_SOURCES,
    )
