"""Letters, onboarding tasks, daily status reports and HR queries.

The paperwork side of the HRMS. All read-only: the chatbot reports what has been
issued, submitted or raised, and never issues, submits or closes anything.
"""
from __future__ import annotations

from sqlalchemy import func

from app.chatbot.formatting import bullet_list, capitalize_first, join_names, pretty_date, sentence
from app.chatbot.period import in_words
from app.chatbot.registry import SkillContext, SkillResult, skill
from app.models.dsr import DailyStatusReport
from app.models.employee import Employee
from app.models.hr_query import HRQuery
from app.models.letter import LetterInstance, LetterTemplate
from app.models.onboarding import OnboardingTask

_ACTIVE = "Active"


@skill(
    name="documents.letters",
    topic="documents",
    summary="Letters issued to an employee, or recently across the company.",
    keywords={
        "letter": 4.5, "offer letter": 5.0, "appointment letter": 5.0,
        "experience letter": 5.0, "relieving": 4.5, "increment letter": 5.0,
        "document issued": 4.0, "issued": 2.5, "certificate": 3.5,
    },
    examples=("What letters has Priya been issued?", "Show recent letters"),
)
def letters(ctx: SkillContext) -> SkillResult | None:
    query = (
        ctx.db.query(
            Employee.first_name, Employee.last_name,
            LetterTemplate.name, LetterInstance.generated_at,
        )
        .join(LetterInstance, LetterInstance.employee_id == Employee.id)
        .join(LetterTemplate, LetterTemplate.id == LetterInstance.template_id)
    )
    if ctx.employee_named and ctx.employee:
        query = query.filter(LetterInstance.employee_id == ctx.employee.id)
    else:
        query = ctx.actor.scope_query(query)

    rows = query.order_by(LetterInstance.generated_at.desc()).limit(25).all()
    if not rows:
        who = ctx.subject() if ctx.employee_named else "anyone in scope"
        return SkillResult(
            text=f"No letters have been issued to {who}.",
            data={"count": 0}, sources=("letter_instances",),
        )

    if ctx.employee_named:
        lines = [
            f"{name} — {pretty_date(at.date() if hasattr(at, 'date') else at)}"
            for _f, _l, name, at in rows
        ]
        text = (
            f"{capitalize_first(ctx.subject())} has been issued {len(rows)} letter(s):\n"
            + bullet_list(lines, limit=12)
        )
    else:
        lines = [
            f"{' '.join(p for p in (f, l) if p)} — {name} "
            f"({pretty_date(at.date() if hasattr(at, 'date') else at)})"
            for f, l, name, at in rows
        ]
        text = f"{len(rows)} recent letter(s):\n" + bullet_list(lines, limit=12)

    return SkillResult(
        text=text, data={"count": len(rows)}, sources=("letter_instances", "letter_templates"),
    )


@skill(
    name="documents.onboarding",
    topic="documents",
    summary="Onboarding task progress for an employee, or what is outstanding.",
    keywords={
        "onboarding": 5.0, "onboard": 4.5, "induction": 4.0, "joining formalities": 5.0,
        "checklist": 4.0, "tasks": 2.5, "paperwork": 3.5, "documents pending": 4.0,
    },
    examples=("Is Priya's onboarding complete?", "What onboarding is outstanding?"),
)
def onboarding(ctx: SkillContext) -> SkillResult | None:
    query = (
        ctx.db.query(
            Employee.first_name, Employee.last_name,
            OnboardingTask.title, OnboardingTask.is_completed, OnboardingTask.due_date,
        )
        .join(OnboardingTask, OnboardingTask.employee_id == Employee.id)
    )
    if ctx.employee_named and ctx.employee:
        query = query.filter(OnboardingTask.employee_id == ctx.employee.id)
    else:
        query = ctx.actor.scope_query(query).filter(OnboardingTask.is_completed.is_(False))

    rows = query.order_by(OnboardingTask.sort_order, OnboardingTask.id).limit(60).all()
    if not rows:
        who = ctx.subject() if ctx.employee_named else "anyone in scope"
        return SkillResult(
            text=f"There are no onboarding tasks on record for {who}."
                 if ctx.employee_named else
                 "No onboarding tasks are outstanding.",
            data={"count": 0}, sources=("onboarding_tasks",),
        )

    done = sum(1 for r in rows if r[3])
    pending = [r for r in rows if not r[3]]

    if ctx.employee_named:
        parts = [
            f"{capitalize_first(ctx.possessive())} onboarding is {done} of {len(rows)} "
            f"task(s) complete."
        ]
        if pending:
            parts.append("Outstanding:\n" + bullet_list([r[2] for r in pending], limit=10))
        else:
            parts.append("Everything is done.")
        text = sentence(parts)
    else:
        lines = [f"{' '.join(p for p in (r[0], r[1]) if p)} — {r[2]}" for r in pending]
        text = f"{len(pending)} outstanding onboarding task(s):\n" + bullet_list(lines, limit=12)

    return SkillResult(
        text=text,
        data={"total": len(rows), "completed": done, "pending": len(pending)},
        sources=("onboarding_tasks",),
    )


@skill(
    name="documents.dsr",
    topic="documents",
    summary="Daily status report submissions — who filed, who has not.",
    keywords={
        "dsr": 5.0, "daily status": 5.0, "status report": 4.5, "work done": 3.5,
        "daily report": 4.5, "submitted": 2.5, "filed": 2.5,
    },
    examples=("Who hasn't submitted their DSR today?", "Did Priya file a DSR?"),
)
def dsr(ctx: SkillContext) -> SkillResult | None:
    when = ctx.period.on()
    label = ctx.period.label if ctx.period.is_day else when.strftime("%d %b %Y")

    if ctx.employee_named and ctx.employee:
        report = (
            ctx.db.query(DailyStatusReport)
            .filter(
                DailyStatusReport.employee_id == ctx.employee.id,
                DailyStatusReport.report_date == when,
            )
            .first()
        )
        if report is None:
            return SkillResult(
                text=f"{ctx.subject()} has not filed a DSR for {label}.",
                data={"date": when.isoformat(), "filed": False},
                sources=("daily_status_reports",),
            )
        parts = [f"{ctx.subject()} filed a DSR for {label} ({report.status.lower()})."]
        if report.total_hours:
            parts.append(f"Hours logged: {report.total_hours}.")
        if report.project_work:
            parts.append(f"Project: {report.project_work}.")
        return SkillResult(
            text=sentence(parts),
            data={"date": when.isoformat(), "filed": True, "status": report.status},
            sources=("daily_status_reports",),
        )

    # Company view: who is missing one.
    submitted_ids = {
        row[0]
        for row in ctx.db.query(DailyStatusReport.employee_id)
        .filter(DailyStatusReport.report_date == when)
        .all()
    }
    rows = ctx.actor.scope_query(
        ctx.db.query(Employee.id, Employee.first_name, Employee.last_name)
        .filter(Employee.employment_status == _ACTIVE)
    ).all()
    missing = [
        " ".join(p for p in (r[1], r[2]) if p).strip()
        for r in rows if r[0] not in submitted_ids
    ]
    filed = len(rows) - len(missing)

    if not missing:
        text = f"All {len(rows)} active employee(s) filed a DSR for {label}."
    else:
        text = sentence([
            f"{filed} of {len(rows)} employee(s) filed a DSR for {label}.",
            f"Missing: {join_names(missing)}.",
        ])
    return SkillResult(
        text=text,
        data={"date": when.isoformat(), "filed": filed, "missing": len(missing),
              "missing_names": missing[:20]},
        sources=("daily_status_reports", "employees"),
    )


@skill(
    name="documents.queries",
    topic="documents",
    summary="HR queries raised by employees and their status.",
    keywords={
        "hr query": 5.0, "queries": 4.0, "query": 3.0, "ticket": 4.0,
        "raised": 3.0, "complaint": 3.5, "open queries": 5.0, "grievance": 4.0,
        "unresolved": 3.5,
    },
    examples=("How many HR queries are open?", "What has Priya raised?"),
)
def queries(ctx: SkillContext) -> SkillResult | None:
    query = (
        ctx.db.query(
            Employee.first_name, Employee.last_name,
            HRQuery.subject, HRQuery.status, HRQuery.category, HRQuery.created_at,
        )
        .join(HRQuery, HRQuery.employee_id == Employee.id)
    )
    if ctx.employee_named and ctx.employee:
        query = query.filter(HRQuery.employee_id == ctx.employee.id)
    else:
        query = ctx.actor.scope_query(query)

    rows = query.order_by(HRQuery.created_at.desc()).limit(40).all()
    if not rows:
        who = ctx.subject() if ctx.employee_named else "anyone in scope"
        return SkillResult(
            text=f"No HR queries are on record for {who}.",
            data={"count": 0}, sources=("hr_queries",),
        )

    open_rows = [r for r in rows if (r[3] or "").upper() != "RESOLVED"]
    parts = [f"{len(rows)} HR quer(y/ies) on record, {len(open_rows)} still open."]
    if open_rows:
        lines = [
            f"{' '.join(p for p in (r[0], r[1]) if p)} — {r[2]} ({(r[3] or '').lower()})"
            for r in open_rows
        ]
        parts.append("Open:\n" + bullet_list(lines, limit=10))

    return SkillResult(
        text=sentence(parts),
        data={"count": len(rows), "open": len(open_rows)},
        sources=("hr_queries",),
    )
