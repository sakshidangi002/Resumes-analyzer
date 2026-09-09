"""Holidays, weekly offs and the financial year.

Named `company_calendar` rather than `calendar` so it cannot shadow the standard
library module that half this package imports.
"""
from __future__ import annotations

from datetime import timedelta

from app.chatbot.formatting import bullet_list, pretty_date, sentence
from app.chatbot.period import in_words
from app.chatbot.registry import SkillContext, SkillResult, skill
from app.core.datetime_utils import get_ist_now
from app.models.company import CompanyConfig, FinancialYear, Holiday


@skill(
    name="calendar.holidays",
    topic="calendar",
    summary="Company holidays — upcoming, or for a given month or year.",
    keywords={
        "holiday": 4.5, "next holiday": 5.0, "public holiday": 4.5,
        "days off": 3.0, "festival": 3.0, "holiday list": 5.0,
        "calendar": 4.0, "company calendar": 5.5, "leave calendar": 5.0,
    },
    examples=("When is the next holiday?", "What holidays are there in August?"),
)
def holidays(ctx: SkillContext) -> SkillResult | None:
    today = get_ist_now().date()
    wants_upcoming = any(
        word in ctx.low for word in ("next", "upcoming", "coming", "when is")
    )

    if wants_upcoming and ctx.period.assumed:
        rows = (
            ctx.db.query(Holiday.date, Holiday.name, Holiday.is_optional)
            .filter(Holiday.date >= today)
            .order_by(Holiday.date)
            .limit(6)
            .all()
        )
        if not rows:
            return SkillResult(
                text="There are no upcoming holidays on the calendar.",
                data={"count": 0}, sources=("holidays",),
            )
        first = rows[0]
        away = (first[0] - today).days
        lines = [
            f"{pretty_date(d)} — {name}" + (" (optional)" if optional else "")
            for d, name, optional in rows
        ]
        return SkillResult(
            text=sentence([
                f"The next holiday is {first[1]} on {pretty_date(first[0])}"
                + (f", {away} day(s) away." if away > 0 else " — today."),
                "\nComing up:\n" + bullet_list(lines[:6]),
            ]),
            data={"next": {"date": first[0].isoformat(), "name": first[1]},
                  "count": len(rows)},
            sources=("holidays",),
        )

    start, end = ctx.period.bounds()
    rows = (
        ctx.db.query(Holiday.date, Holiday.name, Holiday.is_optional)
        .filter(Holiday.date >= start, Holiday.date <= end)
        .order_by(Holiday.date)
        .all()
    )
    if not rows:
        return SkillResult(
            text=f"There are no holidays {in_words(ctx.period.label)}.",
            data={"count": 0, "period": ctx.period.label},
            sources=("holidays",),
        )
    lines = [
        f"{pretty_date(d)} — {name}" + (" (optional)" if optional else "")
        for d, name, optional in rows
    ]
    return SkillResult(
        text=f"{len(rows)} holiday(s) {in_words(ctx.period.label)}:\n"
             + bullet_list(lines, limit=15),
        data={"count": len(rows), "period": ctx.period.label,
              "holidays": [{"date": d.isoformat(), "name": n} for d, n, _ in rows]},
        sources=("holidays",),
    )


@skill(
    name="calendar.working_rules",
    topic="calendar",
    summary="Weekly offs, grace time and the half-day threshold.",
    keywords={
        "weekly off": 4.5, "week off": 4.5, "working day": 3.0, "grace": 4.0,
        "grace time": 4.5, "late after": 4.0, "office timing": 4.0,
        "half day threshold": 4.5, "working hours": 3.5, "shift": 2.5,
        "which days": 2.5,
    },
    examples=("What are our weekly offs?", "How much grace time do we get?"),
)
def working_rules(ctx: SkillContext) -> SkillResult | None:
    config = ctx.db.query(CompanyConfig).first()
    if config is None:
        return None

    parts = []
    if config.weekly_off_days:
        parts.append(f"Weekly offs are {config.weekly_off_days}.")
    if config.grace_time_minutes is not None:
        parts.append(
            f"Arrivals are marked late after {config.grace_time_minutes} minute(s) of grace."
        )
    if config.half_day_threshold_hours is not None:
        parts.append(
            f"A day under {config.half_day_threshold_hours} hour(s) counts as a half day."
        )
    if config.default_working_days_per_month:
        parts.append(
            f"Payroll assumes {config.default_working_days_per_month} working days a month."
        )
    if not parts:
        return None

    return SkillResult(
        text=sentence(parts),
        data={
            "weekly_off_days": config.weekly_off_days,
            "grace_time_minutes": config.grace_time_minutes,
            "half_day_threshold_hours": config.half_day_threshold_hours,
            "default_working_days_per_month": config.default_working_days_per_month,
        },
        sources=("company_config",),
    )


@skill(
    name="calendar.financial_year",
    topic="calendar",
    summary="The current financial year and how much of it is left.",
    keywords={
        "financial year": 5.0, "fiscal year": 5.0, "fy": 3.5, "current fy": 5.0,
        "year end": 3.0, "when does the year": 3.5,
    },
    examples=("What is the current financial year?", "When does this FY end?"),
)
def financial_year(ctx: SkillContext) -> SkillResult | None:
    today = get_ist_now().date()
    fy = (
        ctx.db.query(FinancialYear)
        .filter(FinancialYear.start_date <= today, FinancialYear.end_date >= today)
        .first()
    )
    if fy is None:
        return None

    remaining = (fy.end_date - today).days
    return SkillResult(
        text=(
            f"The current financial year is {fy.name}, running "
            f"{pretty_date(fy.start_date)} to {pretty_date(fy.end_date)} — "
            f"{max(0, remaining)} day(s) left."
        ),
        data={"name": fy.name, "start": fy.start_date.isoformat(),
              "end": fy.end_date.isoformat(), "days_remaining": max(0, remaining)},
        sources=("financial_years",),
    )
