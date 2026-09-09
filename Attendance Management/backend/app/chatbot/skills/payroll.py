"""Payroll: payslips, salary structures, advances and company run totals.

These **read** payslips; none of them runs payroll. `run_payroll_for_period`
computes and writes a run, and a question must never trigger one.

Every skill here is marked `salary=True`, so `service.ask` requires
`Actor.may_see_salary` before running it for a *named* employee. Asking about
your own pay never needs that right — you can already open your payslip.
"""
from __future__ import annotations

from sqlalchemy import and_, func, or_

from app.chatbot.formatting import capitalize_first, days, inr, pretty_date, sentence, to_float
from app.chatbot.period import MONTH_NAMES, in_words
from app.chatbot.registry import SkillContext, SkillResult, skill
from app.models.employee import Employee
from app.models.payroll import Payslip, PayrollPeriod, SalaryStructure
from app.models.salary_advance import SalaryAdvance


@skill(
    name="payroll.payslip",
    topic="payroll",
    summary="One employee's payslip for a month: net, earnings, deductions, LOP.",
    keywords={
        "payslip": 4.5, "pay slip": 4.5, "salary slip": 4.5, "net pay": 4.0,
        "take home": 4.0, "paid": 2.0, "earnings": 3.0, "deduction": 3.0,
        "salary for": 4.0, "salary in": 4.0, "salary of the month": 4.5,
        "how much was": 2.5, "credited": 3.0,
        "gross": 2.5, "tds": 2.5, "pf": 2.0,
    },
    about_employee=True,
    salary=True,
    examples=("What was my net pay last month?",
              "Show Priya's payslip for July"),
)
def payslip(ctx: SkillContext) -> SkillResult | None:
    employee = ctx.employee
    if employee is None:
        return None

    period = ctx.period
    row = (
        ctx.db.query(Payslip, PayrollPeriod)
        .join(PayrollPeriod, PayrollPeriod.id == Payslip.payroll_period_id)
        .filter(
            Payslip.employee_id == employee.id,
            PayrollPeriod.month == period.month,
            PayrollPeriod.year == period.year,
        )
        .first()
    )

    if row is None:
        # Not an error: payroll for the current month usually has not run. Offer
        # the most recent payslip rather than dead-ending.
        latest = (
            ctx.db.query(Payslip, PayrollPeriod)
            .join(PayrollPeriod, PayrollPeriod.id == Payslip.payroll_period_id)
            .filter(Payslip.employee_id == employee.id)
            .order_by(PayrollPeriod.year.desc(), PayrollPeriod.month.desc())
            .first()
        )
        if latest is None:
            return SkillResult(
                text=f"There are no payslips on record for {ctx.subject()}.",
                data={"found": False},
                sources=("payslips",),
            )
        slip, per = latest
        return SkillResult(
            text=(
                f"No payslip has been generated for {period.month_label} yet. "
                f"{capitalize_first(ctx.possessive())} most recent is "
                f"{MONTH_NAMES[per.month - 1]} {per.year}: net "
                f"{inr(slip.net_salary)}."
            ),
            data={"found": False, "latest": {"month": per.month, "year": per.year,
                                             "net_salary": to_float(slip.net_salary)}},
            sources=("payslips", "payroll_periods"),
        )

    slip, per = row
    parts = [
        f"{capitalize_first(ctx.possessive())} net pay for {period.month_label} was "
        f"{inr(slip.net_salary)} — earnings {inr(slip.total_earnings)}, "
        f"deductions {inr(slip.total_deductions)}."
    ]
    if slip.paid_days:
        parts.append(f"Paid days {days(slip.paid_days)}.")
    if slip.lop_days:
        parts.append(f"Loss of pay {days(slip.lop_days)} day(s).")
    if per.status and per.status != "LOCKED":
        parts.append(f"This payroll period is still marked {per.status.lower()}.")

    return SkillResult(
        text=sentence(parts),
        data={
            "found": True, "month": per.month, "year": per.year, "status": per.status,
            "net_salary": to_float(slip.net_salary),
            "total_earnings": to_float(slip.total_earnings),
            "total_deductions": to_float(slip.total_deductions),
            "gross_salary": to_float(slip.gross_salary),
            "paid_days": to_float(slip.paid_days), "lop_days": to_float(slip.lop_days),
        },
        sources=("payslips", "payroll_periods"),
    )


@skill(
    name="payroll.salary_structure",
    topic="payroll",
    summary="An employee's current salary structure and CTC.",
    keywords={
        "ctc": 4.5, "salary structure": 4.5, "annual salary": 4.0, "package": 3.5,
        "basic": 2.5, "hra": 3.0, "allowance": 3.0, "compensation": 3.5,
        "what is the salary": 3.5, "how much does": 2.5, "earn": 2.5,
        # Bare "salary" and "pay details" previously matched nothing at all, so
        # "show me Priya salary" fell through to the model. This skill owns the
        # plain word; `payroll.payslip` wins once a payslip term is present.
        "salary": 3.5, "salary detail": 4.5, "pay detail": 4.5, "pay": 2.0,
        "remuneration": 4.0, "wage": 3.0,
    },
    about_employee=True,
    salary=True,
    examples=("What is Priya's CTC?", "Show Amit's salary structure"),
)
def salary_structure(ctx: SkillContext) -> SkillResult | None:
    employee = ctx.employee
    if employee is None:
        return None

    structure = (
        ctx.db.query(SalaryStructure)
        .filter(SalaryStructure.employee_id == employee.id)
        .order_by(SalaryStructure.id.desc())
        .first()
    )
    if structure is None:
        return SkillResult(
            text=f"No salary structure is on record for {ctx.subject()}.",
            data={"found": False},
            sources=("salary_structures",),
        )

    # Column names vary between deployments, so read defensively and report only
    # what is actually present rather than inventing a zero.
    fields = {
        label: to_float(getattr(structure, attr, None))
        for label, attr in (
            ("CTC", "ctc"), ("gross", "gross_salary"), ("basic", "basic"),
            ("HRA", "hra"), ("monthly gross", "monthly_gross"),
        )
        if getattr(structure, attr, None) is not None
    }
    if not fields:
        return SkillResult(
            text=f"A salary structure exists for {ctx.subject()}, but it has no "
                 f"figures I can read. Check the Payroll page.",
            data={"found": True},
            sources=("salary_structures",),
        )

    detail = ", ".join(f"{label} {inr(value)}" for label, value in fields.items())
    return SkillResult(
        text=f"{capitalize_first(ctx.possessive())} current salary structure — {detail}.",
        data={"found": True, **{k.lower(): v for k, v in fields.items()}},
        sources=("salary_structures",),
    )


@skill(
    name="payroll.company_run",
    topic="payroll",
    summary="Totals for a payroll run: employees covered, gross, deductions, net.",
    keywords={
        "payroll": 3.5, "payroll total": 5.0, "total payroll": 5.0,
        "payroll cost": 4.5, "total salary": 4.0, "salary bill": 4.5,
        "wage bill": 4.5, "in total": 2.5, "overall": 2.5, "altogether": 3.0,
        "how much did we pay": 4.5, "did we pay": 4.0, "payroll run": 4.0,
        "processed": 2.0, "total net": 3.5, "company salary": 4.0,
        "salary expense": 5.0, "salary cost": 5.0, "wage cost": 4.5,
        "salary budget": 4.5, "payout": 4.0, "disbursed": 4.0,
    },
    company_wide=True,
    salary=True,
    examples=("What did the payroll total last month?",
              "How much did we pay out in July?"),
)
def company_run(ctx: SkillContext) -> SkillResult | None:
    period = ctx.period

    # "Total payroll paid to date" spans every run, not one month.
    if period.cumulative:
        totals = (
            ctx.db.query(
                func.count(Payslip.id),
                func.sum(Payslip.total_earnings),
                func.sum(Payslip.total_deductions),
                func.sum(Payslip.net_salary),
                func.count(func.distinct(PayrollPeriod.id)),
            )
            .join(PayrollPeriod, PayrollPeriod.id == Payslip.payroll_period_id)
            .filter(
                or_(
                    PayrollPeriod.year < period.year,
                    and_(
                        PayrollPeriod.year == period.year,
                        PayrollPeriod.month <= period.month,
                    ),
                )
            )
            .first()
        )
        slips = int(totals[0] or 0) if totals else 0
        if not slips:
            return SkillResult(
                text="No payroll has been run yet, so there is nothing paid to date.",
                data={"found": False, "cumulative": True},
                sources=("payslips", "payroll_periods"),
            )
        return SkillResult(
            text=(
                f"Total payroll paid {period.label}: {inr(totals[3])} net across "
                f"{int(totals[4] or 0)} payroll month(s) and {slips} payslip(s) — "
                f"earnings {inr(totals[1])}, deductions {inr(totals[2])}."
            ),
            data={
                "found": True, "cumulative": True, "payslip_count": slips,
                "months_processed": int(totals[4] or 0),
                "total_earnings": to_float(totals[1]),
                "total_deductions": to_float(totals[2]),
                "total_net": to_float(totals[3]),
            },
            sources=("payslips", "payroll_periods"),
        )

    run = (
        ctx.db.query(PayrollPeriod)
        .filter(PayrollPeriod.month == period.month, PayrollPeriod.year == period.year)
        .first()
    )

    year_rows = (
        ctx.db.query(func.sum(Payslip.net_salary), func.count(func.distinct(PayrollPeriod.id)))
        .join(PayrollPeriod, PayrollPeriod.id == Payslip.payroll_period_id)
        .filter(PayrollPeriod.year == period.year)
        .first()
    )
    year_net = to_float(year_rows[0]) if year_rows else 0.0
    months_processed = int(year_rows[1] or 0) if year_rows else 0

    if run is None:
        return SkillResult(
            text=(
                f"No payroll period exists for {period.month_label} yet."
                + (f" So far in {period.year}, {months_processed} month(s) have been "
                   f"processed, totalling {inr(year_net)}." if months_processed else "")
            ),
            data={"found": False, "year_net": year_net,
                  "months_processed": months_processed},
            sources=("payroll_periods", "payslips"),
        )

    totals = (
        ctx.db.query(
            func.count(Payslip.id),
            func.sum(Payslip.total_earnings),
            func.sum(Payslip.total_deductions),
            func.sum(Payslip.net_salary),
        )
        .filter(Payslip.payroll_period_id == run.id)
        .first()
    )
    count = int(totals[0] or 0)
    if count == 0:
        return SkillResult(
            text=f"The {period.month_label} payroll period exists but has no "
                 f"payslips generated yet (status {run.status.lower()}).",
            data={"found": False, "status": run.status},
            sources=("payroll_periods", "payslips"),
        )

    parts = [
        f"The {period.month_label} payroll covered {count} employee(s): "
        f"net {inr(totals[3])}, earnings {inr(totals[1])}, "
        f"deductions {inr(totals[2])}."
    ]
    if run.status and run.status != "LOCKED":
        parts.append(f"This period is not locked yet (status {run.status.lower()}).")
    if months_processed:
        parts.append(
            f"Year to date ({period.year}): {inr(year_net)} across "
            f"{months_processed} processed month(s)."
        )

    return SkillResult(
        text=sentence(parts),
        data={
            "found": True, "month": period.month, "year": period.year,
            "status": run.status, "payslip_count": count,
            "total_earnings": to_float(totals[1]),
            "total_deductions": to_float(totals[2]),
            "total_net": to_float(totals[3]),
            "year_net": year_net, "months_processed": months_processed,
        },
        sources=("payroll_periods", "payslips"),
    )


@skill(
    name="payroll.advances",
    topic="payroll",
    summary="Salary advances taken and whether they have been recovered.",
    keywords={
        "advance": 4.5, "salary advance": 6.0, "loan": 3.5, "borrowed": 3.0,
        "recovered": 3.0, "deducted": 2.5,
    },
    salary=True,
    examples=("Are there any pending salary advances?",
              "Has Priya taken a salary advance?"),
)
def advances(ctx: SkillContext) -> SkillResult | None:
    query = (
        ctx.db.query(
            Employee.first_name, Employee.last_name,
            SalaryAdvance.amount, SalaryAdvance.status, SalaryAdvance.date_taken,
        )
        .join(Employee, Employee.id == SalaryAdvance.employee_id)
    )
    # A named employee narrows it; otherwise it is a company-wide question, and
    # `Actor` has already limited which employees are reachable.
    if ctx.employee_named and ctx.employee:
        query = query.filter(SalaryAdvance.employee_id == ctx.employee.id)
    else:
        query = ctx.actor.scope_query(query)

    rows = query.order_by(SalaryAdvance.date_taken.desc()).limit(30).all()
    if not rows:
        who = ctx.subject() if ctx.employee_named else "anyone in scope"
        return SkillResult(
            text=f"No salary advances are on record for {who}.",
            data={"count": 0},
            sources=("salary_advances",),
        )

    pending = [r for r in rows if (r[3] or "").upper() == "PENDING"]
    total_pending = sum(to_float(r[2]) for r in pending)
    parts = [f"{len(rows)} salary advance(s) on record."]
    if pending:
        parts.append(
            f"{len(pending)} still pending recovery, totalling {inr(total_pending)}."
        )
    else:
        parts.append("All of them have been recovered.")
    latest = rows[0]
    parts.append(
        f"Most recent: {inr(latest[2])} on {pretty_date(latest[4])} "
        f"({(latest[3] or '').lower()})."
    )

    return SkillResult(
        text=sentence(parts),
        data={"count": len(rows), "pending_count": len(pending),
              "pending_amount": total_pending},
        sources=("salary_advances",),
    )
