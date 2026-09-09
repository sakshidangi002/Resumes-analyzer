"""Does an HR/Admin question reach the right area of the chatbot?

Routing is keyword scoring, which means it is tuned by editing weights — and a
weight edit for one skill silently changes every other skill's chances. This
file is the safety net for that: a wide spread of real phrasings across the six
areas HR actually asks about, asserted by *topic*.

Topic, not exact skill, on purpose. "Leave balance" vs "leave requests" is a
judgement call; answering a leave question with an employee profile is a bug.

It caught two real regressions when it was written:

* the per-employee breakdown skills carried their own domain word, so "show
  leave" scored higher for leave-per-employee than for a leave balance and the
  breakdowns swallowed their whole domain (fixed with `requires=`);
* bare "salary" matched no skill at all, so "show me Priya salary" fell through
  to the language model.

No database and no model: `registry.best` is a pure function of the text.
"""
from __future__ import annotations

import pytest

from app.chatbot import registry
from app.chatbot.service import _PROCEDURAL
from app.chatbot.skills import load_skills

load_skills()

# (question, a person was named, expected topic)
CASES = [
    # --- pay / salary -------------------------------------------------------
    ("what is the salary of Priya", True, "payroll"),
    ("how much does Priya earn", True, "payroll"),
    ("show me Priya salary", True, "payroll"),
    ("Priya salary details", True, "payroll"),
    ("what is my salary", False, "payroll"),
    ("my last payslip", False, "payroll"),
    ("show payslip of August", False, "payroll"),
    ("how much was deducted from my salary", False, "payroll"),
    ("what is my gross pay", False, "payroll"),
    ("salary details", False, "payroll"),
    ("pay details of Amit", True, "payroll"),
    ("how much salary did we pay in July", False, "payroll"),
    ("monthly salary expense", False, "payroll"),
    ("total salary cost", False, "payroll"),
    ("what is the CTC of Amit", True, "payroll"),
    ("annual package of Priya", True, "payroll"),
    ("salary structure", False, "payroll"),
    ("pending advances", False, "payroll"),
    ("what did the payroll total last month", False, "payroll"),
    ("total payroll paid to date", False, "payroll"),

    # --- leave --------------------------------------------------------------
    ("leave details of Priya", True, "leave"),
    ("Priya leave", True, "leave"),
    ("show leave", False, "leave"),
    ("leave information", False, "leave"),
    ("how many leaves do I have", False, "leave"),
    ("my remaining leaves", False, "leave"),
    ("leave status", False, "leave"),
    ("who all are on leave", False, "leave"),
    ("anyone on leave tomorrow", False, "leave"),
    ("leave applications", False, "leave"),
    ("approve pending leaves", False, "leave"),
    ("how many leaves has Amit taken", True, "leave"),
    ("sick leave balance", False, "leave"),
    ("casual leaves remaining", False, "leave"),
    ("leave report", False, "leave"),
    ("employees on leave this week", False, "leave"),
    ("unpaid leave days", False, "leave"),

    # --- attendance ---------------------------------------------------------
    ("attendance of Priya", True, "attendance"),
    ("Priya attendance report", True, "attendance"),
    ("attendance details", False, "attendance"),
    ("show attendance", False, "attendance"),
    ("attendance report for August", False, "attendance"),
    ("my attendance", False, "attendance"),
    ("how many days did Amit work", True, "attendance"),
    ("who came late today", False, "attendance"),
    ("who is present today", False, "attendance"),
    ("today attendance", False, "attendance"),
    ("absent employees", False, "attendance"),
    ("attendance percentage of Priya", True, "attendance"),
    ("what time did Priya come", True, "attendance"),
    ("who has not marked attendance", False, "attendance"),
    ("how many people are absent today", False, "attendance"),

    # --- holidays / calendar ------------------------------------------------
    ("holiday list", False, "calendar"),
    ("upcoming holidays", False, "calendar"),
    ("how many holidays this year", False, "calendar"),
    ("is tomorrow a holiday", False, "calendar"),
    ("holidays in October", False, "calendar"),
    ("company calendar", False, "calendar"),
    ("which days are week off", False, "calendar"),
    ("office timings", False, "calendar"),
    ("what is the grace period", False, "calendar"),
    ("when does the financial year end", False, "calendar"),

    # --- employee details ---------------------------------------------------
    ("employee details of Priya", True, "people"),
    ("Priya details", True, "people"),
    ("tell me about Priya", True, "people"),
    ("Priya phone number", True, "people"),
    ("Priya email id", True, "people"),
    ("who is the manager of Amit", True, "people"),
    ("Priya designation", True, "people"),
    ("which department is Amit in", True, "people"),
    ("when did Priya join", True, "people"),
    ("employee list", False, "people"),
    ("show employees in Sales", False, "people"),
    ("who reports to Neha", True, "people"),

    # --- workforce ----------------------------------------------------------
    ("total employees", False, "workforce"),
    ("how many employees do we have", False, "workforce"),
    ("new joiners this month", False, "workforce"),
    ("department wise employee count", False, "workforce"),

    # --- per-employee breakdowns -------------------------------------------
    ("salary of each employee", False, "breakdown"),
    ("leave taken by every employee", False, "breakdown"),
    ("attendance of all employees one by one", False, "breakdown"),
    ("employee wise payroll", False, "breakdown"),
    ("who has the highest salary", False, "breakdown"),
]


def _route(question: str, named: bool) -> str | None:
    """Mirror the service: procedural questions bypass skills entirely."""
    if _PROCEDURAL.search(question):
        return None
    match = registry.best(question, employee_named=named, may_see_company=True)
    return match.skill.topic if match else None


@pytest.mark.parametrize("question,named,expected", CASES)
def test_question_reaches_the_right_area(question, named, expected):
    assert _route(question, named) == expected


# --- questions that must NOT reach a lookup ---------------------------------
# A definition or a how-to has no database answer. Answering "what does LOP
# stand for?" with a leave balance is a correct number to the wrong question.
FALL_THROUGH = [
    "How do I add a new employee?",
    "What does LOP stand for?",
    "how to apply for leave",
    "explain the appraisal process",
]


@pytest.mark.parametrize("question", FALL_THROUGH)
def test_process_questions_go_to_the_model(question):
    assert _route(question, False) is None


def test_company_skills_are_flagged_for_the_permission_check():
    """An employee-facing question must never be served by a company skill.

    The gate itself lives in `service.ask`; this asserts the flag it reads is
    actually set, so a new company-wide skill cannot be added without one.
    """
    for question in ("how many people are absent today", "how many employees do we have"):
        match = registry.best(question, employee_named=False, may_see_company=True)
        assert match is not None
        assert match.skill.company_wide, f"{match.skill.name} reads across employees"


def test_every_skill_handler_takes_ctx():
    """Guards the decorator-misalignment bug: a helper defined between
    `@skill(...)` and its function gets registered as the handler."""
    import inspect

    for skill in registry.all_skills():
        first = next(iter(inspect.signature(skill.handler).parameters), None)
        assert first == "ctx", f"{skill.name} -> {skill.handler.__name__}({first})"


# --- follow-up turns --------------------------------------------------------
# A follow-up states only what changed. The previous turn's skill is favoured
# so "and Neha?" means something, but only just: `FOLLOWUP_BOOST` must never be
# strong enough to beat a domain word the person actually typed. At 3.0 it was,
# and "what about her leave?" kept answering with attendance.

def _followup(question: str, prior: str, named: bool = True) -> str | None:
    match = registry.best(
        question, employee_named=named, may_see_company=True, boost_skill=prior,
    )
    return match.skill.name if match else None


@pytest.mark.parametrize(
    "question,prior,named,expected",
    [
        # No domain word at all: the previous lookup carries.
        ("and Neha?", "payroll.payslip", True, "payroll.payslip"),
        ("what about Amit?", "leave.balance", True, "leave.balance"),
        ("same for last month", "attendance.employee_month", True,
         "attendance.employee_month"),
        # `named` is False here: the previous answer was company-wide, so there
        # is no person to carry into this turn.
        ("what about yesterday?", "attendance.company_day", False,
         "attendance.company_day"),
        # A domain word IS present: it must win over the carried skill.
        ("what about her leave?", "attendance.employee_month", True, "leave.balance"),
        ("and her payslip?", "attendance.employee_month", True, "payroll.payslip"),
        ("what about her attendance?", "leave.balance", True,
         "attendance.employee_month"),
    ],
)
def test_followup_carries_only_what_was_not_restated(question, prior, named, expected):
    assert _followup(question, prior, named=named) == expected


def test_followup_boost_is_weaker_than_a_domain_word():
    """The guard behind the case above, stated as the invariant it protects."""
    assert registry.FOLLOWUP_BOOST < 2.0, (
        "a boost this large lets the previous skill beat a domain keyword the "
        "user typed, which is how 'what about her leave?' answered with attendance"
    )
