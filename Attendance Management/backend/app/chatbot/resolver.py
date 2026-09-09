"""Working out which employee a question is about.

The approach is deliberately **database-first**. Rather than guessing at a name
with a regex ("the word after 'for' that starts with a capital"), the question is
broken into candidate tokens and those are matched against the actual employee
table. That handles single names, full names, employee codes, email addresses,
and names that a regex would never recognise as names — which matters, because
HR asks about real people whose names do not follow English capitalisation.

It also removes a whole class of bug: a regex approach reads "the payroll total
for July" as an employee called July, then reports that no such person exists.
Here, "July" is simply not in the employees table, so nothing resolves and the
question is answered as the company-wide one it was.

Every lookup goes through `Actor.scope_query`, so a Manager can only ever
resolve their own reportees and an Employee only themselves. That check is part
of the query, not applied afterwards, so an out-of-scope name is indistinguishable
from a name that does not exist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.chatbot.access import Actor
from app.chatbot.period import MONTHS
from app.models.employee import Department, Designation, Employee

#: Tokens that are never a person, even if someone in the table shares the name.
#: Domain vocabulary dominates these questions, and an employee called "Leave"
#: matching the word "leave" would derail every leave question in the system.
_STOPWORDS = frozenset(
    set(MONTHS)
    | {
        "the", "and", "for", "who", "what", "when", "where", "which", "how",
        "why", "was", "were", "are", "his", "her", "him", "she", "they", "them",
        "their", "our", "you", "your", "this", "that", "with", "from", "about",
        "many", "much", "does", "did", "has", "have", "had", "can", "will",
        "show", "tell", "give", "list", "find", "get", "any", "all", "some",
        "today", "yesterday", "tomorrow", "now", "week", "month", "year",
        "employee", "employees", "staff", "people", "person", "team", "company",
        "attendance", "absent", "present", "late", "leave", "leaves", "holiday",
        "payroll", "payslip", "salary", "pay", "net", "gross", "deduction",
        "department", "designation", "manager", "report", "reports", "status",
        "detail", "details", "profile", "record", "records", "balance", "days",
        "day", "total", "count", "headcount", "letter", "letters", "onboarding",
        "dsr", "query", "queries", "pending", "approved", "rejected", "office",
        "working", "work", "joined", "joining", "left", "resigned", "active",
    }
)

#: "EMP001", "emp-12", "SW_0042"
_CODE = re.compile(r"\b([A-Za-z]{1,6}[-_]?\d{2,8})\b")
_EMAIL = re.compile(r"\b([\w.+-]+@[\w-]+\.[\w.]+)\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’.-]{1,}")

#: "Priya's", "Amit’s" — a possessive can only be a reference to a person, which
#: makes it the one signal strong enough to say "you named someone I can't find"
#: rather than quietly answering about the asker instead.
_POSSESSIVE = re.compile(r"\b([A-Za-z][A-Za-z.-]{1,})['’]s\b")


def _strip_possessive(token: str) -> str:
    """"priya's" -> "priya".

    Without this the most natural phrasing in the whole product — "Priya's
    attendance" — matches no first name, and the caller falls back to the
    asker's own record while the sentence still says "Priya".
    """
    for suffix in ("'s", "’s", "s'", "s’"):
        if token.endswith(suffix):
            return token[: -len(suffix)]
    return token

#: More than this and the question is a list request, not a lookup of one person.
_MAX_CANDIDATES = 6


@dataclass(frozen=True)
class ResolvedEmployee:
    """A person the actor is allowed to ask about."""

    id: int
    employee_code: str
    first_name: str
    last_name: str
    official_email: str | None = None
    phone: str | None = None
    department: str | None = None
    designation: str | None = None
    date_of_joining: date | None = None
    employment_status: str | None = None
    manager_name: str | None = None

    @property
    def full_name(self) -> str:
        return " ".join(p for p in (self.first_name, self.last_name) if p).strip()

    @property
    def short_name(self) -> str:
        return self.first_name or self.full_name


@dataclass(frozen=True)
class Resolution:
    """What the resolver made of the question.

    `ambiguous` matters as much as `employee`: two people called Priya is a
    question the chatbot must ask back, not a coin toss between two payslips.
    """

    employee: ResolvedEmployee | None = None
    ambiguous: tuple[ResolvedEmployee, ...] = ()
    #: The text that looked like a name but matched nobody in scope.
    unmatched_term: str | None = None
    #: True when the question named nobody at all - a company-wide question, or
    #: one about the asker themselves.
    named_nobody: bool = True

    @property
    def found(self) -> bool:
        return self.employee is not None


def _row_to_employee(row) -> ResolvedEmployee:
    emp, dept, desig, mgr_first, mgr_last = row
    manager = " ".join(p for p in (mgr_first, mgr_last) if p).strip() or None
    return ResolvedEmployee(
        id=emp.id,
        employee_code=emp.employee_code,
        first_name=emp.first_name,
        last_name=emp.last_name,
        official_email=emp.official_email,
        phone=emp.phone,
        department=dept,
        designation=desig,
        date_of_joining=emp.date_of_joining,
        employment_status=emp.employment_status,
        manager_name=manager,
    )


def _base_query(db: Session, actor: Actor):
    """Employee joined to the labels every skill wants, scoped to the actor."""
    manager = Employee.__table__.alias("manager")
    query = (
        db.query(
            Employee,
            Department.name,
            Designation.title,
            manager.c.first_name,
            manager.c.last_name,
        )
        .outerjoin(Department, Department.id == Employee.department_id)
        .outerjoin(Designation, Designation.id == Employee.designation_id)
        .outerjoin(manager, manager.c.id == Employee.reporting_manager_id)
    )
    return actor.scope_query(query)


def load(db: Session, actor: Actor, employee_id: int) -> ResolvedEmployee | None:
    """Fetch one employee by id, still subject to what the actor may see."""
    row = _base_query(db, actor).filter(Employee.id == employee_id).first()
    return _row_to_employee(row) if row else None


def self_employee(db: Session, actor: Actor) -> ResolvedEmployee | None:
    if actor.employee_id is None:
        return None
    # An actor can always see their own record even if `employee_filter` is
    # doing something restrictive; the filter already permits their own id.
    return load(db, actor, actor.employee_id)


def _candidate_tokens(text: str) -> list[str]:
    seen: list[str] = []
    for match in _WORD.finditer(text):
        token = _strip_possessive(match.group(0)).strip(".'’-")
        if len(token) < 2:
            continue
        low = token.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.append(low)
    return seen


def _person_reference(text: str) -> str | None:
    """The name in a possessive, if the question used one.

    Only possessives count. A capitalised word mid-sentence is too weak a
    signal — "Who works in Engineering?" would be read as a missing employee
    called Engineering.
    """
    for match in _POSSESSIVE.finditer(text):
        name = match.group(1)
        if name.lower() not in _STOPWORDS:
            return name
    return None


def resolve(db: Session, actor: Actor, question: str) -> Resolution:
    """Find the one employee this question is about, if it names one at all."""
    # 1. An employee code or an email address is unambiguous - take it as given.
    email = _EMAIL.search(question)
    if email:
        row = (
            _base_query(db, actor)
            .filter(func.lower(Employee.official_email) == email.group(1).lower())
            .first()
        )
        if row:
            return Resolution(employee=_row_to_employee(row), named_nobody=False)
        return Resolution(unmatched_term=email.group(1), named_nobody=False)

    code = _CODE.search(question)
    if code:
        row = (
            _base_query(db, actor)
            .filter(func.lower(Employee.employee_code) == code.group(1).lower())
            .first()
        )
        if row:
            return Resolution(employee=_row_to_employee(row), named_nobody=False)
        # Fall through: a bare number like "2024" is not a failed code lookup.

    # 2. Otherwise match the question's words against real names.
    tokens = _candidate_tokens(question)
    referenced = _person_reference(question)
    if not tokens:
        return Resolution()

    rows = (
        _base_query(db, actor)
        .filter(
            or_(
                func.lower(Employee.first_name).in_(tokens),
                func.lower(Employee.last_name).in_(tokens),
            )
        )
        .limit(40)
        .all()
    )
    if not rows:
        # A possessive named somebody. Falling back to the asker's own record
        # here would answer "What was Priya's net pay?" with the asker's salary
        # and label it "Your net pay" — confidently, and about the wrong person.
        return _unresolved(referenced)

    # 3. Rank. Someone whose first *and* last name both appear beats someone
    #    who shares only a first name, so "Priya Sharma" is not ambiguous just
    #    because another Priya exists.
    scored: list[tuple[int, ResolvedEmployee]] = []
    for row in rows:
        employee = _row_to_employee(row)
        first = (employee.first_name or "").lower()
        last = (employee.last_name or "").lower()
        score = 0
        if first in tokens:
            score += 2
        if last in tokens:
            score += 2
        if first in tokens and last in tokens:
            score += 3  # a full-name match is decisive
        if score:
            scored.append((score, employee))

    if not scored:
        return _unresolved(referenced)

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best = scored[0][0]
    winners = [emp for score, emp in scored if score == best]

    if len(winners) == 1:
        return Resolution(employee=winners[0], named_nobody=False)
    return Resolution(
        ambiguous=tuple(winners[:_MAX_CANDIDATES]),
        named_nobody=False,
    )


def _unresolved(referenced: str | None) -> Resolution:
    if referenced:
        return Resolution(unmatched_term=referenced, named_nobody=False)
    return Resolution()


def ambiguity_question(resolution: Resolution) -> str:
    """What to ask when a name matched more than one person."""
    options = ", ".join(
        f"{e.full_name} ({e.employee_code})" for e in resolution.ambiguous
    )
    return f"More than one employee matches that name — did you mean {options}?"


def search(
    db: Session,
    actor: Actor,
    *,
    department: str | None = None,
    designation: str | None = None,
    active_only: bool = True,
    limit: int = 50,
) -> list[ResolvedEmployee]:
    """List employees, for "who works in Engineering" style questions."""
    query = _base_query(db, actor)
    if active_only:
        query = query.filter(Employee.employment_status == "Active")
    if department:
        query = query.filter(func.lower(Department.name).like(f"%{department.lower()}%"))
    if designation:
        query = query.filter(func.lower(Designation.title).like(f"%{designation.lower()}%"))
    rows = query.order_by(Employee.first_name).limit(limit).all()
    return [_row_to_employee(r) for r in rows]


def find_department(db: Session, question: str) -> str | None:
    """Pick out a department the question mentions, matched against real rows."""
    low = question.lower()
    for (name,) in db.query(Department.name).all():
        if name and name.lower() in low:
            return name
    return None
