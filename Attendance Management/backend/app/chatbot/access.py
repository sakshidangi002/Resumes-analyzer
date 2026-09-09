"""Who may ask what.

This module is the chatbot's single authorisation boundary. Skills never check a
role themselves; they call `Actor.employee_filter()` and get back a SQLAlchemy
criterion that already restricts which `Employee` rows they are allowed to
reach. One place to read, one place to audit, and a new skill inherits the rules
for free instead of re-deriving them.

The rules mirror what the REST routes already enforce, so the chatbot can never
be a way around a permission the UI applies:

* **Admin / HR** — every employee, company-wide totals, salary figures.
  (`require_roles(["Admin", "HR"])` on the reports and payroll endpoints.)
* **Manager** — themselves and their direct reportees, no company totals and no
  salary. (`require_roles(["Admin", "HR", "Manager"])` on attendance history,
  scoped by `reporting_manager_id`.)
* **Employee** — themselves only.

A account with no linked employee record and no elevated role can reach nothing,
which is the same answer `POST /api/queries` gives it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import false, or_

from app.models.employee import Employee

#: Full company visibility, including salary.
FULL_ACCESS_ROLES = frozenset({"Admin", "HR"})
#: Own record plus direct reportees.
TEAM_ACCESS_ROLES = frozenset({"Manager"})


@dataclass(frozen=True)
class Actor:
    """The authenticated user, reduced to what authorisation depends on.

    Built once by the route from `get_current_user`. Nothing in the question can
    influence it — a message asking to "act as HR" changes no field here.
    """

    user_id: int
    employee_id: int | None
    roles: frozenset[str]
    display_name: str = ""

    @classmethod
    def from_user(cls, user) -> "Actor":
        return cls(
            user_id=user.id,
            employee_id=user.employee_id,
            roles=frozenset(r.name for r in user.roles),
            display_name=getattr(user, "username", "") or "",
        )

    # --- capability questions ----------------------------------------------
    @property
    def is_hr(self) -> bool:
        return bool(self.roles & FULL_ACCESS_ROLES)

    @property
    def is_manager(self) -> bool:
        return bool(self.roles & TEAM_ACCESS_ROLES)

    @property
    def may_see_company(self) -> bool:
        """Company-wide totals: headcount, payroll runs, today's roll-call."""
        return self.is_hr

    @property
    def may_see_salary(self) -> bool:
        """Another person's pay. Everyone may see their own payslip."""
        return self.is_hr

    @property
    def may_see_others(self) -> bool:
        return self.is_hr or self.is_manager

    def employee_filter(self):
        """A SQLAlchemy criterion limiting `Employee` rows to what this actor may see.

        Returned as a filter rather than a post-hoc check on purpose: a Manager
        asking about someone outside their team gets "no such employee", which
        is indistinguishable from a name that does not exist. The chatbot is
        therefore not an oracle for who works here.
        """
        if self.is_hr:
            return None  # no restriction
        if self.employee_id is None:
            return false()
        if self.is_manager:
            return or_(
                Employee.id == self.employee_id,
                Employee.reporting_manager_id == self.employee_id,
            )
        return Employee.id == self.employee_id

    def scope_query(self, query):
        """Apply `employee_filter()` to a query already selecting from Employee."""
        criterion = self.employee_filter()
        return query if criterion is None else query.filter(criterion)

    def describe_scope(self) -> str:
        if self.is_hr:
            return "the whole company"
        if self.is_manager:
            return "you and your reportees"
        return "your own record"


def missing_permission_message(*, needs_company: bool, needs_salary: bool) -> str:
    """What to say instead of an answer. Deliberately states the rule, not the data."""
    if needs_salary:
        return (
            "Salary details for other employees are available to Admin and HR only. "
            "I can still show you your own payslips."
        )
    if needs_company:
        return (
            "Company-wide figures are available to Admin and HR only. "
            "I can still answer about your own record."
        )
    return "You don't have permission to view that."


def roles_summary(roles: Iterable[str]) -> str:
    ordered = sorted(set(roles))
    return ", ".join(ordered) if ordered else "no role"
