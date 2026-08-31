"""Tool layer: thin, typed wrappers over the existing HRMS services.

A "tool" here is a plain function `(db, **kwargs) -> dict`. It exists to give the
graph a stable, testable boundary — NOT to re-implement business logic. Every
tool delegates to `app.services.*`, which is the same code the REST routes use,
so attendance rules stay in exactly one place.

Tools never take an employee id from user text. The caller passes the id that
`scope_resolver` already authorised.
"""
from app.assistant.tools.attendance_tools import (  # noqa: F401
    ATTENDANCE_TOOLS,
    employee_period_context,
    monthly_summary,
)
