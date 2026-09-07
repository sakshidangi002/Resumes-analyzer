"""Pagination must never truncate a caller that did not ask for a page.

Adding `page: int = Query(1)` with `page_size: int = Query(50)` to /employees
silently capped the directory at 50 rows. Every caller is a selector that needs
the full set — attendance, payroll, payslips, leave allocation, calendar, user
management — and none of them passes a page. Payroll quietly skipped employees
with nothing on screen to say so.

These endpoints therefore have `page` defaulting to None: paginate only on
request. This test reads the actual FastAPI signatures, so re-introducing a
default fails here rather than in production payroll.
"""
import inspect

import pytest

# Importing anything under app.api.routes pulls in the vision stack (cv2, torch,
# PIL, numpy) via the face/recognition services. Where that is not installed,
# skip cleanly instead of erroring the whole collection. CI runs pytest with -rs
# so the skip is REPORTED — a test that silently stops running is worse than no
# test, because it still looks like coverage.
try:
    from app.api.routes import company, employees, leave, letters, users
except ImportError as exc:  # pragma: no cover - depends on the environment
    pytest.skip(
        f"requires the vision stack pulled in by app.api.routes ({exc}); "
        "install 'requirements.txt' to run",
        allow_module_level=True,
    )

ENDPOINTS = [
    pytest.param(employees.list_employees, id="employees"),
    pytest.param(users.list_users, id="users"),
    pytest.param(leave.list_leave_types, id="leave-types"),
    pytest.param(company.list_financial_years, id="financial-years"),
    pytest.param(letters.list_templates, id="letter-templates"),
]


def _default_of(fn, name):
    param = inspect.signature(fn).parameters[name]
    # FastAPI wraps defaults in a Query() object; the real value is .default.
    return getattr(param.default, "default", param.default)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_page_has_no_default_so_all_rows_are_returned(endpoint):
    assert _default_of(endpoint, "page") is None, (
        f"{endpoint.__name__} defaults `page`, which silently truncates every "
        "caller that does not paginate"
    )


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_pagination_is_still_available_when_asked_for(endpoint):
    params = inspect.signature(endpoint).parameters
    assert "page" in params and "page_size" in params
    assert _default_of(endpoint, "page_size") is not None
