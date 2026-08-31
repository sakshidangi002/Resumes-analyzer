"""Non-employee staff are on a fixed salary; attendance must never reduce it.

Housekeeping, security, drivers and the like work variable hours against no
daily target. Before this rule existed they were run through the normal
attendance-linked payroll, which measures every day against
expected_working_hours (defaulting to 9.0) and rounds the shortfall into
half/full LOP days — so a genuine 2-hour shift scored as a full day unpaid, and
the half-day shortfalls were additionally charged against Short/Paid Leave
buffers they hold no allocation for.

Only salary ADVANCES are deducted for these staff, and that happens outside the
LOP branch this guards.
"""
import pytest
from app.core.staff_policy import is_fixed_salary_staff


class _Employee:
    def __init__(self, staff_type):
        self.staff_type = staff_type


@pytest.mark.parametrize("staff_type", [
    "Housekeeping",
    "Security",
    "Driver",
    "Staff",
    "housekeeping",     # case must not matter
    "  Housekeeping  ",  # nor surrounding whitespace
])
def test_non_employee_staff_are_fixed_salary(staff_type):
    assert is_fixed_salary_staff(_Employee(staff_type))


@pytest.mark.parametrize("staff_type", ["Employee", "employee", "  EMPLOYEE "])
def test_regular_employees_are_not_fixed_salary(staff_type):
    assert not is_fixed_salary_staff(_Employee(staff_type))


def test_missing_staff_type_defaults_to_regular_employee():
    """Rows predating the staff_type column must keep normal payroll.

    Defaulting the other way would silently stop deducting LOP for the entire
    existing workforce — a payroll error affecting everyone at once.
    """
    assert not is_fixed_salary_staff(_Employee(None))
    assert not is_fixed_salary_staff(_Employee(""))

    class _NoAttribute:
        pass

    assert not is_fixed_salary_staff(_NoAttribute())
