"""Zero expected hours means "no fixed daily target", not "nine".

`create_support_staff` has always written `expected_working_hours=0.0` with the
comment "no work-hour expectation for support staff". Every reader then wrote
`float(emp.expected_working_hours or 9.0)` -- and `0.0 or 9.0` is `9.0`, so the
intent never survived contact with a single caller. Somebody on no fixed hours
was measured against a nine-hour day: marked SHORT for a six-hour shift, and
charged the shortfall against Short Leave they had no allocation for.

These tests pin the distinction end to end: 0 is a value, None is an absence.
"""
from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.core.staff_policy import (
    DEFAULT_EXPECTED_HOURS,
    expected_daily_hours,
    is_fixed_salary_staff,
)


class _Emp:
    def __init__(self, hours, staff_type="Employee"):
        self.id = 55
        self.expected_working_hours = hours
        self.staff_type = staff_type


# --- the helper ----------------------------------------------------------

def test_zero_means_no_target_rather_than_the_default():
    """The whole point. `0 or 9` is 9, which is why this needs a helper."""
    assert expected_daily_hours(_Emp(0)) is None
    assert expected_daily_hours(_Emp(0.0)) is None


def test_unset_still_falls_back_to_the_default():
    """A NULL is a genuine absence of information; 0 is a decision."""
    assert expected_daily_hours(_Emp(None)) == DEFAULT_EXPECTED_HOURS


@pytest.mark.parametrize("hours", [7.5, 8, 9.0, 12])
def test_a_real_target_is_returned_unchanged(hours):
    assert expected_daily_hours(_Emp(hours)) == float(hours)


def test_a_negative_target_is_treated_as_no_target():
    """Not meaningful, and it must never reach a half-day threshold as -4.5."""
    assert expected_daily_hours(_Emp(-3)) is None


def test_unparseable_hours_fall_back_rather_than_raising():
    assert expected_daily_hours(_Emp("nonsense")) == DEFAULT_EXPECTED_HOURS


def test_seemas_staff_type_already_marks_her_as_having_no_daily_target():
    """staff_policy already says these people have no fixed daily target."""
    assert is_fixed_salary_staff(_Emp(9.0, staff_type="Housekeeping")) is True
    assert is_fixed_salary_staff(_Emp(9.0, staff_type="Employee")) is False


# --- attendance classification -------------------------------------------

class _Rec:
    def __init__(self, hours, rec_date):
        self.employee_id = 55
        self.date = rec_date
        self.total_work_hours = hours
        self.status = None
        self.source = "CAMERA"
        self.sign_in_time = None
        self.is_weekly_off = False
        self.is_holiday = False


class _Db:
    def __init__(self, emp):
        self._emp = emp

    def query(self, _model):
        emp = self._emp

        class _Q:
            def filter(self, *_a, **_k):
                return self

            def first(self):
                return emp

        return _Q()


def _yesterday():
    return date.today() - timedelta(days=2)


def test_no_target_means_any_worked_hours_count_as_present():
    """There is nothing to fall short OF, so SHORT/HALF_DAY cannot apply."""
    from app.services.attendance_service import apply_status_from_hours

    rec = _Rec(Decimal("2.5"), _yesterday())
    apply_status_from_hours(_Db(_Emp(0.0, "Housekeeping")), rec)
    assert rec.status == "PRESENT", "a short day was still scored against a target"


def test_no_target_with_no_hours_is_still_absent():
    from app.services.attendance_service import apply_status_from_hours

    rec = _Rec(Decimal("0"), _yesterday())
    apply_status_from_hours(_Db(_Emp(0.0, "Housekeeping")), rec)
    assert rec.status == "ABSENT"


def test_a_normal_employee_is_still_measured_against_their_target():
    """The change must not loosen the rule for people who do have hours."""
    from app.services.attendance_service import apply_status_from_hours

    rec = _Rec(Decimal("2.5"), _yesterday())
    apply_status_from_hours(_Db(_Emp(9.0, "Employee")), rec)
    assert rec.status == "HALF_DAY"


def test_a_full_day_is_still_present_for_a_normal_employee():
    from app.services.attendance_service import apply_status_from_hours

    rec = _Rec(Decimal("9"), _yesterday())
    apply_status_from_hours(_Db(_Emp(9.0, "Employee")), rec)
    assert rec.status == "PRESENT"
