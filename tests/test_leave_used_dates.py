"""The dates behind a "Used: N" figure must add up to N.

Leave allocation showed a used-days total with nothing to back it, so an
employee disputing their balance had no way to see which days were charged and
HR had no way to check. The dates are now listed — and the only way that listing
is worth anything is if it is the SAME arithmetic the total came from, not a
second implementation that drifts from it.

That is what these tests pin: for each of the three leave families, the entries
returned by the listing sum to exactly the number the allocation endpoint
displays. Every one of them is a pure-logic test — the buffer rules and the
paid/unpaid split are where the bugs live, not in the SQL.
"""
from datetime import date
from decimal import Decimal

from app.services.leave_service import (
    approved_request_days,
    count_hr_direct_paid_leave_days,
    hr_direct_paid_leave_days,
    short_leave_days_in_month,
)
from app.models import Employee, LeaveRequest
from app.models.attendance import AttendanceRecord


class _Att:
    def __init__(self, day, status):
        self.date = day
        self.status = status


class _Req:
    def __init__(self, start, end, is_half_day=False, paid=None, unpaid=None):
        self.start_date = start
        self.end_date = end
        self.is_half_day = is_half_day
        self.paid_days = paid
        self.unpaid_days = unpaid


class _FakeSession:
    """Serves canned rows per model. Filters are ignored on purpose.

    Every predicate these helpers build is a date-range or status restriction,
    so the tests supply only rows that are already in range; what is under test
    is the buffer arithmetic and the paid/unpaid split, which SQL never sees.
    """

    def __init__(self, rows):
        self._rows = rows

    def query(self, model):
        session = self

        class _Q:
            def filter(self, *_a, **_k):
                return self

            def order_by(self, *_a, **_k):
                return self

            def all(self):
                return session._rows.get(model, [])

            def first(self):
                rows = session._rows.get(model, [])
                return rows[0] if rows else None

        return _Q()


FY_START, FY_END = date(2026, 4, 1), date(2027, 3, 31)


class _Emp:
    """Plain stub, not an ORM instance.

    `Employee.__new__` skips SQLAlchemy's instrumentation, so assigning to its
    mapped columns raises. Only `staff_type` is read here (by
    `is_fixed_salary_staff`), and the fake session keys on the model CLASS, not
    on what it returns.
    """

    def __init__(self, staff_type="Employee"):
        self.id = 55
        self.staff_type = staff_type


def _employee(staff_type="Employee"):
    return _Emp(staff_type)


# ── HR-marked paid leave ────────────────────────────────────────────────────

def test_hr_direct_dates_sum_to_the_hr_direct_count():
    """The count is defined as the sum of the listing — never a second rule."""
    db = _FakeSession({
        Employee: [_employee()],
        LeaveRequest: [],
        AttendanceRecord: [
            _Att(date(2026, 4, 6), "PAID_LEAVE"),
            _Att(date(2026, 4, 9), "SHORT"),      # spends 1 of April's buffer
            _Att(date(2026, 4, 14), "SHORT"),     # spends the 2nd
            _Att(date(2026, 4, 20), "HALF_DAY"),  # buffer gone -> 0.5 charged
            _Att(date(2026, 5, 11), "PAID_LEAVE"),
        ],
    })
    entries = hr_direct_paid_leave_days(db, 55, FY_START, FY_END, 3)
    total = count_hr_direct_paid_leave_days(db, 55, FY_START, FY_END, 3)

    assert sum((e["days"] for e in entries), Decimal("0")) == total
    assert total == Decimal("2.5")
    assert [e["date"] for e in entries] == [
        date(2026, 4, 6), date(2026, 4, 20), date(2026, 5, 11),
    ]


def test_half_day_inside_the_monthly_buffer_is_not_charged_to_paid_leave():
    """A half day is free while the 2-a-month short-leave buffer is untouched."""
    db = _FakeSession({
        Employee: [_employee()],
        LeaveRequest: [],
        AttendanceRecord: [_Att(date(2026, 4, 20), "HALF_DAY")],
    })
    assert hr_direct_paid_leave_days(db, 55, FY_START, FY_END, 3) == []


def test_fixed_salary_staff_are_charged_nothing_and_so_listed_nothing():
    """Housekeeping/security hold a zero allocation; charging them read as -3.5."""
    db = _FakeSession({
        Employee: [_employee(staff_type="Housekeeping")],
        LeaveRequest: [],
        AttendanceRecord: [_Att(date(2026, 4, 6), "PAID_LEAVE")],
    })
    assert hr_direct_paid_leave_days(db, 55, FY_START, FY_END, 3) == []
    assert count_hr_direct_paid_leave_days(db, 55, FY_START, FY_END, 3) == Decimal("0")


# ── Approved requests ───────────────────────────────────────────────────────

def test_only_the_paid_portion_of_a_request_is_charged_to_the_allocation():
    """Approval adds `paid_days` to used_days, so the dates must total that.

    A 4-day request with only 2 days of earned balance is 2 paid + 2 LOP. The
    paid days go to the EARLIEST dates, matching what approval writes onto the
    attendance rows and what payroll then deducts.
    """
    db = _FakeSession({
        LeaveRequest: [
            _Req(date(2026, 6, 1), date(2026, 6, 4), paid=Decimal("2"), unpaid=Decimal("2")),
        ],
    })
    entries = approved_request_days(db, 55, 3, FY_START, FY_END)

    assert [e["days"] for e in entries] == [
        Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0"),
    ]
    assert sum((e["days"] for e in entries), Decimal("0")) == Decimal("2")
    assert "unpaid" in entries[-1]["detail"]


def test_a_request_approved_before_the_split_existed_charges_every_day():
    """paid_days/unpaid_days are `default=0`, so legacy rows read 0/0, not NULL.

    The bug this pins reached the screen: a 3-day June request approved by the
    old path (which added the full day count to `used_days`) came back with
    paid_days=0, an `is None` check that never fired, and all three dates
    rendered "LOP" beneath a heading that read "Used: 4". Both zero means no
    split was ever recorded, NOT a request taken entirely as Loss-Of-Pay.
    """
    db = _FakeSession({
        LeaveRequest: [
            _Req(date(2026, 6, 1), date(2026, 6, 3), paid=Decimal("0"), unpaid=Decimal("0")),
        ],
    })
    entries = approved_request_days(db, 55, 3, FY_START, FY_END)

    assert [e["days"] for e in entries] == [Decimal("1"), Decimal("1"), Decimal("1")]
    assert all("unpaid" not in e["detail"] for e in entries)


def test_a_request_genuinely_all_unpaid_still_charges_nothing():
    """unpaid_days > 0 means the split WAS recorded and really was all LOP."""
    db = _FakeSession({
        LeaveRequest: [
            _Req(date(2026, 6, 1), date(2026, 6, 2), paid=Decimal("0"), unpaid=Decimal("2")),
        ],
    })
    entries = approved_request_days(db, 55, 3, FY_START, FY_END)

    assert [e["days"] for e in entries] == [Decimal("0"), Decimal("0")]
    assert all("unpaid" in e["detail"] for e in entries)


def test_a_half_day_request_charges_half_a_day():
    db = _FakeSession({
        LeaveRequest: [
            _Req(date(2026, 6, 1), date(2026, 6, 1), is_half_day=True, paid=Decimal("0.5")),
        ],
    })
    entries = approved_request_days(db, 55, 3, FY_START, FY_END)
    assert len(entries) == 1
    assert entries[0]["days"] == Decimal("0.5")


# ── Short leave (monthly) ───────────────────────────────────────────────────

def test_short_leave_half_day_spends_both_of_the_months_allowance():
    db = _FakeSession({
        LeaveRequest: [],
        AttendanceRecord: [
            _Att(date(2026, 4, 3), "HALF_DAY"),   # spends both
            _Att(date(2026, 4, 17), "SHORT"),     # nothing left -> not charged
        ],
    })
    entries = short_leave_days_in_month(db, 55, 4, date(2026, 4, 1), date(2026, 5, 1))

    assert [(e["date"], e["days"]) for e in entries] == [
        (date(2026, 4, 3), Decimal("2")),
    ]


def test_short_leave_counts_an_approved_request_once_however_long_it_is():
    """The monthly Short-Leave total counts REQUESTS, so the listing must too."""
    db = _FakeSession({
        LeaveRequest: [_Req(date(2026, 4, 8), date(2026, 4, 10))],
        AttendanceRecord: [],
    })
    entries = short_leave_days_in_month(db, 55, 4, date(2026, 4, 1), date(2026, 5, 1))

    assert len(entries) == 1
    assert entries[0]["days"] == Decimal("1")
    assert entries[0]["date"] == date(2026, 4, 8)
