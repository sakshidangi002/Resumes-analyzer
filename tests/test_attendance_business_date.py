"""The attendance business-day boundary.

Regression test for the night-shift bug: events were keyed on the plain
calendar date, so an employee leaving at 00:30 was evaluated against a brand
new day where their state was ABSENT. resolve_camera_event() then returned
CHECK_IN (because attendance_checkin_on_missing_in defaults to True), so
LEAVING the building created a check-in for the next day — and the day they
actually worked was never closed.

Pure logic tests: no database, no camera.
"""
from datetime import date, datetime

import pytest
from app.services.attendance_event_service import (
    business_date,
    current_state,
    resolve_camera_event,
)


@pytest.mark.parametrize(
    "clock,expected",
    [
        ("2026-07-28T09:15:00", "2026-07-28"),  # normal morning
        ("2026-07-28T18:30:00", "2026-07-28"),  # normal evening
        ("2026-07-28T23:50:00", "2026-07-28"),  # late, before midnight
        ("2026-07-29T00:00:00", "2026-07-28"),  # midnight exactly -> prior day
        ("2026-07-29T00:30:00", "2026-07-28"),  # the bug: exit after midnight
        ("2026-07-29T04:59:00", "2026-07-28"),  # still the night shift
        ("2026-07-29T05:00:00", "2026-07-29"),  # boundary: new day starts
    ],
)
def test_business_date_boundary(clock, expected):
    assert business_date(datetime.fromisoformat(clock), 5).isoformat() == expected


@pytest.mark.parametrize(
    "clock",
    ["2026-07-28T09:15:00", "2026-07-29T00:30:00", "2026-07-29T04:59:00"],
)
def test_day_start_zero_is_plain_calendar_date(clock):
    """day_start_hour=0 must reproduce the old behaviour exactly.

    This is the safe deploy value: ship with 0, confirm no report shifts, then
    raise it. If this ever diverges, the "no behaviour change" promise is broken.
    """
    dt = datetime.fromisoformat(clock)
    assert business_date(dt, 0) == dt.date()


def test_after_midnight_exit_resolves_as_out_not_checkin():
    """The actual defect: an OUT read at 00:30 must not become a CHECK_IN.

    With the business-day boundary the employee's last event (yesterday 21:00
    IN) is still on the same attendance day, so the state is WORKING and the
    OUT camera correctly produces a BREAK_OUT.
    """
    last_type = "IN"                       # they checked in at 21:00 yesterday
    assert current_state(last_type) == "WORKING"

    event_type, reject = resolve_camera_event(
        "OUT", last_type, allow_missing_in=True,
    )
    assert reject is None
    assert event_type == "BREAK_OUT"


def test_absent_at_out_camera_still_recovers_a_missed_checkin():
    """Guard the behaviour we did NOT want to change.

    Someone genuinely first seen at the OUT camera (their entrance read was
    missed) must still get a recovered CHECK_IN — that is what
    attendance_checkin_on_missing_in is for. The night-shift fix must not
    remove it, only stop it firing after midnight for someone already inside.
    """
    event_type, reject = resolve_camera_event("OUT", None, allow_missing_in=True)
    assert (event_type, reject) == ("CHECK_IN", None)

    event_type, reject = resolve_camera_event("OUT", None, allow_missing_in=False)
    assert event_type is None
    assert reject == "check_out_without_check_in"


def test_business_date_accepts_a_date_across_month_end():
    # Naive on purpose — the whole attendance pipeline works in naive IST
    # (to_naive_ist), matching TIMESTAMP WITHOUT TIME ZONE columns.
    assert business_date(datetime(2026, 8, 1, 0, 30), 5) == date(2026, 7, 31)  # noqa: DTZ001
