"""get_ist_now() must stay timezone-NAIVE.

Every attendance timestamp column is TIMESTAMP WITHOUT TIME ZONE holding IST
wall-clock, and the service layer compares those naive values against
get_ist_now(). A well-meaning "fix" that made it aware broke two things at once:

  * /attendance/details returned 500 for anyone currently checked in, because
    calculate_intervals_from_events() subtracts the last event time from now to
    compute the open interval — the Attendance Details modal rendered blank and
    read "Absent" while the daily row showed the person Present;
  * validate_event_time() raised on EVERY new event, and _mark_attendance's
    broad `except Exception` turned that into a silent "attendance_failed", so
    faces kept being recognised while nothing was recorded.

Both failures were invisible: one behind a generic 500 body, one behind a
catch-all. Hence this test.
"""
from datetime import datetime, timedelta

from app.core.datetime_utils import get_ist_now, get_utc_now


def test_ist_now_is_naive():
    now = get_ist_now()
    assert now.tzinfo is None, (
        "get_ist_now() must be naive: it is compared against naive DB "
        "timestamps throughout the attendance services"
    )


def test_ist_now_can_be_compared_with_a_naive_db_timestamp():
    """The exact operation that raised TypeError in production."""
    # The suppression below is deliberate: this simulates a value read back
    # from a TIMESTAMP WITHOUT TIME ZONE column, which IS naive. Attaching a
    # tzinfo would make the test pass against the very bug it exists to catch.
    # (Ruff's DTZ rules are plausibly what prompted the aware-datetime change
    # that caused this outage — right in general, wrong for columns that
    # deliberately store local wall-clock.)
    stored_event_time = datetime(2026, 7, 28, 9, 36, 4)  # noqa: DTZ001
    elapsed = get_ist_now() - stored_event_time            # must not raise
    assert isinstance(elapsed, timedelta)


def test_ist_is_five_and_a_half_hours_ahead_of_utc():
    """The tz-database lookup must still yield the right wall clock."""
    offset = get_ist_now() - get_utc_now().replace(tzinfo=None)
    assert abs(offset.total_seconds() - 5.5 * 3600) < 5


def test_utc_helper_stays_aware():
    """get_utc_now() is for tokens/audit, where aware is correct."""
    assert get_utc_now().tzinfo is not None
