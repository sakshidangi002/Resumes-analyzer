"""Attendance auto-closeout: pick the right day, and the right time to close at.

Regression test for days the OUT camera never closed. Those days accrued an
open interval until midnight and then froze with sign_out_time = NULL and an
inflated total_work_hours, because nothing in the application ever closed them.

These are pure-logic tests over the two decisions that carry the risk:
  * WHICH day gets closed (never one that is legitimately still running)
  * WHAT time is recorded (last-seen, never the cutoff)
The DB-touching parts are covered by find_open_days' query shape, exercised in
integration.
"""
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

from app.services.attendance_closeout import CLOSEOUT_SOURCE, _close_at
from app.services.attendance_event_service import business_date


def _event(when: str, source: str = "AUTO"):
    return SimpleNamespace(
        event_time=datetime.fromisoformat(when),
        source=source,
        attendance_record_id=1,
    )


def test_closes_at_last_seen_not_at_cutoff():
    """The money test.

    Someone last seen at 14:05 must be closed at 14:05, NOT at the 23:00
    cutoff. Closing at the cutoff would silently credit nine hours nobody
    worked, straight into payroll.
    """
    target = date(2026, 7, 27)
    # Naive on purpose: attendance_events.event_time is TIMESTAMP WITHOUT TIME
    # ZONE and the pipeline works in naive IST throughout (to_naive_ist).
    # Attaching a tzinfo here would test something the code never sees.
    assert _close_at(_event("2026-07-27T14:05:00"), target, 23) == datetime(  # noqa: DTZ001
        2026, 7, 27, 14, 5
    )


def test_caps_at_cutoff_when_last_seen_is_later():
    """A last-seen time past the cutoff is capped, not honoured."""
    target = date(2026, 7, 27)
    assert _close_at(_event("2026-07-27T23:45:00"), target, 23) == datetime(  # noqa: DTZ001
        2026, 7, 27, 23, 0
    )


def test_target_day_is_the_previous_business_day():
    """Running at 02:30 must close the day before, never the day in progress.

    With a 05:00 business-day start, 02:30 on the 28th is still inside the
    27th's business day. Closing business_date(now) would truncate a night
    shift that is legitimately still running, so the job steps back one day.
    """
    now = datetime(2026, 7, 28, 2, 30)  # noqa: DTZ001 — naive IST, see above
    target = business_date(now, 5) - timedelta(days=1)
    assert target == date(2026, 7, 26)

    # And with the boundary disabled (the safe deploy value) it is simply
    # yesterday relative to the calendar date.
    assert business_date(now, 0) - timedelta(days=1) == date(2026, 7, 27)


def test_closeout_source_is_identifiable():
    """Every synthetic row must be reversible with one DELETE.

    If this constant ever changes, previously written closeout events become
    unfindable and the job stops being idempotent against them.
    """
    assert CLOSEOUT_SOURCE == "AUTO_CLOSE"


def test_cutoff_hour_is_applied_on_the_target_day_not_today():
    """The cap must be built from the day being closed, not the current date."""
    target = date(2026, 7, 20)
    closed_at = _close_at(_event("2026-07-20T23:59:00"), target, 23)
    assert closed_at.date() == target
    assert closed_at.time() == time(23, 0)
