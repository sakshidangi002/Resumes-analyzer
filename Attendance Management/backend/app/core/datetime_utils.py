
from datetime import datetime
from zoneinfo import ZoneInfo


def get_utc_now() -> datetime:
    """Return an aware UTC timestamp for persistence and token/audit events."""
    return datetime.now(ZoneInfo("UTC"))

def get_ist_now() -> datetime:
    """Current IST wall-clock time as a timezone-NAIVE datetime.

    NAIVE IS DELIBERATE — do not "fix" this to return an aware datetime.

    Every attendance timestamp column (attendance_events.event_time,
    attendance_records.sign_in_time, …) is TIMESTAMP WITHOUT TIME ZONE holding
    IST wall-clock, and the service layer compares those values against this
    function's result. Returning an aware datetime made every such comparison
    raise `TypeError: can't subtract offset-naive and offset-aware datetimes`,
    which broke:
      * calculate_intervals_from_events() — 500 on /attendance/details for
        anyone currently checked in (the open-interval "time worked so far");
      * validate_event_time() — raised on EVERY new event, and the caller's
        broad `except Exception` turned that into a silent "attendance_failed",
        so recognition kept working while nothing was recorded.

    The genuine improvement from the timezone cleanup is kept: the IST offset
    comes from the tz database via ZoneInfo instead of a hardcoded +5:30 on top
    of the deprecated utcnow(). Only the tzinfo is dropped, at the boundary.

    Use get_utc_now() where an AWARE timestamp is actually wanted (tokens,
    audit) — not for attendance.
    """
    return datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
