"""A review-queue write must never be able to break a camera.

`unknown_attendance.record` is called from the recognition thread inside the
per-track loop, for EVERY unrecognised face on an IN/OUT camera. The caller's
`except` sits at the top of the whole analysis tick, so an exception escaping
`record` does not merely lose one review row — it skips the rest of the tick
INCLUDING the step that publishes tracks to the display thread.

That is not hypothetical. Migration 037 creates `unknown_attendance_events`;
with the database still at 036 the table did not exist, so every unknown person
at the entrance raised UndefinedTable and froze the overlay boxes and the track
counter on cameras 57/58 — several times a second, because 46 of 50 employees
had no face enrolled and were therefore all "unknown".

These tests pin the contract that makes the pipeline robust to that class of
fault regardless of schema state: record() fails closed and returns None.
"""
import logging

import pytest

from app.services import unknown_attendance


class _Boom:
    """Stands in for SessionLocal; explodes the way a missing table does."""

    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise self._exc


@pytest.fixture(autouse=True)
def _reset_throttle():
    """The failure log is rate-limited by module state; isolate each test."""
    unknown_attendance._fail_last_logged = 0.0
    unknown_attendance._fail_suppressed = 0
    yield
    unknown_attendance._fail_last_logged = 0.0
    unknown_attendance._fail_suppressed = 0


def _record(camera_id="57", purpose="IN"):
    return unknown_attendance.record(
        camera_id=camera_id, purpose=purpose, event_time=None,
        track_id=1, unknown_face_id=None,
    )


def test_missing_table_returns_none_instead_of_raising(monkeypatch):
    """The exact 036-vs-037 fault: the table is not there."""
    boom = _Boom(RuntimeError('relation "unknown_attendance_events" does not exist'))
    monkeypatch.setattr("app.db.session.SessionLocal", boom)

    assert _record() is None
    assert boom.calls == 1, "the write was attempted, not skipped"


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("database is down"),
        ValueError("bad payload"),
        OSError("connection reset by peer"),
    ],
)
def test_any_database_failure_is_contained(monkeypatch, exc):
    monkeypatch.setattr("app.db.session.SessionLocal", _Boom(exc))
    assert _record() is None


def test_monitor_cameras_are_refused_without_touching_the_database(monkeypatch):
    """MONITOR is rejected on purpose alone — no session is ever opened."""
    boom = _Boom(RuntimeError("must not be reached"))
    monkeypatch.setattr("app.db.session.SessionLocal", boom)

    assert _record(purpose="MONITOR") is None
    assert _record(purpose="") is None
    assert boom.calls == 0


def test_failure_logging_is_throttled(monkeypatch, caplog):
    """A persistent fault must not write one traceback per unknown face.

    The entrance camera analyses at ~0.12s, so an unthrottled logger.exception
    here is thousands of tracebacks an hour — the unbounded-log failure this
    codebase has already been bitten by once.
    """
    monkeypatch.setattr("app.db.session.SessionLocal", _Boom(RuntimeError("down")))

    with caplog.at_level(logging.ERROR, logger="app.services.unknown_attendance"):
        for _ in range(50):
            assert _record() is None

    assert len(caplog.records) == 1, "expected exactly one traceback for 50 failures"
    # The suppressed ones are counted, not lost.
    assert unknown_attendance._fail_suppressed == 49


def test_throttle_reopens_after_the_interval(monkeypatch, caplog):
    monkeypatch.setattr("app.db.session.SessionLocal", _Boom(RuntimeError("down")))

    with caplog.at_level(logging.ERROR, logger="app.services.unknown_attendance"):
        assert _record() is None
        # Pretend a full interval has elapsed rather than sleeping through it.
        unknown_attendance._fail_last_logged -= unknown_attendance._FAIL_LOG_INTERVAL + 1
        assert _record() is None

    assert len(caplog.records) == 2
