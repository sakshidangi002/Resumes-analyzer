"""A person who could not be identified must still be counted.

The rule this pins: "did somebody pass through?" and "who was it?" are separate
questions. Detection and tracking answer the first; recognition answers only the
second. A face that was never visible downgrades an event from Employee to
Unknown - it must never remove the person from the corridor total.

Before this, an UNKNOWN transit was only recorded from inside the
`decision.reason == "no_match"` branch, which is reached only after a face has
been found, passed the quality gate and been matched against the gallery. So
somebody walking away from the lens produced no event of either kind and simply
vanished from the count.

The other invariant here is that an unknown transit is never given a name.
Filling `identity` with a best guess on a payroll-adjacent record is the failure
this system has already suffered once.
"""
import pytest

pytest.importorskip("cv2", reason="needs the vision stack")

from app.services import corridor_counts


class _Q:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def outerjoin(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def all(self):
        return self._rows

    def count(self):
        return len(self._rows)


class _Session:
    """Serves query results in the order corridor_counts asks for them."""

    def __init__(self, *result_sets):
        self._sets = list(result_sets)

    def query(self, *a, **k):
        return _Q(self._sets.pop(0) if self._sets else [])

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


DOORWAYS = [(57, "IN"), (58, "OUT")]


@pytest.fixture
def summary(monkeypatch):
    def _run(doorways, attendance_rows, unknown_in, unknown_out):
        sess = _Session(doorways, attendance_rows, unknown_in, unknown_out)
        monkeypatch.setattr("app.db.session.SessionLocal", lambda: sess)
        import datetime
        return corridor_counts.corridor_summary(datetime.date(2026, 8, 24))
    return _run


def test_unknown_people_are_included_in_the_in_count(summary):
    """The whole point: 5 entered, 3 named, 2 unknown -> in_count is 5."""
    r = summary(
        DOORWAYS,
        [("57", "IN"), ("57", "IN"), ("57", "IN")],   # 3 employees in
        [object(), object()],                          # 2 unknown in
        [],
    )
    assert r["in_count"] == 5
    assert r["employees_recognized"] == 3
    assert r["unknown_people"] == 2


def test_in_count_always_equals_employees_plus_unknown(summary):
    r = summary(DOORWAYS, [("57", "IN"), ("58", "OUT")], [object()], [object(), object()])
    b = r["breakdown"]
    assert r["in_count"] == b["employees_in"] + b["unknown_in"]
    assert r["out_count"] == b["employees_out"] + b["unknown_out"]


def test_direction_comes_from_the_camera_not_the_event_name(summary):
    """attendance_events mixes IN/OUT with CHECK_IN/BREAK_OUT; the camera is
    the reliable signal for which side of the threshold somebody was on."""
    r = summary(DOORWAYS, [("58", "CHECK_IN"), ("58", "BREAK_OUT")], [], [])
    assert r["breakdown"]["employees_out"] == 2
    assert r["breakdown"]["employees_in"] == 0


def test_monitor_cameras_cannot_contribute(summary):
    """A room camera watches desks, never a threshold. Counting someone sitting
    down as someone arriving would be worse than not counting at all."""
    r = summary([], [("59", "IN")], [], [])
    assert r["in_count"] == 0
    assert r["out_count"] == 0
    assert "note" in r


def test_an_empty_corridor_reports_zero(summary):
    r = summary(DOORWAYS, [], [], [])
    assert r["in_count"] == 0 and r["out_count"] == 0
    assert r["employees_recognized"] == 0 and r["unknown_people"] == 0
