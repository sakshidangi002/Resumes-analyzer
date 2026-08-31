"""A manual check-in must be visible to the camera state machine.

The bug this covers: HR adds a check-in through the attendance grid, the record
gets a sign_in_time, but no AttendanceEvent is written. The camera logic reads
only events, so the employee still looks ABSENT and the next recognition is
recorded as a second CHECK_IN instead of the BREAK_OUT / BREAK_IN it really is.
The details modal then shows "Break outs 0 / Break ins 0" and a timeline holding
a single stray Check-In.
"""
import ast
import pathlib

import pytest

from app.services.attendance_event_service import (
    count_attendance_events,
    current_state,
    resolve_camera_event,
)

ROUTES = pathlib.Path(__file__).resolve().parents[1] / \
    "Attendance Management" / "backend" / "app" / "api" / "routes" / "attendance.py"


def _endpoint(name: str) -> ast.FunctionDef:
    tree = ast.parse(ROUTES.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"endpoint {name} not found")


def _writes_times(fn: ast.FunctionDef) -> bool:
    return any(
        isinstance(n, ast.Attribute) and n.attr in ("sign_in_time", "sign_out_time")
        and isinstance(n.ctx, ast.Store)
        for n in ast.walk(fn)
    )


def _publishes_boundary(fn: ast.FunctionDef) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "sync_manual_boundary_event"
        for n in ast.walk(fn)
    )


@pytest.mark.parametrize(
    "endpoint",
    ["auto_mark_attendance", "approve_correction_request", "admin_set_attendance"],
)
def test_every_manual_time_writer_publishes_a_boundary_event(endpoint):
    """Any endpoint that sets sign_in/sign_out by hand must also emit the event,
    or the camera state machine cannot see it."""
    fn = _endpoint(endpoint)
    if not _writes_times(fn):
        pytest.skip(f"{endpoint} does not write sign_in/sign_out directly")
    assert _publishes_boundary(fn), (
        f"{endpoint} writes sign_in/sign_out without calling "
        f"sync_manual_boundary_event; the camera will treat the employee as ABSENT"
    )


def test_after_a_manual_check_in_the_camera_records_a_break_not_a_check_in():
    """The behaviour the user reported: the second event must not be a CHECK_IN."""
    assert current_state("IN") == "WORKING"
    # An exit camera while working is a break-out, never another check-in.
    assert resolve_camera_event("OUT", "IN", allow_missing_in=True) == ("BREAK_OUT", None)
    # Returning from that break is a break-in.
    assert resolve_camera_event("IN", "BREAK_OUT", allow_missing_in=True) == ("BREAK_IN", None)
    # And an entry camera while already inside is rejected, not double-counted.
    assert resolve_camera_event("IN", "IN", allow_missing_in=True)[0] is None


def test_break_counts_are_non_zero_once_a_manual_in_precedes_camera_events():
    """With the boundary event present the modal shows real break counts."""
    class _E:
        def __init__(self, t):
            self.event_type = t

    events = [_E("IN"), _E("BREAK_OUT"), _E("BREAK_IN"), _E("OUT")]
    counts = count_attendance_events(events, has_final_checkout=True)
    assert counts == {
        "check_in_count": 2, "check_out_count": 2,
        "break_in_count": 1, "break_out_count": 1,
    }
