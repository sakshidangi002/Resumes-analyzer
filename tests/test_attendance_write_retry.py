"""Attendance writes must not be lost silently.

`_submit_attendance` queues the DB write on a background executor and used to
DISCARD the result. That was unsafe by construction: the caller sets
`attendance_marked` on the track and records the per-camera cooldown BEFORE
submitting, so a failed write could never be retried by the pipeline and never
surfaced anywhere — the event simply vanished.

Now a transient failure is retried with backoff and an exhausted one is logged
at ERROR so it can be alerted on and keyed in by hand.
"""
import logging
import time

import pytest

camera_service = pytest.importorskip(
    "app.services.camera_service", reason="needs the vision stack (cv2)"
)


class _Recorder:
    """Captures mark_cctv_attendance calls and scripts its return values.

    `actions` is consumed one entry per attempt; the last entry repeats, so
    ["attendance_failed"] fails every attempt while
    ["attendance_failed", "CHECK_IN"] fails once then succeeds.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.actions: list[str] = ["CHECK_IN"]

    def __call__(self, employee_id, camera_id=None, camera_purpose=None,
                 evidence=None, event_time=None):
        self.calls.append({
            "employee_id": employee_id, "camera_id": camera_id,
            "camera_purpose": camera_purpose, "evidence": evidence,
            "event_time": event_time,
        })
        action = self.actions[min(len(self.calls) - 1, len(self.actions) - 1)]
        if action == "RAISE":
            raise RuntimeError("database is down")
        return ({}, action)

    def wait_for(self, expected: int, timeout: float = 8.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self.calls) >= expected:
                return True
            time.sleep(0.05)
        return False

    def __len__(self) -> int:
        return len(self.calls)


@pytest.fixture
def calls(monkeypatch):
    # _submit_attendance imports mark_cctv_attendance INSIDE the function, so
    # patching the module attribute is picked up at call time.
    from app.services import recognition

    recorder = _Recorder()
    monkeypatch.setattr(recognition, "mark_cctv_attendance", recorder)
    return recorder


def test_successful_write_is_not_retried(calls):
    calls.actions = ["CHECK_IN"]
    camera_service._submit_attendance(7, camera_id="53", camera_purpose="IN")
    assert calls.wait_for(1)
    time.sleep(0.4)
    assert len(calls) == 1


def test_evidence_and_event_time_reach_the_writer(calls):
    """The whole point of the evidence plumbing."""
    evidence = {"match_score": 0.71, "match_margin": 0.2,
                "track_id": 4, "snapshot_path": "2026-07-28/7_53_091530.jpg"}
    camera_service._submit_attendance(
        7, camera_id="53", camera_purpose="IN", evidence=evidence, event_time="T",
    )
    assert calls.wait_for(1)
    assert calls.calls[0]["evidence"] == evidence
    assert calls.calls[0]["event_time"] == "T"


@pytest.mark.parametrize("failure", ["attendance_failed", "validation_failed", "RAISE"])
def test_transient_failure_is_retried_then_succeeds(calls, failure):
    calls.actions = [failure, "CHECK_IN"]
    camera_service._submit_attendance(
        7, camera_id="53", camera_purpose="IN", max_attempts=3,
    )
    # First attempt fails, retry is scheduled ~2s later.
    assert calls.wait_for(2)
    assert len(calls) == 2


@pytest.mark.parametrize("rejection", [
    "cooldown",
    "monitor_camera",
    "duplicate_check_in_already_working",
    "duplicate_out_already_away",
    "check_out_without_check_in",
])
def test_business_rejections_are_never_retried(calls, rejection):
    """These mean the state machine refused it. Retrying cannot change that,
    and hammering the DB with them would hide real failures."""
    calls.actions = [rejection]
    camera_service._submit_attendance(7, camera_id="53", camera_purpose="IN")
    assert calls.wait_for(1)
    time.sleep(0.4)
    assert len(calls) == 1


def test_exhausted_retries_log_an_error(calls, caplog):
    """A lost event must be loud — somebody has to key it in by hand."""
    calls.actions = ["attendance_failed"]
    with caplog.at_level(logging.ERROR, logger=camera_service.logger.name):
        camera_service._submit_attendance(
            7, camera_id="53", camera_purpose="IN", attempt=3, max_attempts=3,
        )
        assert calls.wait_for(1)
        time.sleep(0.3)

    assert any("ATTN-WRITE LOST" in r.message for r in caplog.records), (
        "an exhausted write produced no ERROR — the event vanishes silently"
    )


def test_queue_overflow_is_refused_loudly(monkeypatch, caplog):
    """An unbounded queue turns a DB stall into an out-of-memory crash."""
    class _FullQueue:
        def qsize(self):
            return camera_service._ATTENDANCE_QUEUE_MAX + 1

    monkeypatch.setattr(
        camera_service._attendance_executor, "_work_queue", _FullQueue(),
    )
    with caplog.at_level(logging.ERROR, logger=camera_service.logger.name):
        camera_service._submit_attendance(7, camera_id="53", camera_purpose="IN")

    assert any("queue overflow" in r.message for r in caplog.records)
