"""/health tells the truth about the subsystems it depends on.

The endpoint shipped as `return {"status": "ok"}` — unconditional. It never
touched the database and never looked at a camera, so it reported a healthy
service while every camera was dead. A supervisor configured against it would
never restart anything, and an operator reading it would be actively misled.

These tests pin the behaviour that replaced it:

  * /health/live stays trivially 200 (it is the RESTART target, and restarting
    cannot fix an unplugged camera — see the module comment in app/main.py);
  * /health returns 503 when the database, a camera, or the attendance writer
    is in trouble, and names the reason;
  * a static scene does NOT count as a fault, because the motion gate is
    supposed to suspend inference on one;
  * the probes never raise, however broken the thing they are probing.

No database and no cameras are required: every probe is exercised against a
fake camera manager and a monkeypatched DB probe, in keeping with the pure-unit
design stated in conftest.py.
"""
import asyncio
import threading

import pytest

pytest.importorskip("fastapi", reason="needs fastapi")

from app import main as app_main


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeThread:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


def _camera(
    camera_id=57,
    name="Entrance",
    status="running",
    frame_age_sec=0.4,
    inference_age_sec=3.5,
    stream_alive=True,
    recog_alive=True,
):
    return {
        "camera_id": camera_id,
        "name": name,
        "purpose": "IN",
        "status": status,
        "last_error": None,
        "frame_age_sec": frame_age_sec,
        "inference_age_sec": inference_age_sec,
        "reconnect_count": 0,
        "stream_thread_alive": stream_alive,
        "recognition_thread_alive": recog_alive,
    }


class _FakeManager:
    """Stands in for camera_manager. Only health_snapshot is used."""

    def __init__(self, cameras=None, queue_depth=0, queue_max=500, raises=False):
        self._cameras = cameras if cameras is not None else [_camera()]
        self._queue_depth = queue_depth
        self._queue_max = queue_max
        self._raises = raises

    def health_snapshot(self):
        if self._raises:
            raise RuntimeError("snapshot exploded")
        return {
            "ffmpeg_ok": True,
            "cameras": list(self._cameras),
            "attendance_queue_depth": self._queue_depth,
            "attendance_queue_max": self._queue_max,
        }


@pytest.fixture
def cameras_on(monkeypatch):
    """This process runs the camera pipeline."""
    monkeypatch.setattr(app_main, "_cctv_workers_enabled", lambda: True)


@pytest.fixture
def db_ok(monkeypatch):
    monkeypatch.setattr(app_main, "_check_database", lambda: {"ok": True})


def _install_manager(monkeypatch, manager):
    """Patch the module the probes import camera_manager FROM.

    `_probe_cameras` does a function-local `from app.services.camera_service
    import camera_manager`, so the name has to be replaced on that module.
    """
    import app.services.camera_service as cs

    monkeypatch.setattr(cs, "camera_manager", manager)


def _health():
    """Call the async endpoint and return (status_code, body)."""
    result = asyncio.run(app_main.health())
    if isinstance(result, dict):
        return 200, result
    import json

    return result.status_code, json.loads(bytes(result.body))


# ---------------------------------------------------------------------------
# Liveness is not health
# ---------------------------------------------------------------------------
def test_liveness_is_unconditional():
    """The restart target must not depend on cameras or the database.

    If it did, a dead camera would restart the process, which does not
    reconnect the camera — it just adds an outage on top of the fault.
    """
    assert app_main.health_live() == {"status": "ok"}


def test_liveness_stays_ok_while_health_is_degraded(monkeypatch, cameras_on):
    monkeypatch.setattr(app_main, "_check_database", lambda: {"ok": False, "error": "OperationalError"})
    _install_manager(monkeypatch, _FakeManager())

    code, _ = _health()
    assert code == 503
    assert app_main.health_live() == {"status": "ok"}


# ---------------------------------------------------------------------------
# The healthy case
# ---------------------------------------------------------------------------
def test_healthy_system_returns_200(monkeypatch, cameras_on, db_ok):
    _install_manager(monkeypatch, _FakeManager())

    code, body = _health()
    assert code == 200
    assert body["status"] == "ok"
    assert body["checks"]["database"]["ok"]
    assert body["checks"]["cameras"]["ok"]
    assert body["checks"]["attendance_writer"]["ok"]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def test_unreachable_database_is_degraded(monkeypatch, cameras_on):
    monkeypatch.setattr(
        app_main, "_check_database", lambda: {"ok": False, "error": "OperationalError"}
    )
    _install_manager(monkeypatch, _FakeManager())

    code, body = _health()
    assert code == 503
    assert body["status"] == "degraded"
    assert not body["checks"]["database"]["ok"]


def test_hung_database_times_out_rather_than_hanging_the_probe(monkeypatch):
    """A probe that blocks is indistinguishable from a dead process."""
    import time

    release = threading.Event()
    monkeypatch.setattr(app_main, "_HEALTH_DB_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(app_main, "_health_db_inflight", None)
    monkeypatch.setattr(app_main, "_probe_database", lambda: release.wait(30))
    try:
        started = time.monotonic()
        result = app_main._check_database()
        elapsed = time.monotonic() - started

        assert result["ok"] is False
        assert "timeout" in result["error"]
        assert elapsed < 2.0, "the probe waited for the hung database"
    finally:
        release.set()


def test_a_second_probe_does_not_start_while_one_is_hung(monkeypatch):
    """The reason this uses a private executor instead of asyncio.to_thread.

    A blocking socket read cannot be cancelled, so a timed-out probe keeps
    running. If each poll started a fresh one, a supervisor hitting /health
    every 10s against a hung database would park a new thread every 10s — in
    the executor the rest of the app shares. One in flight, ever.
    """
    import time

    release = threading.Event()
    calls = []

    def _hang():
        calls.append(time.monotonic())
        release.wait(30)

    monkeypatch.setattr(app_main, "_HEALTH_DB_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(app_main, "_health_db_inflight", None)
    monkeypatch.setattr(app_main, "_probe_database", _hang)
    try:
        first = app_main._check_database()
        second = app_main._check_database()
        third = app_main._check_database()

        assert first["ok"] is False
        assert second["ok"] is False and "in flight" in second["error"]
        assert third["ok"] is False and "in flight" in third["error"]
        assert len(calls) == 1, f"started {len(calls)} probes against a hung database"
    finally:
        release.set()


def test_database_error_body_never_leaks_the_connection_string(monkeypatch):
    """A connection failure's message carries the DSN, password included."""
    def _explode():
        raise RuntimeError(
            "could not connect to postgresql://hrms:hunter2@10.0.0.9/postgres"
        )

    monkeypatch.setattr(app_main, "SessionLocal", _explode)

    result = app_main._probe_database()
    assert result["ok"] is False
    assert "hunter2" not in str(result)
    assert "postgresql://" not in str(result)
    assert result["error"] == "RuntimeError"


def test_probe_survives_an_exhausted_connection_pool(monkeypatch):
    """SessionLocal() itself raises when the pool is exhausted.

    "FATAL: sorry, too many clients already" is a failure this deployment has
    actually hit, and it happens at checkout — before any query runs. The probe
    must report it, not propagate it.
    """
    def _exhausted():
        raise RuntimeError("QueuePool limit of size 20 overflow 20 reached")

    monkeypatch.setattr(app_main, "SessionLocal", _exhausted)

    result = app_main._probe_database()
    assert result["ok"] is False
    assert result["error"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Cameras — the failure the old endpoint could not see
# ---------------------------------------------------------------------------
def test_a_camera_with_no_recent_frame_is_degraded(monkeypatch, cameras_on, db_ok):
    """The regression the whole ticket exists for: cameras dead, /health "ok"."""
    _install_manager(monkeypatch, _FakeManager(cameras=[_camera(frame_age_sec=120.0)]))

    code, body = _health()
    assert code == 503
    unhealthy = body["checks"]["cameras"]["unhealthy"]
    assert len(unhealthy) == 1
    assert unhealthy[0]["camera_id"] == 57
    assert any("frame" in reason for reason in unhealthy[0]["reasons"])


def test_a_camera_that_never_produced_a_frame_is_degraded(monkeypatch, cameras_on, db_ok):
    """None is 'never delivered anything', not 'no data yet'."""
    _install_manager(monkeypatch, _FakeManager(cameras=[_camera(frame_age_sec=None)]))

    code, body = _health()
    assert code == 503
    assert "no frame ever received" in body["checks"]["cameras"]["unhealthy"][0]["reasons"]


def test_a_dead_capture_thread_is_degraded(monkeypatch, cameras_on, db_ok):
    _install_manager(monkeypatch, _FakeManager(cameras=[_camera(stream_alive=False)]))

    code, body = _health()
    assert code == 503
    assert "capture thread dead" in body["checks"]["cameras"]["unhealthy"][0]["reasons"]


def test_a_dead_recognition_thread_is_degraded(monkeypatch, cameras_on, db_ok):
    _install_manager(monkeypatch, _FakeManager(cameras=[_camera(recog_alive=False)]))

    code, body = _health()
    assert code == 503
    assert "recognition thread dead" in body["checks"]["cameras"]["unhealthy"][0]["reasons"]


def test_a_camera_in_error_status_is_degraded(monkeypatch, cameras_on, db_ok):
    _install_manager(monkeypatch, _FakeManager(cameras=[_camera(status="error")]))

    code, body = _health()
    assert code == 503


def test_a_worker_without_recognition_threads_is_not_a_fault(monkeypatch, cameras_on, db_ok):
    """HCNetSDK workers have no recognition/display thread.

    None means 'not applicable' and must not be confused with False, which
    means 'died'.
    """
    _install_manager(monkeypatch, _FakeManager(
        cameras=[_camera(stream_alive=None, recog_alive=None)]
    ))

    code, body = _health()
    assert code == 200
    assert body["checks"]["cameras"]["ok"]


# --- the false-alarm this design has to avoid ------------------------------
def test_an_idle_scene_is_not_a_fault(monkeypatch, cameras_on, db_ok):
    """Inference pausing on a static scene is correct behaviour, not a fault.

    The motion gate coasts monitor cameras for _MONITOR_COAST_SEC and forces a
    doorway pass only every _MAX_IDLE_SKIP_QUIET_SEC, on top of a p99 cycle
    around 25s. An empty corridor at night is SUPPOSED to look idle. Alarming
    on it would make the endpoint cry wolf every night, which is how a health
    check gets ignored.
    """
    _install_manager(monkeypatch, _FakeManager(
        cameras=[_camera(frame_age_sec=0.3, inference_age_sec=45.0)]
    ))

    code, body = _health()
    assert code == 200, "a quiet corridor must not read as a broken camera"
    assert body["checks"]["cameras"]["ok"]


def test_inference_stopping_for_far_too_long_is_a_fault(monkeypatch, cameras_on, db_ok):
    """Camera 60 produced a 3,360-second gap (host sleep) that nothing caught.

    Frames keep arriving in that case, so frame age alone cannot see it.
    """
    _install_manager(monkeypatch, _FakeManager(
        cameras=[_camera(frame_age_sec=0.3, inference_age_sec=3360.0)]
    ))

    code, body = _health()
    assert code == 503
    assert any(
        "no analysis" in reason
        for reason in body["checks"]["cameras"]["unhealthy"][0]["reasons"]
    )


def test_only_the_failing_camera_is_named(monkeypatch, cameras_on, db_ok):
    _install_manager(monkeypatch, _FakeManager(cameras=[
        _camera(camera_id=57, name="Entrance"),
        _camera(camera_id=58, name="Exit", frame_age_sec=300.0),
        _camera(camera_id=59, name="Dev-room"),
    ]))

    code, body = _health()
    assert code == 503
    unhealthy = body["checks"]["cameras"]["unhealthy"]
    assert [c["camera_id"] for c in unhealthy] == [58]
    assert body["checks"]["cameras"]["total"] == 3


def test_an_api_only_instance_reports_cameras_as_not_applicable(monkeypatch, db_ok):
    """CCTV_WORKERS_ENABLED=0 is a deliberate deployment, not a fault."""
    monkeypatch.setattr(app_main, "_cctv_workers_enabled", lambda: False)

    code, body = _health()
    assert code == 200
    assert body["checks"]["cameras"]["enabled"] is False


def test_a_broken_camera_manager_is_degraded_not_a_500(monkeypatch, cameras_on, db_ok):
    """A probe that raises must report a fault, never take down the endpoint."""
    _install_manager(monkeypatch, _FakeManager(raises=True))

    code, body = _health()
    assert code == 503
    assert not body["checks"]["cameras"]["ok"]
    assert body["checks"]["cameras"]["error"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Attendance writer
# ---------------------------------------------------------------------------
def test_a_backed_up_attendance_queue_is_degraded(monkeypatch, cameras_on, db_ok):
    """Fail BEFORE the queue is full — at 100% writes are already dropped."""
    _install_manager(monkeypatch, _FakeManager(queue_depth=450, queue_max=500))

    code, body = _health()
    assert code == 503
    assert not body["checks"]["attendance_writer"]["ok"]
    assert body["checks"]["attendance_writer"]["queue_depth"] == 450


def test_a_shallow_attendance_queue_is_healthy(monkeypatch, cameras_on, db_ok):
    _install_manager(monkeypatch, _FakeManager(queue_depth=12, queue_max=500))

    code, body = _health()
    assert code == 200


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------
def test_health_strict_off_restores_the_old_behaviour(monkeypatch, cameras_on):
    """The rollback path: an env var, not a code change.

    Everything is broken here — the old endpoint still says ok, which is
    exactly the behaviour being rolled back TO.
    """
    monkeypatch.setattr(app_main, "_HEALTH_STRICT", False)
    monkeypatch.setattr(app_main, "_check_database", lambda: {"ok": False, "error": "nope"})
    _install_manager(monkeypatch, _FakeManager(cameras=[_camera(frame_age_sec=9999.0)]))

    code, body = _health()
    assert code == 200
    assert body == {"status": "ok"}


# ---------------------------------------------------------------------------
# The snapshot itself
# ---------------------------------------------------------------------------
def test_health_snapshot_does_not_take_the_frame_lock_or_read_the_profile():
    """The probe must not contend with capture threads or hit the database.

    A supervisor polls this on a schedule. list_statuses() would have been the
    obvious source, but it calls serialize_state(), which takes each worker's
    _frame_lock AND reads the camera profile (a database-backed lookup) — the
    two things a degraded system has least of. This test pins that choice by
    exploding if either is touched.
    """
    pytest.importorskip("cv2", reason="camera_service needs OpenCV")
    from app.services.camera_service import CameraManager

    class _Boom:
        def __enter__(self):
            raise AssertionError("health_snapshot took the frame lock")

        def __exit__(self, *a):
            return False

    class _Worker:
        camera_id = 57
        name = "Entrance"
        camera_purpose = "IN"

        def __init__(self):
            self.state = type("S", (), {
                "status": "running", "last_error": None,
                "last_frame_time": 1000.0, "updated_at": 1000.0,
                "reconnect_count": 0,
            })()
            self._frame_lock = _Boom()
            self._stream_thread = _FakeThread(True)
            self._recog_thread = _FakeThread(True)

        def serialize_state(self):
            raise AssertionError("health_snapshot called serialize_state")

        def profile(self):
            raise AssertionError("health_snapshot read the camera profile")

    manager = CameraManager()
    manager._workers = {57: _Worker()}

    snapshot = manager.health_snapshot()
    assert snapshot["cameras"][0]["camera_id"] == 57
    assert snapshot["cameras"][0]["stream_thread_alive"] is True


def test_health_snapshot_survives_an_unreadable_worker():
    pytest.importorskip("cv2", reason="camera_service needs OpenCV")
    from app.services.camera_service import CameraManager

    class _BadWorker:
        camera_id = 60
        name = "Dev-room2"

        @property
        def state(self):
            raise RuntimeError("worker is wedged")

    manager = CameraManager()
    manager._workers = {60: _BadWorker()}

    snapshot = manager.health_snapshot()
    assert snapshot["cameras"][0]["status"] == "unreadable"
    assert "RuntimeError" in snapshot["cameras"][0]["last_error"]
