"""/api/metrics — the read side of T-02.

The endpoint is a READ of state the pipeline already keeps. Two properties
matter more than the payload:

  * it must never start work, take a frame lock, hit the database or run
    inference — the system being measured has no spare CPU, and a probe that
    competes with the cameras distorts what it reports;
  * one broken source must not empty the response. A metrics endpoint that
    500s because a single counter is unavailable tells the operator nothing
    about the other nine.
"""
import pytest

pytest.importorskip("fastapi", reason="needs fastapi")

from app.api.routes import metrics as metrics_route


def test_the_route_is_registered_and_admin_only():
    from app.api.routes import api_router

    paths = {r.path: r for r in api_router.routes if getattr(r, "path", "") == "/metrics"}
    assert paths, "the metrics route is not registered on api_router"

    # The guard is a Depends(require_roles([...])) in the signature.
    import inspect

    source = inspect.getsource(metrics_route.get_metrics)
    assert 'require_roles(["Admin"])' in source, "metrics must be Admin-only"


def test_payload_has_every_top_level_section(monkeypatch):
    body = metrics_route.get_metrics(current_user=object())
    for key in ("uptime_sec", "cameras", "gates", "pipeline", "streams", "health"):
        assert key in body, f"missing section: {key}"
    assert set(body["gates"]) == {"yolo", "face"}


def test_one_broken_source_does_not_empty_the_response(monkeypatch):
    """The reason every source goes through _safe()."""
    from app.services import bytetrack_engine

    def _explode():
        raise RuntimeError("perf store is wedged")

    monkeypatch.setattr(bytetrack_engine, "get_perf_stats", _explode)

    body = metrics_route.get_metrics(current_user=object())
    assert body["cameras"] == {}, "the broken source should degrade to empty"
    assert "pipeline" in body and "streams" in body, "other sections were lost"


def test_safe_returns_the_default_on_failure():
    def _explode():
        raise ValueError("nope")

    assert metrics_route._safe("x", _explode, {"fallback": True}) == {"fallback": True}
    assert metrics_route._safe("x", lambda: {"real": 1}, {}) == {"real": 1}


def test_the_endpoint_reports_counters_the_pipeline_recorded(monkeypatch):
    """End-to-end through the real singleton: record, then read it back."""
    from app.services import pipeline_metrics

    pipeline_metrics.metrics.reset()
    try:
        pipeline_metrics.metrics.record_decision(
            57, reason="insufficient_observations", allowed=False, face_px=33.0
        )
        pipeline_metrics.metrics.record_attendance_write(57, "lost")

        body = metrics_route.get_metrics(current_user=object())
        pipeline = body["pipeline"]

        assert pipeline["decisions_total"]["insufficient_observations"] == 1
        assert pipeline["face_px_total"]["28-40"] == 1
        assert pipeline["attendance_writes_total"]["lost"] == 1
    finally:
        pipeline_metrics.metrics.reset()


def test_the_endpoint_does_not_touch_the_database(monkeypatch):
    """A metrics read must not take a pooled connection.

    The audit found request traffic starving the cameras for exactly this
    resource; a metrics endpoint polled on a schedule must not add to it.
    """
    from app.db import session as db_session

    def _forbidden(*_a, **_kw):
        raise AssertionError("/api/metrics opened a database session")

    monkeypatch.setattr(db_session, "SessionLocal", _forbidden)
    metrics_route.get_metrics(current_user=object())


def test_the_endpoint_does_not_take_a_camera_frame_lock(monkeypatch):
    """serialize_state() takes _frame_lock and reads the profile; get_stats and
    health_snapshot deliberately do not. Pin that the endpoint stays on the
    cheap path."""
    pytest.importorskip("cv2", reason="camera_service needs OpenCV")
    from app.services import camera_service

    class _Boom:
        def __enter__(self):
            raise AssertionError("/api/metrics took a camera frame lock")

        def __exit__(self, *a):
            return False

    class _Worker:
        camera_id = 57
        name = "Entrance"
        camera_purpose = "IN"
        use_person_tracking = True

        def __init__(self):
            self.state = type("S", (), {
                "status": "running", "last_error": None, "last_frame_time": 1000.0,
                "updated_at": 1000.0, "reconnect_count": 0, "total_frames": 10,
                "frame_age_ms": 12.0, "active_tracks": 1,
            })()
            self._frame_lock = _Boom()
            self._stream_thread = None
            self._recog_thread = None

        def serialize_state(self):
            raise AssertionError("/api/metrics called serialize_state")

        def profile(self):
            raise AssertionError("/api/metrics read the camera profile")

    manager = camera_service.CameraManager()
    manager._workers = {57: _Worker()}
    monkeypatch.setattr(camera_service, "camera_manager", manager)

    body = metrics_route.get_metrics(current_user=object())
    assert body["health"]["cameras"][0]["camera_id"] == 57
