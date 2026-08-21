"""Per-camera recognition profiles.

These replaced process-wide CCTV_* environment constants, where one value had to
serve both the check-in camera (whose matches become payroll rows) and the
ceiling-mounted room cameras (which can never mark attendance). The code history
shows the shared value repeatedly loosened for the room cameras — silently
loosening the payroll path with it.
"""
import pytest

from app.services import camera_profile
from app.services.camera_profile import CameraProfile, defaults_for


class _Row:
    """Stand-in for a CameraConfig row; only attribute access is used."""

    def __init__(self, **fields):
        defaults = {
            name: None
            for name in (
                "match_margin", "min_face_px", "min_det_score", "max_yaw_deg",
                "max_pitch_deg", "max_landmark_asym", "min_blur_var",
                "min_observations", "min_quality", "min_consensus",
                "analysis_interval", "face_crop_scale", "attendance_cooldown",
            )
        }
        defaults.update({"threshold": 0.45, "camera_purpose": "IN", "camera_type": "IN"})
        defaults.update(fields)
        for name, value in defaults.items():
            setattr(self, name, value)


@pytest.fixture(autouse=True)
def _clear_cache():
    camera_profile.reset_for_tests()
    yield
    camera_profile.reset_for_tests()


def _load(monkeypatch, row, camera_id="1", purpose="IN"):
    """Resolve a profile from a fake row without touching the database.

    Patches SessionLocal rather than _load, so the overlay logic, the safety
    floors and the known-good bookkeeping are all exercised for real.
    """

    class _Query:
        def filter(self, *_a, **_k):
            return self

        def first(self):
            return row

    class _DB:
        def query(self, *_a, **_k):
            return _Query()

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    import app.db.session as session_mod

    monkeypatch.setattr(session_mod, "SessionLocal", lambda: _DB())
    return camera_profile.get_profile(camera_id, purpose)


# ---------------------------------------------------------------------------
# Role defaults
# ---------------------------------------------------------------------------
def test_attendance_defaults_are_stricter_than_monitor():
    attendance = defaults_for("IN")
    monitor = defaults_for("MONITOR")

    assert attendance["margin"] > monitor["margin"]
    assert attendance["min_observations"] >= monitor["min_observations"]
    assert attendance["min_quality"] > monitor["min_quality"]
    assert attendance["limits"].min_face_px > monitor["limits"].min_face_px
    assert attendance["limits"].max_yaw_deg < monitor["limits"].max_yaw_deg


def test_attendance_margin_is_above_the_old_global():
    """0.10 was the global default and was never tuned.

    It is the knob that separates this system's documented failure (wrong person
    at 0.77, correct matches 0.73-0.79 — inseparable by score).
    """
    assert defaults_for("IN")["margin"] > 0.10


def test_attendance_requires_more_than_one_observation():
    """CCTV_CONFIRM_FRAMES shipped as 1: one frame wrote a payroll row."""
    assert defaults_for("IN")["min_observations"] >= 3


def test_out_camera_uses_attendance_defaults():
    assert defaults_for("OUT") == defaults_for("IN")


# ---------------------------------------------------------------------------
# Overlay and floors
# ---------------------------------------------------------------------------
def test_null_columns_inherit_the_purpose_default(monkeypatch):
    profile = _load(monkeypatch, _Row())
    assert profile.margin == defaults_for("IN")["margin"]
    assert profile.min_observations == defaults_for("IN")["min_observations"]


def test_stored_values_override_the_default(monkeypatch):
    profile = _load(monkeypatch, _Row(match_margin=0.31, min_observations=6))
    assert profile.margin == pytest.approx(0.31)
    assert profile.min_observations == 6


def test_legacy_threshold_is_clamped_to_the_floor(monkeypatch):
    """Old rows still carry 0.05, which accepts essentially random faces."""
    profile = _load(monkeypatch, _Row(threshold=0.05))
    assert profile.threshold >= 0.35


def test_absurd_min_face_px_is_clamped(monkeypatch):
    profile = _load(monkeypatch, _Row(min_face_px=2))
    assert profile.limits.min_face_px >= 14.0


def test_min_observations_never_drops_below_one(monkeypatch):
    profile = _load(monkeypatch, _Row(min_observations=0))
    assert profile.min_observations >= 1


def test_monitor_row_resolves_as_monitor(monkeypatch):
    profile = _load(
        monkeypatch, _Row(camera_purpose="MONITOR", camera_type="MONITOR"),
        purpose="MONITOR",
    )
    assert profile.purpose == "MONITOR"
    assert profile.marks_attendance is False


def test_unrecognised_purpose_falls_back_to_attendance_strictness(monkeypatch):
    """An unknown role must not accidentally become the permissive one."""
    profile = _load(monkeypatch, _Row(camera_purpose="LOBBY", camera_type="LOBBY"))
    assert profile.purpose == "IN"
    assert profile.marks_attendance is True


# ---------------------------------------------------------------------------
# Failure behaviour
# ---------------------------------------------------------------------------
def test_database_failure_keeps_the_previous_profile(monkeypatch):
    """Never widen the gates because Postgres hiccuped.

    Falling back to a permissive default on a transient error is a way to write
    attendance from a face the strict profile would have rejected.
    """
    good = _load(monkeypatch, _Row(match_margin=0.29), camera_id="55")
    assert good.margin == pytest.approx(0.29)

    import app.db.session as session_mod

    def explode():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(session_mod, "SessionLocal", explode)
    camera_profile.invalidate("55")

    recovered = camera_profile.get_profile("55", "IN")
    assert recovered.margin == pytest.approx(0.29)


def test_describe_is_a_single_line(monkeypatch):
    profile = _load(monkeypatch, _Row())
    assert "\n" not in profile.describe()
    assert "thr=" in profile.describe()


def test_profiles_compare_by_value():
    """The reload log depends on equality detecting a real change."""
    base = defaults_for("IN")
    a = CameraProfile(camera_id="1", purpose="IN", threshold=base["threshold"],
                      margin=base["margin"], limits=base["limits"],
                      min_observations=base["min_observations"],
                      min_quality=base["min_quality"],
                      min_consensus=base["min_consensus"],
                      analysis_interval=base["analysis_interval"],
                      face_crop_scale=base["face_crop_scale"],
                      attendance_cooldown=base["attendance_cooldown"])
    b = CameraProfile(**{**a.__dict__})
    assert a == b
