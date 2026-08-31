"""The live people count must not overstate what it knows.

Counting BODIES rather than faces is the point: a person walking away from the
lens has no visible face, so the face pipeline reports zero for them, which is
indistinguishable from an empty room. But a body count carries two ways to
mislead, and both are pinned here:

  * A face-only camera reports 0 body tracks because it is not measuring, not
    because the room is empty. Summing it in would drag the total down and read
    as fact.
  * A count is only as fresh as the last COMPLETED analysis pass, and a pass
    costs seconds on this hardware. The age has to travel with the number so a
    stale count can be shown as stale -- presenting a several-second-old zero as
    current is exactly how a plainly visible person came to read "People: 0".
"""
import pytest

camera_service = pytest.importorskip(
    "app.services.camera_service", reason="needs the vision stack (cv2)"
)


class _State:
    def __init__(self, people, status="running", updated_at=1000.0):
        self.active_tracks = people
        self.status = status
        self.updated_at = updated_at
        self.total_frames = 0
        self.reconnect_count = 0
        self.frame_age_ms = 0.0


class _Worker:
    def __init__(self, cid, name, purpose, people, tracking=True,
                 status="running", updated_at=1000.0):
        self.camera_id = cid
        self.name = name
        self.camera_purpose = purpose
        self.use_person_tracking = tracking
        self.state = _State(people, status, updated_at)


@pytest.fixture
def stats(monkeypatch):
    def _run(workers, now=1000.0):
        mgr = camera_service.CameraManager()
        mgr._workers = {w.camera_id: w for w in workers}
        mgr._ffmpeg_ok = True
        monkeypatch.setattr(camera_service.time, "time", lambda: now)
        return mgr.get_stats()
    return _run


def test_counts_bodies_across_tracking_cameras(stats):
    s = stats([
        _Worker(57, "Entrance", "IN", 2),
        _Worker(59, "Dev-room", "MONITOR", 3),
    ])
    assert s["people_detected"] == 5
    assert s["cameras_body_tracking"] == 2


def test_face_only_camera_is_excluded_from_the_total(stats):
    """Its 0 means "not measured", not "nobody there"."""
    s = stats([
        _Worker(57, "Entrance", "IN", 4),
        _Worker(58, "Exit", "OUT", 0, tracking=False),
    ])
    assert s["people_detected"] == 4, "a non-tracking camera must not dilute the total"
    assert s["cameras_body_tracking"] == 1
    # It is still listed, so the UI can say why it shows no number.
    row = next(r for r in s["people_by_camera"] if r["camera_id"] == 58)
    assert row["body_tracking"] is False


def test_analysis_age_travels_with_the_count(stats):
    s = stats([_Worker(57, "Entrance", "IN", 1, updated_at=990.0)], now=1000.0)
    assert s["people_by_camera"][0]["analysis_age_sec"] == 10.0


def test_never_analysed_camera_reports_null_age_not_zero(stats):
    """0.0 would render as 'just now' — the opposite of the truth."""
    s = stats([_Worker(57, "Entrance", "IN", 0, updated_at=0.0)])
    assert s["people_by_camera"][0]["analysis_age_sec"] is None


def test_offline_camera_is_reported_with_its_status(stats):
    s = stats([_Worker(57, "Entrance", "IN", 0, status="error")])
    row = s["people_by_camera"][0]
    assert row["status"] == "error"
    assert s["running_cameras"] == 0


def test_rows_carry_identity_for_the_ui(stats):
    s = stats([_Worker(59, "Dev-room", "MONITOR", 2)])
    row = s["people_by_camera"][0]
    assert row["name"] == "Dev-room"
    assert row["purpose"] == "MONITOR"
    assert row["people"] == 2


def test_no_cameras_does_not_crash(stats):
    s = stats([])
    assert s["people_detected"] == 0
    assert s["people_by_camera"] == []
