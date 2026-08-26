"""Reading V1's tracks to compute V2 occupancy, without touching V1.

WHY A BRIDGE AT ALL. The live feed is V1's, and V2 is not instantiated in the
API process. Putting chair occupancy on that feed had three routes: edit V1's
renderer (V1 must stay untouched), run V2 alongside it (a second RTSP stream and
a second YOLO pass, purely to draw an overlay), or reuse the person boxes V1 has
already computed. This is the third.

THE TWO PROPERTIES THAT MATTER. It must add no inference -- otherwise the cheap
option becomes the expensive one it was chosen over -- and it must not carry
identity across. V1's tracks know employee names; chair occupancy must not,
because a seat is occupied whether or not anyone knows who is in it, and letting
identity in here rebuilds the coupling that made V1's counting depend on
recognition working.
"""
import pytest

from app.cctv_v2.pipeline import v1_bridge
from app.cctv_v2.pipeline.track import TrackState


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeV1Track:
    """Shaped like V1's PersonTrack, identity fields included on purpose."""

    def __init__(self, track_id, box, employee_id=None, employee_name=None):
        self.track_id = track_id
        self.box = box
        self.confidence = 0.7
        self.last_seen = 1000.0
        self.employee_id = employee_id
        self.employee_name = employee_name
        self.identity_source = "face" if employee_id else None


class FakeWorker:
    def __init__(self, tracks, frame=None):
        self._latest_tracks = tracks
        self._latest_frame = frame
        self._frame_lock = FakeLock()


def install(monkeypatch, worker):
    monkeypatch.setattr(v1_bridge, "_v1_worker", lambda cid: worker)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def test_v1_person_boxes_become_v2_tracks(monkeypatch):
    install(monkeypatch, FakeWorker([FakeV1Track(7, (100, 200, 180, 600))]))
    tracks = v1_bridge.read_v1_tracks(59)

    assert len(tracks) == 1
    t = tracks[0]
    assert t.camera_id == 59
    assert t.track_id == 7
    assert t.bbox == (100.0, 200.0, 180.0, 600.0)
    assert t.state is TrackState.CONFIRMED


def test_identity_never_crosses_the_bridge(monkeypatch):
    """A seat is occupied whether or not anyone knows who is in it. Letting a
    name through here is how counting came to depend on recognition."""
    install(monkeypatch, FakeWorker([
        FakeV1Track(1, (10, 10, 90, 400), employee_id=42, employee_name="Adarsh"),
    ]))
    t = v1_bridge.read_v1_tracks(59)[0]

    assert not hasattr(t, "employee_id")
    assert not hasattr(t, "employee_name")
    assert not hasattr(t, "identity_source")
    assert "Adarsh" not in repr(t)


def test_a_camera_v1_is_not_running_yields_no_tracks(monkeypatch):
    install(monkeypatch, None)
    assert v1_bridge.read_v1_tracks(59) == []


def test_not_running_is_distinguishable_from_an_empty_room(monkeypatch):
    """They look identical in the track list and mean very different things on
    a dashboard."""
    install(monkeypatch, None)
    assert v1_bridge.v1_is_running(59) is False
    assert v1_bridge.occupancy_snapshot(59) is None

    install(monkeypatch, FakeWorker([]))
    assert v1_bridge.v1_is_running(59) is True
    assert v1_bridge.occupancy_snapshot(59) is not None


def test_malformed_v1_tracks_are_skipped_not_crashed_on(monkeypatch):
    """This runs inside a request handler; a bad box must not 500 the page."""
    class Broken:
        track_id = 3
        box = None

    class Degenerate(FakeV1Track):
        pass

    install(monkeypatch, FakeWorker([
        Broken(),
        Degenerate(4, (50, 50, 50, 400)),      # zero width
        FakeV1Track(5, (10, 10, 90, 400)),     # the only good one
    ]))
    tracks = v1_bridge.read_v1_tracks(59)
    assert [t.track_id for t in tracks] == [5]


def test_a_failure_inside_v1_does_not_propagate(monkeypatch):
    class Exploding:
        _frame_lock = FakeLock()

        @property
        def _latest_tracks(self):
            raise RuntimeError("V1 is having a bad time")

    install(monkeypatch, Exploding())
    assert v1_bridge.read_v1_tracks(59) == []


# ---------------------------------------------------------------------------
# No extra inference -- the whole reason this exists
# ---------------------------------------------------------------------------
def test_the_bridge_runs_no_detector():
    """If this ever imports a detector, the cheap option has become the
    expensive one it was chosen over.

    Checks the IMPORT GRAPH, not the text. The module's docstring names YOLO
    repeatedly while explaining why it does not run one, so a substring search
    would fail on the very comment that documents the guarantee.
    """
    import ast

    tree = ast.parse(open(v1_bridge.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{a.name}" for a in node.names)

    for forbidden in ("ultralytics", "app.cctv_v2.pipeline.detect",
                      "app.cctv_v2.capture.grabber",
                      "app.cctv_v2.scheduler.loop"):
        assert not any(m.startswith(forbidden) for m in imported), (
            f"{forbidden} was imported into the bridge"
        )

    # And no detector is constructed anywhere in it.
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "PersonDetector" not in called
    assert "CameraGrabber" not in called


def test_the_bridge_never_writes_to_v1(monkeypatch):
    """Read-only. V1 is production and stays untouched."""
    tracks = [FakeV1Track(1, (10, 10, 90, 400))]
    worker = FakeWorker(tracks)
    install(monkeypatch, worker)

    before = [(t.track_id, t.box) for t in tracks]
    v1_bridge.occupancy_snapshot(59)
    assert [(t.track_id, t.box) for t in tracks] == before
    assert worker._latest_tracks is tracks


# ---------------------------------------------------------------------------
# The snapshot the UI consumes
# ---------------------------------------------------------------------------
def test_the_snapshot_carries_what_an_overlay_needs(monkeypatch):
    install(monkeypatch, FakeWorker([FakeV1Track(1, (400, 300, 520, 900))]))
    data = v1_bridge.occupancy_snapshot(59)

    for key in ("camera_id", "people_count", "chairs_total", "chairs_occupied",
                "chairs_free", "chairs", "unassigned_people",
                "frame_width", "frame_height"):
        assert key in data, f"missing {key}"
    assert data["camera_id"] == 59
    assert data["source"] == "v1_tracks"


def test_every_chair_reports_whether_its_occupant_is_visible_now(monkeypatch):
    """Occupancy is smoothed, so a chair stays OCCUPIED while its occupant is
    briefly undetected. The UI has to be able to tell those apart rather than
    infer it."""
    install(monkeypatch, FakeWorker([FakeV1Track(1, (400, 300, 520, 900))]))
    data = v1_bridge.occupancy_snapshot(59)

    assert data["chairs"], "camera 59 should have a configured chair map"
    for chair in data["chairs"]:
        assert "track_visible" in chair
        assert isinstance(chair["track_visible"], bool)


def test_unassigned_people_are_reported(monkeypatch):
    """Camera 59 has ~10 physical chairs and 4 mapped, so most people sit in
    seats the map does not know. They must be visible as unassigned rather than
    forced into the nearest mapped chair."""
    install(monkeypatch, FakeWorker([
        FakeV1Track(1, (10, 10, 90, 400)),
        FakeV1Track(2, (110, 10, 190, 400)),
    ]))
    data = v1_bridge.occupancy_snapshot(59)

    assert data["people_count"] == 2
    assert data["unassigned_people"] == 2 - data["chairs_occupied"]


def test_occupancy_state_persists_between_calls(monkeypatch):
    """The registry is module-level on purpose: smoothing needs consecutive
    observations, so rebuilding it per request would make every chair flicker."""
    worker = FakeWorker([FakeV1Track(1, (400, 300, 520, 900))])
    install(monkeypatch, worker)

    first = v1_bridge.occupancy_snapshot(59)
    second = v1_bridge.occupancy_snapshot(59)
    assert v1_bridge._registry.get(59).observations >= 2
    assert first["chairs_total"] == second["chairs_total"]


def test_coordinates_are_normalised_so_the_overlay_can_be_scaled(monkeypatch):
    """Pixel coordinates would be wrong the moment the <img> is resized."""
    from app.cctv_v2.config.geometry import room_geometry

    install(monkeypatch, FakeWorker([]))
    data = v1_bridge.occupancy_snapshot(59)
    assert data["frame_width"] > 0 and data["frame_height"] > 0
    for zone in room_geometry(59).chairs:
        assert all(0.0 <= v <= 1.0 for v in zone.box)
