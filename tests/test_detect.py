"""Detection runs at the settings its camera's ROLE demands, on fresh frames only.

Two rules are pinned here, and both have already failed in production once.

THE ROLE SETTINGS ARE NOT INTERCHANGEABLE. A doorway runs at 640px and a room at
960px, and each was measured against real frames from those cameras. At 480 a
walking person on the Exit camera scored 0.148 -- under the 0.20 track threshold,
so they vanished and the overlay read "People: 0" with somebody walking through
shot. The rooms were ALSO on 480 until 2026-08-26, on an earlier measurement that
scored configurations per frame against no fixed ground truth; re-measured
against 16 labelled frames and 41 hand-boxed people, one of the four people on
camera 59 turned out to produce no detection at 480 at ANY confidence. The two
roles still want different sizes -- a doorway subject is close and walking, a
room subject is distant, seated and half hidden -- so a single shared imgsz would
still break one of them.

STALE FRAMES MUST NEVER REACH YOLO. Killing camera 59 mid-run showed the
scheduler spending inference passes on a picture that aged to 71 seconds. The
guard lives in the scheduler, but the guarantee belongs to the detector's
callers, so it is tested end-to-end here: scheduler plus detector, dead camera,
zero inference.

No ultralytics import anywhere. A fake model records the kwargs it was called
with, which is exactly what these tests are about -- and it keeps the suite
runnable on a machine with no weights file.
"""
import time

import pytest

from app.cctv_v2.capture.grabber import CameraGrabber, FrameSnapshot
from app.cctv_v2.pipeline.detect import (
    PERSON_CLASS_ID,
    DetectionResult,
    PersonDetection,
    PersonDetector,
)
from app.cctv_v2.scheduler.loop import InferenceScheduler


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeBoxes:
    def __init__(self, rows):
        # rows: (x1, y1, x2, y2, conf, cls)
        self._rows = rows
        self.xyxy = _Arr([list(r[:4]) for r in rows])
        self.conf = _Arr([r[4] for r in rows])
        self.cls = _Arr([r[5] for r in rows])

    def __len__(self):
        return len(self._rows)


class _Arr:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class FakeResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeModel:
    """Records every predict() call and returns whatever it was told to."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append({"frame": frame, **kwargs})
        return [FakeResult(self.rows)]


def detector(model=None):
    model = model if model is not None else FakeModel()
    return PersonDetector(model_loader=lambda: model), model


def snap(camera_id=57, age=0.0, sequence=1):
    return FrameSnapshot(
        camera_id=camera_id, frame="pixels",
        timestamp=time.time() - age, sequence=sequence,
    )


# ---------------------------------------------------------------------------
# Role-specific model settings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("camera_id", [57, 58])
def test_a_doorway_runs_at_640_and_conf_015(camera_id):
    det, model = detector()
    det.detect(camera_id, snap(camera_id))

    call = model.calls[0]
    assert call["imgsz"] == 640
    assert call["conf"] == pytest.approx(0.15)


@pytest.mark.parametrize("camera_id", [59, 60])
def test_a_room_runs_at_960_and_conf_0015(camera_id):
    det, model = detector()
    det.detect(camera_id, snap(camera_id))

    call = model.calls[0]
    assert call["imgsz"] == 960
    assert call["conf"] == pytest.approx(0.015)


def test_the_two_roles_do_not_share_settings():
    """The whole point of the profile split. If these ever converge, one role is
    running at numbers that were measured for the other."""
    det, model = detector()
    det.detect(57, snap(57))
    det.detect(59, snap(59))

    doorway, room = model.calls[0], model.calls[1]
    assert doorway["imgsz"] != room["imgsz"]
    assert doorway["conf"] != room["conf"]


def test_settings_come_from_the_role_not_the_camera_id():
    """Four cameras, two configurations. Adding a fifth camera must not mean a
    fifth copy of these numbers."""
    det, model = detector()
    for cid in (57, 58, 59, 60):
        det.detect(cid, snap(cid))

    by_imgsz = {c["imgsz"] for c in model.calls}
    assert by_imgsz == {640, 960}, "there should be exactly two configurations"
    assert model.calls[0]["imgsz"] == model.calls[1]["imgsz"]     # both doorways
    assert model.calls[2]["imgsz"] == model.calls[3]["imgsz"]     # both rooms


def test_the_settings_used_are_recorded_on_every_result():
    """So a surprising detection count can be traced to what it was run at,
    rather than to what the config says today."""
    det, _ = detector()
    result = det.detect(59, snap(59))
    assert result.imgsz == 960
    assert result.conf == pytest.approx(0.015)
    assert result.role == "room"


def test_an_unknown_camera_is_refused_rather_than_guessed():
    """Defaulting would mean a camera nobody configured gets doorway settings --
    and doorway is the role that writes attendance."""
    det, _ = detector()
    with pytest.raises(ValueError):
        det.detect(999, snap(999))


# ---------------------------------------------------------------------------
# Person class only
# ---------------------------------------------------------------------------
def test_only_the_person_class_is_requested():
    det, model = detector()
    det.detect(57, snap(57))
    assert model.calls[0]["classes"] == [PERSON_CLASS_ID]


def test_non_person_boxes_are_dropped_even_if_the_model_returns_them():
    """The `classes=[0]` argument is a request; this filter is the guarantee.
    They fail differently, so both exist."""
    model = FakeModel(rows=[
        (10, 10, 50, 200, 0.90, 0),      # person
        (60, 60, 90, 90, 0.80, 56),      # chair
        (99, 99, 120, 130, 0.70, 0),     # person
    ])
    det, _ = detector(model)
    result = det.detect(57, snap(57))

    assert result.count == 2
    assert all(d.class_id == PERSON_CLASS_ID for d in result.detections)


def test_degenerate_boxes_are_dropped():
    """A zero-area box is not a person and would divide by zero downstream."""
    model = FakeModel(rows=[
        (10, 10, 10, 200, 0.9, 0),       # zero width
        (10, 10, 50, 10, 0.9, 0),        # zero height
        (10, 10, 50, 200, 0.9, 0),       # real
    ])
    det, _ = detector(model)
    assert det.detect(57, snap(57)).count == 1


def test_a_detection_carries_the_frame_it_came_from():
    """camera_id, frame_timestamp, bbox, confidence, class_id -- the minimum a
    later step needs to reason about a box without re-deriving its context."""
    model = FakeModel(rows=[(10.0, 20.0, 50.0, 200.0, 0.77, 0)])
    det, _ = detector(model)
    s = snap(58, sequence=42)
    result = det.detect(58, s)

    d = result.detections[0]
    assert d.camera_id == 58
    assert d.frame_timestamp == s.timestamp
    assert d.frame_sequence == 42
    assert d.bbox == (10.0, 20.0, 50.0, 200.0)
    assert d.confidence == pytest.approx(0.77)
    assert d.class_id == PERSON_CLASS_ID
    assert d.width == 40.0 and d.height == 180.0
    assert d.centroid == (30.0, 110.0)


# ---------------------------------------------------------------------------
# Stale frames never reach YOLO
# ---------------------------------------------------------------------------
class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _wired(clock, cameras=(57, 58, 59, 60)):
    """A scheduler whose process callback is the real detector."""
    grabbers = {cid: CameraGrabber(cid, f"test://{cid}") for cid in cameras}
    model = FakeModel(rows=[(10, 10, 50, 200, 0.9, 0)])
    det = PersonDetector(model_loader=lambda: model, clock=clock)
    sched = InferenceScheduler(
        grabbers.values(),
        process=lambda cid, s: det.detect(cid, s),
        clock=clock,
    )
    return sched, grabbers, det, model


def test_a_stale_doorway_frame_never_reaches_yolo():
    clock = Clock()
    sched, grabbers, _, model = _wired(clock, cameras=(57,))
    grabbers[57].publish("old", timestamp=clock.t - 1.01)     # over the 1.0s cutoff

    assert sched.run_once() is None
    assert model.calls == [], "YOLO was run on a stale frame"


def test_a_stale_room_frame_never_reaches_yolo():
    clock = Clock()
    sched, grabbers, _, model = _wired(clock, cameras=(59,))
    grabbers[59].publish("old", timestamp=clock.t - 5.01)     # over the 5.0s cutoff

    assert sched.run_once() is None
    assert model.calls == []


def test_a_dead_camera_stops_reaching_yolo_while_the_others_continue():
    """The live failure, wired to a real detector: 59 dies, and the inference it
    would have wasted goes to cameras that have something to show."""
    clock = Clock()
    sched, grabbers, det, model = _wired(clock)
    for cid in grabbers:
        grabbers[cid].publish("f", timestamp=clock.t)

    for _ in range(12):                       # 6s -- past the 5s room cutoff
        for cid in (57, 58, 60):
            grabbers[cid].publish("fresh", timestamp=clock.t)
        sched.run_once()
        clock.advance(0.5)

    settled = sum(1 for c in model.calls if c["frame"] == "f")

    for _ in range(40):                       # 20s more with 59 dark
        for cid in (57, 58, 60):
            grabbers[cid].publish("fresh", timestamp=clock.t)
        sched.run_once()
        clock.advance(0.5)

    still = sum(1 for c in model.calls if c["frame"] == "f")
    assert still == settled, "a dead camera's frame kept reaching YOLO"
    assert det.timing["doorway"].passes > 0
    assert sched.stats[59].stale_skips > 0


def test_when_every_camera_is_stale_yolo_is_never_called():
    clock = Clock()
    sched, grabbers, _, model = _wired(clock)
    for cid in grabbers:
        grabbers[cid].publish("ancient", timestamp=clock.t - 90.0)

    for _ in range(30):
        assert sched.run_once() is None

    assert model.calls == []


def test_every_frame_yolo_sees_is_within_its_role_cutoff():
    """The invariant behind all of the above, asserted directly over a run in
    which cameras drop out and come back."""
    clock = Clock()
    sched, grabbers, det, _ = _wired(clock)
    seen: list[DetectionResult] = []
    sched._process = lambda cid, s: seen.append(det.detect(cid, s))

    for i in range(80):
        for cid in (57, 58, 60):
            grabbers[cid].publish("fresh", timestamp=clock.t)
        if i % 20 < 5:                        # 59 flaps in and out
            grabbers[59].publish("fresh", timestamp=clock.t)
        sched.run_once()
        clock.advance(0.5)

    assert seen, "nothing was detected at all"
    limits = {"doorway": 1.0, "room": 5.0}
    for r in seen:
        assert r.frame_age_ms / 1000.0 <= limits[r.role], (
            f"camera {r.camera_id} ({r.role}) ran on a "
            f"{r.frame_age_ms / 1000.0:.1f}s frame"
        )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
def test_latency_is_recorded_per_role_not_pooled():
    """640px and 960px passes are not comparable; a shared average would
    describe no camera that exists."""
    clock = Clock()
    model = FakeModel()

    def slow_predict(frame, **kw):
        clock.advance(0.100 if kw["imgsz"] == 640 else 0.040)
        model.calls.append({"frame": frame, **kw})
        return [FakeResult([])]

    model.predict = slow_predict
    det = PersonDetector(model_loader=lambda: model, clock=clock)

    det.detect(57, FrameSnapshot(57, "f", clock.t, 1))
    det.detect(59, FrameSnapshot(59, "f", clock.t, 1))

    assert det.timing["doorway"].avg_ms == pytest.approx(100.0, abs=1)
    assert det.timing["room"].avg_ms == pytest.approx(40.0, abs=1)


def test_detections_per_frame_is_tracked():
    model = FakeModel(rows=[(0, 0, 10, 10, 0.9, 0), (20, 20, 30, 40, 0.8, 0)])
    det, _ = detector(model)
    det.detect(57, snap(57))
    det.detect(57, snap(57))

    t = det.timing["doorway"]
    assert t.passes == 2
    assert t.total_detections == 4
    assert t.detections_per_frame == pytest.approx(2.0)
    assert t.frames_with_people == 2


def test_empty_frames_count_as_passes_but_not_as_people():
    det, _ = detector(FakeModel(rows=[]))
    det.detect(59, snap(59))

    t = det.timing["room"]
    assert t.passes == 1
    assert t.frames_with_people == 0
    assert t.detections_per_frame == 0.0


def test_the_latency_window_is_bounded():
    """This runs for days. An unbounded list of every latency is a slow leak."""
    det, _ = detector()
    for i in range(500):
        det.detect(57, snap(57, sequence=i))

    assert det.timing["doorway"].passes == 500
    assert len(det.timing["doorway"].recent_ms) <= 200


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
def test_a_failed_inference_returns_empty_rather_than_raising():
    """One bad frame must not stop the scheduler serving the other three."""
    class Exploding:
        def predict(self, frame, **kw):
            raise RuntimeError("CUDA is on fire")

    det = PersonDetector(model_loader=Exploding)
    result = det.detect(57, snap(57))

    assert result.count == 0
    assert det.timing["doorway"].failures == 1


def test_a_missing_model_disables_detection_without_retrying_every_frame():
    """A retry per frame turns one missing file into a traceback at 12fps, which
    is how V1 hid a broken model behind a wall of identical errors."""
    attempts = []

    def failing_loader():
        attempts.append(1)
        raise FileNotFoundError("no weights")

    det = PersonDetector(model_loader=failing_loader)
    for _ in range(10):
        assert det.detect(57, snap(57)).count == 0

    assert len(attempts) == 1, f"model load was retried {len(attempts)} times"
    assert det.available is False
