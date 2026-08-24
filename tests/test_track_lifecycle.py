"""The published person count must describe the room NOW, not minutes ago.

Measured on the running system before this test existed: 55% of analysis passes
published MORE tracks than the detector had found, and the Exit camera reported

    detections=0  ->  tracks=5

on an empty corridor. That is the count the UI shows and the number an operator
would act on.

Two mechanisms combine to cause it, and both are exercised here:

  * Held tracks. A track that stops being detected is still published until it
    times out, so an operator can see boxes for people who have walked away.
    The window is `max(_INTERIOR_HOLD_MIN_SEC, _INTERIOR_HOLD_CYCLES * cycle)`
    -- it scales WITH the analysis cycle, so the slower the machine gets the
    longer stale people linger. Measured cycles of 5-10s on this deployment
    turned a 6s floor into a 30-60s hold.

  * Provisional adoption. A confident detection ByteTrack has not yet confirmed
    is adopted as a track so a person crossing a doorway in one pass is not
    lost. But a walker is somewhere else on the next pass, fails the IoU check
    against their own held track, and gets adopted AGAIN under a new id.

These tests use a fake detector so the lifecycle is exercised deterministically,
without YOLO timing or camera availability entering into it.
"""
import time

import pytest

bytetrack_engine = pytest.importorskip(
    "app.services.bytetrack_engine", reason="needs the vision stack (cv2)"
)


class _Arr:
    """Minimal stand-in for the torch tensors ultralytics returns."""

    def __init__(self, data):
        self._d = data

    def cpu(self):
        return self

    def numpy(self):
        return self._d

    def tolist(self):
        return list(self._d)

    def int(self):
        return self

    def __len__(self):
        return len(self._d)


class _Boxes:
    def __init__(self, boxes, confs, ids):
        self.xyxy = _Arr(boxes)
        self.conf = _Arr(confs)
        self.id = _Arr(ids) if ids is not None else None

    def __len__(self):
        return len(self.xyxy)


class _Result:
    def __init__(self, boxes, confs, ids):
        self.boxes = _Boxes(boxes, confs, ids)


class _FakeModel:
    """Returns whatever the test scripts for this pass."""

    def __init__(self):
        self.script = []

    def track(self, frame, **kw):
        boxes, confs, ids = self.script.pop(0) if self.script else ([], [], None)
        return [_Result(boxes, confs, ids)]

    def predict(self, *a, **k):
        return self.track(None)


@pytest.fixture
def engine(monkeypatch):
    """A ByteTrackEngine wired to a scriptable fake detector."""
    def _make(**kw):
        eng = bytetrack_engine.ByteTrackEngine(
            conf=0.03, max_misses=5, camera_id=kw.pop("camera_id", "58"), **kw
        )
        fake = _FakeModel()
        monkeypatch.setattr(eng, "_get_model", lambda: fake)
        eng._fake = fake
        return eng
    monkeypatch.setattr(bytetrack_engine, "_CROP_ASSIST_CAMERAS", set())
    return _make


PERSON = ([[100.0, 100.0, 200.0, 400.0]], [0.85], None)   # confident, unconfirmed
EMPTY = ([], [], None)


def _frame():
    import numpy as np
    return np.zeros((1080, 960, 3), dtype=np.uint8)


def test_confident_detection_is_counted_immediately(engine):
    """A doorway transit is seen on ONE pass; it must not be discarded."""
    eng = engine()
    eng._fake.script = [PERSON]
    assert len(eng.update(_frame())) == 1


def test_empty_scene_reports_zero(engine):
    eng = engine()
    eng._fake.script = [EMPTY, EMPTY]
    eng.update(_frame())
    assert len(eng.update(_frame())) == 0


def test_count_returns_to_zero_after_the_person_leaves(engine):
    """The acceptance criterion: person leaves -> count becomes 0.

    Not "eventually, after a minute" -- an operator reading the count while a
    corridor is visibly empty is being told something false.
    """
    eng = engine()
    eng._fake.script = [PERSON] + [EMPTY] * 12
    eng.update(_frame())                       # person present

    counts = [len(eng.update(_frame())) for _ in range(12)]
    assert counts[-1] == 0, f"count never returned to zero: {counts}"

    passes_to_clear = next(i for i, c in enumerate(counts, 1) if c == 0)
    # Generous, but bounded: a stale box may bridge a miss or two, never a minute.
    assert passes_to_clear <= 4, (
        f"took {passes_to_clear} passes to clear a departed person: {counts}"
    )


def test_a_walker_does_not_accumulate_ids(engine):
    """One person crossing must stay one track, not one per pass.

    A walker is in a different place each pass, so they cannot be matched to
    their own held track by IoU. Without a bound this mints a new provisional
    id every pass -- the mechanism behind detections=0, tracks=5.
    """
    eng = engine()
    # Same person, walking left to right across five passes.
    eng._fake.script = [
        ([[100.0 + i * 120, 100.0, 200.0 + i * 120, 400.0]], [0.85], None)
        for i in range(5)
    ]
    counts = [len(eng.update(_frame())) for _ in range(5)]
    assert max(counts) <= 2, f"one walker produced {max(counts)} concurrent tracks: {counts}"


def test_two_people_report_two(engine):
    eng = engine()
    eng._fake.script = [(
        [[100.0, 100.0, 200.0, 400.0], [500.0, 100.0, 600.0, 400.0]],
        [0.85, 0.80], None,
    )]
    assert len(eng.update(_frame())) == 2


def test_published_count_never_exceeds_what_was_detected_on_an_empty_scene(engine):
    """detections=0 must never publish tracks once the hold has elapsed."""
    eng = engine()
    eng._fake.script = [PERSON, PERSON] + [EMPTY] * 10
    eng.update(_frame()); eng.update(_frame())
    final = [len(eng.update(_frame())) for _ in range(10)][-1]
    assert final == 0
