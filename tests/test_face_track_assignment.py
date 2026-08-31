"""A detected face belongs to exactly ONE person.

Person boxes overlap constantly on a top-down office view — someone standing
behind a seated colleague yields two boxes over the same pixels. The pipeline
asked each track independently "is there a face inside my box?", so both were
handed the SAME face and both were recognised as the same employee.

Observed live on the dev-room camera: person 1 (conf 0.77) and person 3
(conf 0.18) received the same 30px face and produced identical match scores
against Rakhi Channa. Two different people, one name.
"""
import pytest

camera_service = pytest.importorskip(
    "app.services.camera_service", reason="needs the vision stack (cv2)"
)
_assign = camera_service._assign_faces_to_tracks


class _Track:
    def __init__(self, track_id, box):
        self.track_id, self.box = track_id, box


def _face(box, conf=0.6):
    return {"box": list(box), "confidence": conf}


def test_overlapping_tracks_do_not_share_one_face():
    """The exact live failure."""
    tight = _Track(1, (100, 100, 200, 400))          # the face's real owner
    loose = _Track(3, (50, 50, 600, 900))            # a big box that also covers it
    face = _face((130, 120, 160, 150))

    assigned = _assign([face], [tight, loose])

    assert len(assigned) == 1, "the same face was handed to two people"
    assert 1 in assigned, "the tighter box should win the face"


def test_each_face_goes_to_its_own_track():
    a = _Track(1, (0, 0, 100, 300))
    b = _Track(2, (200, 0, 300, 300))
    fa, fb = _face((20, 20, 50, 50)), _face((220, 20, 250, 50))

    assigned = _assign([fa, fb], [a, b])

    assert assigned[1] is fa
    assert assigned[2] is fb


def test_track_with_no_face_inside_gets_nothing():
    """Someone facing away must stay Unknown, not inherit a neighbour's face."""
    near = _Track(1, (0, 0, 100, 300))
    away = _Track(2, (500, 500, 600, 800))

    assigned = _assign([_face((20, 20, 50, 50))], [near, away])

    assert 1 in assigned
    assert 2 not in assigned


def test_more_faces_than_tracks_is_safe():
    track = _Track(1, (0, 0, 100, 300))
    assigned = _assign([_face((10, 10, 40, 40)), _face((50, 50, 80, 80))], [track])
    assert len(assigned) == 1


def test_empty_inputs():
    assert _assign([], [_Track(1, (0, 0, 10, 10))]) == {}
    assert _assign([_face((1, 1, 2, 2))], []) == {}


def test_malformed_face_box_is_ignored():
    assigned = _assign([{"box": [1, 2], "confidence": 0.9}], [_Track(1, (0, 0, 100, 100))])
    assert assigned == {}
