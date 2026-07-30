"""One person must be published as ONE box.

A permissive new_track_thresh is required so seated people are detected at all,
but it makes ByteTrack spawn a fresh id instead of re-associating when a
detection is weak. Combined with publishing held tracks, one person ends up
wearing several overlapping boxes. Observed live: "#36" stacked with "2" and
"3", and a People count of 6 in a room containing 4.

It is not cosmetic. Each duplicate accumulates its OWN embedding fusion, so the
face observations that should combine into one confident identity are split
across fragments that each stay too weak to match — duplicates actively prevent
recognition.
"""
import pytest

bytetrack_engine = pytest.importorskip(
    "app.services.bytetrack_engine", reason="needs the vision stack"
)
from app.services.person_tracker import PersonTrack

_dedupe = bytetrack_engine.ByteTrackEngine._dedupe_overlapping
_overlap = bytetrack_engine._overlap_ratio


class _Engine:
    """Minimal stand-in — _dedupe_overlapping only touches .tracks and .camera_id."""

    def __init__(self, tracks):
        self.tracks = {t.track_id: t for t in tracks}
        self.camera_id = "test"


def _track(tid, box, age=1, employee=None, observations=0):
    t = PersonTrack(track_id=tid, box=box)
    t.age = age
    if employee:
        t.employee_id, t.employee_name, t.matched = employee, f"emp{employee}", True
        t.identity_source = "face"
    for _ in range(observations):
        t.fuser.observations += 1
    return t


def test_nested_box_counts_as_full_overlap():
    """A tight torso box inside a bloated body box has low IoU but is obviously
    the same person — which is why this is not IoU."""
    assert _overlap(( 0, 0, 100, 400), (10, 10, 60, 120)) == pytest.approx(1.0)


def test_separate_people_are_not_merged():
    assert _overlap((0, 0, 100, 300), (400, 0, 500, 300)) == 0.0


def test_stacked_boxes_collapse_to_one_person():
    """The live failure: three ids on the yellow-shirt person."""
    tracks = [
        _track(36, (100, 400, 200, 700), age=20),
        _track(2, (105, 410, 195, 690), age=3),
        _track(3, (110, 420, 190, 680), age=2),
    ]
    kept = _dedupe(_Engine(tracks), tracks, seen={36})
    assert len(kept) == 1
    assert kept[0].track_id == 36, "the established, currently-detected track should survive"


def test_detected_track_beats_a_merely_held_one():
    held = _track(9, (100, 400, 200, 700), age=99)
    detected = _track(10, (105, 405, 205, 705), age=2)
    kept = _dedupe(_Engine([held, detected]), [held, detected], seen={10})
    assert kept[0].track_id == 10


def test_identity_survives_a_track_id_change():
    """The valuable part.

    ByteTrack renames people constantly on this view. Without merging, every id
    change discarded the name that had been established by a real face match.
    """
    old_named = _track(5, (100, 400, 200, 700), age=50, employee=7)
    new_unnamed = _track(61, (105, 405, 205, 705), age=1)

    kept = _dedupe(_Engine([old_named, new_unnamed]), [old_named, new_unnamed], seen={61})

    assert len(kept) == 1
    assert kept[0].track_id == 61, "the freshly-detected track should be the survivor"
    assert kept[0].employee_id == 7, "the established identity was thrown away"
    assert kept[0].identity_source == "face"


def test_richer_fusion_is_carried_over():
    """Observations must not be lost when a duplicate is dropped — that is what
    keeps recognition from being permanently reset."""
    rich = _track(5, (100, 400, 200, 700), age=50, observations=14)
    fresh = _track(61, (105, 405, 205, 705), age=1, observations=1)

    kept = _dedupe(_Engine([rich, fresh]), [rich, fresh], seen={61})

    assert kept[0].fuser.observations == 14


def test_dropped_duplicates_are_removed_from_engine_state():
    """Otherwise they linger and re-appear on the next frame."""
    a = _track(1, (100, 400, 200, 700), age=10)
    b = _track(2, (105, 405, 205, 705), age=2)
    engine = _Engine([a, b])
    _dedupe(engine, [a, b], seen={1})
    assert 2 not in engine.tracks


def test_single_and_empty_inputs_are_untouched():
    solo = [_track(1, (0, 0, 10, 10))]
    assert _dedupe(_Engine(solo), solo, seen=set()) == solo
    assert _dedupe(_Engine([]), [], seen=set()) == []


def test_a_crowd_of_distinct_people_is_preserved():
    """Dedup must not collapse a genuinely busy room."""
    tracks = [_track(i, (i * 200, 0, i * 200 + 100, 300)) for i in range(1, 5)]
    kept = _dedupe(_Engine(tracks), tracks, seen={1, 2, 3, 4})
    assert len(kept) == 4
