"""Recognition failing must never remove somebody from the people count.

This is the single most load-bearing rule in the room-camera stack, and the one
the system has already broken once: a dev room with people in it read
"People: 0" on the overlay while every one of them was plainly visible.

The shape of that failure is always the same -- a count that is really measuring
identity gets labelled as if it were measuring presence. So the count is taken
from BODY tracks, `unknown` is defined as the remainder rather than counted, and
the overlay is required to say which of the two it is showing.

    People: 10
    Recognised: 3
    Unknown: 7          <- seven failures of recognition, zero missing people
"""
import pytest

camera_service = pytest.importorskip(
    "app.services.camera_service", reason="needs the vision stack (cv2)"
)

people_split = camera_service.people_split


class Track:
    """Shaped like V1's PersonTrack for the fields this rule touches."""

    def __init__(self, track_id, matched=False, employee_name=None):
        self.track_id = track_id
        self.box = (10, 10, 60, 200)
        self.matched = matched
        self.employee_name = employee_name


# ---------------------------------------------------------------------------
# The count is bodies
# ---------------------------------------------------------------------------
def test_nobody_recognised_still_counts_everybody():
    """The exact scenario behind "People: 0" -- a room seen from behind, where
    not one face is readable."""
    people, recognised, unknown = people_split([Track(i) for i in range(10)])
    assert people == 10
    assert recognised == 0
    assert unknown == 10


def test_a_mix_splits_without_losing_anyone():
    tracks = [Track(1, matched=True, employee_name="A"),
              Track(2, matched=True, employee_name="B"),
              Track(3, matched=True, employee_name="C")]
    tracks += [Track(i) for i in range(4, 11)]
    people, recognised, unknown = people_split(tracks)
    assert (people, recognised, unknown) == (10, 3, 7)


def test_everybody_recognised_leaves_nobody_unknown():
    tracks = [Track(i, matched=True, employee_name=str(i)) for i in range(4)]
    assert people_split(tracks) == (4, 4, 0)


def test_unknown_is_the_remainder_so_the_three_always_agree():
    """`unknown` is never counted independently. If it were, it could drift from
    the other two and reintroduce the ambiguity this split exists to remove."""
    for n_matched in range(6):
        tracks = [Track(i, matched=i < n_matched) for i in range(5)]
        people, recognised, unknown = people_split(tracks)
        assert recognised + unknown == people == 5


def test_recognition_going_away_does_not_change_the_count():
    """Same bodies, identity lost between passes -- as happens the moment
    somebody turns away from the lens."""
    named = [Track(i, matched=True, employee_name=str(i)) for i in range(6)]
    turned_away = [Track(t.track_id) for t in named]

    assert people_split(named)[0] == people_split(turned_away)[0] == 6
    assert people_split(turned_away)[1] == 0


def test_an_empty_room_is_zero_of_everything():
    assert people_split([]) == (0, 0, 0)
    assert people_split(None) == (0, 0, 0)


def test_a_track_with_no_identity_fields_at_all_is_still_a_person():
    """Body tracking can run without the recognition stage having touched the
    track. A missing attribute must read as 'not recognised', never as 'not
    there'."""
    class Bare:
        track_id = 1

    assert people_split([Bare()]) == (1, 0, 1)


# ---------------------------------------------------------------------------
# What the API publishes
# ---------------------------------------------------------------------------
def test_the_status_payload_carries_all_three_numbers():
    """The overlay used to have only `active_tracks` to work with, and labelled
    it "Faces" -- so on a body-tracking camera it reported bodies under a name
    that says the opposite. Three named fields remove the guess."""
    import inspect

    src = inspect.getsource(camera_service.CameraWorker.serialize_state)
    for field in ("people_tracked", "people_recognised", "people_unknown"):
        assert f'"{field}"' in src, f"{field} missing from the status payload"
    assert "people_split(" in src, (
        "serialize_state must use the shared split, not recount identity "
        "inline -- two definitions is how the numbers drift"
    )


def test_the_count_comes_from_body_tracks_not_face_tracks():
    """A MONITOR camera always goes through the person path, where
    `active_tracks` is set from the person-track list."""
    import inspect

    src = inspect.getsource(camera_service._RecognitionThread._analyze_person)
    assert "w.state.active_tracks = len(ptracks)" in src, (
        "the room count must be published from body tracks; if this moved, "
        "check it did not become a count of faces"
    )
