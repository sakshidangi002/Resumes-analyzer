"""Four cameras, two profiles - and room cameras can never write attendance.

Two things are pinned here, for different reasons.

ATTENDANCE SAFETY. `marks_attendance = False` on the room profile is a
configuration value, and a configuration value is a single point of failure for
something that writes payroll data. `attendance_allowed()` is the independent
check the attendance service makes for itself, so a caller that resolved the
wrong profile - or was handed one by a bug - still cannot write attendance from
a monitoring camera. An unknown camera id is refused for the same reason: the
failure mode of a typo must not be "a camera nobody configured started writing
attendance".

PARAMETER DRIFT. Every CV number in the profiles was measured against real
frames from these cameras, and several are counter-intuitive (640 for doorways
but 480 for rooms; 0.03 confidence in the rooms). The V2 rebuild is an
architecture and scheduling change: if a threshold moves at the same time, a
V1/V2 comparison can no longer attribute a difference to either. The exact-value
tests below exist so that a change is a deliberate act with a failing test
attached, rather than a silent edit nobody notices until the comparison is
meaningless.
"""
import pytest

from app.cctv_v2.config import cameras, profiles


# ---------------------------------------------------------------------------
# Role mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("camera_id,expected", [(57, "doorway"), (58, "doorway"),
                                                (59, "room"), (60, "room")])
def test_each_camera_maps_to_its_role(camera_id, expected):
    assert cameras.role_for(camera_id) == expected


def test_there_are_two_profiles_not_four_configurations():
    """Four cameras share two Profile objects - by identity, not by equality."""
    assert profiles.PROFILES.keys() == {"doorway", "room"}
    assert cameras.profile_for(57) is cameras.profile_for(58)
    assert cameras.profile_for(59) is cameras.profile_for(60)
    assert cameras.profile_for(57) is not cameras.profile_for(59)


def test_an_unknown_camera_raises_rather_than_defaulting():
    """Defaulting to `doorway` would give attendance rights to a typo."""
    with pytest.raises(ValueError):
        cameras.role_for(99)


# ---------------------------------------------------------------------------
# Attendance safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("camera_id", [57, 58])
def test_doorway_cameras_may_write_attendance(camera_id):
    assert cameras.attendance_allowed(camera_id) is True


@pytest.mark.parametrize("camera_id", [59, 60])
def test_room_cameras_can_never_write_attendance(camera_id):
    """The rule that protects payroll. Independent of any profile the caller holds."""
    assert cameras.attendance_allowed(camera_id) is False
    assert cameras.profile_for(camera_id).marks_attendance is False


def test_an_unknown_camera_cannot_write_attendance():
    assert cameras.attendance_allowed(99) is False
    assert cameras.attendance_allowed(None) is False
    assert cameras.attendance_allowed("not-a-camera") is False


def test_only_doorway_cameras_are_listed_as_attendance_capable():
    assert cameras.cameras_with_role("doorway") == (57, 58)
    assert cameras.cameras_with_role("room") == (59, 60)


# ---------------------------------------------------------------------------
# Same-room grouping (cross-camera de-duplication comes later)
# ---------------------------------------------------------------------------
def test_the_two_room_cameras_know_they_share_a_room():
    """59 and 60 watch one room. Anything counting PEOPLE rather than TRACKS
    must know, or one person at one desk becomes two."""
    assert cameras.same_room_peers(59) == {60}
    assert cameras.same_room_peers(60) == {59}


def test_doorway_cameras_have_no_room_peers():
    assert cameras.same_room_peers(57) == frozenset()
    assert cameras.same_room_peers(58) == frozenset()


# ---------------------------------------------------------------------------
# Measured parameters must not drift during the rebuild
# ---------------------------------------------------------------------------
def test_doorway_profile_matches_the_measured_values():
    p = profiles.DOORWAY
    assert (p.input_size, p.predict_conf, p.new_track_thresh) == (640, 0.15, 0.20)
    assert (p.match_threshold, p.match_margin) == (0.45, 0.18)
    assert (p.min_face_px, p.max_yaw) == (28.0, 40.0)
    assert (p.observations_required, p.quality_required, p.consensus_required) == (3, 0.28, 0.55)
    assert p.marks_attendance is True
    assert p.line_crossing is True and p.line_position == 0.35
    assert p.show_today_total is True


def test_room_profile_matches_the_measured_values():
    p = profiles.ROOM
    assert (p.input_size, p.predict_conf, p.new_track_thresh) == (480, 0.03, 0.03)
    assert (p.match_threshold, p.match_margin) == (0.42, 0.10)
    assert (p.min_face_px, p.max_yaw) == (16.0, 75.0)
    assert (p.observations_required, p.quality_required, p.consensus_required) == (2, 0.15, 0.40)
    assert p.marks_attendance is False
    assert p.line_crossing is False and p.line_position is None
    assert p.show_today_total is False


def test_doorways_outrank_rooms_for_scheduling():
    """A person crosses a doorway in ~2s; a seated person is still there next pass."""
    assert profiles.DOORWAY.priority > profiles.ROOM.priority


def test_profiles_are_immutable():
    """Read from several threads; a value that can change mid-pass produces
    decisions that disagree with the log line explaining them."""
    with pytest.raises(Exception):
        profiles.DOORWAY.input_size = 999


# ---------------------------------------------------------------------------
# Rollback switch
# ---------------------------------------------------------------------------
def test_the_pipeline_flag_defaults_to_v1():
    """V2 must not become live by accident. This stays v1 until the V1/V2
    comparison says otherwise."""
    from app.core.config import get_settings

    assert get_settings().cctv_pipeline == "v1"
