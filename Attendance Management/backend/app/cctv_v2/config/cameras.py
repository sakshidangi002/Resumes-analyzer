"""Which camera plays which role, and the safety rules that follow from it.

The role map is the ONLY place a camera's behaviour is decided. Everything else
- detection size, thresholds, whether attendance may be written - is derived
from the profile that role names. A new camera is one line here.

WHY THIS MODULE OWNS THE ATTENDANCE RULE
----------------------------------------
`marks_attendance = False` on the room profile is a configuration value, and a
configuration value is a single point of failure for something that writes
payroll data. If a room camera ever reached the attendance writer - through a
refactor, a mistaken caller, an API request naming camera 59 - one flag would be
all that stood between a monitoring camera and somebody's pay.

So the rule is expressed as a function, `attendance_allowed`, that the
attendance service calls independently of any profile lookup the caller may have
done. Combined with `policy/room.py` not importing the attendance writer at all,
a room camera has to get past two separate defences that fail in different ways.

UNKNOWN CAMERAS
---------------
A camera id that is not in the map is not assumed to be anything. Attendance is
refused for it, and callers asking for its role get an explicit error rather
than a default. Defaulting an unknown camera to `doorway` would mean the failure
mode of a typo is "a camera nobody configured started writing attendance".
"""
from __future__ import annotations

import logging

from app.cctv_v2.config.profiles import Profile, Role, get_profile

logger = logging.getLogger(__name__)

# camera id -> role. The single source of truth.
CAMERA_ROLE: dict[int, Role] = {
    57: "doorway",   # Entrance
    58: "doorway",   # Exit
    59: "room",      # Dev-room
    60: "room",      # Dev-room2  (same physical room as 59)
}

# Cameras 59 and 60 watch the SAME room from different angles. Tracking is kept
# independent per camera for now - a track id means nothing across cameras - but
# anything that later counts PEOPLE rather than TRACKS has to know these two
# overlap, or one person at one desk becomes two.
SAME_ROOM_GROUPS: tuple[frozenset[int], ...] = (frozenset({59, 60}),)


def role_for(camera_id: int) -> Role:
    """The role of a camera. Raises for an unknown camera; see module docstring."""
    try:
        return CAMERA_ROLE[int(camera_id)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"camera {camera_id!r} has no configured role; "
            f"known cameras are {sorted(CAMERA_ROLE)}"
        ) from None


def profile_for(camera_id: int) -> Profile:
    """The profile a camera behaves according to."""
    return get_profile(role_for(camera_id))


def attendance_allowed(camera_id: int) -> bool:
    """May this camera create an attendance record?

    The authoritative answer, independent of any profile the caller has already
    resolved. The attendance service calls this rather than trusting a
    `marks_attendance` flag handed to it, so a caller that constructed a profile
    incorrectly - or was handed one by a bug - still cannot write payroll data
    from a monitoring camera.

    Unknown cameras are refused. An unrecognised id is a mistake, and the safe
    response to a mistake near payroll is "no".
    """
    try:
        role = role_for(camera_id)
    except ValueError:
        logger.warning(
            "ATTENDANCE-BLOCKED camera=%r is not a configured camera", camera_id
        )
        return False

    allowed = get_profile(role).marks_attendance
    if not allowed:
        logger.warning(
            "ATTENDANCE-BLOCKED camera=%s role=%s — %s cameras never create "
            "attendance records", camera_id, role, role,
        )
    return allowed


def cameras_with_role(role: Role) -> tuple[int, ...]:
    """All camera ids playing a role, ascending."""
    return tuple(sorted(cid for cid, r in CAMERA_ROLE.items() if r == role))


def same_room_peers(camera_id: int) -> frozenset[int]:
    """Other cameras watching the same physical room as this one.

    Empty when the camera watches somewhere no other camera does. Provided so
    that cross-camera de-duplication can be added later without every caller
    having to rediscover which cameras overlap.
    """
    cid = int(camera_id)
    for group in SAME_ROOM_GROUPS:
        if cid in group:
            return frozenset(group - {cid})
    return frozenset()
