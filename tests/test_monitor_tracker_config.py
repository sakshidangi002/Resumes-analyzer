"""MONITOR cameras must get the permissive person tracker by default.

A ceiling view of seated people yields detection scores far below the strict
config's new_track_thresh of 0.30 — measured on the dev room: 0.58 / 0.25 /
0.19 / 0.16, so one of four people became a track and the overlay read
"People: 1".

The permissive config existed but had to be enabled per-camera via
CCTV_STEEP_CAMERAS, so any monitor camera the operator forgot to list silently
saw a fraction of the room. A MONITOR camera can never mark attendance, so a
spurious box costs nothing while a missed person defeats its only purpose —
recall must win there. IN/OUT cameras keep the strict config, because there a
false track feeds payroll.
"""
import inspect

import pytest

bytetrack_engine = pytest.importorskip(
    "app.services.bytetrack_engine", reason="needs the vision stack (cv2)"
)
camera_service = pytest.importorskip(
    "app.services.camera_service", reason="needs the vision stack (cv2)"
)


def _worker_source() -> str:
    return inspect.getsource(camera_service.CameraWorker.__init__)


def test_monitor_cameras_get_the_permissive_tracker_without_env_config():
    src = _worker_source()
    assert "steep = self.is_monitor or" in src, (
        "MONITOR cameras no longer default to the permissive tracker — a room "
        "camera not listed in CCTV_STEEP_CAMERAS will silently miss most people"
    )


def test_explicit_steep_camera_list_still_honoured():
    """IN/OUT cameras on a bad angle can still opt in by id."""
    assert "_STEEP_CAMERAS" in _worker_source()


def test_strict_config_is_more_restrictive_than_permissive():
    """Guards the two YAMLs against being edited into agreement."""
    import re
    from pathlib import Path

    root = Path(camera_service.__file__).resolve().parents[2]

    def new_track_thresh(name: str) -> float:
        text = (root / "models" / name).read_text(encoding="utf-8")
        m = re.search(r"^new_track_thresh:\s*([0-9.]+)", text, re.MULTILINE)
        assert m, f"new_track_thresh missing from {name}"
        return float(m.group(1))

    strict = new_track_thresh("bytetrack_person.yaml")
    permissive = new_track_thresh("bytetrack_person_lowconf.yaml")
    assert permissive < strict, (
        f"the permissive tracker ({permissive}) is not more permissive than "
        f"the strict one ({strict}) — monitor cameras gain nothing"
    )


# ---------------------------------------------------------------------------
# The two floors are not the same floor
# ---------------------------------------------------------------------------
# The DETECTOR floor decides what ByteTrack is allowed to SEE. new_track_thresh
# decides what may CREATE a person. Setting them equal -- which is how the room
# cameras shipped -- defeats the two-stage association the permissive config
# exists for: stage 2 re-associates weak boxes to tracks that already exist, and
# with both bars at the same height those boxes were discarded before stage 2
# ever saw them. A seated person who dipped for one pass could then only be
# re-CREATED, at the higher bar.
def _lowconf_cfg() -> dict:
    import pathlib
    import re

    path = (pathlib.Path(__file__).resolve().parents[1] / "Attendance Management"
            / "backend" / "models" / "bytetrack_person_lowconf.yaml")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([a-z_]+):\s*([0-9.]+)\s*$", line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def test_the_room_tracker_admits_weaker_boxes_than_it_will_create_from():
    cfg = _lowconf_cfg()
    assert cfg["track_low_thresh"] < cfg["new_track_thresh"], (
        "stage 2 must accept boxes too weak to create a track, or it has "
        "nothing left to re-associate"
    )


def test_the_room_creation_floor_is_low_enough_for_a_seated_person():
    """Measured on these cameras: seated staff seen from behind score
    0.035-0.057, and on the labelled benchmark the person at camera 59's far
    desk is seen twice as often at 0.02 as at 0.03."""
    cfg = _lowconf_cfg()
    assert cfg["new_track_thresh"] <= 0.02


def test_a_size_floor_exists_to_pay_for_the_low_confidence_floor():
    """At these confidences the detector emits few-pixel fragments on frame
    edges and decode-corruption bands, and SOME OUTSCORE a real seated person --
    so confidence cannot separate them and size must. Smallest labelled real
    person: 46px tall."""
    assert 0 < bytetrack_engine._MIN_PERSON_PX <= 46


def test_the_size_floor_is_applied_to_the_crop_pass_too():
    """The crop runs at HALF the already-low confidence, so it is the likelier
    source of fragments, not the less likely one."""
    import inspect

    src = inspect.getsource(bytetrack_engine.ByteTrackEngine._add_crop_assist_tracks)
    assert "_MIN_PERSON_PX" in src


# ---------------------------------------------------------------------------
# The adoption bar is per-role too
# ---------------------------------------------------------------------------
# ByteTrack issues an id only after matching a detection across TWO passes.
# Between those passes a room camera's person may simply not be detected, so the
# engine adopts a not-yet-tracked detection directly -- above a confidence bar.
#
# That bar was global, and set from DOORWAY evidence: empty corridors score
# 0.00-0.01 and walking people 0.43-0.78, so 0.35 separates them cleanly. Seated
# staff on a room camera score 0.02-0.20 and fail it universally, so on a room
# the rescue path never fired for anybody it existed to rescue.
#
# Measured over 10 labelled camera-59 frames and 6 camera-60 frames, running the
# real V1 engine:
#
#                                     camera 59      camera 60
#     480/0.03, adopt 0.35            11/35  31%      0/6    0%
#     960/0.015, adopt 0.02           14/35  40%      5/6   83%
def test_a_room_adopts_at_a_far_lower_bar_than_a_doorway():
    assert (bytetrack_engine._STEEP_ADOPT_UNTRACKED_MIN_CONF
            < bytetrack_engine._ADOPT_UNTRACKED_MIN_CONF), (
        "a room camera's people score 0.02-0.20; a doorway's score 0.43-0.78. "
        "One bar cannot serve both."
    )


def test_the_room_adoption_bar_reaches_a_seated_person():
    """Seated staff seen from behind measure 0.035-0.057 on these cameras."""
    assert bytetrack_engine._STEEP_ADOPT_UNTRACKED_MIN_CONF <= 0.035


def test_the_doorway_adoption_bar_is_unchanged():
    """The doorways mark attendance and their behaviour was validated; this
    change must not reach them."""
    assert bytetrack_engine._ADOPT_UNTRACKED_MIN_CONF == 0.35


def test_the_engine_takes_its_adoption_bar_per_camera():
    """Not a module global read at use time -- two cameras with different roles
    run in the same process."""
    doorway = bytetrack_engine.ByteTrackEngine(camera_id="57")
    room = bytetrack_engine.ByteTrackEngine(
        camera_id="59",
        adopt_min_conf=bytetrack_engine._STEEP_ADOPT_UNTRACKED_MIN_CONF)
    assert doorway.adopt_min_conf == bytetrack_engine._ADOPT_UNTRACKED_MIN_CONF
    assert room.adopt_min_conf == bytetrack_engine._STEEP_ADOPT_UNTRACKED_MIN_CONF


def test_monitor_cameras_are_given_the_room_bar():
    source = _worker_source()
    assert "adopt_min_conf=" in source
    assert "_STEEP_ADOPT_UNTRACKED_MIN_CONF" in source
