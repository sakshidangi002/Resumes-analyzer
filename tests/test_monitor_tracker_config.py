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
