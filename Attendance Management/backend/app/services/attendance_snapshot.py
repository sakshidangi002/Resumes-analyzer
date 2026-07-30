"""Persist the face crop that produced an attendance event.

Roughly 5 KB per event. It is the only artefact that makes a disputed record
reviewable, and it is the labelled data set that lets recognition thresholds be
tuned from measurement instead of anecdote — every threshold in the CCTV
pipeline is currently justified by a single mis-labelled person recalled in a
code comment.

PRIVACY. These are face images: biometric personal data. They are written inside
the existing access-controlled `data/` tree, never into a web-served static
directory, and pruned after RETENTION_DAYS. Check the retention window against
your local obligations before enabling capture.
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Alongside the enrolment photos, under the same access controls.
SNAPSHOT_ROOT = Path(__file__).resolve().parents[2] / "data" / "attendance_snapshots"

# Context around the face box. A bare 112x112 crop is useless to a human
# reviewer — they need enough of the person to recognise them.
_PAD_RATIO = float(os.getenv("ATTENDANCE_SNAPSHOT_PAD", "0.35"))
_JPEG_QUALITY = int(os.getenv("ATTENDANCE_SNAPSHOT_QUALITY", "85"))
RETENTION_DAYS = int(os.getenv("ATTENDANCE_SNAPSHOT_RETENTION_DAYS", "90"))

# Capture can be turned off entirely without touching the event columns.
ENABLED = os.getenv("ATTENDANCE_SNAPSHOT_ENABLED", "true").lower() in {"1", "true", "yes"}


def _safe_component(value: object) -> str:
    """Filename-safe fragment. camera_id is operator-supplied, so never trust it
    to be free of path separators."""
    text = str(value)
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:40] or "unknown"


def save_face_snapshot(
    frame_bgr: np.ndarray,
    box,
    employee_id: int,
    camera_id: str,
    when: Optional[datetime] = None,
) -> Optional[str]:
    """Write the cropped face; return a path RELATIVE to SNAPSHOT_ROOT.

    Never raises. Failing to save evidence must not block the attendance write —
    a missing image is a gap in the audit trail, a lost event is a missing day's
    pay.
    """
    if not ENABLED:
        return None
    if frame_bgr is None or box is None or len(box) < 4:
        return None

    try:
        import cv2

        when = when or datetime.now()
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        pad_x = int(abs(x2 - x1) * _PAD_RATIO)
        pad_y = int(abs(y2 - y1) * _PAD_RATIO)

        crop = frame_bgr[
            max(0, min(y1, y2) - pad_y):min(height, max(y1, y2) + pad_y),
            max(0, min(x1, x2) - pad_x):min(width, max(x1, x2) + pad_x),
        ]
        if crop.size == 0:
            return None

        relative = Path(when.strftime("%Y-%m-%d")) / (
            f"{_safe_component(employee_id)}_{_safe_component(camera_id)}"
            f"_{when.strftime('%H%M%S_%f')[:-3]}.jpg"
        )
        destination = SNAPSHOT_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), crop, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]):
            logger.warning("snapshot write returned false for %s", destination)
            return None
        return relative.as_posix()
    except Exception:
        logger.exception(
            "snapshot failed employee_id=%s camera=%s (attendance still recorded)",
            employee_id, camera_id,
        )
        return None


def resolve_snapshot(relative_path: str) -> Optional[Path]:
    """Absolute path for a stored snapshot, or None if it escapes the root.

    `snapshot_path` comes back out of the database and will eventually be fed to
    a download route, so the containment check belongs here rather than being
    re-implemented (or forgotten) at each call site.
    """
    if not relative_path:
        return None
    try:
        candidate = (SNAPSHOT_ROOT / relative_path).resolve()
        candidate.relative_to(SNAPSHOT_ROOT.resolve())
    except (ValueError, OSError):
        logger.warning("rejected snapshot path outside the root: %r", relative_path)
        return None
    return candidate if candidate.is_file() else None


def prune_snapshots(retention_days: int = RETENTION_DAYS) -> int:
    """Delete snapshot day-folders older than the retention window.

    Returns the number of day-folders removed. Called from the nightly closeout
    tick — without it this directory grows without bound.
    """
    if not SNAPSHOT_ROOT.exists():
        return 0

    cutoff = date.today() - timedelta(days=max(1, retention_days))
    removed = 0
    for day_dir in SNAPSHOT_ROOT.iterdir():
        if not day_dir.is_dir():
            continue
        try:
            folder_date = date.fromisoformat(day_dir.name)
        except ValueError:
            continue          # not a date-named folder — leave it alone
        if folder_date < cutoff:
            shutil.rmtree(day_dir, ignore_errors=True)
            removed += 1

    if removed:
        logger.info(
            "Attendance snapshots: pruned %d day-folder(s) older than %s",
            removed, cutoff.isoformat(),
        )
    return removed
