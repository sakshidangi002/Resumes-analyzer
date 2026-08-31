"""Per-camera recognition profile, resolved from the database, hot-reloadable.

Replaces the process-wide CCTV_* environment constants. A profile answers three
separate questions for one camera:

  * **Which faces may influence identity at all?** — the QualityLimits handed to
    face_quality.assess (size, detector score, yaw, pitch, landmark asymmetry,
    face-crop sharpness).
  * **What counts as a match?** — threshold and margin.
  * **What counts as enough evidence to touch payroll?** — minimum observations,
    minimum best-observation quality, minimum track self-consensus.

WHY PER CAMERA
--------------
This deployment has two irreconcilable camera roles behind one codebase:

  IN / OUT   Ceiling-mounted at the entrance. Their matches become attendance
             rows. A false accept writes the wrong person's payroll data and is
             discovered, if ever, weeks later. A miss is recoverable — HR keys
             it in. So: strict everything, and demand real evidence.

  MONITOR    Ceiling-mounted over desks. Hard-blocked from marking attendance
             (recognition._mark_attendance). Their worst failure is a wrong name
             on a live overlay for a few seconds. A miss, by contrast, defeats
             the camera's entire purpose. So: relaxed everything.

The old global constants had to be one or the other, and the code history shows
them being loosened for the room cameras — which silently loosened the payroll
cameras too. This module is what makes "strict where it costs money, permissive
where it does not" expressible.

HOT RELOAD
----------
Profiles are cached for _TTL_SECONDS and re-read from the database after that,
so an administrator changing a threshold in the UI sees it applied within
~10 seconds with no restart. Camera worker threads call get_profile() once per
analysis tick; the cache keeps that to one dictionary lookup in the common case.

A database failure NEVER changes recognition behaviour: the last good profile is
kept, and if there has never been one, the purpose defaults apply. Falling back
to a permissive default on a DB blip would be a way to accidentally mark
attendance from a face the strict profile would have rejected.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace

from app.services.face_quality import QualityLimits

logger = logging.getLogger(__name__)

_TTL_SECONDS = float(10.0)


@dataclass(frozen=True)
class CameraProfile:
    camera_id: str
    purpose: str                     # IN | OUT | MONITOR

    # Matching
    threshold: float                 # min cosine score against the gallery
    margin: float                    # min (best - runner_up)

    # Quality gates for individual observations
    limits: QualityLimits

    # Evidence required before an attendance write
    min_observations: int
    min_quality: float               # best single observation's quality score
    min_consensus: float             # fused-template self-agreement

    # Runtime
    analysis_interval: float
    face_crop_scale: int
    attendance_cooldown: float

    @property
    def marks_attendance(self) -> bool:
        """MONITOR cameras label people; they never create attendance rows."""
        return self.purpose in {"IN", "OUT"}

    def describe(self) -> str:
        """One-line summary for the startup log — makes drift visible."""
        return (
            f"purpose={self.purpose} thr={self.threshold:.2f} margin={self.margin:.2f} "
            f"min_face={self.limits.min_face_px:.0f}px yaw<={self.limits.max_yaw_deg:.0f} "
            f"pitch<={self.limits.max_pitch_deg:.0f} blur>={self.limits.min_blur_var:.0f} "
            f"obs>={self.min_observations} q>={self.min_quality:.2f} "
            f"consensus>={self.min_consensus:.2f} interval={self.analysis_interval:.2f}s"
        )


# ---------------------------------------------------------------------------
# Purpose defaults
# ---------------------------------------------------------------------------
# ATTENDANCE cameras. Every number here is deliberately stricter than the
# constants it replaces, because those constants were relaxed to keep the room
# cameras working and nobody re-tightened the payroll path.
#
#   threshold 0.45   unchanged from settings.default_threshold
#   margin    0.18   was 0.10. The documented mislabelling had the WRONG person
#                    at 0.77 while correct matches sat at 0.73-0.79 — score alone
#                    cannot separate those, but a noisy embedding does not pull
#                    clear of the runner-up the way a real match does. This is
#                    the single most effective knob against false accepts and is
#                    the one that was never tuned.
#   min_face  28px   was 16. 16 was chosen because raising it left the ROOM
#                    cameras identifying nobody — a room-camera problem applied
#                    to the payroll cameras. At the entrance a person walks
#                    toward the camera and does reach ~30-40px.
#   obs       3      was effectively 1 (CCTV_CONFIRM_FRAMES=1). Not 3 CONSECUTIVE
#                    frames — 3 accepted observations anywhere in the track. That
#                    distinction is why the previous attempt at 2 failed and had
#                    to be reverted: a face visible for one moment can never
#                    produce two consecutive reads, but it can contribute to a
#                    fused template over the approach.
_ATTENDANCE_DEFAULTS = dict(
    threshold=0.45,
    margin=0.18,
    limits=QualityLimits(
        min_face_px=28.0,
        min_det_score=0.50,
        max_yaw_deg=40.0,
        max_pitch_deg=35.0,
        max_landmark_asym=0.55,
        min_blur_var=18.0,
        good_face_px=70.0,
        good_blur_var=90.0,
    ),
    min_observations=3,
    min_quality=0.28,
    min_consensus=0.55,
    analysis_interval=0.12,
    face_crop_scale=4,
    attendance_cooldown=20.0,
)

# MONITOR cameras. Cannot mark attendance, so the cost of being wrong is a
# briefly-wrong label. Optimised for coverage instead: see people at all, name
# them when possible. The slow analysis interval is not a quality choice — it
# reserves the shared inference gate for the attendance cameras.
_MONITOR_DEFAULTS = dict(
    threshold=0.42,
    margin=0.10,
    limits=QualityLimits(
        min_face_px=16.0,
        min_det_score=0.35,
        # Monitoring labels cannot write payroll. Permit the oblique/downward
        # views this deployment actually produces; IN/OUT keeps the stricter
        # 40/35 degree limits above.
        max_yaw_deg=75.0,
        max_pitch_deg=70.0,
        max_landmark_asym=0.75,
        min_blur_var=8.0,
        good_face_px=50.0,
        good_blur_var=60.0,
    ),
    min_observations=2,
    min_quality=0.15,
    min_consensus=0.40,
    analysis_interval=1.5,
    face_crop_scale=4,
    attendance_cooldown=20.0,
)

# Absolute floors. A stored profile may never go below these, whatever an
# operator types into the UI or a stale row holds. Older camera rows still carry
# the legacy threshold of 0.05, which accepts essentially random faces; that row
# must not be able to write payroll data because someone forgot to clean it up.
_FLOORS = dict(
    threshold=0.35,
    margin=0.05,
    min_face_px=14.0,
    min_observations=1,
)


def defaults_for(purpose: str) -> dict:
    return dict(_MONITOR_DEFAULTS if (purpose or "").upper() == "MONITOR" else _ATTENDANCE_DEFAULTS)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_lock = threading.Lock()

# TTL cache — cleared by invalidate() so an operator's change lands at once.
_cache: dict[str, tuple[float, CameraProfile]] = {}

# Last profile successfully resolved from the database, per camera. Deliberately
# SEPARATE from _cache and never cleared by invalidate(): it is the fallback for
# a database failure, and if invalidate() wiped it too then an admin edit
# immediately followed by a DB blip would silently drop the camera back to the
# purpose defaults — widening the gates at exactly the moment nobody is
# watching. Found by test_database_failure_keeps_the_previous_profile.
_last_good: dict[str, CameraProfile] = {}


def get_profile(camera_id, purpose: str = "IN") -> CameraProfile:
    """Resolved profile for a camera. Cheap enough to call every analysis tick."""
    key = str(camera_id)
    now = time.time()

    with _lock:
        entry = _cache.get(key)
        if entry is not None and (now - entry[0]) < _TTL_SECONDS:
            return entry[1]
        previous = entry[1] if entry is not None else None

    profile = _load(key, purpose)

    with _lock:
        _cache[key] = (now, profile)

    if previous is not None and previous != profile:
        logger.info("CAMERA-PROFILE camera=%s reloaded: %s", key, profile.describe())
    return profile


def invalidate(camera_id=None) -> None:
    """Drop cached profiles so the next read hits the database immediately.

    Called by the camera admin routes on update, so a settings change applies at
    once rather than after the TTL. Does NOT clear _last_good — see above.
    """
    with _lock:
        if camera_id is None:
            _cache.clear()
        else:
            _cache.pop(str(camera_id), None)


def reset_for_tests() -> None:
    """Clear both caches. Test helper only."""
    with _lock:
        _cache.clear()
        _last_good.clear()


def _load(camera_id: str, purpose: str) -> CameraProfile:
    """Read the row and overlay any non-NULL overrides onto the purpose default."""
    row = None
    try:
        from app.db.session import SessionLocal
        from app.models.camera import CameraConfig

        with SessionLocal() as db:
            try:
                numeric_id = int(camera_id)
            except (TypeError, ValueError):
                numeric_id = None
            if numeric_id is not None:
                row = db.query(CameraConfig).filter(CameraConfig.id == numeric_id).first()
    except Exception:
        # Keep whatever we last resolved. Silently widening the gates because
        # Postgres hiccuped is exactly how a false attendance gets written.
        with _lock:
            last = _last_good.get(str(camera_id))
        logger.warning(
            "CAMERA-PROFILE camera=%s database read failed — %s",
            camera_id,
            "keeping the last known-good profile" if last is not None
            else "no previous profile, falling back to purpose defaults",
            exc_info=True,
        )
        if last is not None:
            return last

    resolved_purpose = (
        (getattr(row, "camera_purpose", None) or getattr(row, "camera_type", None) or purpose)
        if row is not None else purpose
    ) or "IN"
    resolved_purpose = str(resolved_purpose).strip().upper()
    if resolved_purpose not in {"IN", "OUT", "MONITOR"}:
        resolved_purpose = "IN"

    base = defaults_for(resolved_purpose)
    limits: QualityLimits = base["limits"]

    def pick(attr: str, fallback):
        if row is None:
            return fallback
        value = getattr(row, attr, None)
        return fallback if value is None else value

    limits = replace(
        limits,
        min_face_px=max(
            _FLOORS["min_face_px"], float(pick("min_face_px", limits.min_face_px))
        ),
        min_det_score=float(pick("min_det_score", limits.min_det_score)),
        max_yaw_deg=float(pick("max_yaw_deg", limits.max_yaw_deg)),
        max_pitch_deg=float(pick("max_pitch_deg", limits.max_pitch_deg)),
        max_landmark_asym=float(pick("max_landmark_asym", limits.max_landmark_asym)),
        min_blur_var=float(pick("min_blur_var", limits.min_blur_var)),
    )

    # `threshold` predates this migration and is NOT NULL with a default, so a
    # legacy value of 0.05 is a real possibility — clamp it.
    stored_threshold = float(pick("threshold", base["threshold"]) or base["threshold"])
    threshold = max(_FLOORS["threshold"], stored_threshold)
    if stored_threshold < _FLOORS["threshold"]:
        logger.warning(
            "CAMERA-PROFILE camera=%s stored threshold %.3f below floor %.3f — using %.3f",
            camera_id, stored_threshold, _FLOORS["threshold"], threshold,
        )

    profile = CameraProfile(
        camera_id=str(camera_id),
        purpose=resolved_purpose,
        threshold=threshold,
        margin=max(_FLOORS["margin"], float(pick("match_margin", base["margin"]))),
        limits=limits,
        min_observations=max(
            int(_FLOORS["min_observations"]),
            int(pick("min_observations", base["min_observations"])),
        ),
        min_quality=float(pick("min_quality", base["min_quality"])),
        min_consensus=float(pick("min_consensus", base["min_consensus"])),
        analysis_interval=max(
            0.02, float(pick("analysis_interval", base["analysis_interval"]))
        ),
        face_crop_scale=max(1, int(pick("face_crop_scale", base["face_crop_scale"]))),
        attendance_cooldown=float(
            pick("attendance_cooldown", base["attendance_cooldown"])
        ),
    )

    # Only remember it as known-good when it actually came from a row. A profile
    # built entirely from purpose defaults (camera not in the table yet) is not
    # something to fall back to later as though it had been configured.
    if row is not None:
        with _lock:
            _last_good[str(camera_id)] = profile
    return profile
