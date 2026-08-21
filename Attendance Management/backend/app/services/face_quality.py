"""Single source of truth for "is this face good enough to trust?".

Every place that turns pixels into an identity decision must agree on what a
usable face is: enrollment, the entrance pipeline, the room-camera pipeline, the
fusion weighting, and the calibration scripts. Before this module they did not.
Enrollment checked size + detector score + landmark asymmetry + blur, while the
CCTV path checked only ``_MIN_FACE_PX`` and a WHOLE-FRAME blur value — so a
side-profile at 20px passed straight into a payroll write, and a face the
enrolment gate would have rejected outright was allowed to name an employee.

Design rules:

* **Face crop only, never the whole frame.** A frame is "sharp" because the
  door frame and floor tiles are sharp; that says nothing about the 24px face in
  the corner. Whole-frame blur is a stream-health signal, not a face signal.
* **Hard gates and a soft score are different things.** ``ok`` answers "may this
  observation influence identity at all?". ``score`` answers "how much should it
  count relative to other observations?" — that is the fusion weight. Collapsing
  them into one number is what made the old ``quality=face_px`` weighting treat a
  sharp frontal 30px face and a motion-blurred 30px profile as equals.
* **Pose is a first-class gate.** These cameras are ceiling-mounted and mostly
  see people from above and from the side, which is exactly the regime where
  ArcFace embeddings degrade into noise that lands on an arbitrary employee.
* **Nothing here raises.** A quality check that throws would take down a camera
  thread. Every failure path degrades to "unusable, with a reason".

All limits arrive from the per-camera profile (see ``camera_profile.py``) so a
check-in camera and a room camera can hold different bars without code changes.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityLimits:
    """Hard gates a face must clear before it may influence identity.

    Defaults are the historical enrollment values, so enrollment behaviour is
    unchanged when a caller does not pass a profile. The CCTV callers always
    pass camera-specific values.
    """

    min_face_px: float = 90.0        # real (de-scaled) face width in frame pixels
    min_det_score: float = 0.62      # detector confidence
    max_yaw_deg: float = 45.0        # |yaw| — left/right head turn
    max_pitch_deg: float = 40.0      # |pitch| — looking up/down (monitors!)
    max_landmark_asym: float = 0.60  # normalised nose-between-eyes asymmetry
    min_blur_var: float = 25.0       # Laplacian variance of the FACE CROP

    # Reference points for the soft score. A face at `good_face_px` scores 1.0
    # on size; one at `min_face_px` scores near 0. Same idea for blur. These do
    # not gate anything — they only shape the fusion weight.
    good_face_px: float = 110.0
    good_blur_var: float = 120.0


# The bar used when nobody supplies one. Mirrors the pre-existing enrollment
# gate exactly (employee_face_service._assess_face_quality) so replacing that
# code path with this module is behaviour-preserving.
ENROLLMENT_LIMITS = QualityLimits()


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FaceQuality:
    ok: bool
    reason: str            # "" when ok; a short machine-greppable code otherwise
    detail: str            # human-facing sentence (used by enrollment responses)

    face_px: float
    det_score: float
    blur_var: float
    yaw: float
    pitch: float
    roll: float
    landmark_asym: float

    score: float           # 0..1 fusion weight. 0 when not ok.

    def as_log_fields(self) -> dict:
        """Flat dict for structured logging — see camera_service DECISION lines."""
        return {
            "face_px": round(self.face_px, 1),
            "det": round(self.det_score, 3),
            "blur": round(self.blur_var, 1),
            "yaw": round(self.yaw, 1),
            "pitch": round(self.pitch, 1),
            "roll": round(self.roll, 1),
            "asym": round(self.landmark_asym, 3),
            "q": round(self.score, 3),
        }


_UNUSABLE = FaceQuality(
    ok=False, reason="no_face", detail="no clear face found",
    face_px=0.0, det_score=0.0, blur_var=0.0,
    yaw=0.0, pitch=0.0, roll=0.0, landmark_asym=1.0, score=0.0,
)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def face_px_width(face: dict, scale: float = 1.0) -> float:
    """Face width in ORIGINAL frame pixels.

    ``scale`` is the upscale factor applied before detection (the room-camera
    zoom path enlarges a head crop 3x). Dividing it back out is what stops an
    interpolated 20px face from being scored as a real 60px one.
    """
    box = face.get("box") or []
    if len(box) < 4:
        return 0.0
    return abs(float(box[2]) - float(box[0])) / max(1.0, float(scale))


def _pose_from_face(face: dict) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees, or zeros when the model did not supply them.

    The pose dict is produced by face_service from InsightFace's ``face.pose``
    (pitch, yaw, roll). It used to be read as ``face.yaw`` — an attribute
    InsightFace never sets — so every face in the system reported 0/0/0 and pose
    could not gate anything. See face_service._extract_faces_insightface.
    """
    pose = face.get("pose") or {}
    try:
        return (
            float(pose.get("yaw") or 0.0),
            float(pose.get("pitch") or 0.0),
            float(pose.get("roll") or 0.0),
        )
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0)


def _landmark_asymmetry(kps: Optional[Sequence]) -> float:
    """Normalised nose-between-eyes asymmetry in [0, 1].

    0.00  nose centred between the eyes (dead-on frontal)
    ~0.50 clear left/right turn, both eyes still visible
    ~1.00 nose aligned with one eye (true profile)

    Bounded on purpose. The original metric was max(dl,dr)/min(dl,dr), which is
    unbounded and explodes long before the face is unusable, so no threshold
    could separate a usable 35-degree turn from a full profile.

    Returns 1.0 (worst) when landmarks are unavailable, but callers treat a
    missing-landmark face as "asymmetry unknown" rather than failing it — see
    ``assess``: the gate only fires when landmarks actually exist. A detector
    that gives no landmarks (the YOLO fallback without keypoints) must not have
    every one of its faces rejected.
    """
    if not kps or len(kps) < 3:
        return 0.0
    try:
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        dl = abs(float(nose[0]) - float(left_eye[0]))
        dr = abs(float(right_eye[0]) - float(nose[0]))
        span = dl + dr
        if span <= 1e-3:
            return 1.0
        return float(abs(dl - dr) / span)
    except (TypeError, ValueError, IndexError):
        return 0.0


def _blur_variance(rgb: np.ndarray, box: Sequence, scale: float = 1.0) -> float:
    """Laplacian variance of the FACE CROP, corrected for prior upscaling.

    Bicubic upsampling invents no detail: it spreads the same edge energy over
    more pixels, so the Laplacian variance of a 3x-enlarged crop is far below
    that of a natively-sharp crop of the same pixel size. Without the correction
    every zoomed room-camera face would look "blurry" and be rejected, taking
    the room cameras' only identification path with it.

    The correction is empirical, not exact — high-frequency energy falls roughly
    with the square of the interpolation factor, so we scale the measurement
    back up by ``scale**2`` and cap it, which restores comparability with
    natively-captured crops without letting the zoom manufacture quality.
    """
    try:
        import cv2
    except Exception:  # pragma: no cover - OpenCV absence is fatal elsewhere
        return 0.0

    try:
        if len(box) < 4:
            return 0.0
        h, w = rgb.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2]))
        y2 = min(h, int(box[3]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return 0.0
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if scale > 1.0:
            var *= float(scale) ** 2
        return var
    except Exception:
        # A blur check must never cost a camera its recognition pass.
        logger.debug("blur variance failed", exc_info=True)
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def assess(
    face: dict,
    rgb: Optional[np.ndarray] = None,
    *,
    limits: QualityLimits = ENROLLMENT_LIMITS,
    scale: float = 1.0,
) -> FaceQuality:
    """Grade one detected face.

    ``face``   detection dict from face_service (box / confidence / kps / pose).
    ``rgb``    the image the face was detected IN. Optional: without it the blur
               term is skipped rather than guessed, because a fabricated blur
               value would silently change the fusion weighting.
    ``scale``  upscale factor already applied to ``rgb`` (room-camera zoom).
    """
    if not face:
        return _UNUSABLE

    box = face.get("box") or []
    if len(box) < 4:
        return _UNUSABLE

    px = face_px_width(face, scale)
    det = float(face.get("confidence") or 0.0)
    yaw, pitch, roll = _pose_from_face(face)
    kps = face.get("kps")
    asym = _landmark_asymmetry(kps)
    blur = _blur_variance(rgb, box, scale) if rgb is not None else float("nan")

    # ── Hard gates, cheapest and most decisive first ────────────────────────
    if px < limits.min_face_px:
        return _fail(
            "face_too_small",
            f"face is too small ({px:.0f}px, need {limits.min_face_px:.0f}px)",
            px, det, blur, yaw, pitch, roll, asym,
        )

    if det < limits.min_det_score:
        return _fail(
            "low_det_score",
            f"face is not clearly visible (detector {det:.2f} < {limits.min_det_score:.2f})",
            px, det, blur, yaw, pitch, roll, asym,
        )

    # Pose. Only enforced when the model actually reported a pose — a detector
    # that supplies none must not have all its faces rejected.
    has_pose = abs(yaw) > 1e-6 or abs(pitch) > 1e-6 or abs(roll) > 1e-6
    if has_pose:
        if abs(yaw) > limits.max_yaw_deg:
            return _fail(
                "pose_yaw",
                f"face is turned too far sideways (yaw {yaw:.0f}deg, limit {limits.max_yaw_deg:.0f}deg)",
                px, det, blur, yaw, pitch, roll, asym,
            )
        if abs(pitch) > limits.max_pitch_deg:
            return _fail(
                "pose_pitch",
                f"face is tilted too far up/down (pitch {pitch:.0f}deg, limit {limits.max_pitch_deg:.0f}deg)",
                px, det, blur, yaw, pitch, roll, asym,
            )

    # Landmark asymmetry — a second, model-independent read on side-profiles.
    # Only applied when landmarks exist (see _landmark_asymmetry).
    if kps and len(kps) >= 3 and asym > limits.max_landmark_asym:
        return _fail(
            "landmark_asym",
            f"face is turned too far to the side (turn {asym:.2f}, limit {limits.max_landmark_asym:.2f})",
            px, det, blur, yaw, pitch, roll, asym,
        )

    if rgb is not None and not math.isnan(blur) and blur < limits.min_blur_var:
        return _fail(
            "blurry",
            f"face is too blurry (sharpness {blur:.0f}, need {limits.min_blur_var:.0f})",
            px, det, blur, yaw, pitch, roll, asym,
        )

    # ── Soft score → fusion weight ──────────────────────────────────────────
    score = _soft_score(px, det, blur, yaw, pitch, asym, limits)
    return FaceQuality(
        ok=True, reason="", detail="",
        face_px=px, det_score=det, blur_var=(0.0 if math.isnan(blur) else blur),
        yaw=yaw, pitch=pitch, roll=roll, landmark_asym=asym, score=score,
    )


def _fail(reason, detail, px, det, blur, yaw, pitch, roll, asym) -> FaceQuality:
    return FaceQuality(
        ok=False, reason=reason, detail=detail,
        face_px=px, det_score=det,
        blur_var=(0.0 if math.isnan(blur) else blur),
        yaw=yaw, pitch=pitch, roll=roll, landmark_asym=asym, score=0.0,
    )


def _soft_score(
    px: float, det: float, blur: float,
    yaw: float, pitch: float, asym: float,
    limits: QualityLimits,
) -> float:
    """Weighted geometric mean of the quality terms, in (0, 1].

    Geometric, not arithmetic: a face that is large and sharp but nearly in
    profile should NOT average out to "good". Any term approaching zero must
    drag the whole score down, because that is exactly the observation whose
    embedding is unreliable.

    Weights reflect measured impact on ArcFace similarity for this deployment:
    size dominates (a 20px face is unrecoverable), pose next (these cameras see
    profiles constantly), then sharpness, then detector confidence.
    """
    size_span = max(1.0, limits.good_face_px - limits.min_face_px)
    size_t = _sat((px - limits.min_face_px) / size_span)

    if math.isnan(blur):
        blur_t = 0.75  # unknown: neither rewarded nor punished
    else:
        blur_span = max(1.0, limits.good_blur_var - limits.min_blur_var)
        blur_t = _sat((blur - limits.min_blur_var) / blur_span)

    # Pose term combines the model pose and the landmark reading, taking the
    # WORSE of the two. They fail in different situations — the 3D pose model is
    # unreliable on tiny faces, the landmark ratio is unreliable under roll —
    # and trusting the optimistic one defeats the point of checking both.
    yaw_t = _sat(1.0 - abs(yaw) / max(1.0, limits.max_yaw_deg))
    pitch_t = _sat(1.0 - abs(pitch) / max(1.0, limits.max_pitch_deg))
    asym_t = _sat(1.0 - asym / max(1e-3, limits.max_landmark_asym))
    pose_t = min(yaw_t, pitch_t, asym_t)

    det_t = _sat(det)

    # Floors keep the geometric mean from collapsing to exactly 0 for an
    # observation that legitimately passed every hard gate.
    terms = (
        (max(size_t, 0.05), 0.40),
        (max(pose_t, 0.05), 0.30),
        (max(blur_t, 0.05), 0.20),
        (max(det_t, 0.05), 0.10),
    )
    log_sum = sum(weight * math.log(value) for value, weight in terms)
    return _sat(math.exp(log_sum))


def _sat(v: float) -> float:
    """Saturate to [0, 1]."""
    if v != v:      # NaN
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)
