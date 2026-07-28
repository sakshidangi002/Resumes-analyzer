from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Optional
from urllib.parse import unquote

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from PIL import Image
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db.session import get_db
from app.services.audit_service import log_audit
from app.services.recognition import DEFAULT_THRESHOLD, recognize_faces, recognize_from_rgb

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# These endpoints write attendance, which feeds payroll, so a caller must not be
# able to set threshold=0 and have every face match every employee.
#
# The floor is deliberately LOW. An earlier version pinned it to
# DEFAULT_THRESHOLD (0.45), which silently overrode any operator who had turned
# the threshold DOWN to catch difficult faces -- it made recognition stricter
# than configured and employees stopped matching. Tuning the threshold is a
# legitimate operator action; these endpoints now require a logged-in user,
# which is what actually closes the abuse hole. This floor only blocks the
# degenerate "match anything" values.
# NOTE: deliberately NOT called CCTV_MIN_THRESHOLD -- camera_service.py already
# uses that name for a different thing (a per-camera safety floor, default 0.35).
MIN_ALLOWED_THRESHOLD = float(os.getenv("RECOGNITION_API_MIN_THRESHOLD", "0.15"))


# ---------------------------------------------------------------------------
# Upload-based attendance: anti-proxy control (S-04, partial)
# ---------------------------------------------------------------------------
# /recognize-frame accepts an arbitrary JPEG and records attendance for whoever
# is recognised in it. That is a photo-attack with no camera involved: any
# logged-in employee could POST a picture of a colleague and mark them present.
#
# There is no liveness model in this system, so a still image cannot be told
# from a live capture. What CAN be enforced without one is WHO the frame is
# allowed to mark: by default an uploaded frame may only record attendance for
# the uploader themselves. That removes the "mark my friend in" abuse entirely
# while leaving normal self-service check-in working.
#
# Shared kiosks (one tablet at reception marking many people) legitimately need
# the old behaviour. Set KIOSK_ATTENDANCE_ROLES to the roles allowed to mark
# OTHER people from an upload -- e.g. "Admin,HR". Empty (default) = nobody.
#
# RESIDUAL RISK, stated plainly: this does NOT stop someone holding a printed
# photo up to a real IN/OUT camera. That is the same class of attack and it
# still works. Closing it needs a passive-liveness model in the CCTV pipeline
# (a texture/depth anti-spoof pass before _mark_attendance); this control is
# the part that can be done correctly without one, not a substitute for it.
KIOSK_ATTENDANCE_ROLES = {
    role.strip()
    for role in os.getenv("KIOSK_ATTENDANCE_ROLES", "").split(",")
    if role.strip()
}


def _enforce_upload_attendance_policy(data: dict, current_user, db) -> dict:
    """Drop attendance recorded from an upload for someone other than the uploader.

    The recognition result is left intact -- the caller still sees who was
    matched -- only the attendance side effect is refused, and the refusal is
    audited so proxy attempts are visible rather than merely blocked.
    """
    attendance = data.get("attendance")
    if not attendance:
        return data

    marked_employee_id = attendance.get("employee_id")
    if marked_employee_id is None:
        return data

    if current_user.employee_id == marked_employee_id:
        return data                                   # self check-in: allowed

    role_names = {r.name for r in current_user.roles}
    if KIOSK_ATTENDANCE_ROLES and (role_names & KIOSK_ATTENDANCE_ROLES):
        log_audit(
            db, current_user.id, "ATTENDANCE_MARKED_FOR_OTHER", "Employee",
            str(marked_employee_id),
            f"Kiosk upload by {current_user.username} marked employee "
            f"{marked_employee_id} ({attendance.get('event_type')})",
        )
        return data

    logger.warning(
        "Blocked upload attendance: user=%s (employee_id=%s) tried to mark employee_id=%s",
        current_user.username, current_user.employee_id, marked_employee_id,
    )
    log_audit(
        db, current_user.id, "ATTENDANCE_PROXY_BLOCKED", "Employee",
        str(marked_employee_id),
        f"{current_user.username} uploaded a frame that matched employee "
        f"{marked_employee_id}; attendance refused (not the uploader)",
    )
    data = dict(data)
    data["attendance"] = None
    data["attendance_blocked_reason"] = (
        "An uploaded image can only record attendance for the person signed in. "
        "Contact HR if you need kiosk check-in for others."
    )
    return data


def _clamp_threshold(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    if v < MIN_ALLOWED_THRESHOLD:
        logger.warning(
            "threshold %.3f is below the minimum %.3f; using the minimum. "
            "Set CCTV_MIN_THRESHOLD to change this floor.",
            v, MIN_ALLOWED_THRESHOLD,
        )
        return MIN_ALLOWED_THRESHOLD
    return v


class CCTVRecognitionRequest(BaseModel):
    stream_url: str = Field(..., min_length=1, description="RTSP/HTTP camera stream URL")
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    camera_id: str | None = None
    camera_type: str = Field(default="IN", description="Camera purpose: IN, OUT, BREAK_IN, or BREAK_OUT")


@router.post("/recognize-frame")
async def recognize_frame(
    file: UploadFile = File(...),
    threshold: float = DEFAULT_THRESHOLD,
    camera_id: Optional[str] = Query(default=None, description="Camera ID for attendance tracking"),
    camera_purpose: Optional[str] = Query(default=None, description="Camera purpose: IN or OUT"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Accept a webcam frame and recognize the face inside the HRMS backend.

    Optional query params:
      - camera_id: identifies which camera is sending the frame
      - camera_purpose: 'IN' forces Check-In, 'OUT' forces Check-Out;
        omit to use auto-toggle logic (IN → OUT → IN ...)
    """
    threshold = _clamp_threshold(threshold)
    logger.info(
        "STEP-1 webcam_frame_received source=webcam camera_id=%s camera_purpose=%s threshold=%.3f",
        camera_id, camera_purpose, threshold,
    )
    image_bytes = await file.read()
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("recognize_frame: invalid image upload: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid image uploaded") from exc

    try:
        data = recognize_faces(
            image,
            threshold=threshold,
            source="webcam",
            camera_id=camera_id,
            camera_purpose=camera_purpose,
        )
    except RuntimeError as exc:
        logger.error("recognize_frame: RuntimeError: %s", exc)
        raise HTTPException(status_code=503, detail="Recognition service is temporarily unavailable.") from exc
    except Exception as exc:
        logger.exception("recognize_frame: unexpected error")
        raise HTTPException(status_code=500, detail="Face recognition failed. Please try again later.") from exc

    data = _enforce_upload_attendance_policy(data, current_user, db)

    # Log outcome
    matched = [f for f in data.get("faces", []) if f.get("matched")]
    logger.info(
        "recognize_frame DONE matched=%d attendance_recorded=%s",
        len(matched), data.get("attendance") is not None,
    )
    return data


@router.post("/recognize-cctv-frame")
def recognize_cctv_frame(
    payload: CCTVRecognitionRequest,
    current_user=Depends(get_current_user),
):
    """Read one frame from a CCTV/IP camera stream and run attendance recognition with improved error handling."""
    payload.threshold = _clamp_threshold(payload.threshold)
    try:
        import cv2
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="OpenCV is required for CCTV recognition.",
        ) from exc

    logger.info(f"CCTV recognition request - URL: {payload.stream_url}, Camera ID: {payload.camera_id}, Type: {payload.camera_type}")
    
    # Decode URL-encoded characters (e.g., %40 -> @) for RTSP
    stream_url = unquote(payload.stream_url)
    logger.info(f"Decoded URL: {stream_url}")
    
    capture = None
    try:
        # Use FFmpeg backend for RTSP streams
        if stream_url.startswith("rtsp://"):
            logger.info("Using FFmpeg backend for RTSP stream")
            capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            # Set timeout and buffer settings
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5 second read timeout
        else:
            logger.info("Using default backend for non-RTSP stream")
            capture = cv2.VideoCapture(stream_url)

        if not capture.isOpened():
            logger.error(f"Could not open CCTV stream: {payload.stream_url}")
            raise HTTPException(
                status_code=503, 
                detail="Could not open the CCTV stream. Check URL format, network connectivity, and camera availability."
            )

        logger.info("CCTV stream opened successfully, attempting to read frame...")
        ok, frame = capture.read()
        
        if not ok or frame is None:
            logger.error("Could not read frame from CCTV stream")
            raise HTTPException(
                status_code=503, 
                detail="Could not read a frame from the CCTV stream. Camera may be offline or using unsupported codec."
            )

        height, width = frame.shape[:2]
        logger.info(f"Frame read successfully: {width}x{height}")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        data = recognize_from_rgb(
            rgb_frame,
            threshold=payload.threshold,
            source="cctv",
            camera_id=payload.camera_id,
            camera_purpose=payload.camera_type,
        )
        
        logger.info(f"Recognition completed: {data.get('status')}, faces detected: {len(data.get('faces', []))}")
        return data
        
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.error(f"Runtime error in CCTV recognition: {exc}")
        raise HTTPException(status_code=503, detail="Recognition service is temporarily unavailable.") from exc
    except Exception as exc:
        logger.exception(f"Unexpected error in CCTV recognition: {exc}")
        raise HTTPException(status_code=500, detail="CCTV recognition failed. Please try again later.") from exc
    finally:
        if capture is not None:
            capture.release()
            logger.info("VideoCapture released")
