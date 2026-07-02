from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional
from urllib.parse import unquote

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

from app.core.config import get_settings
from app.services.recognition import DEFAULT_THRESHOLD, recognize_faces, recognize_from_rgb

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


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
):
    """Accept a webcam frame and recognize the face inside the HRMS backend.

    Optional query params:
      - camera_id: identifies which camera is sending the frame
      - camera_purpose: 'IN' forces Check-In, 'OUT' forces Check-Out;
        omit to use auto-toggle logic (IN → OUT → IN ...)
    """
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
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("recognize_frame: unexpected error")
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {exc}") from exc

    # Log outcome
    matched = [f for f in data.get("faces", []) if f.get("matched")]
    logger.info(
        "recognize_frame DONE matched=%d attendance_recorded=%s",
        len(matched), data.get("attendance") is not None,
    )
    return data


@router.post("/recognize-cctv-frame")
def recognize_cctv_frame(payload: CCTVRecognitionRequest):
    """Read one frame from a CCTV/IP camera stream and run attendance recognition with improved error handling."""
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
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Unexpected error in CCTV recognition: {exc}")
        raise HTTPException(status_code=500, detail=f"CCTV recognition failed: {exc}") from exc
    finally:
        if capture is not None:
            capture.release()
            logger.info("VideoCapture released")
