from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

# pyrefly: ignore [missing-import]
import cv2
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.camera import Camera
from app.services.camera_service import camera_manager
from app.services.recognition import DEFAULT_THRESHOLD

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cameras", tags=["cameras"])

ALLOWED_SOURCE_TYPES = {"rtsp", "usb", "http"}
ALLOWED_CAMERA_TYPES = {"IN", "OUT"}


class CameraPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    location: str | None = None
    source_url: str | None = None
    stream_url: str | None = None
    source_type: str | None = None
    camera_type: str | None = None
    camera_purpose: str | None = None
    threshold: float | None = Field(None, ge=0.30, le=0.90)
    interval_sec: float | None = Field(None, ge=0.5, le=10.0)
    enabled: bool | None = None


class CameraUpdate(CameraPayload):
    pass


def serialize_camera_row(row) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "source_url": row.source_url,
        "source_type": row.source_type,
        "camera_type": row.camera_type,
        "enabled": row.enabled,
        "threshold": row.threshold,
        "interval_sec": row.interval_sec,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_camera_type(value: Any) -> str:
    camera_type = _normalize_text(value).upper()
    if camera_type in {"CHECK-IN", "CHECKIN", "ENTRY"}:
        return "IN"
    if camera_type in {"CHECK-OUT", "CHECKOUT", "EXIT"}:
        return "OUT"
    return camera_type or "IN"


def _normalize_source_type(value: Any) -> str:
    source_type = _normalize_text(value).lower() or "rtsp"
    return source_type


def _validate_stream_url(source_url: str, source_type: str) -> str:
    if source_type == "usb":
        return source_url

    parsed = urlparse(source_url)
    if parsed.scheme.lower() not in {"rtsp", "http", "https"}:
        raise HTTPException(status_code=400, detail="Stream URL must start with rtsp://, http://, or https://")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="Stream URL is missing the camera host/IP address")
    try:
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="RTSP credentials or port are malformed. Use rtsp://user:pass@ip:554/path",
        ) from exc
    if parsed.scheme.lower() == "rtsp" and port is None:
        raise HTTPException(status_code=400, detail="RTSP URL must include the port, for example rtsp://user:pass@ip:554/path")
    return source_url


async def _read_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        return dict(form)
    body = await request.json()
    if isinstance(body, dict):
        return body
    raise HTTPException(status_code=400, detail="Request body must be a JSON object")


async def _parse_payload(request: Request, model: type[CameraPayload]) -> CameraPayload:
    raw = await _read_payload(request)
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        logger.warning("Invalid camera payload: %s", exc.errors())
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _ensure_single_active_camera(
    db: Session,
    camera_type: str,
    *,
    exclude_id: int | None = None,
) -> int:
    if camera_type not in ALLOWED_CAMERA_TYPES:
        return 0

    query = db.query(Camera).filter(Camera.camera_type == camera_type, Camera.enabled.is_(True))
    if exclude_id is not None:
        query = query.filter(Camera.id != exclude_id)

    disabled = query.update({Camera.enabled: False}, synchronize_session=False)
    if disabled:
        logger.info("Disabled %s other active %s camera(s)", disabled, camera_type)
    return int(disabled or 0)


@router.get("")
def list_cameras(db: Session = Depends(get_db)) -> dict:
    cameras_qs = db.query(Camera).order_by(Camera.created_at.desc(), Camera.id.desc()).all()
    cameras = [serialize_camera_row(cam) for cam in cameras_qs]
    live_states = {item["camera_id"]: item for item in camera_manager.list_states()}

    for camera in cameras:
        live = live_states.get(camera["id"])
        camera["runtime"] = live or {
            "status": "stopped" if not camera["enabled"] else "starting",
            "last_error": None,
            "latest_result": {"faces": []},
        }

    return {"cameras": cameras}


@router.post("")
async def create_camera(request: Request, db: Session = Depends(get_db)) -> dict:
    payload = await _parse_payload(request, CameraPayload)

    clean_name = _normalize_text(payload.name)
    clean_source = _normalize_text(payload.source_url or payload.stream_url)
    clean_type = _normalize_source_type(payload.source_type)
    camera_type = _normalize_camera_type(payload.camera_type or payload.camera_purpose)
    enabled = _normalize_bool(payload.enabled, True)
    threshold = float(payload.threshold if payload.threshold is not None else DEFAULT_THRESHOLD)
    interval_sec = float(payload.interval_sec if payload.interval_sec is not None else 1.5)

    if not clean_name:
        raise HTTPException(status_code=400, detail="Camera name is required")
    if not clean_source:
        raise HTTPException(status_code=400, detail="Camera source URL is required")
    if clean_type not in ALLOWED_SOURCE_TYPES:
        raise HTTPException(status_code=400, detail="source_type must be rtsp, http, or usb")

    clean_source = _validate_stream_url(clean_source, clean_type)
    if camera_type not in ALLOWED_CAMERA_TYPES:
        raise HTTPException(status_code=400, detail="camera_type must be IN or OUT")

    clean_source = _validate_stream_url(clean_source, clean_type)

    try:
        if enabled:
            _ensure_single_active_camera(db, camera_type)

        camera = Camera(
            name=clean_name,
            source_url=clean_source,
            source_type=clean_type,
            camera_type=camera_type,
            enabled=bool(enabled),
            threshold=threshold,
            interval_sec=interval_sec,
        )
        db.add(camera)
        db.commit()
        db.refresh(camera)
        logger.info("Created camera id=%s name=%s type=%s enabled=%s", camera.id, camera.name, camera.camera_type, camera.enabled)
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to create camera: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save camera") from exc

    camera_manager.sync_from_db()
    return {"message": "Camera added", "camera": serialize_camera_row(camera)}


@router.patch("/{camera_id}")
async def update_camera(camera_id: int, request: Request, db: Session = Depends(get_db)) -> dict:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    payload = await _parse_payload(request, CameraUpdate)
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    try:
        if "source_type" in updates:
            clean_type = _normalize_source_type(updates["source_type"])
            if clean_type not in ALLOWED_SOURCE_TYPES:
                raise HTTPException(status_code=400, detail="source_type must be rtsp, http, or usb")
            updates["source_type"] = clean_type

        if "camera_type" in updates or "camera_purpose" in updates:
            camera_type = _normalize_camera_type(updates.get("camera_type") or updates.get("camera_purpose"))
            if camera_type not in ALLOWED_CAMERA_TYPES:
                raise HTTPException(status_code=400, detail="camera_type must be IN or OUT")
            updates["camera_type"] = camera_type
            updates.pop("camera_purpose", None)

        if "enabled" in updates:
            updates["enabled"] = bool(_normalize_bool(updates["enabled"], camera.enabled))

        if "name" in updates:
            clean_name = _normalize_text(updates["name"])
            if not clean_name:
                raise HTTPException(status_code=400, detail="Camera name cannot be empty")
            updates["name"] = clean_name

        if "source_url" in updates:
            clean_source = _normalize_text(updates["source_url"])
            if not clean_source:
                raise HTTPException(status_code=400, detail="Camera source URL cannot be empty")
            updates["source_url"] = clean_source

        if "stream_url" in updates and "source_url" not in updates:
            clean_source = _normalize_text(updates["stream_url"])
            if not clean_source:
                raise HTTPException(status_code=400, detail="Camera source URL cannot be empty")
            updates["source_url"] = clean_source
            updates.pop("stream_url", None)

        if (updates.get("enabled") is True) or (updates.get("camera_type") and updates.get("camera_type") != camera.camera_type):
            _ensure_single_active_camera(db, updates.get("camera_type", camera.camera_type), exclude_id=camera.id)

        if "source_url" in updates or "source_type" in updates:
            clean_source = _validate_stream_url(
                _normalize_text(updates.get("source_url", camera.source_url)),
                _normalize_source_type(updates.get("source_type", camera.source_type)),
            )
            updates["source_url"] = clean_source

        for key, value in updates.items():
            if hasattr(camera, key):
                setattr(camera, key, value)

        db.commit()
        db.refresh(camera)
        logger.info("Updated camera id=%s", camera.id)
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to update camera %s: %s", camera_id, exc)
        raise HTTPException(status_code=500, detail="Failed to update camera") from exc

    camera_manager.sync_from_db()
    return {"message": "Camera updated", "camera": serialize_camera_row(camera)}


@router.post("/test-connection")
@router.post("/test")
async def test_camera_connection(request: Request) -> dict:
    """Test camera connection with detailed logging and frame capture."""
    payload = await _parse_payload(request, CameraPayload)
    clean_source = _normalize_text(payload.source_url or payload.stream_url)
    clean_type = _normalize_source_type(payload.source_type)

    if not clean_source:
        raise HTTPException(status_code=400, detail="Camera source URL is required")
    if clean_type not in ALLOWED_SOURCE_TYPES:
        raise HTTPException(status_code=400, detail="source_type must be rtsp, http, or usb")

    clean_source = _validate_stream_url(clean_source, clean_type)

    logger.info("Testing camera connection: type=%s source=%s", clean_type, clean_source)

    cap = None
    try:
        if clean_type == "usb" or clean_source.isdigit():
            logger.info("Opening USB camera index=%s", clean_source)
            cap = cv2.VideoCapture(int(clean_source))
        else:
            logger.info("Opening stream using FFmpeg backend")
            cap = cv2.VideoCapture(clean_source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        if not cap.isOpened():
            logger.warning("VideoCapture could not open source=%s", clean_source)
            return {
                "ok": False,
                "message": "Could not open camera stream",
                "details": "VideoCapture failed to initialize. Check URL format, network connectivity, and camera availability.",
            }

        ok, frame = cap.read()
        if not ok or frame is None:
            logger.warning("VideoCapture opened but no frame available source=%s", clean_source)
            return {
                "ok": False,
                "message": "Connected but could not read a frame",
                "details": "Stream opened but no frames were returned. The camera may be offline or using an unsupported codec.",
            }

        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        codec = cap.get(cv2.CAP_PROP_FOURCC)
        logger.info("Camera test succeeded source=%s size=%sx%s fps=%s", clean_source, width, height, fps)
        return {
            "ok": True,
            "message": "Camera connection successful",
            "frame_size": {"width": width, "height": height},
            "fps": fps,
            "codec": int(codec) if codec else None,
        }
    except Exception as exc:
        logger.exception("Exception during camera test for source=%s: %s", clean_source, exc)
        return {
            "ok": False,
            "message": f"Exception during connection test: {exc}",
            "details": "An unexpected error occurred. Check server logs for details.",
        }
    finally:
        if cap is not None:
            cap.release()
            logger.info("VideoCapture released for source=%s", clean_source)


@router.post("/{camera_id}/start")
def start_camera(camera_id: int, db: Session = Depends(get_db)) -> dict:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    _ensure_single_active_camera(db, camera.camera_type, exclude_id=camera.id)
    camera.enabled = True
    db.commit()
    db.refresh(camera)
    camera_manager.sync_from_db()
    return {"message": "Camera started", "camera_id": camera_id}


@router.post("/{camera_id}/stop")
def stop_camera(camera_id: int, db: Session = Depends(get_db)) -> dict:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera.enabled = False
    db.commit()
    camera_manager.stop_camera(camera_id)
    return {"message": "Camera stopped", "camera_id": camera_id}


@router.post("/{camera_id}/restart")
def restart_camera(camera_id: int, db: Session = Depends(get_db)) -> dict:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_manager.stop_camera(camera_id)
    _ensure_single_active_camera(db, camera.camera_type, exclude_id=camera.id)
    camera.enabled = True
    db.commit()
    db.refresh(camera)
    camera_manager.sync_from_db()
    return {"message": "Camera restarted", "camera_id": camera_id}


@router.delete("/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_db)) -> dict:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    name = camera.name
    try:
        db.delete(camera)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to delete camera %s: %s", camera_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete camera") from exc

    camera_manager.stop_camera(camera_id)
    return {"message": "Camera deleted", "camera_id": camera_id, "name": name}


@router.get("/{camera_id}/status")
def camera_status(camera_id: int, db: Session = Depends(get_db)) -> dict:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    payload = serialize_camera_row(camera)
    payload["runtime"] = camera_manager.get_state(camera_id) or {
        "status": "stopped",
        "last_error": None,
        "latest_result": {"faces": []},
    }
    return payload


@router.get("/{camera_id}/preview.jpg")
def camera_preview(camera_id: int):
    from fastapi.responses import Response

    preview = camera_manager.get_preview(camera_id)
    if preview is None:
        raise HTTPException(status_code=404, detail="No preview frame available yet")
    return Response(content=preview, media_type="image/jpeg")


@router.get("/{camera_id}/preview")
def camera_preview_alias(camera_id: int):
    return camera_preview(camera_id)






