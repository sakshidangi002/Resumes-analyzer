from __future__ import annotations

# pyrefly: ignore [missing-import]
import cv2
from fastapi import APIRouter, Form, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.camera import Camera
from app.services.camera_service import camera_manager
from app.services.recognition import DEFAULT_THRESHOLD

router = APIRouter(prefix="/api/cameras", tags=["cameras"])

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

class CameraUpdate(BaseModel):
    name: str | None = None
    source_url: str | None = None
    source_type: str | None = None
    camera_type: str | None = None
    threshold: float | None = Field(None, ge=0.30, le=0.90)
    interval_sec: float | None = Field(None, ge=0.5, le=10.0)
    enabled: bool | None = None

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
def create_camera(
    name: str = Form(...),
    source_url: str = Form(...),
    source_type: str = Form("rtsp"),
    camera_type: str = Form("IN"),
    threshold: float = Form(DEFAULT_THRESHOLD),
    interval_sec: float = Form(1.5),
    enabled: bool = Form(True),
    db: Session = Depends(get_db),
) -> dict:
    clean_name = name.strip()
    clean_source = source_url.strip()
    clean_type = source_type.strip().lower()

    if not clean_name:
        raise HTTPException(status_code=400, detail="Camera name is required")
    if not clean_source:
        raise HTTPException(status_code=400, detail="Camera source URL is required")
    if clean_type not in {"rtsp", "usb"}:
        raise HTTPException(status_code=400, detail="source_type must be rtsp or usb")
    if camera_type not in {"IN", "OUT"}:
        raise HTTPException(status_code=400, detail="camera_type must be IN or OUT")

    camera = Camera(
        name=clean_name,
        source_url=clean_source,
        source_type=clean_type,
        camera_type=camera_type,
        enabled=enabled,
        threshold=threshold,
        interval_sec=interval_sec
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)

    camera_manager.sync_from_db()
    return {"message": "Camera added", "camera": serialize_camera_row(camera)}

@router.patch("/{camera_id}")
def update_camera(camera_id: int, payload: CameraUpdate, db: Session = Depends(get_db)) -> dict:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    if "source_type" in updates:
        clean_type = updates["source_type"].strip().lower()
        if clean_type not in {"rtsp", "usb"}:
            raise HTTPException(status_code=400, detail="source_type must be rtsp or usb")
        updates["source_type"] = clean_type

    if "name" in updates and not updates["name"].strip():
        raise HTTPException(status_code=400, detail="Camera name cannot be empty")
    if "source_url" in updates and not updates["source_url"].strip():
        raise HTTPException(status_code=400, detail="Camera source URL cannot be empty")

    for key, value in updates.items():
        setattr(camera, key, value)
    
    db.commit()
    db.refresh(camera)

    camera_manager.sync_from_db()
    return {"message": "Camera updated", "camera": serialize_camera_row(camera)}

@router.post("/test")
def test_camera_connection(
    source_url: str = Form(...),
    source_type: str = Form("rtsp"),
) -> dict:
    clean_source = source_url.strip()
    clean_type = source_type.strip().lower()
    if not clean_source:
        raise HTTPException(status_code=400, detail="Camera source URL is required")
    if clean_type not in {"rtsp", "usb"}:
        raise HTTPException(status_code=400, detail="source_type must be rtsp or usb")

    if clean_type == "usb" or clean_source.isdigit():
        cap = cv2.VideoCapture(int(clean_source))
    else:
        cap = cv2.VideoCapture(clean_source, cv2.CAP_FFMPEG)

    try:
        if not cap.isOpened():
            return {"ok": False, "message": "Could not open camera stream"}

        ok, frame = cap.read()
        if not ok or frame is None:
            return {"ok": False, "message": "Connected but could not read a frame"}

        height, width = frame.shape[:2]
        return {
            "ok": True,
            "message": "Camera connection successful",
            "frame_size": {"width": width, "height": height},
        }
    finally:
        cap.release()

@router.post("/{camera_id}/start")
def start_camera(camera_id: int, db: Session = Depends(get_db)) -> dict:
    if db.query(Camera).filter(Camera.id == camera_id).first() is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_manager.start_camera(camera_id)
    return {"message": "Camera started", "camera_id": camera_id}

@router.post("/{camera_id}/stop")
def stop_camera(camera_id: int, db: Session = Depends(get_db)) -> dict:
    if db.query(Camera).filter(Camera.id == camera_id).first() is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_manager.stop_camera(camera_id)
    return {"message": "Camera stopped", "camera_id": camera_id}

@router.delete("/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_db)) -> dict:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    name = camera.name
    db.delete(camera)
    db.commit()

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

