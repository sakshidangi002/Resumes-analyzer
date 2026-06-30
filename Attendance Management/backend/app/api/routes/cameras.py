"""
Camera management API routes.

Endpoints
---------
GET    /api/cameras                  – list all cameras (DB + live status)
POST   /api/cameras                  – add camera
GET    /api/cameras/{id}             – single camera details
PUT    /api/cameras/{id}             – update camera config
DELETE /api/cameras/{id}             – remove camera
POST   /api/cameras/{id}/start       – enable + start stream
POST   /api/cameras/{id}/stop        – disable + stop stream
POST   /api/cameras/{id}/restart     – force reconnect
GET    /api/cameras/{id}/preview     – latest JPEG frame (binary)
GET    /api/cameras/{id}/status      – live runtime metrics
POST   /api/cameras/test-connection  – test RTSP URL without saving
GET    /api/cameras/stats            – overall system stats
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.camera import CameraConfig
from app.services.camera_service import camera_manager
from app.services.hikvision_discovery import discover_cameras
from app.services.dvr_manager import get_dvr_manager

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CameraCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    location: Optional[str] = None
    source_url: str = Field(..., min_length=1, max_length=500)  # Database uses source_url
    source_type: str = Field(default="rtsp", pattern="^(rtsp|usb|http|hcnetsdk)$")
    camera_purpose: str = Field(default="IN", pattern="^(IN|OUT)$")
    threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    interval_sec: float = Field(default=2.0, ge=0.5, le=60.0)
    enabled: bool = False


class DVRDiscoveryRequest(BaseModel):
    """Request for DVR camera discovery."""
    ip: str = Field(..., min_length=1, max_length=50)
    port: int = Field(default=8000, ge=1, le=65535)
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=100)


class DiscoveredChannelResponse(BaseModel):
    """Response for a discovered channel."""
    id: int
    name: str
    status: str
    channel_type: str
    resolution: Optional[str] = None


class DiscoveredDeviceResponse(BaseModel):
    """Response for a discovered device."""
    model: str
    firmware: str
    serial: str
    total_channels: int
    analog_channels: int
    ip_channels: int
    channels: list[DiscoveredChannelResponse]


class DVRDiscoveryResponse(BaseModel):
    """Response for DVR discovery."""
    success: bool
    device: Optional[DiscoveredDeviceResponse] = None
    error: Optional[str] = None


class CameraUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    location: Optional[str] = None
    source_url: Optional[str] = Field(None, min_length=1, max_length=500)  # Database uses source_url
    source_type: Optional[str] = Field(None, pattern="^(rtsp|usb|http|hcnetsdk)$")
    camera_purpose: Optional[str] = Field(None, pattern="^(IN|OUT)$")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    interval_sec: Optional[float] = Field(None, ge=0.5, le=60.0)
    enabled: Optional[bool] = None


class TestConnectionRequest(BaseModel):
    stream_url: str = Field(..., min_length=1)
    source_type: str = Field(default="rtsp", pattern="^(rtsp|usb|http|hcnetsdk)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_camera(cam: CameraConfig, live: Optional[dict] = None) -> dict:
    base = {
        "id": cam.id,
        "name": cam.name,
        "location": cam.location,
        "stream_url": cam.source_url,  # Map source_url to stream_url for API consistency
        "source_type": cam.source_type,
        "camera_purpose": cam.camera_purpose,
        "threshold": cam.threshold,
        "interval_sec": cam.interval_sec,
        "enabled": cam.enabled,
        "created_at": cam.created_at.isoformat() if cam.created_at else None,
        "updated_at": cam.updated_at.isoformat() if cam.updated_at else None,
    }
    if live:
        base["live"] = live
    else:
        base["live"] = None
    return base


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/cameras/stats", tags=["cameras"])
def get_camera_stats(current_user=Depends(get_current_user)):
    """Return global camera system statistics."""
    return camera_manager.get_stats()


@router.post("/cameras/test-connection", tags=["cameras"])
def test_camera_connection(
    payload: TestConnectionRequest,
    current_user=Depends(get_current_user),
):
    """
    Test an RTSP/stream URL without persisting it.
    Returns stream properties on success or a detailed error on failure.
    """
    if not camera_manager.is_ffmpeg_ok():
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenCV is built without FFmpeg — RTSP streams will NOT work. "
                "Run fix_opencv.bat in the backend directory then restart the server."
            ),
        )

    try:
        import cv2
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="OpenCV not available") from exc

    stream_url = payload.stream_url.strip()
    source_type = payload.source_type

    # Open capture
    try:
        if source_type == "usb" or stream_url.isdigit():
            cap = cv2.VideoCapture(int(stream_url))
        else:
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"VideoCapture exception: {exc}") from exc

    if not cap.isOpened():
        cap.release()
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not open stream. "
                "Check: DVR IP, RTSP port 554, username/password, H.264 codec, "
                "firewall rules, and that the DVR's RTSP service is enabled."
            ),
        )

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise HTTPException(
            status_code=503,
            detail=(
                "Stream opened but could not read a frame. "
                "Camera may be offline, codec unsupported (use H.264 not H.265), "
                "or channel number is wrong."
            ),
        )

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return {
        "success": True,
        "message": "Connection successful",
        "width": width,
        "height": height,
        "fps": fps,
        "stream_url": stream_url,
    }


@router.get("/cameras", tags=["cameras"])
def list_cameras(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Return all cameras from DB merged with live runtime status."""
    cameras = db.query(CameraConfig).order_by(CameraConfig.id).all()
    result = []
    for cam in cameras:
        live = camera_manager.get_status(cam.id)
        result.append(_serialize_camera(cam, live))
    return result


@router.post("/cameras", tags=["cameras"])
def create_camera(
    payload: CameraCreateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Add a new camera. If enabled=True, stream starts immediately."""
    try:
        logger.info(f"Creating camera with payload: {payload.model_dump()}")
        
        # Validate payload manually for debugging
        if not payload.name or len(payload.name) < 1:
            raise ValueError("Camera name is required")
        if not payload.source_url or len(payload.source_url) < 1:
            raise ValueError("Source URL is required")
        if payload.source_type not in ["rtsp", "usb", "http", "hcnetsdk"]:
            raise ValueError(f"Invalid source_type: {payload.source_type}")
        if payload.camera_purpose not in ["IN", "OUT"]:
            raise ValueError(f"Invalid camera_purpose: {payload.camera_purpose}")
        
        cam = CameraConfig(
            name=payload.name,
            location=payload.location,
            source_url=payload.source_url,  # Database uses source_url
            source_type=payload.source_type,
            camera_purpose=payload.camera_purpose,
            threshold=payload.threshold,
            interval_sec=payload.interval_sec,
            enabled=payload.enabled,
        )
        db.add(cam)
        db.commit()
        db.refresh(cam)
        logger.info("Camera created: id=%d name=%s purpose=%s", cam.id, cam.name, cam.camera_purpose)

        if cam.enabled:
            camera_manager.add_camera(cam.id)

        return _serialize_camera(cam, camera_manager.get_status(cam.id))
    except ValueError as exc:
        logger.error(f"Validation error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Failed to create camera: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to create camera: {str(exc)}") from exc


@router.get("/cameras/{camera_id}", tags=["cameras"])
def get_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    cam = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    live = camera_manager.get_status(camera_id)
    return _serialize_camera(cam, live)


@router.put("/cameras/{camera_id}", tags=["cameras"])
def update_camera(
    camera_id: int,
    payload: CameraUpdateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    cam = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    update_data = payload.model_dump(exclude_unset=True)
    was_enabled = cam.enabled
    for field, value in update_data.items():
        setattr(cam, field, value)
    db.commit()
    db.refresh(cam)
    logger.info("Camera updated: id=%d", camera_id)

    # Restart worker to apply config changes
    if cam.enabled:
        camera_manager.add_camera(cam.id)   # replaces existing worker
    elif was_enabled and not cam.enabled:
        camera_manager.remove_camera(cam.id)

    return _serialize_camera(cam, camera_manager.get_status(camera_id))


@router.delete("/cameras/{camera_id}", tags=["cameras"])
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    cam = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_manager.remove_camera(camera_id)
    db.delete(cam)
    db.commit()
    logger.info("Camera deleted: id=%d", camera_id)
    return {"message": f"Camera {camera_id} deleted"}


@router.post("/cameras/{camera_id}/start", tags=["cameras"])
def start_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    cam = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam.enabled = True
    db.commit()
    camera_manager.add_camera(camera_id)
    logger.info("Camera started: id=%d", camera_id)
    return {"message": f"Camera {camera_id} started", "status": "starting"}


@router.post("/cameras/{camera_id}/stop", tags=["cameras"])
def stop_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    cam = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam.enabled = False
    db.commit()
    camera_manager.remove_camera(camera_id)
    logger.info("Camera stopped: id=%d", camera_id)
    return {"message": f"Camera {camera_id} stopped", "status": "stopped"}


@router.post("/cameras/{camera_id}/restart", tags=["cameras"])
def restart_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    cam = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    camera_manager.restart_camera(camera_id)
    logger.info("Camera restarted: id=%d", camera_id)
    return {"message": f"Camera {camera_id} is reconnecting", "status": "reconnecting"}


@router.get("/cameras/{camera_id}/status", tags=["cameras"])
def get_camera_status(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Return live runtime metrics for a camera."""
    cam = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    live = camera_manager.get_status(camera_id)
    if not live:
        return {"camera_id": camera_id, "status": "stopped", "message": "Camera is not running"}
    return live


@router.get("/cameras/{camera_id}/preview", tags=["cameras"])
def get_camera_preview(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Return the latest annotated JPEG frame as binary image/jpeg."""
    cam = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    jpeg = camera_manager.get_latest_jpeg(camera_id)
    if jpeg is None:
        raise HTTPException(
            status_code=503,
            detail="No frame available. Camera may be offline or not yet started.",
        )
    return Response(content=jpeg, media_type="image/jpeg")


@router.post("/dvr/discover", tags=["cameras"])
def discover_dvr_cameras(
    request: DVRDiscoveryRequest,
    current_user=Depends(get_current_user),
):
    """
    Discover cameras on a Hikvision DVR using HCNetSDK.
    
    This endpoint logs into the DVR and retrieves information about all
    available channels, including channel names and status.
    """
    success, discovered, error = discover_cameras(
        ip=request.ip,
        port=request.port,
        username=request.username,
        password=request.password,
    )
    
    if not success:
        return DVRDiscoveryResponse(success=False, error=error)
    
    # Convert discovered device to response format
    device_response = DiscoveredDeviceResponse(
        model=discovered.model,
        firmware=discovered.firmware,
        serial=discovered.serial,
        total_channels=discovered.total_channels,
        analog_channels=discovered.analog_channels,
        ip_channels=discovered.ip_channels,
        channels=[
            DiscoveredChannelResponse(
                id=ch.id,
                name=ch.name,
                status=ch.status,
                channel_type=ch.channel_type,
                resolution=ch.resolution
            )
            for ch in discovered.channels
        ]
    )
    
    return DVRDiscoveryResponse(success=True, device=device_response)


# ---------------------------------------------------------------------------
# DVR Manager API endpoints
# ---------------------------------------------------------------------------

class DVRConnectRequest(BaseModel):
    ip: str = Field(..., min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class DVRConnectResponse(BaseModel):
    success: bool
    message: str
    device_info: Optional[dict] = None
    cameras: Optional[list] = None


@router.post("/dvr/connect")
def dvr_connect(payload: DVRConnectRequest):
    """Connect to DVR and discover cameras."""
    dvr_manager = get_dvr_manager()
    
    logger.info(f"DVR connect request: {payload.ip}:{payload.port}")
    
    success, message, device = dvr_manager.connect(
        ip=payload.ip,
        port=payload.port,
        username=payload.username,
        password=payload.password
    )
    
    logger.info(f"DVR connect result: success={success}, message={message}, device={device}")
    
    if success and device:
        cameras = dvr_manager.get_all_cameras()
        logger.info(f"Returning {len(cameras)} cameras to frontend")
        return DVRConnectResponse(
            success=True,
            message="Connected successfully",
            device_info={
                "model": device.model,
                "firmware": device.firmware,
                "serial": device.serial,
                "total_channels": device.total_channels,
            },
            cameras=cameras
        )
    else:
        logger.error(f"DVR connect failed: {message}")
        return DVRConnectResponse(
            success=False,
            message=message or "Connection failed"
        )


@router.post("/dvr/disconnect")
def dvr_disconnect():
    """Disconnect from DVR and stop all streams."""
    dvr_manager = get_dvr_manager()
    dvr_manager.disconnect()
    return {"success": True, "message": "Disconnected"}


@router.get("/dvr/status")
def dvr_status():
    """Get DVR connection status and all camera statuses."""
    dvr_manager = get_dvr_manager()
    
    logger.info(f"DVR status check: connected={dvr_manager.is_connected()}")
    
    if not dvr_manager.is_connected():
        return {
            "connected": False,
            "cameras": []
        }
    
    try:
        cameras = dvr_manager.get_all_cameras()
        logger.info(f"DVR status returning {len(cameras)} cameras")
        
        return {
            "connected": True,
            "connection_info": dvr_manager.get_connection_info(),
            "cameras": cameras
        }
    except Exception as e:
        logger.exception(f"Error getting DVR camera statuses: {e}")
        return {
            "connected": True,
            "connection_info": dvr_manager.get_connection_info(),
            "cameras": [],
            "error": str(e)
        }


@router.post("/dvr/cameras/{channel_id}/start")
def dvr_start_camera(channel_id: int):
    """Start live stream for a specific camera."""
    dvr_manager = get_dvr_manager()
    
    if not dvr_manager.is_connected():
        raise HTTPException(status_code=400, detail="Not connected to DVR")
    
    success = dvr_manager.start_camera_stream(channel_id)
    
    if success:
        return {"success": True, "message": f"Camera {channel_id} started"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to start camera {channel_id}")


@router.post("/dvr/cameras/{channel_id}/stop")
def dvr_stop_camera(channel_id: int):
    """Stop live stream for a specific camera."""
    dvr_manager = get_dvr_manager()
    
    if not dvr_manager.is_connected():
        raise HTTPException(status_code=400, detail="Not connected to DVR")
    
    success = dvr_manager.stop_camera_stream(channel_id)
    
    if success:
        return {"success": True, "message": f"Camera {channel_id} stopped"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to stop camera {channel_id}")


@router.post("/dvr/cameras/{channel_id}/recognition")
def dvr_set_recognition(channel_id: int, enabled: bool = Query(...)):
    """Enable or disable recognition for a camera."""
    dvr_manager = get_dvr_manager()
    
    if not dvr_manager.is_connected():
        raise HTTPException(status_code=400, detail="Not connected to DVR")
    
    success = dvr_manager.set_recognition_enabled(channel_id, enabled)
    
    if success:
        return {"success": True, "message": f"Camera {channel_id} recognition set to {enabled}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to set recognition for camera {channel_id}")


@router.post("/dvr/cameras/start-all")
def dvr_start_all():
    """Start streams for all online cameras."""
    dvr_manager = get_dvr_manager()
    
    if not dvr_manager.is_connected():
        raise HTTPException(status_code=400, detail="Not connected to DVR")
    
    started = dvr_manager.start_all_streams()
    return {"success": True, "message": f"Started {started} cameras"}


@router.post("/dvr/cameras/stop-all")
def dvr_stop_all():
    """Stop all camera streams."""
    dvr_manager = get_dvr_manager()
    
    if not dvr_manager.is_connected():
        raise HTTPException(status_code=400, detail="Not connected to DVR")
    
    stopped = dvr_manager.stop_all_streams()
    return {"success": True, "message": f"Stopped {stopped} cameras"}


@router.get("/dvr/cameras/{channel_id}/preview")
def dvr_camera_preview(channel_id: int):
    """Get latest JPEG frame from DVR camera."""
    dvr_manager = get_dvr_manager()
    
    if not dvr_manager.is_connected():
        raise HTTPException(status_code=400, detail="Not connected to DVR")
    
    camera_status = dvr_manager.get_camera_status(channel_id)
    if not camera_status:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    if not camera_status.get("worker_status", {}).get("is_alive"):
        raise HTTPException(status_code=400, detail="Camera not streaming")
    
    # Get the camera worker
    with dvr_manager._connection.lock:
        camera = dvr_manager._connection.cameras.get(channel_id)
        if not camera:
            raise HTTPException(status_code=400, detail="Camera not available")
        
        # Get latest frame from worker (HCNetSDK or RTSP)
        frame = None
        if camera.worker:
            frame = camera.worker.get_latest_frame()
        elif camera.rtsp_worker:
            frame = camera.rtsp_worker.get_latest_frame()
        
        if frame is None:
            raise HTTPException(status_code=503, detail="Stream connected, waiting for first frame")
        
        # Convert to JPEG
        import cv2
        _, jpeg = cv2.imencode('.jpg', frame)
        
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.get("/dvr/cameras/{channel_id}/stream")
async def dvr_camera_stream(channel_id: int):
    """MJPEG streaming endpoint for live video feed."""
    import cv2
    import time
    
    dvr_manager = get_dvr_manager()
    
    if not dvr_manager._connection or not dvr_manager._connection.connected:
        raise HTTPException(status_code=400, detail="Not connected to DVR")
    
    camera_status = dvr_manager.get_camera_status(channel_id)
    if not camera_status:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    if not camera_status.get("worker_status", {}).get("is_alive"):
        raise HTTPException(status_code=400, detail="Camera not streaming")
    
    async def generate_frames():
        """Generator function that yields JPEG frames for MJPEG stream."""
        while True:
            with dvr_manager._connection.lock:
                camera = dvr_manager._connection.cameras.get(channel_id)
                if not camera:
                    break
                
                frame = None
                if camera.worker:
                    frame = camera.worker.get_latest_frame()
                elif camera.rtsp_worker:
                    frame = camera.rtsp_worker.get_latest_frame()
            
            if frame is not None:
                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            # Control frame rate (~10 FPS)
            await asyncio.sleep(0.1)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
