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
import socket
from urllib.parse import urlparse
from typing import Optional

from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Response, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_user,
    require_media_access,
    require_media_or_bearer,
    require_roles,
)
from app.core.net import is_allowed_camera_host_ip
from app.core.security import MEDIA_TOKEN_TTL_SECONDS, create_media_token
from app.db.session import get_db
from app.models.camera import CameraConfig
from app.services.camera_service import (
    _redact_url,
    camera_manager,
    open_capture_with_timeout,
)
from app.services.hikvision_discovery import discover_cameras
from app.services.dvr_manager import get_dvr_manager
from app.services.audit_service import log_audit

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
    camera_purpose: str = Field(default="IN", pattern="^(IN|OUT|MONITOR)$")
    threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    interval_sec: float = Field(default=2.0, ge=0.5, le=60.0)
    frame_skip: int = Field(default=0, ge=0, le=10)
    tracking_max_distance: float = Field(default=100.0, ge=10.0, le=500.0)
    tracking_cooldown: float = Field(default=3.0, ge=0.5, le=30.0)
    enabled: bool = False
    # Doorway line crossing. Fully implemented in person_tracker.check_line_crossing
    # and honoured by CameraWorker, but previously absent from every request
    # schema — so the feature could only be enabled with a manual UPDATE against
    # the cameras table.
    crossing_enabled: bool = False
    line_orientation: str = Field(default="horizontal", pattern="^(horizontal|vertical)$")
    line_position: float = Field(default=0.5, ge=0.0, le=1.0)
    entry_direction: str = Field(default="down", pattern="^(up|down|left|right)$")


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
    camera_purpose: Optional[str] = Field(None, pattern="^(IN|OUT|MONITOR)$")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    interval_sec: Optional[float] = Field(None, ge=0.5, le=60.0)
    frame_skip: Optional[int] = Field(None, ge=0, le=10)
    tracking_max_distance: Optional[float] = Field(None, ge=10.0, le=500.0)
    tracking_cooldown: Optional[float] = Field(None, ge=0.5, le=30.0)
    enabled: Optional[bool] = None
    # See CameraCreateRequest — these were configurable only via direct SQL.
    crossing_enabled: Optional[bool] = None
    line_orientation: Optional[str] = Field(None, pattern="^(horizontal|vertical)$")
    line_position: Optional[float] = Field(None, ge=0.0, le=1.0)
    entry_direction: Optional[str] = Field(None, pattern="^(up|down|left|right)$")


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
        # NOTE: `stream_url` is the RAW url, password included, because the edit
        # form parses it (parseRtspUrl) and rebuilds it on save (buildRtspUrl) —
        # redacting it here would write "***" back as the password and break the
        # stream. Anything that merely DISPLAYS the url must use
        # `stream_url_display` instead; the camera cards were rendering the raw
        # value, so the DVR password was visible on screen.
        "stream_url": cam.source_url,  # Map source_url to stream_url for API consistency
        "stream_url_display": _redact_url(cam.source_url),
        "source_type": cam.source_type,
        "camera_purpose": cam.camera_purpose,
        "threshold": cam.threshold,
        "interval_sec": cam.interval_sec,
        "enabled": cam.enabled,
        # Round-trip the tuning + line-crossing config so the edit form can
        # show what is actually stored. These were writable-but-invisible
        # (and, for frame_skip and the tracking pair, not even written).
        "frame_skip": cam.frame_skip,
        "tracking_max_distance": cam.tracking_max_distance,
        "tracking_cooldown": cam.tracking_cooldown,
        "crossing_enabled": cam.crossing_enabled,
        "line_orientation": cam.line_orientation,
        "line_position": cam.line_position,
        "entry_direction": cam.entry_direction,
        "created_at": cam.created_at.isoformat() if cam.created_at else None,
        "updated_at": cam.updated_at.isoformat() if cam.updated_at else None,
    }
    if live:
        base["live"] = live
    else:
        base["live"] = None
    return base


def _validate_camera_source(source_url: str, source_type: str) -> str:
    """Reject camera URLs that could make the server fetch public/metadata hosts."""
    value = source_url.strip()
    if source_type == "usb" or value.isdigit():
        return value
    if source_type == "hcnetsdk":
        parsed = urlparse(value)
        if parsed.scheme != "hcnetsdk" or not parsed.hostname:
            raise HTTPException(status_code=422, detail="Invalid HCNetSDK camera source.")
        host = parsed.hostname
    else:
        parsed = urlparse(value)
        if parsed.scheme not in {"rtsp", "rtsps", "http", "https"} or not parsed.hostname:
            raise HTTPException(status_code=422, detail="Camera source must use a supported URL scheme.")
        host = parsed.hostname
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
        for address in addresses:
            # Rule lives in app/core/net.py so it is unit-testable without
            # importing the vision stack. See it for why link-local matters.
            if not is_allowed_camera_host_ip(address):
                # Name the host and what it resolved to. "Camera sources must
                # resolve to a private network" alone gives the operator no way
                # to tell WHICH part of the URL was wrong — and a typo in the
                # host is the most common cause.
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Camera sources must be on a private network. "
                        f"'{host}' resolves to {address}, which is public. "
                        f"Use the camera's LAN address (e.g. 192.168.x.x)."
                    ),
                )
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Camera source hostname '{host}' could not be resolved. "
                f"Check for a typo, or use the camera's IP address directly."
            ),
        ) from exc
    return value


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/cameras/stats", tags=["cameras"])
def get_camera_stats(current_user=Depends(get_current_user)):
    """Return global camera system statistics."""
    return camera_manager.get_stats()


@router.get("/corridor/summary", tags=["cameras"])
def corridor_summary_endpoint(
    day: Optional[date] = Query(default=None, description="IST date; defaults to today"),
    current_user=Depends(get_current_user),
):
    """How many people passed through the corridor today, and how many were named.

    in_count = employees_in + unknown_in. Counting depends on detection and
    tracking only; recognition decides identity, never whether the person is
    counted. A transit nobody could name is Unknown, not absent.
    """
    from app.services.corridor_counts import corridor_summary

    return corridor_summary(day)


@router.get("/corridor/events", tags=["cameras"])
def corridor_events_endpoint(
    day: Optional[date] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    current_user=Depends(get_current_user),
):
    """Individual corridor transits, newest first.

    `identity` is null for an unknown transit and is never filled with a
    best-guess employee.
    """
    from app.services.corridor_counts import corridor_events

    return {"events": corridor_events(day, limit=limit)}


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

    # Bounded open: cv2.VideoCapture() blocks synchronously on the connect, so
    # CAP_PROP_OPEN_TIMEOUT_MSEC (previously set afterwards) never bounded it
    # and an unreachable DVR held this request worker for FFmpeg's default.
    cap = open_capture_with_timeout(
        stream_url, source_type, camera_id=0, timeout_sec=12.0,
    )
    if cap is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not open stream within 12s. "
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
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Add a new camera. If enabled=True, stream starts immediately."""
    try:
        # Log the payload with the source URL redacted — it carries the DVR
        # password, and this line ran on every create.
        _safe_payload = payload.model_dump()
        _safe_payload["source_url"] = _redact_url(_safe_payload.get("source_url", ""))
        logger.info("Creating camera with payload: %s", _safe_payload)


        # Validate payload manually for debugging
        if not payload.name or len(payload.name) < 1:
            raise ValueError("Camera name is required")
        if not payload.source_url or len(payload.source_url) < 1:
            raise ValueError("Source URL is required")
        if payload.source_type not in ["rtsp", "usb", "http", "hcnetsdk"]:
            raise ValueError(f"Invalid source_type: {payload.source_type}")
        if payload.camera_purpose not in ["IN", "OUT", "MONITOR"]:
            raise ValueError(f"Invalid camera_purpose: {payload.camera_purpose}")
        
        source_url = _validate_camera_source(payload.source_url, payload.source_type)

        # Build from the validated payload rather than naming fields by hand.
        # The hand-written version listed only 8 of them, so frame_skip,
        # tracking_max_distance and tracking_cooldown were accepted by the API,
        # returned in the 200 response, and silently discarded. Constructing
        # from model_dump() means a schema field with no column raises here
        # instead of vanishing.
        data = payload.model_dump()
        data["source_url"] = source_url
        # `camera_type` is a legacy NOT NULL column kept in step with
        # camera_purpose until it can be dropped (see CCTV_REVIEW.md §9).
        data["camera_type"] = data["camera_purpose"]
        cam = CameraConfig(**data)
        db.add(cam)
        db.commit()
        db.refresh(cam)
        log_audit(db, current_user.id, "CAMERA_CREATED", "Camera", str(cam.id), f"Created camera {cam.name}", request.client.host if request.client else None)
        logger.info("Camera created: id=%d name=%s purpose=%s", cam.id, cam.name, cam.camera_purpose)

        if cam.enabled:
            camera_manager.add_camera(cam.id)

        return _serialize_camera(cam, camera_manager.get_status(cam.id))
    except HTTPException:
        # MUST come before `except Exception`. HTTPException subclasses
        # Exception, so the generic handler below was swallowing the precise
        # 422 from _validate_camera_source and re-raising it as a bare 500
        # "Unable to create the camera." The operator was told nothing about
        # WHY — the actual reason ("must resolve to a private network") only
        # ever appeared in the server log.
        raise
    except ValueError as exc:
        logger.error("Camera validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create camera: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to create the camera.") from exc


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
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    cam = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    update_data = payload.model_dump(exclude_unset=True)
    if "source_url" in update_data:
        update_data["source_url"] = _validate_camera_source(update_data["source_url"], update_data.get("source_type", cam.source_type))
    if "camera_purpose" in update_data:
        # Keep the legacy NOT NULL camera_type column in step, or the two
        # disagree and whichever one a future query happens to read wins.
        update_data["camera_type"] = update_data["camera_purpose"]
    was_enabled = cam.enabled
    for field, value in update_data.items():
        setattr(cam, field, value)
    db.commit()
    db.refresh(cam)
    log_audit(db, current_user.id, "CAMERA_UPDATED", "Camera", str(camera_id), "Updated camera configuration", request.client.host if request.client else None)
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
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    cam = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_manager.remove_camera(camera_id)
    db.delete(cam)
    db.commit()
    log_audit(db, current_user.id, "CAMERA_DELETED", "Camera", str(camera_id), "Deleted camera configuration", request.client.host if request.client else None)
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


@router.post("/cameras/media-token", tags=["cameras"])
def issue_camera_media_token(current_user=Depends(require_roles(["Admin", "HR"]))):
    """Mint a short-lived token for <img>-rendered camera media.

    Browsers cannot attach an Authorization header to an <img src>, so the live
    MJPEG feeds and JPEG previews take `?t=<token>` instead. This endpoint is
    the authenticated chokepoint: only an Admin/HR session can obtain one, and
    what it grants expires in ~2 minutes and covers camera media only.
    """
    return {
        "token": create_media_token(current_user.id, [r.name for r in current_user.roles]),
        "expires_in": MEDIA_TOKEN_TTL_SECONDS,
    }


@router.get("/cameras/{camera_id}/preview", tags=["cameras"])
def get_camera_preview(
    camera_id: int,
    db: Session = Depends(get_db),
    _auth=Depends(require_media_or_bearer),
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


@router.get("/cameras/{camera_id}/occupancy", tags=["cameras"])
def get_camera_occupancy(
    camera_id: int,
    current_user=Depends(get_current_user),
):
    """Chair occupancy for a room camera, for drawing on the live feed.

    Computed by the CCTV V2 occupancy layer from the person boxes V1 HAS
    ALREADY produced -- see app/cctv_v2/pipeline/v1_bridge.py. No extra RTSP
    connection and NO ADDITIONAL INFERENCE: this reads a list that is already in
    memory, so polling it costs essentially nothing and cannot slow the feed.

    Coordinates are NORMALISED 0..1, with the frame size alongside them, so the
    caller can scale the overlay to however the video is being displayed. Pixel
    coordinates would be wrong the moment the <img> is resized.

    POLLING THIS IS FREE, AND ALSO INERT. Calling it more often does not make
    occupancy change faster: the smoothing advances once per completed ANALYSIS
    PASS, not once per request. It used to advance per request, which meant the
    dashboard's 2s poll spent three "consecutive observations" on a single pass
    and chairs flipped state while nobody moved.

    Four freshness fields, because they answer different questions:

        observation_updated_at   when V1 last completed an analysis pass
        observation_age_sec      how old that pass is now
        state_updated_at         when a chair last actually CHANGED
        from_new_observation     whether THIS response folded in a new pass

    A room settled for ten minutes has a FRESH observation and an OLD state
    change. Collapsing the two is how a steady answer gets read as a stuck one.

    HOW FAST CAN IT CHANGE? Not faster than
    `confirmations x observation interval`. Measured on this deployment from
    the app's own PERF log (n=2572 passes), a room camera's observation interval
    is 8.1s median, so a seat becomes FREE about 40s after its occupant leaves.
    Polling harder cannot improve that and never could; the levers are the
    confirmation counts (bounded by how long the detector loses a seated
    person -- see geometry.py) and the pass rate (bounded by CPU).

    `chairs` are the seats THIS CAMERA OWNS. `observed_elsewhere` are seats it
    can see but another camera controls -- a chair has exactly one owner, so the
    two cameras' `chairs_total` must never be added. `room_chairs_total` is the
    room's real count, each chair once.

    404 when V1 is not running this camera -- a different thing from an empty
    room, and a dashboard must not show them identically.
    """
    from app.cctv_v2.pipeline.v1_bridge import occupancy_snapshot

    data = occupancy_snapshot(camera_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"camera {camera_id} is not running; no occupancy available",
        )
    return data


@router.get("/cameras/{camera_id}/stream.mjpg", tags=["cameras"])
async def stream_camera_mjpeg(
    camera_id: int,
    request: Request,
    _auth=Depends(require_media_access),
):
    """Continuous MJPEG stream of the annotated live feed.

    Rendered directly by an <img> tag in the browser, so the video plays at the
    backend display FPS with no client-side polling. <img> cannot send a Bearer
    header, so this is gated by a short-lived `?t=` media token from
    POST /api/cameras/media-token rather than being left open.
    """
    async def generate_frames():
        # ~30 FPS ceiling; the display thread produces frames at CCTV_DISPLAY_FPS.
        frame_period = 1.0 / 30.0
        last_sent = None
        while True:
            # Stop as soon as the browser navigates away, so the connection and
            # this task are released promptly (avoids piling up open streams).
            if await request.is_disconnected():
                break
            jpeg = camera_manager.get_latest_jpeg(camera_id)
            if jpeg is not None and jpeg is not last_sent:
                last_sent = jpeg
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(frame_period)

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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
def dvr_connect(payload: DVRConnectRequest, current_user=Depends(get_current_user)):
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
def dvr_disconnect(current_user=Depends(get_current_user)):
    """Disconnect from DVR and stop all streams."""
    dvr_manager = get_dvr_manager()
    dvr_manager.disconnect()
    return {"success": True, "message": "Disconnected"}


@router.get("/dvr/status")
def dvr_status(current_user=Depends(get_current_user)):
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
def dvr_start_camera(channel_id: int, current_user=Depends(get_current_user)):
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
def dvr_stop_camera(channel_id: int, current_user=Depends(get_current_user)):
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
def dvr_set_recognition(channel_id: int, enabled: bool = Query(...), current_user=Depends(get_current_user)):
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
def dvr_start_all(current_user=Depends(get_current_user)):
    """Start streams for all online cameras."""
    dvr_manager = get_dvr_manager()
    
    if not dvr_manager.is_connected():
        raise HTTPException(status_code=400, detail="Not connected to DVR")
    
    started = dvr_manager.start_all_streams()
    return {"success": True, "message": f"Started {started} cameras"}


@router.post("/dvr/cameras/stop-all")
def dvr_stop_all(current_user=Depends(get_current_user)):
    """Stop all camera streams."""
    dvr_manager = get_dvr_manager()
    
    if not dvr_manager.is_connected():
        raise HTTPException(status_code=400, detail="Not connected to DVR")
    
    stopped = dvr_manager.stop_all_streams()
    return {"success": True, "message": f"Stopped {stopped} cameras"}


@router.get("/dvr/cameras/{channel_id}/preview")
def dvr_camera_preview(
    channel_id: int,
    request: Request,
    _auth=Depends(require_media_or_bearer),
):
    """Get latest JPEG frame from DVR camera."""
    dvr_manager = get_dvr_manager()
    
    if not dvr_manager.is_connected():
        raise HTTPException(status_code=400, detail="Not connected to DVR")
    
    camera_status = dvr_manager.get_camera_status(channel_id)
    if not camera_status:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    if not camera_status.get("worker_status", {}).get("is_alive"):
        raise HTTPException(status_code=400, detail="Camera not streaming")
    
    # Resolve the worker through the manager: it returns whichever worker type
    # this channel uses, both of which implement get_latest_jpeg(). Reaching
    # into .worker / .rtsp_worker here is what produced an AttributeError (500)
    # for HCNetSDK channels, whose worker lacked that method.
    worker = dvr_manager.get_worker(channel_id)
    if worker is None:
        raise HTTPException(status_code=400, detail="Camera not available")

    jpeg = worker.get_latest_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="Stream connected, waiting for first frame")

    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/dvr/cameras/{channel_id}/stream")
async def dvr_camera_stream(
    channel_id: int,
    request: Request,
    _auth=Depends(require_media_access),
):
    """MJPEG streaming endpoint for live video feed.

    Gated by a short-lived `?t=` media token -- see stream_camera_mjpeg.
    """
    dvr_manager = get_dvr_manager()

    if not dvr_manager._connection or not dvr_manager._connection.connected:
        raise HTTPException(status_code=400, detail="Not connected to DVR")

    camera_status = dvr_manager.get_camera_status(channel_id)
    if not camera_status:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not camera_status.get("worker_status", {}).get("is_alive"):
        raise HTTPException(status_code=400, detail="Camera not streaming")

    # Resolve the worker ONCE, up front, so the per-frame loop never has to take
    # the connection lock (which a blocking connect/start could be holding —
    # taking it here would stall the whole event loop).
    worker = dvr_manager.get_worker(channel_id)

    async def generate_frames():
        while worker is not None:
            if await request.is_disconnected():
                break
            jpeg = worker.get_latest_jpeg()
            if jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            await asyncio.sleep(0.1)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
