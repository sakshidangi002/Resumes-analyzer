"""
camera_service.py
=================
Production-ready CCTV camera manager for Hikvision DVR.

Supports both RTSP streams (via OpenCV) and HCNetSDK (direct DVR connection).

Architecture
------------
  CameraManager            – singleton, manages N cameras
    └── CameraWorker       – one per camera; runs two daemon threads:
          ├── StreamThread – opens RTSP / USB, reads frames continuously
          └── RecognitionThread – picks latest frame, runs face recognition

    OR (for HCNetSDK):
    └── HCNetSDKCameraWorker – uses HCNetSDK for direct DVR connection
          ├── DVR Login
          ├── Live Preview
          ├── Stream Callback
          └── PlayCtrl Decoder

Design decisions
----------------
* Persistent VideoCapture – one capture per camera, NOT per request.
* Auto-reconnect with exponential back-off (2 s → 30 s max).
* Stale-frame watchdog – if no frame for 15 s, forces reconnect.
* Frame buffer protected by threading.Lock – recognition thread always
  gets the latest JPEG without blocking the stream reader.
* camera_purpose ("IN"/"OUT") is forwarded to the recognition service
  so the correct attendance event type is forced, regardless of any
  previous event for that employee.
* Environment variable CCTV_FFMPEG_OPTS allows passing extra FFmpeg
  options for network tuning without code changes.
* HCNetSDK support for direct DVR connection when source_type="hcnetsdk"
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# Import HCNetSDK components (will be used when source_type="hcnetsdk")
try:
    from app.services.hcnetsdk_camera import HCNetSDKCameraWorker
    HCNETSDK_AVAILABLE = True
except ImportError:
    HCNETSDK_AVAILABLE = False
    logger.warning("HCNetSDK camera module not available")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants (all overridable via env vars)
# ---------------------------------------------------------------------------
_RECONNECT_INIT_DELAY = float(os.getenv("CCTV_RECONNECT_INIT", "2.0"))   # seconds
_RECONNECT_MAX_DELAY  = float(os.getenv("CCTV_RECONNECT_MAX",  "30.0"))
_STALE_TIMEOUT        = float(os.getenv("CCTV_STALE_TIMEOUT",  "15.0"))  # force reconnect
_OPEN_TIMEOUT_MS      = int(os.getenv("CCTV_OPEN_TIMEOUT_MS",  "10000")) # 10 s
_READ_TIMEOUT_MS      = int(os.getenv("CCTV_READ_TIMEOUT_MS",  "5000"))  # 5 s
_JPEG_QUALITY         = int(os.getenv("CCTV_JPEG_QUALITY",     "80"))
_FPS_WINDOW           = 30  # frames used to compute rolling FPS


# ---------------------------------------------------------------------------
# HCNetSDK Configuration Parser
# ---------------------------------------------------------------------------
def parse_hcnetsdk_config(source_url: str) -> dict:
    """Parse HCNetSDK configuration from source_url.
    
    Expected format: hcnetsdk://ip:port@username:password?channel=X
    Example: hcnetsdk://192.168.1.100:8000@admin:password123?channel=1
    
    Returns dict with keys: dvr_ip, dvr_port, dvr_username, dvr_password, dvr_channel
    """
    try:
        # Pattern: hcnetsdk://ip:port@username:password?channel=X
        pattern = r'hcnetsdk://([^:]+):(\d+)@([^:]+):([^?]+)\?channel=(\d+)'
        match = re.match(pattern, source_url)
        
        if match:
            return {
                "dvr_ip": match.group(1),
                "dvr_port": int(match.group(2)),
                "dvr_username": match.group(3),
                "dvr_password": match.group(4),
                "dvr_channel": int(match.group(5)),
            }
        else:
            logger.error(f"Invalid HCNetSDK URL format: {source_url}")
            return None
    except Exception as e:
        logger.error(f"Error parsing HCNetSDK config: {e}")
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CameraRuntimeState:
    """Live runtime metrics for a single camera."""
    status: str = "stopped"             # stopped | connecting | running | reconnecting | error
    last_error: Optional[str] = None
    total_frames: int = 0
    reconnect_count: int = 0
    last_frame_time: float = 0.0        # time.time() of last successful frame
    updated_at: float = 0.0
    fps: float = 0.0
    latest_jpeg: Optional[bytes] = None
    latest_result: dict = field(default_factory=dict)
    _fps_ts: deque = field(default_factory=lambda: deque(maxlen=_FPS_WINDOW))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _draw_boxes(frame: np.ndarray, result: dict) -> np.ndarray:
    """Overlay recognition results on frame."""
    annotated = frame.copy()
    for face in result.get("faces", []):
        box = face.get("box") or []
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        matched = bool(face.get("matched"))
        state   = face.get("state", "")

        if state == "cooldown":
            color = (255, 165, 0)   # orange
        elif state in ("in", "out"):
            color = (34, 197, 94)   # green
        elif matched:
            color = (34, 197, 94)
        else:
            color = (68, 68, 239)   # blue for unknown

        label = face.get("employee_name") or "Unknown"
        score = face.get("score")
        if isinstance(score, (float, int)):
            label = f"{label} {score:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_bg_x2 = x1 + max(120, len(label) * 9)
        cv2.rectangle(annotated, (x1, max(0, y1 - 24)), (text_bg_x2, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _open_capture(stream_url: str, source_type: str, camera_id: int) -> cv2.VideoCapture:
    """Open a VideoCapture with appropriate backend and timeouts."""
    source = stream_url.strip()
    logger.info("Camera %s: Opening stream: %s", camera_id, source)

    if source_type == "usb" or source.isdigit():
        logger.info("Camera %s: USB/webcam mode, index=%s", camera_id, source)
        return cv2.VideoCapture(int(source))

    # RTSP / HTTP – use FFmpeg backend with Hikvision-compatible options
    # These options help handle non-standard H.264 encoding from older DVRs
    # Append FFmpeg options to the URL for older OpenCV versions
    ffmpeg_options = {
        'rtsp_transport': 'tcp',  # Use TCP instead of UDP for reliability
        'fflags': 'nobuffer',     # Disable buffering
        'flags': 'low_delay',     # Low latency mode
        'rtsp_flags': 'prefer_tcp',  # Prefer TCP for RTSP
        'analyzeduration': '5000000',  # Analyze 5 seconds of stream for better SPS/PPS detection
        'probesize': '5000000',   # Probe 5 MB of stream
        'max_delay': '0',         # No delay
    }
    
    # Build FFmpeg options string and append to URL
    options_str = '&'.join([f'{k}={v}' for k, v in ffmpeg_options.items()])
    source_with_options = f"{source}?{options_str}"
    
    cap = cv2.VideoCapture(source_with_options, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _OPEN_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, _READ_TIMEOUT_MS)
    
    logger.info(f"Camera {camera_id}: Using FFmpeg options: {options_str}")
    return cap


def _check_ffmpeg() -> bool:
    """Return True if OpenCV was built with FFmpeg support."""
    build = cv2.getBuildInformation()
    ok = "FFmpeg" in build
    if not ok:
        logger.error(
            "CRITICAL: OpenCV is built WITHOUT FFmpeg – RTSP streams will NOT work! "
            "Fix: run fix_opencv.bat to reinstall opencv-python (full build)."
        )
    return ok


def _is_blurry(frame: np.ndarray, threshold: float = 80.0) -> bool:
    """Return True when the frame is too blurry for reliable recognition."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


# ---------------------------------------------------------------------------
# StreamThread – reads frames from RTSP continuously
# ---------------------------------------------------------------------------
class _StreamThread(threading.Thread):
    def __init__(self, worker: "CameraWorker") -> None:
        super().__init__(daemon=True, name=f"stream-{worker.camera_id}")
        self._w = worker
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        w = self._w
        cap: Optional[cv2.VideoCapture] = None
        reconnect_delay = _RECONNECT_INIT_DELAY
        consecutive_failures = 0

        logger.info("Camera %s [%s]: Stream thread started", w.camera_id, w.name)

        while not self._stop_evt.is_set():
            # ── connect ────────────────────────────────────────────────────
            if cap is None or not cap.isOpened():
                w.state.status = "connecting"
                logger.warning(
                    "Camera %s [%s]: Connecting (attempt %d) ...",
                    w.camera_id, w.name, w.state.reconnect_count + 1,
                )
                try:
                    cap = _open_capture(w.stream_url, w.source_type, w.camera_id)
                except Exception as exc:
                    logger.error("Camera %s: Open exception: %s", w.camera_id, exc)
                    cap = None

                if cap is None or not cap.isOpened():
                    consecutive_failures += 1
                    w.state.reconnect_count += 1
                    w.state.status = "error"
                    w.state.last_error = (
                        f"Cannot open stream (attempt {w.state.reconnect_count}). "
                        "Check DVR IP, RTSP port, credentials, and H.264 codec."
                    )
                    logger.error("Camera %s: %s", w.camera_id, w.state.last_error)
                    reconnect_delay = min(reconnect_delay * 1.5, _RECONNECT_MAX_DELAY)
                    logger.info(
                        "Camera %s: Retrying in %.1fs", w.camera_id, reconnect_delay
                    )
                    self._stop_evt.wait(reconnect_delay)
                    continue

                consecutive_failures = 0
                reconnect_delay = _RECONNECT_INIT_DELAY
                w.state.status = "running"
                w.state.last_error = None
                logger.info("Camera %s [%s]: Connected successfully", w.camera_id, w.name)

            # ── stale watchdog ─────────────────────────────────────────────
            if (
                w.state.last_frame_time > 0
                and time.time() - w.state.last_frame_time > _STALE_TIMEOUT
            ):
                logger.warning(
                    "Camera %s: No frame for %.0fs – forcing reconnect",
                    w.camera_id, _STALE_TIMEOUT,
                )
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = None
                w.state.status = "reconnecting"
                continue

            # ── read frame ─────────────────────────────────────────────────
            try:
                ok, frame = cap.read()
            except Exception as exc:
                logger.warning("Camera %s: Read exception: %s", w.camera_id, exc)
                ok, frame = False, None

            if not ok or frame is None:
                consecutive_failures += 1
                logger.warning(
                    "Camera %s: Frame read failed (%d consecutive)",
                    w.camera_id, consecutive_failures,
                )
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = None
                w.state.status = "reconnecting"
                if consecutive_failures > 3:
                    reconnect_delay = min(reconnect_delay * 2, _RECONNECT_MAX_DELAY)
                self._stop_evt.wait(reconnect_delay)
                continue

            # ── frame received ─────────────────────────────────────────────
            consecutive_failures = 0
            now = time.time()
            w.state.total_frames += 1
            w.state.last_frame_time = now
            w.state.status = "running"

            # Rolling FPS calculation
            w.state._fps_ts.append(now)
            if len(w.state._fps_ts) >= 2:
                span = w.state._fps_ts[-1] - w.state._fps_ts[0]
                w.state.fps = round((len(w.state._fps_ts) - 1) / span, 1) if span > 0 else 0.0

            # Store latest frame for RecognitionThread
            with w._frame_lock:
                w._latest_frame = frame.copy()

            if w.state.total_frames % 200 == 0:
                logger.info(
                    "Camera %s: %d frames, %.1f FPS, reconnects=%d",
                    w.camera_id, w.state.total_frames, w.state.fps,
                    w.state.reconnect_count,
                )

        # cleanup
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        logger.info(
            "Camera %s: Stream thread stopped. frames=%d reconnects=%d",
            w.camera_id, w.state.total_frames, w.state.reconnect_count,
        )


# ---------------------------------------------------------------------------
# RecognitionThread – face detection on latest frame
# ---------------------------------------------------------------------------
class _RecognitionThread(threading.Thread):
    def __init__(self, worker: "CameraWorker") -> None:
        super().__init__(daemon=True, name=f"recog-{worker.camera_id}")
        self._w = worker
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        w = self._w
        logger.info("Camera %s: Recognition thread started", w.camera_id)

        while not self._stop_evt.is_set():
            self._stop_evt.wait(max(0.5, float(w.interval_sec)))
            if self._stop_evt.is_set():
                break

            with w._frame_lock:
                frame = w._latest_frame

            if frame is None:
                continue

            if _is_blurry(frame):
                logger.debug("Camera %s: Skipping blurry frame", w.camera_id)
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Import here to avoid circular imports at module load
                from app.services.recognition import recognize_from_rgb

                result = recognize_from_rgb(
                    rgb,
                    threshold=w.threshold,
                    source="cctv",
                    camera_id=str(w.camera_id),
                    camera_purpose=w.camera_purpose,
                )

                # Annotate frame and encode JPEG for preview
                annotated = _draw_boxes(frame, result)
                ok_enc, jpeg_buf = cv2.imencode(
                    ".jpg", annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
                )
                with w._frame_lock:
                    if ok_enc:
                        w.state.latest_jpeg = jpeg_buf.tobytes()
                    w.state.latest_result = result
                    w.state.updated_at = time.time()

                faces = result.get("faces", [])
                matched = [f for f in faces if f.get("matched")]
                if matched:
                    logger.info(
                        "Camera %s [%s]: %d face(s) matched – %s",
                        w.camera_id, w.camera_purpose,
                        len(matched),
                        ", ".join(f.get("employee_name", "?") for f in matched),
                    )
                elif faces:
                    logger.debug("Camera %s: %d unknown face(s)", w.camera_id, len(faces))

            except Exception as exc:
                logger.error(
                    "Camera %s: Recognition error: %s", w.camera_id, exc, exc_info=True
                )

        logger.info("Camera %s: Recognition thread stopped", w.camera_id)


# ---------------------------------------------------------------------------
# CameraWorker – owns one StreamThread + one RecognitionThread
# ---------------------------------------------------------------------------
class CameraWorker:
    """Manages a single camera's stream + recognition lifecycle."""

    def __init__(
        self,
        *,
        camera_id: int,
        name: str,
        source_url: str,  # Database uses source_url
        source_type: str,
        camera_purpose: str,
        threshold: float,
        interval_sec: float,
    ) -> None:
        self.camera_id = camera_id
        self.name = name
        self.stream_url = source_url  # Keep as stream_url internally for consistency
        self.source_url = source_url  # Database field
        self.source_type = source_type
        self.camera_purpose = camera_purpose.upper()   # "IN" | "OUT"
        self.threshold = threshold
        self.interval_sec = interval_sec

        self.state = CameraRuntimeState()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        self._stream_thread: Optional[_StreamThread] = None
        self._recog_thread: Optional[_RecognitionThread] = None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._stream_thread and self._stream_thread.is_alive():
            return
        self._stream_thread = _StreamThread(self)
        self._recog_thread  = _RecognitionThread(self)
        self._stream_thread.start()
        self._recog_thread.start()
        logger.info(
            "Camera %s [%s]: Worker started (purpose=%s url=%s)",
            self.camera_id, self.name, self.camera_purpose, self.stream_url,
        )

    def stop(self) -> None:
        if self._stream_thread:
            self._stream_thread.stop()
        if self._recog_thread:
            self._recog_thread.stop()
        if self._stream_thread:
            self._stream_thread.join(timeout=6)
        if self._recog_thread:
            self._recog_thread.join(timeout=6)
        self.state.status = "stopped"
        logger.info("Camera %s [%s]: Worker stopped", self.camera_id, self.name)

    def restart(self) -> None:
        logger.info("Camera %s: Restarting...", self.camera_id)
        self.stop()
        self.state = CameraRuntimeState()
        with self._frame_lock:
            self._latest_frame = None
        self.start()
    
    def is_alive(self) -> bool:
        """Check if the worker is alive."""
        return self._stream_thread is not None and self._stream_thread.is_alive()

    # ── query helpers ───────────────────────────────────────────────────────
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame as numpy array."""
        with self._frame_lock:
            return self._latest_frame
    
    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self.state.latest_jpeg

    def serialize_state(self) -> dict:
        s = self.state
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "location": getattr(self, "location", None),
            "stream_url": self.stream_url,
            "source_type": self.source_type,
            "camera_purpose": self.camera_purpose,
            "threshold": self.threshold,
            "interval_sec": self.interval_sec,
            "status": s.status,
            "last_error": s.last_error,
            "fps": s.fps,
            "total_frames": s.total_frames,
            "reconnect_count": s.reconnect_count,
            "last_frame_time": s.last_frame_time,
            "updated_at": s.updated_at,
            "last_result_faces": len(s.latest_result.get("faces", [])),
        }


# ---------------------------------------------------------------------------
# CameraManager – singleton
# ---------------------------------------------------------------------------
class CameraManager:
    """Thread-safe manager for all active camera workers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers: dict[int, CameraWorker] = {}
        self._ffmpeg_ok: Optional[bool] = None

    # ── startup / shutdown ──────────────────────────────────────────────────
    def start_all_from_db(self) -> None:
        """Load enabled cameras from PostgreSQL and start workers.
        Called once from the FastAPI lifespan on application start.
        """
        self._ffmpeg_ok = _check_ffmpeg()
        if not self._ffmpeg_ok:
            logger.error(
                "Camera startup skipped: OpenCV lacks FFmpeg support. "
                "Run fix_opencv.bat then restart the server."
            )
            return

        try:
            from app.db.session import SessionLocal
            from app.models.camera import CameraConfig

            with SessionLocal() as db:
                rows = db.query(CameraConfig).filter(CameraConfig.enabled.is_(True)).all()

            logger.info("Camera startup: %d enabled camera(s) found", len(rows))
            for row in rows:
                self._start_worker_from_model(row)
        except Exception:
            logger.exception("Camera startup: failed to load cameras from DB")

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for w in workers:
            w.stop()
        logger.info("CameraManager: all cameras stopped")

    def _start_worker_from_model(self, model) -> None:
        source_type = (model.source_type or "rtsp").lower()
        
        # Use HCNetSDK if source_type is "hcnetsdk"
        if source_type == "hcnetsdk":
            if not HCNETSDK_AVAILABLE:
                logger.error(
                    f"Camera {model.id}: HCNetSDK requested but not available, "
                    "falling back to RTSP"
                )
                source_type = "rtsp"
            else:
                config = parse_hcnetsdk_config(model.source_url)
                if not config:
                    logger.error(
                        f"Camera {model.id}: Failed to parse HCNetSDK config from {model.source_url}"
                    )
                    return
                
                worker = HCNetSDKCameraWorker(
                    camera_id=model.id,
                    name=model.name,
                    dvr_ip=config["dvr_ip"],
                    dvr_port=config["dvr_port"],
                    dvr_username=config["dvr_username"],
                    dvr_password=config["dvr_password"],
                    dvr_channel=config["dvr_channel"],
                    camera_purpose=model.camera_purpose or "IN",
                    threshold=float(model.threshold),
                    interval_sec=float(model.interval_sec),
                )
                worker.location = getattr(model, "location", None)
                with self._lock:
                    old = self._workers.pop(model.id, None)
                    if old:
                        old.stop()
                    self._workers[model.id] = worker
                worker.start()
                return
        
        # Use RTSP/USB for other source types
        worker = CameraWorker(
            camera_id=model.id,
            name=model.name,
            source_url=model.source_url,  # Database uses source_url
            source_type=source_type,
            camera_purpose=model.camera_purpose or "IN",
            threshold=float(model.threshold),
            interval_sec=float(model.interval_sec),
        )
        worker.location = getattr(model, "location", None)
        with self._lock:
            old = self._workers.pop(model.id, None)
            if old:
                old.stop()
            self._workers[model.id] = worker
        worker.start()

    # ── CRUD operations ─────────────────────────────────────────────────────
    def add_camera(self, camera_id: int) -> None:
        """Start a camera that was just added/enabled in the DB."""
        try:
            from app.db.session import SessionLocal
            from app.models.camera import CameraConfig

            with SessionLocal() as db:
                model = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()

            if not model:
                logger.warning("add_camera: camera_id=%d not found in DB", camera_id)
                return
            self._start_worker_from_model(model)
        except Exception:
            logger.exception("add_camera failed for camera_id=%d", camera_id)

    def remove_camera(self, camera_id: int) -> None:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()
            logger.info("CameraManager: camera %d removed", camera_id)

    def restart_camera(self, camera_id: int) -> bool:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            worker.restart()
            return True
        # Not running yet – try to start from DB
        self.add_camera(camera_id)
        return True

    # ── queries ─────────────────────────────────────────────────────────────
    def get_latest_jpeg(self, camera_id: int) -> Optional[bytes]:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            # Handle both CameraWorker and HCNetSDKCameraWorker
            if hasattr(worker, 'get_latest_jpeg'):
                return worker.get_latest_jpeg()
            elif hasattr(worker, 'state') and hasattr(worker.state, 'latest_jpeg'):
                return worker.state.latest_jpeg
        return None

    def get_status(self, camera_id: int) -> Optional[dict]:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            # Handle both CameraWorker and HCNetSDKCameraWorker
            if hasattr(worker, 'serialize_state'):
                return worker.serialize_state()
            elif hasattr(worker, 'state'):
                # Fallback for HCNetSDKCameraWorker
                return {
                    "camera_id": worker.camera_id,
                    "name": worker.name,
                    "status": worker.state.status,
                    "last_error": worker.state.last_error,
                    "fps": worker.state.fps,
                    "total_frames": worker.state.total_frames,
                    "reconnect_count": worker.state.reconnect_count,
                    "last_frame_time": worker.state.last_frame_time,
                }
        return None

    def list_statuses(self) -> list[dict]:
        with self._lock:
            workers = list(self._workers.values())
        statuses = []
        for w in workers:
            if hasattr(w, 'serialize_state'):
                statuses.append(w.serialize_state())
            elif hasattr(w, 'state'):
                # Fallback for HCNetSDKCameraWorker
                statuses.append({
                    "camera_id": w.camera_id,
                    "name": w.name,
                    "status": w.state.status,
                    "last_error": w.state.last_error,
                    "fps": w.state.fps,
                    "total_frames": w.state.total_frames,
                    "reconnect_count": w.state.reconnect_count,
                    "last_frame_time": w.state.last_frame_time,
                })
        return statuses

    def get_stats(self) -> dict:
        with self._lock:
            workers = list(self._workers.values())
        total    = len(workers)
        running  = sum(1 for w in workers if w.state.status == "running")
        error    = sum(1 for w in workers if w.state.status == "error")
        frames   = sum(w.state.total_frames for w in workers)
        reconnects = sum(w.state.reconnect_count for w in workers)
        return {
            "ffmpeg_ok": self._ffmpeg_ok,
            "total_cameras": total,
            "running_cameras": running,
            "error_cameras": error,
            "total_frames_processed": frames,
            "total_reconnects": reconnects,
        }

    def is_ffmpeg_ok(self) -> bool:
        if self._ffmpeg_ok is None:
            self._ffmpeg_ok = _check_ffmpeg()
        return bool(self._ffmpeg_ok)


# ---------------------------------------------------------------------------
# Module-level singleton used by all API routes and lifespan hooks
# ---------------------------------------------------------------------------
camera_manager = CameraManager()
