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

from app.services.face_tracker import FaceTracker

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
    active_tracks: int = 0  # Number of active face tracks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _draw_enhanced_overlay(
    frame: np.ndarray,
    tracks: list,
    camera_name: str,
    fps: float,
) -> np.ndarray:
    """Enhanced overlay with green/red boxes, labels, confidence, and metadata."""
    annotated = frame.copy()
    
    # Draw face overlays
    for track in tracks:
        display_info = track.get_display_info()
        box = display_info["box"]
        if len(box) < 4:
            continue
        
        x1, y1, x2, y2 = box
        matched = display_info["matched"]
        employee_name = display_info["employee_name"]
        confidence = display_info["confidence"]
        employee_id = display_info["employee_id"]
        employee_code = display_info["employee_code"]
        
        logger.debug(
            "STAGE-overlay track=%s matched=%s employee=%s confidence=%.4f",
            display_info.get("track_id", "unknown"),
            matched,
            employee_name,
            confidence,
        )

        # Color based on recognition status
        if matched:
            color = (34, 197, 94)  # Green for known employees
        else:
            color = (239, 68, 68)  # Red for unknown
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Build label text
        label_lines = [employee_name]
        
        if matched:
            # Add confidence percentage
            confidence_pct = int(confidence * 100)
            label_lines.append(f"Confidence: {confidence_pct}%")
            
            # Add employee ID if available
            if employee_id:
                id_display = employee_code or str(employee_id)
                label_lines.append(f"ID: {id_display}")
        
        # Draw label background and text
        label_height = 24 * len(label_lines)
        text_bg_x2 = x1 + max(180, max(len(line) for line in label_lines) * 9)
        
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - label_height - 4)),
            (text_bg_x2, y1 - 2),
            color,
            -1,
        )
        
        for i, line in enumerate(label_lines):
            y_pos = max(16, y1 - label_height + 4 + i * 24)
            cv2.putText(
                annotated,
                line,
                (x1 + 4, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    
    # Draw camera info overlay (top-left)
    overlay_lines = [
        f"Camera: {camera_name}",
        f"FPS: {fps:.1f}",
        f"Faces: {len(tracks)}",
    ]
    
    # Add timestamp
    from datetime import datetime
    overlay_lines.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Draw overlay background
    overlay_height = 24 * len(overlay_lines) + 8
    overlay_width = 220
    cv2.rectangle(
        annotated,
        (10, 10),
        (10 + overlay_width, 10 + overlay_height),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        annotated,
        (10, 10),
        (10 + overlay_width, 10 + overlay_height),
        (255, 255, 255),
        1,
    )
    
    # Draw overlay text
    for i, line in enumerate(overlay_lines):
        y_pos = 30 + i * 24
        cv2.putText(
            annotated,
            line,
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
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
                from app.services.recognition import recognize_face
                from app.services.recognition import extract_faces_from_rgb

                # Step 1: Detect all faces ONCE on the full frame. Each face dict
                # already carries its embedding, so recognition never needs a
                # second detection pass.
                faces = extract_faces_from_rgb(rgb)
                logger.debug(
                    "STAGE-detect camera=%s faces_detected=%d", w.camera_id, len(faces)
                )

                # Carry the full face dict (with embedding) into the tracker so
                # each track can be recognised directly from its embedding.
                detections = [{"box": face.get("box"), "face": face} for face in faces]

                # Step 2: Update face tracker (always update for smooth tracking)
                tracks = w.face_tracker.update(detections)
                logger.debug(
                    "STAGE-track camera=%s active_tracks=%d", w.camera_id, len(tracks)
                )

                # Frame skipping for recognition only (not tracking)
                with w._frame_lock:
                    w._frame_counter += 1
                    skip_recognition = w.frame_skip > 0 and w._frame_counter % (w.frame_skip + 1) != 0

                # Step 3: Recognise tracks seen in THIS frame whose cooldown elapsed.
                if not skip_recognition:
                    for track in tracks:
                        # Only recognise a track that was matched to a detection
                        # this frame (fresh embedding) and is off cooldown.
                        if track.consecutive_misses != 0 or track.face is None:
                            continue
                        if not track.can_recognize():
                            continue

                        logger.debug(
                            "STAGE-recognize camera=%s track=%d box=%s",
                            w.camera_id, track.track_id, track.box,
                        )

                        # Match from the already-computed embedding (no re-detect).
                        result = recognize_face(
                            track.face,
                            threshold=w.threshold,
                            source="cctv",
                            camera_id=str(w.camera_id),
                            camera_purpose=w.camera_purpose,
                        )
                        faces_data = result.get("faces", [])
                        face_data = faces_data[0] if faces_data else {}

                        track.update_recognition(
                            employee_id=face_data.get("employee_id"),
                            employee_name=face_data.get("employee_name") or "Unknown Person",
                            employee_code=face_data.get("employee_code"),
                            matched=face_data.get("matched", False),
                            confidence=face_data.get("score", 0.0),
                        )

                        if track.matched:
                            logger.info(
                                "Camera %s [%s]: track=%d RECOGNIZED %s (id=%s, conf=%.1f%%)",
                                w.camera_id, w.camera_purpose, track.track_id,
                                track.employee_name, track.employee_id,
                                track.confidence * 100,
                            )
                        else:
                            logger.info(
                                "Camera %s: track=%d REJECTED score=%.4f reason=%s",
                                w.camera_id, track.track_id, track.confidence,
                                face_data.get("state", "no_face_or_below_threshold"),
                            )
                
                # Step 4: Update state with active tracks count
                w.state.active_tracks = len(tracks)
                
                # Step 5: Always annotate frame with enhanced overlay (shows persistent boxes)
                annotated = _draw_enhanced_overlay(frame, tracks, w.name, w.state.fps)
                ok_enc, jpeg_buf = cv2.imencode(
                    ".jpg", annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
                )
                with w._frame_lock:
                    if ok_enc:
                        w.state.latest_jpeg = jpeg_buf.tobytes()
                    w.state.updated_at = time.time()

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
        frame_skip: int = 0,  # Skip N frames between recognition (0 = no skip)
    ) -> None:
        self.camera_id = camera_id
        self.name = name
        self.stream_url = source_url  # Keep as stream_url internally for consistency
        self.source_url = source_url  # Database field
        self.source_type = source_type
        self.camera_purpose = camera_purpose.upper()   # "IN" | "OUT"
        self.threshold = threshold
        self.interval_sec = interval_sec
        self.frame_skip = frame_skip

        self.state = CameraRuntimeState()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_counter = 0  # For frame skipping

        # Face tracking for multi-face recognition
        self.face_tracker = FaceTracker(
            max_distance=100.0,
            recognition_cooldown=3.0,
            max_misses=10,
        )

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
                    frame_skip=getattr(model, 'frame_skip', 0),
                )
                worker.location = getattr(model, "location", None)
                
                # Update tracker config from model (if HCNetSDKCameraWorker has face_tracker)
                if hasattr(worker, 'face_tracker'):
                    tracking_max_distance = getattr(model, 'tracking_max_distance', 100.0)
                    tracking_cooldown = getattr(model, 'tracking_cooldown', 3.0)
                    worker.face_tracker.max_distance = tracking_max_distance
                    worker.face_tracker.recognition_cooldown = tracking_cooldown
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
            frame_skip=getattr(model, 'frame_skip', 0),
        )
        worker.location = getattr(model, "location", None)
        
        # Update tracker config from model
        tracking_max_distance = getattr(model, 'tracking_max_distance', 100.0)
        tracking_cooldown = getattr(model, 'tracking_cooldown', 3.0)
        worker.face_tracker.max_distance = tracking_max_distance
        worker.face_tracker.recognition_cooldown = tracking_cooldown
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
