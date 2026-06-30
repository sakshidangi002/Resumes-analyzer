"""
hcnetsdk_camera.py
==================
HCNetSDK-based camera worker for Hikvision DVR streaming.

This module implements a camera worker that uses HCNetSDK for direct DVR
connection and streaming, replacing RTSP/OpenCV for better compatibility.
"""
from __future__ import annotations

import ctypes
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.services.hcnetsdk_wrapper import (
    _sdk_wrapper,
    NET_DVR_PREVIEWINFO,
    REALDATA_CALLBACK,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RECONNECT_INIT_DELAY = 2.0
_RECONNECT_MAX_DELAY = 30.0
_STALE_TIMEOUT = 15.0
_JPEG_QUALITY = 80
_FPS_WINDOW = 30


# ---------------------------------------------------------------------------
# PlayCtrl Decoder
# ---------------------------------------------------------------------------
class PlayCtrlDecoder:
    """Decodes H.264 stream using PlayCtrl library."""
    
    def __init__(self):
        self.port = -1
        self._initialized = False
        self._frame_ready = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        
    def initialize(self) -> bool:
        """Initialize PlayCtrl decoder port."""
        try:
            if not _sdk_wrapper.playctrl:
                logger.error("PlayCtrl not loaded")
                return False
                
            self.port = _sdk_wrapper.playctrl.PlayM4_GetPort()
            if self.port < 0:
                logger.error("Failed to get PlayCtrl port")
                return False
            
            self._initialized = True
            logger.debug(f"PlayCtrl decoder initialized on port {self.port}")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize PlayCtrl decoder: {e}")
            return False
    
    def open_stream(self, data: bytes, size: int) -> bool:
        """Open stream with initial data."""
        if not self._initialized:
            return False
        try:
            result = _sdk_wrapper.playctrl.PlayM4_OpenStream(
                self.port,
                data,
                size,
                1024 * 1024  # 1MB buffer
            )
            if result:
                # Set display buffer
                _sdk_wrapper.playctrl.PlayM4_SetDisplayBuf(self.port, 5)
                # Start playback
                _sdk_wrapper.playctrl.PlayM4_Play(self.port, None)
                logger.debug(f"PlayCtrl stream opened on port {self.port}")
            return result != 0
        except Exception as e:
            logger.exception(f"Failed to open stream: {e}")
            return False
    
    def input_data(self, data: bytes, size: int) -> bool:
        """Input stream data to decoder."""
        if not self._initialized:
            return False
        try:
            result = _sdk_wrapper.playctrl.PlayM4_InputData(
                self.port,
                ctypes.cast(data, ctypes.POINTER(ctypes.c_byte)),
                size
            )
            return result != 0
        except Exception as e:
            logger.exception(f"Failed to input data: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get decoded frame as OpenCV BGR image."""
        if not self._initialized:
            return None
        try:
            # Allocate buffer for image
            buf_size = ctypes.c_ulong(1024 * 1024 * 4)  # 4MB buffer
            buf = (ctypes.c_byte * buf_size.value)()
            
            result = _sdk_wrapper.playctrl.PlayM4_GetPicture(
                self.port,
                buf,
                ctypes.byref(buf_size)
            )
            
            if result and buf_size.value > 0:
                # Convert buffer to numpy array
                img_data = bytes(buf[:buf_size.value])
                # Decode JPEG/BMP from PlayCtrl
                frame = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
            return None
        except Exception as e:
            logger.exception(f"Failed to get frame: {e}")
            return None
    
    def close(self):
        """Close decoder port."""
        if self._initialized and self.port >= 0:
            try:
                _sdk_wrapper.playctrl.PlayM4_Stop(self.port)
                _sdk_wrapper.playctrl.PlayM4_CloseStream(self.port)
                _sdk_wrapper.playctrl.PlayM4_FreePort(self.port)
                logger.debug(f"PlayCtrl decoder closed on port {self.port}")
            except Exception as e:
                logger.error(f"Error closing decoder: {e}")
            self._initialized = False
            self.port = -1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class HCNetSDKRuntimeState:
    """Live runtime metrics for a single HCNetSDK camera."""
    status: str = "stopped"
    last_error: Optional[str] = None
    total_frames: int = 0
    reconnect_count: int = 0
    last_frame_time: float = 0.0
    updated_at: float = 0.0
    fps: float = 0.0
    latest_jpeg: Optional[bytes] = None
    latest_result: dict = field(default_factory=dict)
    _fps_ts: deque = field(default_factory=lambda: deque(maxlen=_FPS_WINDOW))


# ---------------------------------------------------------------------------
# HCNetSDK Camera Worker
# ---------------------------------------------------------------------------
class HCNetSDKCameraWorker:
    """Manages a single HCNetSDK camera connection and streaming."""
    
    def __init__(
        self,
        *,
        camera_id: int,
        name: str,
        dvr_ip: str,
        dvr_port: int,
        dvr_username: str,
        dvr_password: str,
        dvr_channel: int,
        camera_purpose: str,
        threshold: float,
        interval_sec: float,
    ) -> None:
        self.camera_id = camera_id
        self.name = name
        self.dvr_ip = dvr_ip
        self.dvr_port = dvr_port
        self.dvr_username = dvr_username
        self.dvr_password = dvr_password
        self.dvr_channel = dvr_channel
        self.camera_purpose = camera_purpose.upper()
        self.threshold = threshold
        self.interval_sec = interval_sec
        
        self.state = HCNetSDKRuntimeState()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        
        self._login_id = -1
        self._play_handle = -1
        self._callback_ref: Optional[REALDATA_CALLBACK] = None
        self._decoder: Optional[PlayCtrlDecoder] = None
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream_buffer = bytearray()
        self._start_channel = 1  # Default start channel
        
    def start(self) -> bool:
        """Start the camera worker."""
        if self._thread and self._thread.is_alive():
            return True
            
        if not _sdk_wrapper.initialize():
            self.state.status = "error"
            self.state.last_error = "Failed to initialize HCNetSDK"
            return False
            
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"hcnetsdk-{self.camera_id}",
        )
        self._thread.start()
        logger.info(
            f"Camera {self.camera_id} [{self.name}]: HCNetSDK worker started "
            f"(DVR={self.dvr_ip}:{self.dvr_port}, channel={self.dvr_channel})"
        )
        return True
        
    def stop(self) -> None:
        """Stop the camera worker."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=6)
        self._cleanup()
        self.state.status = "stopped"
        logger.info(f"Camera {self.camera_id}: HCNetSDK worker stopped")
    
    def is_alive(self) -> bool:
        """Check if the worker is alive."""
        return self._thread is not None and self._thread.is_alive()
    
    def restart(self) -> None:
        """Restart the camera worker."""
        logger.info(f"Camera {self.camera_id}: Restarting...")
        self.stop()
        self.state = HCNetSDKRuntimeState()
        with self._frame_lock:
            self._latest_frame = None
        self._stream_buffer.clear()
        self.start()
        
    def _cleanup(self) -> None:
        """Cleanup HCNetSDK resources."""
        # Stop real play
        if self._play_handle >= 0:
            _sdk_wrapper.stop_real_play(self._play_handle)
            self._play_handle = -1
            
        # Close decoder
        if self._decoder:
            self._decoder.close()
            self._decoder = None
            
        # Logout
        if self._login_id >= 0:
            _sdk_wrapper.logout(self._login_id)
            self._login_id = -1
            
    def _login(self) -> bool:
        """Login to DVR."""
        logger.info(
            f"Camera {self.camera_id} [{self.name}]: Attempting login to DVR "
            f"{self.dvr_ip}:{self.dvr_port} as user '{self.dvr_username}'"
        )
        success, login_id, device_info = _sdk_wrapper.login(
            self.dvr_ip,
            self.dvr_port,
            self.dvr_username,
            self.dvr_password
        )
        
        if success:
            self._login_id = login_id
            self._start_channel = getattr(device_info, 'byStartChan', 1)
            logger.info(
                f"Camera {self.camera_id} [{self.name}]: Login successful - "
                f"UserID: {login_id}, StartChannel: {self._start_channel}, "
                f"TotalChannels: {getattr(device_info, 'byChanNum', 'unknown')}"
            )
            return True
        else:
            error_code = _sdk_wrapper.get_last_error()
            self.state.last_error = f"Login failed: {error_code}"
            logger.error(
                f"Camera {self.camera_id} [{self.name}]: Login failed - "
                f"Error Code: {error_code}, DVR: {self.dvr_ip}:{self.dvr_port}"
            )
            return False
            
    def _start_preview(self) -> bool:
        """Start real-time preview with callback."""
        logger.info(
            f"Camera {self.camera_id} [{self.name}]: Initializing PlayCtrl decoder"
        )
        # Initialize decoder
        self._decoder = PlayCtrlDecoder()
        if not self._decoder.initialize():
            self.state.last_error = "Failed to initialize PlayCtrl decoder"
            logger.error(
                f"Camera {self.camera_id} [{self.name}]: PlayCtrl decoder initialization failed"
            )
            return False
        
        # Create callback wrapper
        def data_callback(l_real_handle: int, p_buffer, dw_buf_size: int, dw_user: int, p_reserved):
            """Callback for real-time stream data."""
            try:
                if p_buffer and dw_buf_size > 0:
                    data = ctypes.string_at(p_buffer, dw_buf_size)
                    self._process_stream_data(data)
            except Exception as e:
                logger.error(f"Data callback error: {e}")
        
        self._callback_ref = REALDATA_CALLBACK(data_callback)
        
        logger.info(
            f"Camera {self.camera_id} [{self.name}]: Starting preview - "
            f"Channel: {self.dvr_channel}, StartChannel: {self._start_channel}, "
            f"LoginID: {self._login_id}"
        )
        success, play_handle = _sdk_wrapper.start_real_play(
            self._login_id,
            self.dvr_channel,
            self._callback_ref,
            user_data=self.camera_id,
            start_channel=self._start_channel
        )
        
        if success:
            self._play_handle = play_handle
            logger.info(
                f"Camera {self.camera_id} [{self.name}]: Preview started successfully - "
                f"PlayHandle: {play_handle}"
            )
            return True
        else:
            error_code = _sdk_wrapper.get_last_error()
            self.state.last_error = f"Start real play failed: {error_code}"
            logger.error(
                f"Camera {self.camera_id} [{self.name}]: Preview failed - "
                f"Error Code: {error_code}, Channel: {self.dvr_channel}, "
                f"StartChannel: {self._start_channel}"
            )
            self._decoder.close()
            self._decoder = None
            return False
    
    def _process_stream_data(self, data: bytes) -> None:
        """Process incoming stream data and convert to OpenCV frame."""
        try:
            # Add data to stream buffer
            self._stream_buffer.extend(data)
            
            # If decoder is not initialized, try to open stream with this data
            if self._decoder and not self._decoder._initialized:
                if len(self._stream_buffer) > 100:  # Need some data to start
                    self._decoder.open_stream(self._stream_buffer, len(self._stream_buffer))
                    self._stream_buffer.clear()
            
            # Feed data to decoder
            if self._decoder and self._decoder._initialized:
                self._decoder.input_data(data, len(data))
                
                # Try to get a frame
                frame = self._decoder.get_frame()
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
                    self.state.total_frames += 1
                    self.state.last_frame_time = time.time()
                    
                    # Update FPS
                    self.state._fps_ts.append(time.time())
                    if len(self.state._fps_ts) >= 2:
                        span = self.state._fps_ts[-1] - self.state._fps_ts[0]
                        self.state.fps = round((len(self.state._fps_ts) - 1) / span, 1) if span > 0 else 0.0
                    
                    if self.state.total_frames % 100 == 0:
                        logger.info(
                            f"Camera {self.camera_id}: {self.state.total_frames} frames, "
                            f"{self.state.fps:.1f} FPS"
                        )
        except Exception as e:
            logger.error(f"Stream data processing error: {e}")
    
    def _run(self) -> None:
        """Main worker loop with reconnection logic."""
        reconnect_delay = _RECONNECT_INIT_DELAY
        consecutive_failures = 0
        
        logger.info(f"Camera {self.camera_id}: HCNetSDK worker thread started")
        
        while not self._stop_evt.is_set():
            # Login if not logged in
            if self._login_id < 0:
                self.state.status = "connecting"
                logger.warning(
                    f"Camera {self.camera_id}: Attempting to login "
                    f"(attempt {self.state.reconnect_count + 1})"
                )
                
                if self._login():
                    consecutive_failures = 0
                    reconnect_delay = _RECONNECT_INIT_DELAY
                    self.state.status = "running"
                    self.state.last_error = None
                    logger.info(f"Camera {self.camera_id}: Successfully logged in")
                else:
                    consecutive_failures += 1
                    self.state.reconnect_count += 1
                    self.state.status = "error"
                    logger.error(f"Camera {self.camera_id}: {self.state.last_error}")
                    
                    reconnect_delay = min(reconnect_delay * 1.5, _RECONNECT_MAX_DELAY)
                    logger.info(
                        f"Camera {self.camera_id}: Retrying in {reconnect_delay:.1f}s"
                    )
                    self._stop_evt.wait(reconnect_delay)
                    continue
            
            # Start preview if not started
            if self._play_handle < 0:
                if self._start_preview():
                    logger.info(f"Camera {self.camera_id}: Preview started")
                else:
                    consecutive_failures += 1
                    logger.error(f"Camera {self.camera_id}: {self.state.last_error}")
                    
                    # Cleanup and retry
                    self._cleanup()
                    reconnect_delay = min(reconnect_delay * 2, _RECONNECT_MAX_DELAY)
                    self._stop_evt.wait(reconnect_delay)
                    continue
            
            # Process latest frame for recognition
            frame = self.get_latest_frame()
            if frame is not None:
                self._process_frame_for_recognition(frame)
            
            # Check for stale frames
            if (
                self.state.last_frame_time > 0
                and time.time() - self.state.last_frame_time > _STALE_TIMEOUT
            ):
                logger.warning(
                    f"Camera {self.camera_id}: No frame for {_STALE_TIMEOUT}s – forcing reconnect"
                )
                self._cleanup()
                self.state.status = "reconnecting"
                continue
            
            # Sleep for interval
            self._stop_evt.wait(max(0.5, float(self.interval_sec)))
        
        # Cleanup
        self._cleanup()
        logger.info(
            f"Camera {self.camera_id}: HCNetSDK worker thread stopped. "
            f"Total frames: {self.state.total_frames}, Reconnects: {self.state.reconnect_count}"
        )
    
    def _process_frame_for_recognition(self, frame: np.ndarray) -> None:
        """Process frame for face recognition (mirrors CameraWorker behavior)."""
        try:
            # Check if frame is blurry
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 80.0:
                logger.debug(f"Camera {self.camera_id}: Skipping blurry frame")
                return
            
            # Convert to RGB for recognition
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Import recognition function
            from app.services.recognition import recognize_from_rgb
            
            result = recognize_from_rgb(
                rgb,
                threshold=self.threshold,
                source="cctv",
                camera_id=str(self.camera_id),
                camera_purpose=self.camera_purpose,
            )
            
            # Annotate frame and encode JPEG for preview
            annotated = self._draw_boxes(frame, result)
            ok_enc, jpeg_buf = cv2.imencode(
                ".jpg", annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
            )
            
            if ok_enc:
                self.state.latest_jpeg = jpeg_buf.tobytes()
            
            self.state.latest_result = result
            self.state.updated_at = time.time()
            
            # Log matches
            faces = result.get("faces", [])
            matched = [f for f in faces if f.get("matched")]
            if matched:
                logger.info(
                    f"Camera {self.camera_id} [{self.camera_purpose}]: {len(matched)} face(s) matched – "
                    + ", ".join(f.get("employee_name", "?") for f in matched)
                )
            elif faces:
                logger.debug(f"Camera {self.camera_id}: {len(faces)} unknown face(s)")
                
        except Exception as exc:
            logger.error(
                f"Camera {self.camera_id}: Recognition error: {exc}", exc_info=True
            )
    
    def _draw_boxes(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Overlay recognition results on frame (mirrors CameraWorker behavior)."""
        annotated = frame.copy()
        for face in result.get("faces", []):
            box = face.get("box") or []
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            matched = bool(face.get("matched"))
            state = face.get("state", "")
            
            if state == "cooldown":
                color = (255, 165, 0)  # orange
            elif state in ("in", "out"):
                color = (34, 197, 94)  # green
            elif matched:
                color = (34, 197, 94)
            else:
                color = (68, 68, 239)  # blue for unknown
            
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
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (thread-safe)."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None
    
    def serialize_state(self) -> dict:
        """Serialize camera state for API responses."""
        s = self.state
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "dvr_ip": self.dvr_ip,
            "dvr_port": self.dvr_port,
            "dvr_channel": self.dvr_channel,
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


# Import ctypes for callback
import ctypes
