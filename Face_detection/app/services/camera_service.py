from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

import cv2
import numpy as np

from app.db import db_cursor
from app.services.recognition import recognize_from_rgb

logger = logging.getLogger(__name__)


@dataclass
class CameraState:
    camera_id: int
    name: str
    source_url: str
    source_type: str
    threshold: float
    interval_sec: float
    status: str = "stopped"
    last_error: str | None = None
    latest_result: dict = field(default_factory=dict)
    latest_frame_jpeg: bytes | None = None
    updated_at: float = 0.0
    reconnect_count: int = 0
    total_frames: int = 0
    last_successful_frame_time: float = 0.0


def _draw_boxes(frame: np.ndarray, result: dict) -> np.ndarray:
    annotated = frame.copy()
    for face in result.get("faces", []):
        box = face.get("box") or []
        if len(box) < 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        matched = bool(face.get("matched"))
        state = face.get("state")
        if state == "already_marked":
            color = (235, 99, 37)
        elif state == "checked_out":
            color = (37, 99, 235)
        elif matched:
            color = (34, 197, 94)
        else:
            color = (68, 68, 239)

        label = face.get("employee_name") or "Unknown"
        score = face.get("score")
        if isinstance(score, (float, int)):
            label = f"{label} {score:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - 24)), (x1 + max(120, len(label) * 9), y1), color, -1)
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


class CameraWorker:
    def __init__(self, config: CameraState):
        self.config = config
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"camera-{self.config.camera_id}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.config.status = "stopped"

    def _open_capture(self) -> cv2.VideoCapture:
        """Open video capture with timeout and error handling."""
        source = self.config.source_url.strip()
        logger.info("Camera %s: Opening capture for %s", self.config.camera_id, source)

        if self.config.source_type == "usb" or source.isdigit():
            logger.info("Camera %s: Using USB camera index %s", self.config.camera_id, source)
            return cv2.VideoCapture(int(source))

        parsed = urlparse(source)
        if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
            raise ValueError("Invalid RTSP URL. Use rtsp://username:password@ip:port/path")

        # Hikvision DVRs are more reliable over RTSP/TCP than the default transport.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000",
        )
        logger.info("Camera %s: Using RTSP stream with FFmpeg backend over TCP", self.config.camera_id)
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        return cap

    def _run(self) -> None:
        """Main camera worker loop with reconnection logic"""
        cap: cv2.VideoCapture | None = None
        reconnect_delay = 2.0
        max_reconnect_delay = 30.0
        consecutive_failures = 0
        
        logger.info(f"Camera {self.config.camera_id}: Worker thread started")

        while not self._stop.is_set():
            try:
                # Reconnection logic
                if cap is None or not cap.isOpened():
                    logger.warning(f"Camera {self.config.camera_id}: Attempting to connect (attempt {self.config.reconnect_count + 1})")
                    self.config.status = "connecting"
                    
                    cap = self._open_capture()

                    if not cap.isOpened():
                        consecutive_failures += 1
                        self.config.reconnect_count += 1
                        self.config.status = "error"
                        self.config.last_error = f"Could not connect to camera stream (attempt {self.config.reconnect_count})"
                        logger.error("Camera %s: %s", self.config.camera_id, self.config.last_error)

                        reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                        logger.info("Camera %s: Waiting %.1fs before retry", self.config.camera_id, reconnect_delay)
                        time.sleep(reconnect_delay)
                        continue
                    
                    # Connection successful
                    consecutive_failures = 0
                    reconnect_delay = 2.0
                    self.config.status = "running"
                    self.config.last_error = None
                    logger.info(f"Camera {self.config.camera_id}: Successfully connected")

                # Read frame with timeout handling
                ok, frame = cap.read()
                
                if not ok or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Camera {self.config.camera_id}: Failed to read frame (failure {consecutive_failures})")
                    
                    if cap is not None:
                        cap.release()
                    cap = None
                    self.config.status = "reconnecting"
                    
                    # If too many consecutive failures, increase delay
                    if consecutive_failures > 3:
                        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                        logger.warning(f"Camera {self.config.camera_id}: Multiple failures, increasing reconnect delay to {reconnect_delay:.1f}s")
                    
                    time.sleep(reconnect_delay)
                    continue

                # Frame read successfully
                consecutive_failures = 0
                self.config.total_frames += 1
                self.config.last_successful_frame_time = time.time()
                
                # Log frame info periodically
                if self.config.total_frames % 100 == 0:
                    logger.info(f"Camera {self.config.camera_id}: Processed {self.config.total_frames} frames")

                # Process frame for recognition
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = recognize_from_rgb(
                    rgb,
                    threshold=self.config.threshold,
                    source=f"cctv:{self.config.camera_id}",
                )
                annotated = _draw_boxes(frame, result)
                ok, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    self.config.latest_frame_jpeg = jpeg.tobytes()

                self.config.latest_result = result
                self.config.updated_at = time.time()
                self.config.status = "running"
                
                # Sleep for interval
                time.sleep(max(0.5, float(self.config.interval_sec)))
                
            except Exception as exc:
                logger.exception(f"Camera {self.config.camera_id}: Exception in worker loop: {exc}")
                consecutive_failures += 1
                self.config.last_error = f"Exception: {str(exc)}"
                self.config.status = "error"
                
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass
                cap = None
                
                time.sleep(3)

        # Cleanup
        if cap is not None:
            try:
                cap.release()
                logger.info(f"Camera {self.config.camera_id}: VideoCapture released")
            except:
                pass
        
        logger.info(f"Camera {self.config.camera_id}: Worker thread stopped. Total frames: {self.config.total_frames}, Reconnects: {self.config.reconnect_count}")


class CameraManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: dict[int, CameraWorker] = {}
        self._states: dict[int, CameraState] = {}

    def list_states(self) -> list[dict]:
        with self._lock:
            return [self._serialize_state(state) for state in self._states.values()]

    def get_state(self, camera_id: int) -> dict | None:
        with self._lock:
            state = self._states.get(camera_id)
            return self._serialize_state(state) if state else None

    def get_preview(self, camera_id: int) -> bytes | None:
        with self._lock:
            state = self._states.get(camera_id)
            return state.latest_frame_jpeg if state else None

    def sync_from_db(self) -> None:
        with db_cursor() as (_, cur):
            rows = cur.execute(
                """
                SELECT id, name, source_url, source_type, enabled, threshold, interval_sec
                FROM cameras
                ORDER BY id
                """
            ).fetchall()

        desired: dict[int, CameraState] = {}
        for row in rows:
            if not int(row["enabled"]):
                continue
            desired[int(row["id"])] = CameraState(
                camera_id=int(row["id"]),
                name=row["name"],
                source_url=row["source_url"],
                source_type=row["source_type"],
                threshold=float(row["threshold"]),
                interval_sec=float(row["interval_sec"]),
            )

        with self._lock:
            for camera_id, worker in list(self._workers.items()):
                if camera_id not in desired:
                    worker.stop()
                    self._workers.pop(camera_id, None)
                    self._states.pop(camera_id, None)

            for camera_id, config in desired.items():
                existing = self._states.get(camera_id)
                if existing and self._config_changed(existing, config):
                    worker = self._workers.get(camera_id)
                    if worker:
                        worker.stop()
                        self._workers.pop(camera_id, None)

                if camera_id not in self._workers:
                    self._states[camera_id] = config
                    worker = CameraWorker(config)
                    self._workers[camera_id] = worker
                    worker.start()
                else:
                    self._states[camera_id] = config

    def start_camera(self, camera_id: int) -> None:
        with db_cursor() as (_, cur):
            cur.execute("UPDATE cameras SET enabled = 1 WHERE id = ?", (camera_id,))
        self.sync_from_db()

    def stop_camera(self, camera_id: int) -> None:
        with db_cursor() as (_, cur):
            cur.execute("UPDATE cameras SET enabled = 0 WHERE id = ?", (camera_id,))
        with self._lock:
            worker = self._workers.pop(camera_id, None)
            self._states.pop(camera_id, None)
        if worker:
            worker.stop()

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
            self._states.clear()
        for worker in workers:
            worker.stop()

    @staticmethod
    def _config_changed(left: CameraState, right: CameraState) -> bool:
        return (
            left.source_url != right.source_url
            or left.source_type != right.source_type
            or left.threshold != right.threshold
            or left.interval_sec != right.interval_sec
            or left.name != right.name
        )

    @staticmethod
    def _serialize_state(state: CameraState) -> dict:
        faces = state.latest_result.get("faces") if state.latest_result else []
        return {
            "camera_id": state.camera_id,
            "name": state.name,
            "source_url": state.source_url,
            "source_type": state.source_type,
            "threshold": state.threshold,
            "interval_sec": state.interval_sec,
            "status": state.status,
            "last_error": state.last_error,
            "updated_at": state.updated_at,
            "reconnect_count": state.reconnect_count,
            "total_frames": state.total_frames,
            "last_successful_frame_time": state.last_successful_frame_time,
            "latest_result": {
                "status": bool(state.latest_result.get("status")) if state.latest_result else False,
                "message": state.latest_result.get("message") if state.latest_result else "",
                "faces": faces or [],
            },
        }
    
    def get_camera_stats(self) -> dict:
        """Get overall statistics for all cameras"""
        with self._lock:
            total_cameras = len(self._states)
            running_cameras = sum(1 for s in self._states.values() if s.status == "running")
            error_cameras = sum(1 for s in self._states.values() if s.status == "error")
            total_frames = sum(s.total_frames for s in self._states.values())
            total_reconnects = sum(s.reconnect_count for s in self._states.values())
            
            return {
                "total_cameras": total_cameras,
                "running_cameras": running_cameras,
                "error_cameras": error_cameras,
                "total_frames_processed": total_frames,
                "total_reconnects": total_reconnects,
                "cameras": [
                    {
                        "camera_id": state.camera_id,
                        "name": state.name,
                        "status": state.status,
                        "frames": state.total_frames,
                        "reconnects": state.reconnect_count,
                    }
                    for state in self._states.values()
                ]
            }


camera_manager = CameraManager()
