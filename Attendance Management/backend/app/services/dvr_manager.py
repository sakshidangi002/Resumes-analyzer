"""DVR Manager Service for automatic camera discovery and live streaming."""
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor
import logging

from app.services.hcnetsdk_wrapper import _sdk_wrapper
from app.services.hcnetsdk_camera import HCNetSDKCameraWorker
from app.services.hikvision_discovery import discover_cameras, DiscoveredDevice, DiscoveredChannel
from app.services.camera_service import CameraWorker

logger = logging.getLogger(__name__)


@dataclass
class LiveCamera:
    """Represents a live camera stream."""
    channel_id: int
    name: str
    status: str  # online, offline, error
    worker: Optional[HCNetSDKCameraWorker] = None
    rtsp_worker: Optional[CameraWorker] = None
    use_rtsp: bool = False
    recognition_enabled: bool = False
    last_frame_time: float = 0
    error_message: str = ""


@dataclass
class DVRConnection:
    """Represents a DVR connection."""
    ip: str
    port: int
    username: str
    password: str
    login_id: int = -1
    connected: bool = False
    device_info: Optional[DiscoveredDevice] = None
    cameras: Dict[int, LiveCamera] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)
    start_channel: int = 1


import os as _os

# DVR dashboard streams are for VIEWING only. If a stream has not been requested
# by a browser for this long, it is auto-released to free the RTSP connection.
_DVR_IDLE_STOP_SEC = float(_os.getenv("DVR_IDLE_STOP_SEC", "30"))


class DVRManager:
    """Manages DVR connections and on-demand live camera streams.

    Streams opened from the DVR Dashboard are PREVIEW streams: they open only
    when a channel is started/viewed and auto-release when no browser has viewed
    them for `_DVR_IDLE_STOP_SEC`. Background attendance (Check-In / Check-Out)
    is handled separately by `camera_service.camera_manager` and is unaffected.
    """

    def __init__(self):
        self._connection: Optional[DVRConnection] = None
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._running = False
        # Idle-stream reaper (releases preview streams nobody is watching).
        self._reaper = threading.Thread(target=self._reaper_loop, daemon=True, name="dvr-reaper")
        self._reaper_stop = threading.Event()
        self._reaper.start()

    def _active_count(self) -> int:
        conn = self._connection
        if not conn:
            return 0
        return sum(
            1 for c in conn.cameras.values()
            if (c.worker and c.worker.is_alive()) or (c.rtsp_worker and c.rtsp_worker.is_alive())
        )

    def _reaper_loop(self) -> None:
        """Periodically release preview streams that no browser is viewing."""
        while not self._reaper_stop.wait(5.0):
            conn = self._connection
            if not conn:
                continue
            now = time.time()
            idle: list[int] = []
            with conn.lock:
                for cid, cam in conn.cameras.items():
                    worker = cam.worker or cam.rtsp_worker
                    if worker and worker.is_alive():
                        last_view = getattr(worker, "_last_view_ts", now)
                        if now - last_view > _DVR_IDLE_STOP_SEC:
                            idle.append(cid)
            for cid in idle:
                logger.info(
                    "DVR reaper: releasing idle preview stream channel=%d (no viewer >%.0fs)",
                    cid, _DVR_IDLE_STOP_SEC,
                )
                self.stop_camera_stream(cid)
            if idle:
                logger.info("DVR active streams now: %d", self._active_count())
        
    def connect(self, ip: str, port: int, username: str, password: str) -> tuple[bool, str, Optional[DiscoveredDevice]]:
        """Connect to DVR and discover cameras."""
        if self._connection and self._connection.connected:
            self.disconnect()

        # Fast reachability pre-check. Without this, an unreachable DVR makes the
        # HCNetSDK login block for a long timeout while holding a request thread —
        # a few of those (the UI polls every 5s) exhaust the server threadpool and
        # the WHOLE app hangs. A 3s socket probe fails fast instead.
        import socket
        try:
            with socket.create_connection((ip, int(port)), timeout=3):
                pass
        except OSError as exc:
            logger.warning("DVR %s:%s unreachable (%s) — aborting connect", ip, port, exc)
            return False, f"DVR not reachable at {ip}:{port}. Check IP, port and network.", None

        try:
            # Try HCNetSDK discovery first
            logger.info(f"Attempting HCNetSDK connection to {ip}:{port}")
            success, device, error = discover_cameras(ip, port, username, password)
            
            if success and device:
                # Create connection object
                connection = DVRConnection(
                    ip=ip,
                    port=port,
                    username=username,
                    password=password,
                    connected=True,
                    device_info=device
                )
                
                # Create live camera objects for all channels (online, unknown, and offline)
                for channel in device.channels:
                    if channel.status in ["online", "unknown", "offline"]:
                        connection.cameras[channel.id] = LiveCamera(
                            channel_id=channel.id,
                            name=channel.name,
                            status=channel.status,
                            recognition_enabled=False,
                            use_rtsp=False
                        )
                
                # Store start_channel for HCNetSDK
                connection.start_channel = getattr(device, 'start_channel', 1)
                
                self._connection = connection
                logger.info(f"Connected to DVR {ip}:{port} via HCNetSDK, discovered {len(connection.cameras)} cameras")
                return True, "", device
            else:
                # Fallback to RTSP-based discovery (no admin privileges needed)
                logger.warning(f"HCNetSDK failed ({error}), falling back to RTSP discovery")
                return self._connect_via_rtsp(ip, port, username, password)
            
        except Exception as e:
            logger.exception(f"Failed to connect to DVR {ip}:{port}: {e}")
            return False, str(e), None
    
    def _connect_via_rtsp(self, ip: str, port: int, username: str, password: str) -> tuple[bool, str, Optional[DiscoveredDevice]]:
        """Connect to DVR using RTSP (no admin privileges required)."""
        try:
            logger.info(f"Attempting RTSP fallback connection to {ip}:{port}")
            # Create a mock device info for RTSP mode
            from app.services.hikvision_discovery import DiscoveredDevice, DiscoveredChannel
            from urllib.parse import quote
            
            # URL-encode credentials to handle special characters like @, #, :, %
            encoded_username = quote(username, safe='')
            encoded_password = quote(password, safe='')
            
            # Try standard Hikvision channel range (1-8 for typical DVRs)
            channels = []
            for channel_id in range(1, 9):
                # Use the same RTSP URL format that works in CCTV attendance
                rtsp_url = f"rtsp://{encoded_username}:{encoded_password}@{ip}:554/Streaming/Channels/{channel_id:03d}01"
                channels.append(DiscoveredChannel(
                    id=channel_id,
                    name=f"Channel {channel_id}",
                    status="offline",
                    channel_type="analog",
                    resolution=None
                ))
            
            device = DiscoveredDevice(
                model="Hikvision DVR (RTSP Mode)",
                firmware="unknown",
                serial="unknown",
                total_channels=len(channels),
                analog_channels=len(channels),
                ip_channels=0,
                channels=channels
            )
            
            # Create connection object
            connection = DVRConnection(
                ip=ip,
                port=port,
                username=username,
                password=password,
                connected=True,
                device_info=device
            )
            
            # Create live camera objects for all channels
            for channel in device.channels:
                connection.cameras[channel.id] = LiveCamera(
                    channel_id=channel.id,
                    name=channel.name,
                    status="unknown",
                    recognition_enabled=False,
                    use_rtsp=True
                )
            
            self._connection = connection
            logger.info(f"Connected to DVR {ip}:{port} via RTSP, discovered {len(connection.cameras)} cameras")
            return True, "", device
            
        except Exception as e:
            logger.exception(f"Failed to connect to DVR via RTSP {ip}:{port}: {e}")
            return False, str(e), None
    
    def disconnect(self) -> None:
        """Disconnect from DVR and stop all streams."""
        if not self._connection:
            return
            
        with self._connection.lock:
            # Stop all camera workers
            for camera in self._connection.cameras.values():
                if camera.worker:
                    camera.worker.stop()
                    camera.worker = None
                if camera.rtsp_worker:
                    camera.rtsp_worker.stop()
                    camera.rtsp_worker = None
            
            # Logout from DVR
            if self._connection.login_id >= 0:
                _sdk_wrapper.logout(self._connection.login_id)
                self._connection.login_id = -1
            
            self._connection.connected = False
            self._connection = None
        
        logger.info("Disconnected from DVR")
    
    def start_camera_stream(self, channel_id: int) -> bool:
        """Start live stream for a specific camera."""
        if not self._connection or not self._connection.connected:
            logger.error("Not connected to DVR")
            return False
            
        with self._connection.lock:
            if channel_id not in self._connection.cameras:
                logger.error(f"Camera {channel_id} not found")
                return False
            
            camera = self._connection.cameras[channel_id]
            
            # Check if already streaming → REUSE the existing capture (one
            # VideoCapture per camera, never a duplicate RTSP connection).
            existing = camera.worker or camera.rtsp_worker
            if existing and existing.is_alive():
                existing._last_view_ts = time.time()  # refresh so reaper keeps it
                logger.info(
                    "DVR: reusing existing stream channel=%d (active=%d)",
                    channel_id, self._active_count(),
                )
                return True
            
            try:
                # Use RTSP directly (works with operator account, no admin privileges needed)
                # Use the same URL format as CCTV attendance (proven to work)
                # URL-encode credentials to handle special characters like @, #, :, %
                from urllib.parse import quote
                encoded_username = quote(self._connection.username, safe='')
                encoded_password = quote(self._connection.password, safe='')
                rtsp_url = f"rtsp://{encoded_username}:{encoded_password}@{self._connection.ip}:554/Streaming/Channels/{channel_id:03d}01"
                
                from app.core.config import get_settings
                _s = get_settings()

                # Decide the camera role per channel: MONITOR > OUT > IN.
                def _chan_set(csv: str) -> set[int]:
                    return {int(c.strip()) for c in (csv or "").split(",") if c.strip().isdigit()}
                monitor_channels = _chan_set(_s.dvr_monitor_channels)
                out_channels = _chan_set(_s.dvr_out_channels)
                if channel_id in monitor_channels:
                    purpose = "MONITOR"
                elif channel_id in out_channels:
                    purpose = "OUT"
                else:
                    purpose = "IN"
                entry_dir = _s.dvr_entry_direction
                if purpose == "OUT" and _s.dvr_out_entry_direction:
                    entry_dir = _s.dvr_out_entry_direction
                logger.info("DVR channel %d configured as %s camera", channel_id, purpose)

                camera.rtsp_worker = CameraWorker(
                    camera_id=channel_id,
                    name=camera.name,
                    source_url=rtsp_url,
                    source_type="rtsp",
                    camera_purpose=purpose,
                    threshold=_s.dvr_recognition_threshold,  # 0.05 accepted near-random matches
                    interval_sec=0.5,
                    frame_skip=0,
                    crossing_enabled=_s.dvr_crossing_enabled,
                    line_orientation=_s.dvr_line_orientation,
                    line_position=_s.dvr_line_position,
                    entry_direction=entry_dir,
                )
                
                camera.rtsp_worker.start()
                camera.rtsp_worker._last_view_ts = time.time()  # grace before reaper
                camera.status = "online"
                camera.last_frame_time = time.time()
                camera.use_rtsp = True

                logger.info(
                    "DVR: opened stream channel=%d purpose=%s rtsp=%s (active=%d)",
                    channel_id, purpose, rtsp_url.split('@')[-1], self._active_count(),
                )
                return True
                
            except Exception as e:
                logger.exception(f"Failed to start stream for camera {channel_id}: {e}")
                camera.status = "error"
                camera.error_message = str(e)
                return False
    
    def stop_camera_stream(self, channel_id: int) -> bool:
        """Stop live stream for a specific camera."""
        if not self._connection:
            return False
            
        with self._connection.lock:
            if channel_id not in self._connection.cameras:
                return False
            
            camera = self._connection.cameras[channel_id]
            if camera.worker:
                camera.worker.stop()
                camera.worker = None
                camera.status = "offline"
                logger.info("DVR: closed stream channel=%d (active=%d)", channel_id, self._active_count())
                return True
            if camera.rtsp_worker:
                camera.rtsp_worker.stop()
                camera.rtsp_worker = None
                camera.status = "offline"
                logger.info("DVR: closed stream channel=%d (active=%d)", channel_id, self._active_count())
                return True
            return False
    
    def start_all_streams(self) -> int:
        """Start streams for all online cameras."""
        if not self._connection:
            return 0
            
        started = 0
        for channel_id in self._connection.cameras.keys():
            if self.start_camera_stream(channel_id):
                started += 1
        
        logger.info(f"Started {started} camera streams")
        return started
    
    def stop_all_streams(self) -> int:
        """Stop all camera streams."""
        if not self._connection:
            return 0
            
        stopped = 0
        for channel_id in self._connection.cameras.keys():
            if self.stop_camera_stream(channel_id):
                stopped += 1
        
        logger.info(f"Stopped {stopped} camera streams")
        return stopped
    
    def set_recognition_enabled(self, channel_id: int, enabled: bool) -> bool:
        """Enable or disable recognition for a camera."""
        if not self._connection:
            return False
            
        with self._connection.lock:
            if channel_id not in self._connection.cameras:
                return False
            
            camera = self._connection.cameras[channel_id]
            camera.recognition_enabled = enabled
            
            # Update worker if running
            if camera.worker:
                camera.worker.recognition_enabled = enabled
            if camera.rtsp_worker:
                camera.rtsp_worker.recognition_enabled = enabled
            
            logger.info(f"Camera {channel_id} recognition set to {enabled}")
            return True
    
    def get_camera_status(self, channel_id: int) -> Optional[dict]:
        """Get status of a specific camera."""
        if not self._connection:
            return None
            
        with self._connection.lock:
            if channel_id not in self._connection.cameras:
                return None
            
            camera = self._connection.cameras[channel_id]
            worker_status = None
            is_alive = False
            
            try:
                # Try HCNetSDK worker first
                if camera.worker:
                    is_alive = camera.worker.is_alive()
                    worker_status = {
                        "is_alive": is_alive,
                        "last_error": camera.worker.state.last_error,
                        "fps": camera.worker.state.fps,
                        "total_frames": camera.worker.state.total_frames,
                    }
                # Try RTSP worker
                elif camera.rtsp_worker:
                    is_alive = camera.rtsp_worker.is_alive()
                    worker_status = {
                        "is_alive": is_alive,
                        "last_error": camera.rtsp_worker.state.last_error,
                        "fps": camera.rtsp_worker.state.fps,
                        "total_frames": camera.rtsp_worker.state.total_frames,
                    }
            except Exception as e:
                logger.error(f"Error getting worker status for camera {channel_id}: {e}")
                worker_status = {
                    "is_alive": False,
                    "last_error": str(e),
                    "fps": 0,
                    "total_frames": 0,
                }
            
            return {
                "channel_id": camera.channel_id,
                "name": camera.name,
                "status": camera.status,
                "recognition_enabled": camera.recognition_enabled,
                "last_frame_time": camera.last_frame_time,
                "error_message": camera.error_message,
                "worker_status": worker_status,
            }
    
    def get_all_cameras(self) -> List[dict]:
        """Get status of all cameras."""
        if not self._connection:
            return []
            
        with self._connection.lock:
            return [self.get_camera_status(cid) for cid in self._connection.cameras.keys()]
    
    def is_connected(self) -> bool:
        """Check if connected to DVR."""
        return self._connection is not None and self._connection.connected
    
    def get_connection_info(self) -> Optional[dict]:
        """Get connection information."""
        if not self._connection:
            return None
            
        return {
            "ip": self._connection.ip,
            "port": self._connection.port,
            "username": self._connection.username,
            "connected": self._connection.connected,
            "device_info": {
                "model": self._connection.device_info.model if self._connection.device_info else None,
                "serial": self._connection.device_info.serial if self._connection.device_info else None,
                "total_channels": self._connection.device_info.total_channels if self._connection.device_info else 0,
            } if self._connection.device_info else None,
            "cameras_count": len(self._connection.cameras),
        }


# Singleton instance
_dvr_manager = DVRManager()


def get_dvr_manager() -> DVRManager:
    """Get the DVR Manager singleton instance."""
    return _dvr_manager
