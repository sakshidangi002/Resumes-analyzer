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


class DVRManager:
    """Manages DVR connections and live camera streams."""
    
    def __init__(self):
        self._connection: Optional[DVRConnection] = None
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._running = False
        
    def connect(self, ip: str, port: int, username: str, password: str) -> tuple[bool, str, Optional[DiscoveredDevice]]:
        """Connect to DVR and discover cameras."""
        if self._connection and self._connection.connected:
            self.disconnect()
            
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
            
            # Check if already streaming
            if (camera.worker and camera.worker.is_alive()) or (camera.rtsp_worker and camera.rtsp_worker.is_alive()):
                logger.info(f"Camera {channel_id} already streaming")
                return True
            
            try:
                # Use RTSP directly (works with operator account, no admin privileges needed)
                # Use the same URL format as CCTV attendance (proven to work)
                # URL-encode credentials to handle special characters like @, #, :, %
                from urllib.parse import quote
                encoded_username = quote(self._connection.username, safe='')
                encoded_password = quote(self._connection.password, safe='')
                rtsp_url = f"rtsp://{encoded_username}:{encoded_password}@{self._connection.ip}:554/Streaming/Channels/{channel_id:03d}01"
                
                camera.rtsp_worker = CameraWorker(
                    camera_id=channel_id,
                    name=camera.name,
                    source_url=rtsp_url,
                    source_type="rtsp",
                    camera_purpose="IN",
                    threshold=0.05,  # Very low threshold for poor CCTV footage quality
                    interval_sec=0.5,  # Process frames every 0.5 seconds for faster response
                    frame_skip=0,  # No frame skipping for accurate recognition
                )
                
                camera.rtsp_worker.start()
                camera.status = "online"
                camera.last_frame_time = time.time()
                camera.use_rtsp = True
                
                logger.info(f"Started RTSP stream for camera {channel_id} (Streaming/Channels format)")
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
                logger.info(f"Stopped HCNetSDK stream for camera {channel_id}")
                return True
            if camera.rtsp_worker:
                camera.rtsp_worker.stop()
                camera.rtsp_worker = None
                camera.status = "offline"
                logger.info(f"Stopped RTSP stream for camera {channel_id}")
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
