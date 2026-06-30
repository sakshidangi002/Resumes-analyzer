"""Base worker interface for camera workers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class CameraWorkerBase(ABC):
    """Base class for all camera workers (RTSP, HCNetSDK, etc.)."""
    
    @abstractmethod
    def start(self) -> bool:
        """Start the camera worker. Returns True if successful."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the camera worker."""
        pass
    
    @abstractmethod
    def is_alive(self) -> bool:
        """Check if the worker is alive and running."""
        pass
    
    @abstractmethod
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame as OpenCV BGR image."""
        pass
    
    @property
    @abstractmethod
    def status(self) -> str:
        """Get the current status of the worker."""
        pass
    
    @property
    @abstractmethod
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        pass
    
    @property
    @abstractmethod
    def fps(self) -> float:
        """Get the current FPS."""
        pass
    
    @property
    @abstractmethod
    def total_frames(self) -> int:
        """Get the total number of frames processed."""
        pass
    
    @property
    @abstractmethod
    def reconnect_count(self) -> int:
        """Get the number of reconnection attempts."""
        pass
