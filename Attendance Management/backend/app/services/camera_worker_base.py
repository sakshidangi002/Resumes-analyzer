"""The interface every camera worker must satisfy.

Two worker classes exist — `camera_service.CameraWorker` (RTSP/USB, three
threads) and `hcnetsdk_camera.HCNetSDKCameraWorker` (direct DVR, one thread) —
and API routes hold them interchangeably. They drifted: the HCNetSDK worker was
missing `get_latest_jpeg()`, so every DVR preview/stream request for a channel
using it raised AttributeError and returned HTTP 500.

Routes and the camera manager should depend on this Protocol, never on a
concrete class, and `test_camera_worker_contract.py` pins it so a missing method
fails in CI instead of in production.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


# The methods a route or the CameraManager may call on any worker.
REQUIRED_WORKER_METHODS = (
    "start",
    "stop",
    "is_alive",
    "get_latest_frame",
    "get_latest_jpeg",
    "serialize_state",
)


@runtime_checkable
class CameraWorkerProtocol(Protocol):
    """Structural type implemented by every camera worker."""

    camera_id: int
    name: str
    camera_purpose: str
    # True while AI analysis is suspended; video keeps streaming regardless.
    analysis_paused: bool

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def is_alive(self) -> bool: ...

    def get_latest_frame(self): ...

    def get_latest_jpeg(self) -> Optional[bytes]: ...

    def serialize_state(self) -> dict: ...
