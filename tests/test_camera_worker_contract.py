"""Both camera worker classes must expose the same surface.

Regression test for the DVR preview 500s: `HCNetSDKCameraWorker` was missing
`get_latest_jpeg()`, but the DVR preview and stream routes called it on
whichever worker a channel happened to be using. The result was an
AttributeError -> HTTP 500 for every hcnetsdk-backed channel, while the
RTSP-backed ones worked fine — so it looked like a DVR problem rather than a
missing method.

Pure import-level checks: no camera, no DVR, no OpenCV capture is created.
"""
import pytest
from app.services.camera_worker_base import REQUIRED_WORKER_METHODS

WORKER_CLASSES = [
    ("app.services.camera_service", "CameraWorker"),
    ("app.services.hcnetsdk_camera", "HCNetSDKCameraWorker"),
]


def _load(module_name: str, class_name: str):
    module = pytest.importorskip(
        module_name,
        reason=f"{module_name} needs the vision stack (cv2 / HCNetSDK) installed",
    )
    cls = getattr(module, class_name, None)
    assert cls is not None, f"{class_name} not found in {module_name}"
    return cls


@pytest.mark.parametrize("module_name,class_name", WORKER_CLASSES)
@pytest.mark.parametrize("method", REQUIRED_WORKER_METHODS)
def test_worker_implements_required_method(module_name, class_name, method):
    cls = _load(module_name, class_name)
    attr = getattr(cls, method, None)
    assert callable(attr), (
        f"{class_name} is missing {method}(). Routes hold both worker types "
        f"interchangeably, so a missing method is a 500 in production."
    )


@pytest.mark.parametrize("module_name,class_name", WORKER_CLASSES)
def test_worker_declares_analysis_paused(module_name, class_name):
    """The DVR recognition toggle sets `analysis_paused`; both must honour it.

    Previously the toggle set `recognition_enabled`, an attribute neither class
    read — Python created it and nothing happened, so the control was inert.
    """
    cls = _load(module_name, class_name)
    source = cls.__init__.__code__.co_names + cls.__init__.__code__.co_varnames
    assert "analysis_paused" in source, (
        f"{class_name}.__init__ must set analysis_paused so the DVR "
        f"recognition toggle actually suspends analysis."
    )
