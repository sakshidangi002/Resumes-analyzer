"""CPU-friendly person (body) detector using OpenCV's DNN module.

Uses a MobileNet-SSD (Caffe) model — small, fast on CPU, and loadable through
OpenCV alone (no torch / ultralytics). If the model files are not present the
detector reports unavailable and the caller falls back to face-only tracking.

Model files (place under backend/models/mobilenet_ssd/):
  * deploy.prototxt
  * mobilenet_iter_73000.caffemodel
These are the standard MobileNet-SSD VOC files (widely mirrored, ~23 MB).
"""
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# MobileNet-SSD (Pascal VOC) class index for "person".
_PERSON_CLASS = 15
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_net():
    import cv2

    s = get_settings()
    proto, weights = s.person_model_proto, s.person_model_weights
    if not (proto and weights and os.path.exists(proto) and os.path.exists(weights)):
        logger.error(
            "Person detector disabled: model files not found "
            "(proto=%s, weights=%s). Download MobileNet-SSD (deploy.prototxt + "
            "mobilenet_iter_73000.caffemodel) into backend/models/mobilenet_ssd/ "
            "to enable body tracking.",
            proto, weights,
        )
        return None
    try:
        net = cv2.dnn.readNetFromCaffe(proto, weights)
        logger.info("Person detector (MobileNet-SSD) loaded from %s", weights)
        return net
    except Exception:
        logger.exception("Failed to load person-detector model")
        return None


def is_available() -> bool:
    return _load_net() is not None


def detect_persons(bgr_image: np.ndarray, conf_threshold: float | None = None) -> list[dict]:
    """Return a list of {'box': (x1,y1,x2,y2), 'confidence': float} for people."""
    import cv2

    net = _load_net()
    if net is None:
        return []

    thr = conf_threshold if conf_threshold is not None else get_settings().person_conf
    h, w = bgr_image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr_image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    with _lock:
        net.setInput(blob)
        detections = net.forward()

    people: list[dict] = []
    for i in range(detections.shape[2]):
        cls = int(detections[0, 0, i, 1])
        conf = float(detections[0, 0, i, 2])
        if cls != _PERSON_CLASS or conf < thr:
            continue
        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))
        if x2 > x1 and y2 > y1:
            people.append({"box": (x1, y1, x2, y2), "confidence": conf})
    return people
