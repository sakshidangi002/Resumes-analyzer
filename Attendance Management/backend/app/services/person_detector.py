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
from app.services.inference_gate import inference_slot

logger = logging.getLogger(__name__)

# MobileNet-SSD (Pascal VOC) class index for "person".
_PERSON_CLASS = 15
_lock = threading.Lock()


@lru_cache(maxsize=4)
def _load_net(model_path: str | None = None):
    """Load the configured YOLO ONNX model, or the legacy Caffe fallback.

    Ultralytics is optional in this deployment. OpenCV's DNN backend can run
    the checked-in YOLO11 ONNX model directly, which keeps monitor cameras
    capable of detecting whole bodies when the Python Ultralytics package is
    unavailable.
    """
    import cv2

    s = get_settings()
    requested = model_path or getattr(s, "yolo_person_model_path", "")
    candidates = [requested]
    shared_onnx = getattr(s, "yolo_person_model_path", "")
    if shared_onnx not in candidates:
        candidates.append(shared_onnx)
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and candidate.lower().endswith(".onnx"):
            try:
                net = cv2.dnn.readNetFromONNX(candidate)
                logger.info("Person detector (YOLO ONNX/OpenCV) loaded from %s", candidate)
                return ("yolo", net)
            except Exception:
                logger.exception("Failed to load YOLO ONNX person-detector model: %s", candidate)

    proto, weights = s.person_model_proto, s.person_model_weights
    if proto and weights and os.path.exists(proto) and os.path.exists(weights):
        try:
            net = cv2.dnn.readNetFromCaffe(proto, weights)
            logger.info("Person detector (MobileNet-SSD) loaded from %s", weights)
            return ("caffe", net)
        except Exception:
            logger.exception("Failed to load person-detector model")

    logger.error(
        "Person detector disabled: no usable YOLO ONNX or MobileNet-SSD model "
        "(requested=%s, proto=%s, weights=%s)", requested, proto, weights,
    )
    return None


def is_available(model_path: str | None = None) -> bool:
    return _load_net(model_path) is not None


def detect_persons(
    bgr_image: np.ndarray,
    conf_threshold: float | None = None,
    *,
    model_path: str | None = None,
) -> list[dict]:
    """Return a list of {'box': (x1,y1,x2,y2), 'confidence': float} for people."""
    import cv2

    loaded = _load_net(model_path)
    if loaded is None:
        return []
    backend, net = loaded

    thr = conf_threshold if conf_threshold is not None else get_settings().person_conf
    h, w = bgr_image.shape[:2]

    if backend == "yolo":
        # The checked-in yolo11m.onnx is exported with a STATIC 960x960 input
        # (verified: graph input is [1,3,960,960]). It cannot be run at any
        # other size, which is why no size parameter is accepted here -- passing
        # one would silently do nothing. Boxes are scaled back to the frame.
        size = 960
        blob = cv2.dnn.blobFromImage(
            bgr_image, 1.0 / 255.0, (size, size), swapRB=True, crop=False
        )
        # The priority gate, NOT a module-level lock. A plain lock here would
        # serialise every camera process-wide and serve them first-come-first-
        # served, so a room camera's pass would block the entrance -- exactly the
        # failure inference_gate was written to remove. The gate still bounds
        # concurrency (a single inference already saturates this box) but admits
        # attendance cameras first.
        with inference_slot():
            net.setInput(blob)
            output = net.forward()

        # Ultralytics YOLO11 ONNX export: [1, 4 + classes, candidates], so this
        # is (candidates, 4 + classes) after the transpose -- 18900 rows at
        # 960x960. Decoding that with a Python loop cost ~30 ms per frame per
        # camera; numpy does the same work in well under a millisecond.
        predictions = output[0].T if output.ndim == 3 else output.T
        if predictions.shape[1] <= 4:
            return []

        scores_all = predictions[:, 4]          # COCO class 0 = person
        keep_mask = scores_all >= thr
        if not keep_mask.any():
            return []

        cx = predictions[keep_mask, 0]
        cy = predictions[keep_mask, 1]
        bw = predictions[keep_mask, 2]
        bh = predictions[keep_mask, 3]
        sx, sy = w / float(size), h / float(size)

        x1 = np.clip((cx - bw / 2.0) * sx, 0, w).astype(int)
        y1 = np.clip((cy - bh / 2.0) * sy, 0, h).astype(int)
        x2 = np.clip((cx + bw / 2.0) * sx, 0, w).astype(int)
        y2 = np.clip((cy + bh / 2.0) * sy, 0, h).astype(int)

        valid = (x2 > x1) & (y2 > y1)
        if not valid.any():
            return []

        # NMSBoxes wants [x, y, w, h] and plain Python types.
        boxes = np.stack([x1[valid], y1[valid], x2[valid] - x1[valid], y2[valid] - y1[valid]], axis=1)
        scores = scores_all[keep_mask][valid].astype(float)
        box_list = boxes.tolist()
        score_list = scores.tolist()

        keep = cv2.dnn.NMSBoxes(box_list, score_list, float(thr), 0.50)
        indices = keep.flatten().tolist() if len(keep) else []
        return [
            {
                "box": (box_list[i][0], box_list[i][1],
                        box_list[i][0] + box_list[i][2], box_list[i][1] + box_list[i][3]),
                "confidence": score_list[i],
            }
            for i in indices
        ]

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
