"""Person Re-Identification — body/appearance embeddings (OSNet x0.25, MSMT17).

Gives each detected body a 512-d L2-normalised appearance vector ("what this
person looks like": clothes, build, hair). Used by the Global Identity Manager to
keep an employee's name on their body track when their FACE is not visible —
e.g. seated with their back to a ceiling-mounted dev-room camera.

Runs on onnxruntime (CPU) — no new dependencies, and it does NOT touch torch or
ultralytics. Cost is ~12 ms per person crop, negligible next to the YOLO pass.

NOTE: the exported ONNX graph has a FIXED batch of 16, so crops are padded up to
16 and the outputs sliced back.

Appearance is CLOTHING-based, therefore only valid within a single day — the
gallery in identity_manager is day-scoped for exactly this reason.
"""
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache

import cv2
import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Serialises ONNX inference (the session is shared; keeps CPU use predictable).
_lock = threading.Lock()

_BATCH = 16          # fixed by the exported graph
_H, _W = 256, 128    # OSNet input (h, w)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Crops smaller than this describe nothing useful (a sliver of a chair, etc.).
_MIN_W, _MIN_H = 20, 40


def _resolve(path: str) -> str | None:
    if os.path.exists(path):
        return path
    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    alt = os.path.join(backend_root, path)
    return alt if os.path.exists(alt) else None


@lru_cache(maxsize=1)
def _session():
    path = _resolve(get_settings().reid_model_path)
    if path is None:
        logger.warning(
            "ReID model not found (%s) — cross-camera identity disabled",
            get_settings().reid_model_path,
        )
        return None
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception:
        logger.exception("Failed to load ReID (OSNet) model")
        return None
    logger.info("ReID (OSNet x0.25) loaded from %s", path)
    return sess


def is_available() -> bool:
    return _session() is not None


def _prep(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (_W, _H)).astype(np.float32) / 255.0
    r = (r - _MEAN) / _STD
    return r.transpose(2, 0, 1)  # HWC → CHW


def extract_body_embeddings(frame_bgr: np.ndarray, boxes: list) -> list:
    """Embed each person box. Returns a list aligned with `boxes`; entries are a
    512-d L2-normalised np.float32 vector, or None when the crop is unusable."""
    out: list = [None] * len(boxes)
    sess = _session()
    if sess is None or not boxes:
        return out

    fh, fw = frame_bgr.shape[:2]
    crops: list[np.ndarray] = []
    slots: list[int] = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if (x2 - x1) < _MIN_W or (y2 - y1) < _MIN_H:
            continue
        crops.append(_prep(frame_bgr[y1:y2, x1:x2]))
        slots.append(i)

    if not crops:
        return out

    name = sess.get_inputs()[0].name
    for start in range(0, len(crops), _BATCH):
        chunk = crops[start:start + _BATCH]
        batch = np.zeros((_BATCH, 3, _H, _W), dtype=np.float32)  # pad to fixed batch
        for j, c in enumerate(chunk):
            batch[j] = c
        try:
            with _lock:
                emb = sess.run(None, {name: batch})[0]
        except Exception:
            logger.exception("ReID inference failed")
            return out
        norms = np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6, None)
        emb = (emb / norms).astype(np.float32)
        for j in range(len(chunk)):
            out[slots[start + j]] = emb[j]

    return out
