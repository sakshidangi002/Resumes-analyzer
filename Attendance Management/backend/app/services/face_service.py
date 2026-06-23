from __future__ import annotations

import io
import threading
from functools import lru_cache

import numpy as np
from PIL import Image


EMBEDDING_MODEL_VERSION = "insightface_buffalo_l_v1"
DETECTION_THRESHOLD = 0.32
DETECTION_SIZE = (640, 640)

_inference_lock = threading.Lock()


@lru_cache(maxsize=1)
def _use_cuda() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_face_analyser():
    from insightface.app import FaceAnalysis

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if _use_cuda()
        else ["CPUExecutionProvider"]
    )
    analyser = FaceAnalysis(name="buffalo_l", providers=providers)
    ctx_id = 0 if _use_cuda() else -1
    analyser.prepare(
        ctx_id=ctx_id,
        det_size=DETECTION_SIZE,
        det_thresh=DETECTION_THRESHOLD,
    )
    return analyser


def _bbox_to_list(bbox: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = bbox.astype(float).tolist()
    return [float(x1), float(y1), float(x2), float(y2)]


def extract_faces_from_rgb(rgb_image: np.ndarray) -> list[dict]:
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected an RGB image array with shape (H, W, 3)")

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "Face recognition dependencies are missing. Install opencv-python and insightface."
        ) from exc

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    with _inference_lock:
        detected_faces = get_face_analyser().get(bgr_image)

    faces: list[dict] = []
    for face in detected_faces:
        embedding = np.asarray(face.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm

        faces.append(
            {
                "box": _bbox_to_list(face.bbox),
                "confidence": float(face.det_score),
                "embedding": embedding,
                "pose": {
                    "yaw": float(getattr(face, "yaw", 0.0) or 0.0),
                    "pitch": float(getattr(face, "pitch", 0.0) or 0.0),
                    "roll": float(getattr(face, "roll", 0.0) or 0.0),
                },
            }
        )

    return faces


def extract_faces_from_image(image: Image.Image) -> list[dict]:
    rgb_image = np.asarray(image.convert("RGB"))
    return extract_faces_from_rgb(rgb_image)


def extract_faces_from_bgr(bgr_image: np.ndarray) -> list[dict]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "Face recognition dependencies are missing. Install opencv-python and insightface."
        ) from exc

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return extract_faces_from_rgb(rgb_image)


def extract_faces_from_bytes(image_bytes: bytes) -> list[dict]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return extract_faces_from_image(image)


def extract_single_face_embedding(image_path: str) -> np.ndarray | None:
    try:
        image = Image.open(image_path).convert("RGB")
        faces = extract_faces_from_image(image)
        if not faces:
            return None
        return faces[0]["embedding"]
    except Exception:
        return None


def extract_face_embeddings(image: Image.Image) -> list[dict]:
    return extract_faces_from_image(image)
