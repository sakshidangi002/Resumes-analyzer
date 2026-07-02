from __future__ import annotations

import io
import logging
import threading
from functools import lru_cache

import numpy as np
from PIL import Image

from app.core.config import get_settings


logger = logging.getLogger(__name__)

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


@lru_cache(maxsize=1)
def _get_yolo_model():
    """Load the YOLOv8-face detector (lazy, cached)."""
    from ultralytics import YOLO

    path = get_settings().yolo_face_model_path
    logger.info("Loading YOLO face detector from %s", path)
    return YOLO(path)


@lru_cache(maxsize=1)
def _get_recognizer():
    """Reuse the ArcFace recognition model bundled inside buffalo_l.

    YOLO only detects faces; the identity embedding still comes from ArcFace.
    """
    analyser = get_face_analyser()
    rec = getattr(analyser, "models", {}).get("recognition")
    if rec is None:
        raise RuntimeError("ArcFace recognition model not found in buffalo_l pack")
    return rec


def _normalize(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm > 0 else embedding


def _extract_faces_insightface(rgb_image: np.ndarray) -> list[dict]:
    import cv2

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    with _inference_lock:
        detected_faces = get_face_analyser().get(bgr_image)

    faces: list[dict] = []
    for face in detected_faces:
        faces.append(
            {
                "box": _bbox_to_list(face.bbox),
                "confidence": float(face.det_score),
                "embedding": _normalize(face.embedding),
                "pose": {
                    "yaw": float(getattr(face, "yaw", 0.0) or 0.0),
                    "pitch": float(getattr(face, "pitch", 0.0) or 0.0),
                    "roll": float(getattr(face, "roll", 0.0) or 0.0),
                },
            }
        )
    return faces


def _extract_faces_yolo(rgb_image: np.ndarray) -> list[dict]:
    """Detect with YOLOv8-face, embed with ArcFace.

    Uses the detector's 5 facial landmarks to align each crop before the
    ArcFace model computes the embedding (same alignment SCRFD uses), so the
    embeddings stay comparable to those enrolled via the InsightFace path.
    """
    import cv2
    from insightface.app.common import Face

    settings = get_settings()
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    model = _get_yolo_model()
    recognizer = _get_recognizer()

    with _inference_lock:
        results = model.predict(bgr_image, conf=float(settings.yolo_conf), verbose=False)

    faces: list[dict] = []
    if not results:
        return faces

    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return faces

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    kpts = getattr(res, "keypoints", None)
    all_kps = None
    if kpts is not None and getattr(kpts, "xy", None) is not None:
        all_kps = kpts.xy.cpu().numpy()  # (N, K, 2)

    for i in range(len(xyxy)):
        bbox = xyxy[i].astype(np.float32)
        conf = float(confs[i])

        kps = None
        if all_kps is not None and i < len(all_kps) and all_kps[i].shape[0] >= 5:
            kps = all_kps[i][:5].astype(np.float32)

        try:
            if kps is not None:
                # Aligned embedding via landmarks (preferred, best accuracy).
                face = Face(bbox=bbox, kps=kps, det_score=conf)
                with _inference_lock:
                    recognizer.get(bgr_image, face)
                embedding = face.normed_embedding if getattr(face, "normed_embedding", None) is not None else face.embedding
            else:
                # No landmarks from this model: fall back to a resized crop.
                x1, y1, x2, y2 = (int(max(0, v)) for v in bbox[:4])
                crop = bgr_image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                aligned = cv2.resize(crop, (112, 112))
                with _inference_lock:
                    embedding = recognizer.get_feat(aligned).flatten()
        except Exception as exc:
            logger.warning("YOLO embedding failed for one face: %s", exc)
            continue

        faces.append(
            {
                "box": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "confidence": conf,
                "embedding": _normalize(embedding),
                "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            }
        )

    return faces


def extract_faces_from_rgb(rgb_image: np.ndarray) -> list[dict]:
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected an RGB image array with shape (H, W, 3)")

    try:
        import cv2  # noqa: F401  (ensures OpenCV present for both backends)
    except Exception as exc:
        raise RuntimeError(
            "Face recognition dependencies are missing. Install opencv-python and insightface."
        ) from exc

    backend = (get_settings().face_detector or "insightface").lower()
    if backend == "yolo":
        try:
            return _extract_faces_yolo(rgb_image)
        except Exception as exc:
            # Never let a detector-config problem take down live recognition —
            # fall back to the always-available InsightFace detector.
            logger.error(
                "YOLO detector unavailable (%s); falling back to InsightFace", exc
            )
            return _extract_faces_insightface(rgb_image)

    return _extract_faces_insightface(rgb_image)


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
