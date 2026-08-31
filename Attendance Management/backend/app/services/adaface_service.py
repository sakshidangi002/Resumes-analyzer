"""Optional AdaFace embedding backend.

The backend is deliberately opt-in. AdaFace vectors must never be mixed with
ArcFace vectors, so the model version is part of the gallery compatibility key.
"""
from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "third_party" / "AdaFace"
_WEIGHTS = Path(os.getenv(
    "ADAFACE_MODEL_PATH",
    str(_SOURCE / "pretrained" / "adaface_ir50_webface4m.ckpt"),
))


@lru_cache(maxsize=1)
def _model():
    if not _WEIGHTS.is_file():
        raise RuntimeError(f"AdaFace weights not found: {_WEIGHTS}")
    if str(_SOURCE) not in sys.path:
        sys.path.insert(0, str(_SOURCE))
    try:
        import torch
        import net
    except Exception as exc:
        raise RuntimeError(
            "AdaFace requires PyTorch and the official AdaFace source"
        ) from exc

    model = net.build_model("ir_50")
    checkpoint = torch.load(str(_WEIGHTS), map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    state = {key[6:]: value for key, value in state.items() if key.startswith("model.")}
    model.load_state_dict(state)
    model.eval()
    if os.getenv("ADAFACE_CUDA", "true").lower() in {"1", "true", "yes"} and torch.cuda.is_available():
        model = model.cuda()
    logger.info("Loaded AdaFace model %s", _WEIGHTS)
    return model


def embed(bgr_image: np.ndarray, landmarks) -> np.ndarray:
    """Return a normalized AdaFace vector from a landmark-aligned face."""
    if landmarks is None or len(landmarks) < 5:
        raise ValueError("AdaFace requires five facial landmarks")
    from insightface.utils import face_align
    import torch

    aligned = face_align.norm_crop(bgr_image, np.asarray(landmarks, dtype=np.float32), image_size=112)
    # Official AdaFace expects BGR, [0,1] -> [-1,1].
    tensor = ((aligned.astype(np.float32) / 255.0) - 0.5) / 0.5
    tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0)
    model = _model()
    if next(model.parameters()).is_cuda:
        tensor = tensor.cuda()
    with torch.inference_mode():
        feature, _quality_norm = model(tensor)
    vector = feature.detach().float().cpu().numpy().reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector

