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


# ONNX build of the SAME checkpoint, used when it is present.
#
# Embedding was the most expensive per-face operation in the pipeline. MEASURED
# on the production box, one 112x112 face:
#
#     PyTorch eager    433.7 ms
#     ONNX Runtime     129.1 ms     3.4x faster
#
# The model_version string is deliberately UNCHANGED, and that is only safe
# because it was verified rather than assumed. Compared on 60 real face crops
# from these cameras:
#
#     cosine similarity   min 0.99999988   mean 1.00000000
#     max abs difference  7.75e-07
#     worst possible shift in a gallery match score  1.19e-07
#
# The match threshold is 0.45 and the margin bar 0.18, so the two builds differ
# by six orders of magnitude less than the smallest decision the matcher makes.
# Every enrolled vector stays valid; nobody needs re-enrolling.
#
# Regenerate with scripts/export_adaface_onnx.py if the checkpoint ever changes
# — and re-run the equivalence check, because a silent drift here would look
# exactly like "the cameras got worse".
_ONNX_PATH = Path(os.getenv(
    "ADAFACE_ONNX_PATH",
    str(_ROOT / "models" / "adaface_ir50_webface4m.onnx"),
))


@lru_cache(maxsize=1)
def _onnx_session():
    """The ONNX session, or None if the export is not present."""
    if not _ONNX_PATH.is_file():
        logger.info(
            "AdaFace ONNX build not found at %s — using PyTorch (about 3x "
            "slower per face). Export it with scripts/export_adaface_onnx.py.",
            _ONNX_PATH,
        )
        return None
    try:
        import onnxruntime as ort
    except Exception:
        logger.warning("onnxruntime unavailable; AdaFace falls back to PyTorch")
        return None
    try:
        opts = ort.SessionOptions()
        # One face at a time on a contended 4-core box: extra intra-op threads
        # fight the YOLO pass for the same cores rather than adding throughput.
        opts.intra_op_num_threads = int(os.getenv("ADAFACE_ONNX_THREADS", "2"))
        sess = ort.InferenceSession(
            str(_ONNX_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        logger.info("Loaded AdaFace ONNX model %s", _ONNX_PATH)
        return sess
    except Exception:
        logger.exception("AdaFace ONNX session failed; falling back to PyTorch")
        return None


def _preprocess(bgr_image: np.ndarray, landmarks) -> np.ndarray:
    """Landmark-aligned 112x112 crop as the model wants it: BGR, [-1, 1], NCHW."""
    from insightface.utils import face_align

    aligned = face_align.norm_crop(
        bgr_image, np.asarray(landmarks, dtype=np.float32), image_size=112
    )
    chw = (((aligned.astype(np.float32) / 255.0) - 0.5) / 0.5).transpose(2, 0, 1)
    # Contiguous: AdaFace uses .view() internally and rejects a transposed view.
    return np.ascontiguousarray(chw[None, ...])


def embed(bgr_image: np.ndarray, landmarks) -> np.ndarray:
    """Return a normalized AdaFace vector from a landmark-aligned face."""
    if landmarks is None or len(landmarks) < 5:
        raise ValueError("AdaFace requires five facial landmarks")

    batch = _preprocess(bgr_image, landmarks)

    session = _onnx_session()
    if session is not None:
        vector = np.asarray(
            session.run(["embedding"], {"input": batch})[0]
        ).reshape(-1)
    else:
        import torch

        tensor = torch.from_numpy(batch)
        model = _model()
        if next(model.parameters()).is_cuda:
            tensor = tensor.cuda()
        with torch.inference_mode():
            feature, _quality_norm = model(tensor)
        vector = feature.detach().float().cpu().numpy().reshape(-1)

    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector

