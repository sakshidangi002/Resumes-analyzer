from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.services.face_service import extract_face_embeddings


BASE_DIR = Path(__file__).resolve().parents[2]
FACE_UPLOAD_DIR = BASE_DIR / "data" / "face_uploads"


# ── Enrollment quality gate ─────────────────────────────────────────────────
# A face embedding is only as good as the photo it comes from. buffalo_l/ArcFace
# is a FRONTAL recognizer, so side-profiles, blurry, tiny, or low-confidence
# faces produce weak embeddings that then fail to match at the camera. We reject
# those at upload time with a clear message instead of silently enrolling them.
# All thresholds are env-tunable, and the whole gate can be disabled.
_QUALITY_CHECK      = os.getenv("FACE_ENROLL_QUALITY_CHECK", "1").lower() in {"1", "true", "yes"}
_MIN_FACE_PX        = int(os.getenv("FACE_ENROLL_MIN_FACE_PX", "90"))     # face box min width/height
_MIN_DET_SCORE      = float(os.getenv("FACE_ENROLL_MIN_DET_SCORE", "0.62"))  # detector confidence
_MAX_NOSE_EYE_RATIO = float(os.getenv("FACE_ENROLL_MAX_NOSE_EYE_RATIO", "2.6"))  # frontal-ness (allows guided left/right angles, rejects strong profiles)
_MIN_BLUR_VAR       = float(os.getenv("FACE_ENROLL_MIN_BLUR_VAR", "25.0"))   # Laplacian variance


def _assess_face_quality(face: dict, rgb: np.ndarray) -> tuple[bool, str]:
    """Return (ok, reason). reason is a user-facing message when not ok."""
    box = face.get("box") or []
    if len(box) < 4:
        return False, "no clear face found — use a well-lit, front-facing photo"
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    w, h = x2 - x1, y2 - y1

    # Too small / too far → not enough detail for a reliable embedding.
    if w < _MIN_FACE_PX or h < _MIN_FACE_PX:
        return False, "face is too small — move closer so the face fills more of the frame"

    # Low detector confidence → occluded or extreme angle (back/side of head).
    if float(face.get("confidence", 0.0)) < _MIN_DET_SCORE:
        return False, "face is not clearly visible — look straight at the camera"

    # Frontal check from 5-point landmarks: the nose should sit roughly between
    # the eyes. A strong left/right turn skews this ratio heavily.
    kps = face.get("kps")
    if kps and len(kps) >= 3:
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        dl = abs(nose[0] - left_eye[0])
        dr = abs(right_eye[0] - nose[0])
        lo = min(dl, dr)
        ratio = (max(dl, dr) / lo) if lo > 1e-3 else 999.0
        if ratio > _MAX_NOSE_EYE_RATIO:
            return False, "face is turned to the side — use a front-facing photo (slight turns are fine)"

    # Blur check on the face crop only (background blur is irrelevant).
    try:
        import cv2

        crop = rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < _MIN_BLUR_VAR:
                return False, "photo is too blurry — hold steady and use good lighting"
    except Exception:
        # Never fail enrollment because the optional blur check errored.
        pass

    return True, ""


async def process_face_uploads(files: list[UploadFile]) -> list[dict]:
    prepared_images: list[dict] = []

    for file in files:
        image_bytes = await file.read()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {file.filename}") from exc

        faces = extract_face_embeddings(pil_image)
        if not faces:
            raise HTTPException(
                status_code=400,
                detail=f"No face detected in {file.filename}. Use a clear, front-facing photo.",
            )
        if len(faces) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple faces detected in {file.filename}. Use one face per image.",
            )

        if _QUALITY_CHECK:
            ok, reason = _assess_face_quality(faces[0], np.asarray(pil_image))
            if not ok:
                raise HTTPException(
                    status_code=400,
                    detail=f"{file.filename}: {reason}.",
                )

        prepared_images.append(
            {
                "filename": file.filename or "image.jpg",
                "bytes": image_bytes,
                "embedding": faces[0]["embedding"],
            }
        )

    return prepared_images


def save_employee_photo(employee_id: int, image_bytes: bytes, filename: str) -> str:
    employee_dir = FACE_UPLOAD_DIR / str(employee_id)
    employee_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".jpg"
    image_path = employee_dir / f"photo{suffix}"
    image_path.write_bytes(image_bytes)
    return str(image_path)


def delete_employee_photos(employee_id: int) -> None:
    """Remove all stored face photos for an employee (used when clearing enrollment)."""
    employee_dir = FACE_UPLOAD_DIR / str(employee_id)
    if employee_dir.exists():
        shutil.rmtree(employee_dir, ignore_errors=True)
