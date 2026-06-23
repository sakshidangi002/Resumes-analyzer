from __future__ import annotations

from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.services.recognition import DEFAULT_THRESHOLD, recognize_faces

router = APIRouter()


@router.post("/recognize-frame")
async def recognize_frame(file: UploadFile = File(...), threshold: float = DEFAULT_THRESHOLD):
    """Accept a webcam frame and recognize the face inside the HRMS backend."""
    image_bytes = await file.read()
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image uploaded") from exc

    try:
        data = recognize_faces(image, threshold=threshold, source="webcam")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {exc}") from exc

    return JSONResponse(data)
