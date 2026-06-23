from __future__ import annotations

import io
import shutil
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image


from app.services.recognition import DEFAULT_THRESHOLD, recognize_faces

router = APIRouter(prefix="/api", tags=["recognition"])


@router.post("/recognize-frame")
async def recognize_frame(
    file: UploadFile = File(...),
    threshold: float = DEFAULT_THRESHOLD,
):
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid frame image") from exc

    result = recognize_faces(image, threshold=threshold, source="webcam")
    return JSONResponse(result)


@router.post("/recognize-frame-only")
async def recognize_frame_only(
    file: UploadFile = File(...),
    threshold: float = DEFAULT_THRESHOLD,
):
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid frame image") from exc

    result = recognize_faces(image, threshold=threshold, source="webcam", log_attendance=False)
    return JSONResponse(result)
