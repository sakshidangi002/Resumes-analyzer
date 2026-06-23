from __future__ import annotations

import io
from pathlib import Path
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.db import UPLOAD_DIR
from app.services.face_service import extract_face_embeddings

async def process_face_uploads(files: list[UploadFile]) -> list[dict]:
    prepared_images = []

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
                detail=f"No face detected in {file.filename}. Include front, side, and partial-face photos.",
            )
        if len(faces) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple faces detected in {file.filename}. Use one face per image.",
            )

        face = faces[0]
        prepared_images.append(
            {
                "filename": file.filename or "image.jpg",
                "bytes": image_bytes,
                "embedding": face["embedding"],
            }
        )

    return prepared_images

def save_employee_photo(employee_id: int, image_bytes: bytes, filename: str) -> str:
    employee_dir = UPLOAD_DIR / str(employee_id)
    employee_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".jpg"
    image_path = employee_dir / f"photo{suffix}"
    image_path.write_bytes(image_bytes)
    return str(image_path)

def serialize_employee_row(employee) -> dict:
    return {
        "id": employee.id,
        "name": employee.name,
        "department": employee.department,
        "photo_path": employee.photo_path,
        "sample_count": employee.sample_count,
        "created_at": employee.created_at.isoformat() if employee.created_at else None,
    }

