from __future__ import annotations

import io
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.services.face_service import extract_face_embeddings


BASE_DIR = Path(__file__).resolve().parents[2]
FACE_UPLOAD_DIR = BASE_DIR / "data" / "face_uploads"


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
                detail=f"No face detected in {file.filename}. Include front, side, and partial-face photos.",
            )
        if len(faces) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple faces detected in {file.filename}. Use one face per image.",
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
