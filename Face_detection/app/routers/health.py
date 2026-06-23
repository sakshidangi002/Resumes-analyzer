from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.services.recognition import DEFAULT_THRESHOLD, MIN_MATCH_MARGIN

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "detector": "RetinaFace",
        "recognizer": "ArcFace",
        "default_threshold": DEFAULT_THRESHOLD,
        "min_match_margin": MIN_MATCH_MARGIN,
        "webcam_interval_ms": settings.webcam_recognition_interval_ms,
    }
