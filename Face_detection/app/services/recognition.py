from __future__ import annotations

from datetime import datetime

import numpy as np
from PIL import Image

from app.config import get_settings
from app.db import SessionLocal
from app.services.embedding_cache import get_employee_candidates
from app.services.face_service import extract_face_embeddings, extract_faces_from_rgb
from app.services.match import find_best_match
from app.services.attendance_service import log_attendance_event, log_webcam_attendance

settings = get_settings()
DEFAULT_THRESHOLD = settings.default_threshold
MIN_MATCH_MARGIN = settings.min_match_margin

def _build_face_result(face: dict, candidates: list[dict], threshold: float, camera_id: int | None = None, log_attendance: bool = True) -> dict:
    best = find_best_match(
        face["embedding"],
        candidates,
        threshold=threshold,
        min_margin=MIN_MATCH_MARGIN,
    )
    attendance_event = None
    display_name = "Unknown"
    face_state = "unknown"
    any_match = False

    if best["status"]:
        any_match = True
        display_name = best["employee_name"]
        if not log_attendance:
            face_state = "recognized"
        elif camera_id is not None:
            with SessionLocal() as db:
                attendance_event = log_attendance_event(db, best["employee_id"], camera_id, best["score"])
                if attendance_event:
                    face_state = f"marked_{attendance_event.event_type.lower()}"
                else:
                    face_state = "already_marked"
        else:
            # Webcam mode — log check-in directly to DailyAttendance
            with SessionLocal() as db:
                webcam_record = log_webcam_attendance(db, best["employee_id"], best["score"])
                if webcam_record:
                    face_state = "marked_in"
                else:
                    face_state = "already_marked"

    return {
        "any_match": any_match,
        "attendance_event": attendance_event,
        "face_payload": {
            "box": face["box"],
            "confidence": face["confidence"],
            "employee_id": best["employee_id"],
            "employee_name": display_name,
            "score": round(best["score"], 4),
            "runner_up_score": round(best["runner_up_score"], 4),
            "margin": round(best["margin"], 4),
            "matched": display_name != "Unknown",
            "state": face_state,
            "pose": face.get("pose"),
        },
    }

def recognize_faces(image: Image.Image, threshold: float = DEFAULT_THRESHOLD, source: str = "webcam", camera_id: int | None = None, log_attendance: bool = True) -> dict:
    faces = extract_face_embeddings(image)
    return _recognize_from_faces(faces, threshold=threshold, source=source, camera_id=camera_id, log_attendance=log_attendance)

def recognize_from_rgb(rgb_image: np.ndarray, threshold: float = DEFAULT_THRESHOLD, source: str = "cctv", camera_id: int | None = None, log_attendance: bool = True) -> dict:
    faces = extract_faces_from_rgb(rgb_image)
    return _recognize_from_faces(faces, threshold=threshold, source=source, camera_id=camera_id, log_attendance=log_attendance)

def _recognize_from_faces(faces: list[dict], threshold: float, source: str, camera_id: int | None = None, log_attendance: bool = True) -> dict:
    if not faces:
        return {
            "status": False,
            "message": "No face detected",
            "source": source,
            "faces": [],
            "attendance": None,
        }

    candidates = get_employee_candidates()
    if not candidates:
        return {
            "status": False,
            "message": "No registered employees found",
            "source": source,
            "faces": [],
            "attendance": None,
        }

    results = []
    events = []
    any_match = False

    for face in faces:
        outcome = _build_face_result(face, candidates, threshold, camera_id, log_attendance)
        if outcome["any_match"]:
            any_match = True
        
        evt = outcome["attendance_event"]
        if evt:
            events.append({
                "employee_id": evt.employee_id,
                "camera_id": evt.camera_id,
                "event_type": evt.event_type,
                "timestamp": evt.timestamp.isoformat(),
                "confidence": evt.confidence,
                "employee_name": outcome["face_payload"]["employee_name"]
            })
        results.append(outcome["face_payload"])

    return {
        "status": any_match,
        "message": "Recognition complete",
        "source": source,
        "faces": results,
        "attendance": events[0] if events else None,
    }

