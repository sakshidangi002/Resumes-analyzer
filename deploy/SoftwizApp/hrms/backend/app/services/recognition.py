from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.datetime_utils import get_ist_now
from app.db.session import SessionLocal
from app.models.employee import Employee, EmploymentStatus
from app.services.embedding_cache import get_employee_candidates
from app.services.face_service import extract_face_embeddings, extract_faces_from_rgb
from app.services.match import find_best_match
from app.services.attendance_event_service import record_face_attendance


logger = logging.getLogger(__name__)

settings = get_settings()
DEFAULT_THRESHOLD = settings.default_threshold
MIN_MATCH_MARGIN = settings.min_match_margin


def _attendance_payload(rec, employee, event_type: str | None = None) -> dict:
    return {
        "employee_id": employee.id,
        "employee_code": employee.employee_code,
        "employee_name": employee.full_name,
        "date": rec.date.isoformat(),
        "sign_in_time": rec.sign_in_time.isoformat() if rec.sign_in_time else None,
        "sign_out_time": rec.sign_out_time.isoformat() if rec.sign_out_time else None,
        "total_work_hours": float(rec.total_work_hours) if rec.total_work_hours is not None else None,
        "total_break_hours": float(rec.total_break_hours) if rec.total_break_hours is not None else None,
        "status": rec.status,
        "event_type": event_type,
    }


def _mark_attendance(employee_id: int, now_dt=None) -> tuple[dict | None, str]:
    with SessionLocal() as db:
        employee = (
            db.query(Employee)
            .filter(
                Employee.id == employee_id,
                Employee.employment_status == EmploymentStatus.ACTIVE.value,
            )
            .first()
        )
        if not employee:
            logger.info("face-match employee_id=%s action=unknown_employee", employee_id)
            return None, "unknown"

        try:
            event, rec, action = record_face_attendance(db, employee_id, now_dt=now_dt)
        except ValueError as exc:
            logger.info(
                "face-match employee=%s employee_id=%s action=validation_failed detail=%s",
                employee.full_name,
                employee.id,
                exc,
            )
            return None, "validation_failed"

        if action == "cooldown":
            logger.info(
                "face-match employee=%s employee_id=%s action=cooldown attendance_date=%s",
                employee.full_name,
                employee.id,
                rec.date.isoformat(),
            )
            return _attendance_payload(rec, employee), "cooldown"

        event_type = event.event_type if event else action
        logger.info(
            "face-match employee=%s employee_id=%s action=%s attendance_date=%s event_time=%s",
            employee.full_name,
            employee.id,
            event_type,
            rec.date.isoformat(),
            event.event_time.isoformat() if event else None,
        )
        return _attendance_payload(rec, employee, event_type=event_type), event_type.lower()


def _build_face_result(
    face: dict,
    candidates: list[dict],
    threshold: float,
    source: str,
) -> dict:
    best = find_best_match(
        face["embedding"],
        candidates,
        threshold=threshold,
        min_margin=MIN_MATCH_MARGIN,
    )

    face_state = "unknown"
    attendance_info = None
    matched = False
    employee_name = "Unknown"
    employee_id = best["employee_id"]
    employee_code = best.get("employee_code")

    if best["status"]:
        matched = True
        employee_name = best["employee_name"]
        logger.info(
            "face-match detected_face=match employee_name=%s employee_id=%s employee_code=%s score=%.4f margin=%.4f source=%s",
            employee_name,
            employee_id,
            employee_code,
            float(best["score"]),
            float(best["margin"]),
            source,
        )
        if source == "webcam":
            attendance_info, face_state = _mark_attendance(int(best["employee_id"]), get_ist_now())
        else:
            face_state = "recognized"
    else:
        logger.info(
            "face-match detected_face=unknown employee_name=%s employee_id=%s employee_code=%s score=%.4f margin=%.4f source=%s",
            best.get("employee_name") or "Unknown",
            employee_id,
            employee_code,
            float(best.get("score", -1.0)),
            float(best.get("margin", 0.0)),
            source,
        )

    return {
        "any_match": matched,
        "attendance": attendance_info,
        "face_payload": {
            "box": face["box"],
            "confidence": face["confidence"],
            "employee_id": employee_id,
            "employee_code": employee_code,
            "employee_name": employee_name,
            "score": round(best["score"], 4),
            "runner_up_score": round(best["runner_up_score"], 4),
            "margin": round(best["margin"], 4),
            "matched": matched,
            "state": face_state,
            "pose": face.get("pose"),
        },
    }


def recognize_faces(
    image: Image.Image,
    threshold: float = DEFAULT_THRESHOLD,
    source: str = "webcam",
) -> dict:
    faces = extract_face_embeddings(image)
    return _recognize_from_faces(faces, threshold=threshold, source=source)


def recognize_from_rgb(
    rgb_image: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    source: str = "cctv",
) -> dict:
    faces = extract_faces_from_rgb(rgb_image)
    return _recognize_from_faces(faces, threshold=threshold, source=source)


def _recognize_from_faces(
    faces: list[dict],
    threshold: float,
    source: str,
) -> dict:
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
    attendance = None
    any_match = False

    for face in faces:
        outcome = _build_face_result(face, candidates, threshold, source)
        any_match = any_match or outcome["any_match"]
        if attendance is None and outcome["attendance"] is not None:
            attendance = outcome["attendance"]
        results.append(outcome["face_payload"])

    return {
        "status": any_match,
        "message": "Recognition complete",
        "source": source,
        "faces": results,
        "attendance": attendance,
    }
