from __future__ import annotations

import logging
import threading
from datetime import datetime, time, timedelta

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.datetime_utils import get_ist_now
from app.db.session import SessionLocal
from app.models.employee import Employee, EmploymentStatus
from app.services.attendance_service import (
    apply_status_from_hours,
    calculate_work_hours,
    get_company_config,
    get_or_create_attendance,
)
from app.services.embedding_cache import get_employee_candidates
from app.services.face_service import extract_face_embeddings, extract_faces_from_rgb
from app.services.match import find_best_match


logger = logging.getLogger(__name__)

settings = get_settings()
DEFAULT_THRESHOLD = settings.default_threshold
MIN_MATCH_MARGIN = settings.min_match_margin
WEBCAM_ATTENDANCE_COOLDOWN_SECONDS = 8

_webcam_attendance_lock = threading.Lock()
_webcam_last_marked_at: dict[int, datetime] = {}


def _is_within_webcam_cooldown(employee_id: int, now_dt: datetime) -> bool:
    with _webcam_attendance_lock:
        last_marked_at = _webcam_last_marked_at.get(employee_id)
        if last_marked_at is not None and now_dt - last_marked_at < timedelta(seconds=WEBCAM_ATTENDANCE_COOLDOWN_SECONDS):
            return True
        _webcam_last_marked_at[employee_id] = now_dt
        return False


def _mark_attendance(employee_id: int, now_dt: datetime) -> tuple[dict | None, str]:
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

        today = now_dt.date()
        now_time = now_dt.time().replace(microsecond=0)
        rec = get_or_create_attendance(db, employee_id, today)

        if rec.sign_in_time is not None:
            logger.info(
                "face-match employee=%s employee_id=%s action=already_marked attendance_date=%s",
                employee.full_name,
                employee.id,
                rec.date.isoformat(),
            )
            return {
                "employee_id": employee.id,
                "employee_code": employee.employee_code,
                "employee_name": employee.full_name,
                "date": rec.date.isoformat(),
                "sign_in_time": rec.sign_in_time.isoformat() if rec.sign_in_time else None,
                "sign_out_time": rec.sign_out_time.isoformat() if rec.sign_out_time else None,
                "status": rec.status,
            }, "already_marked"

        if _is_within_webcam_cooldown(employee_id, now_dt):
            logger.info(
                "face-match employee=%s employee_id=%s action=cooldown attendance_date=%s",
                employee.full_name,
                employee.id,
                today.isoformat(),
            )
            return None, "cooldown"

        rec.sign_in_time = now_time
        rec.sign_out_time = None
        rec.total_work_hours = None
        rec.status = "PRESENT"
        config = get_company_config(db)
        grace_min = config.grace_time_minutes if config else 15
        standard_start = time(9, 0)
        t = datetime.combine(today, now_time)
        s = datetime.combine(today, standard_start)
        rec.is_late = (t - s).total_seconds() > grace_min * 60
        rec.is_early_exit = False
        rec.source = "AUTO"
        db.commit()
        db.refresh(rec)

        logger.info(
            "face-match employee=%s employee_id=%s action=marked_in attendance_date=%s sign_in=%s",
            employee.full_name,
            employee.id,
            rec.date.isoformat(),
            rec.sign_in_time.isoformat() if rec.sign_in_time else None,
        )

        return {
            "employee_id": employee.id,
            "employee_code": employee.employee_code,
            "employee_name": employee.full_name,
            "date": rec.date.isoformat(),
            "sign_in_time": rec.sign_in_time.isoformat() if rec.sign_in_time else None,
            "sign_out_time": rec.sign_out_time.isoformat() if rec.sign_out_time else None,
            "status": rec.status,
        }, "marked_in"


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