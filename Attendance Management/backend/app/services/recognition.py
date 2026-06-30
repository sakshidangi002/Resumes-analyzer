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


def _mark_attendance(
    employee_id: int,
    now_dt=None,
    camera_id: str | None = None,
    camera_purpose: str | None = None,   # "IN" | "OUT" | None
) -> tuple[dict | None, str]:
    """
    Step 7: Attendance service called.
    Step 8: Business logic determines event type (IN/OUT/cooldown).
    Step 9: Attendance record inserted into database.
    Step 10: Daily attendance summary updated.
    """
    logger.info(
        "STEP-7 attendance_service_called employee_id=%s camera_id=%s camera_purpose=%s",
        employee_id, camera_id, camera_purpose,
    )

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
            logger.warning(
                "STEP-7 FAILED employee_id=%s reason=not_found_or_inactive",
                employee_id,
            )
            return None, "unknown"

        logger.info(
            "STEP-8 business_logic_start employee=%s employee_id=%s determining_event_type",
            employee.full_name, employee.id,
        )

        try:
            event, rec, action = record_face_attendance(
                db, employee_id, now_dt=now_dt,
                camera_id=camera_id,
                camera_purpose=camera_purpose,
            )
        except ValueError as exc:
            logger.warning(
                "STEP-8 FAILED employee=%s employee_id=%s reason=validation_error detail=%s",
                employee.full_name,
                employee.id,
                exc,
            )
            return None, "validation_failed"
        except Exception as exc:
            logger.exception(
                "STEP-9 FAILED employee=%s employee_id=%s reason=database_error detail=%s",
                employee.full_name,
                employee.id,
                exc,
            )
            return None, "attendance_failed"

        if action == "cooldown":
            logger.info(
                "STEP-8 cooldown employee=%s employee_id=%s attendance_date=%s "
                "reason=duplicate_within_60s",
                employee.full_name,
                employee.id,
                rec.date.isoformat(),
            )
            return _attendance_payload(rec, employee), "cooldown"

        event_type = event.event_type if event else action
        logger.info(
            "STEP-9 database_insert_successful employee=%s employee_id=%s "
            "event_type=%s attendance_date=%s event_time=%s",
            employee.full_name,
            employee.id,
            event_type,
            rec.date.isoformat(),
            event.event_time.isoformat() if event else None,
        )
        logger.info(
            "STEP-10 daily_attendance_updated employee=%s employee_id=%s "
            "sign_in=%s sign_out=%s work_hours=%s status=%s",
            employee.full_name,
            employee.id,
            rec.sign_in_time,
            rec.sign_out_time,
            rec.total_work_hours,
            rec.status,
        )
        return _attendance_payload(rec, employee, event_type=event_type), event_type.lower()


def _build_face_result(
    face: dict,
    candidates: list[dict],
    threshold: float,
    source: str,
    camera_id: str | None = None,
    camera_purpose: str | None = None,
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

        # ── Step 3: Recognition ──────────────────────────────────────────────
        logger.info(
            "STEP-3 face_recognition_match employee_name=%s employee_id=%s "
            "employee_code=%s score=%.4f margin=%.4f source=%s",
            employee_name,
            employee_id,
            employee_code,
            float(best["score"]),
            float(best["margin"]),
            source,
        )

        # ── Step 4: Employee ID returned ────────────────────────────────────
        logger.info(
            "STEP-4 employee_id_returned employee_id=%s source=%s",
            employee_id, source,
        )

        # ── Step 5: Confidence threshold passes ─────────────────────────────
        logger.info(
            "STEP-5 confidence_threshold_passed score=%.4f threshold=%.4f margin=%.4f",
            float(best["score"]), threshold, float(best["margin"]),
        )

        # ── Step 6: Attendance event triggered (webcam/cctv only) ───────────
        if source in {"webcam", "cctv"}:
            logger.info(
                "STEP-6 attendance_trigger employee_name=%s employee_id=%s "
                "source=%s camera_id=%s camera_purpose=%s",
                employee_name, employee_id, source, camera_id, camera_purpose,
            )
            try:
                attendance_info, face_state = _mark_attendance(
                    int(best["employee_id"]),
                    get_ist_now(),
                    camera_id=camera_id,
                    camera_purpose=camera_purpose,
                )
                logger.info(
                    "STEP-6 attendance_trigger_complete employee_name=%s "
                    "employee_id=%s face_state=%s attendance_recorded=%s",
                    employee_name, employee_id, face_state,
                    attendance_info is not None,
                )
            except Exception:
                # This outer catch should never trigger because _mark_attendance
                # handles its own exceptions — but keep it as a safety net and
                # NEVER silently swallow it.
                logger.exception(
                    "STEP-6 CRITICAL attendance_trigger_exception "
                    "employee_name=%s employee_id=%s source=%s",
                    employee_name, employee_id, source,
                )
                attendance_info, face_state = None, "attendance_failed"
        else:
            # Photo upload or other non-live source — do not record attendance
            logger.info(
                "STEP-6 skipped_non_live_source source=%s employee_name=%s",
                source, employee_name,
            )
            face_state = "recognized"
    else:
        logger.info(
            "STEP-3 face_recognition_no_match best_candidate=%s employee_id=%s "
            "score=%.4f margin=%.4f source=%s",
            best.get("employee_name") or "Unknown",
            employee_id,
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
    camera_id: str | None = None,
    camera_purpose: str | None = None,
) -> dict:
    faces = extract_face_embeddings(image)
    return _recognize_from_faces(
        faces, threshold=threshold, source=source,
        camera_id=camera_id, camera_purpose=camera_purpose,
    )


def recognize_from_rgb(
    rgb_image: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    source: str = "cctv",
    camera_id: str | None = None,
    camera_purpose: str | None = None,
) -> dict:
    faces = extract_faces_from_rgb(rgb_image)
    return _recognize_from_faces(
        faces, threshold=threshold, source=source,
        camera_id=camera_id, camera_purpose=camera_purpose,
    )


def _recognize_from_faces(
    faces: list[dict],
    threshold: float,
    source: str,
    camera_id: str | None = None,
    camera_purpose: str | None = None,
) -> dict:
    # ── Step 1: Webcam frame received ────────────────────────────────────────
    logger.info(
        "STEP-1 frame_received source=%s face_count=%d camera_id=%s camera_purpose=%s",
        source, len(faces), camera_id, camera_purpose,
    )

    if not faces:
        logger.debug("STEP-2 face_detection=none source=%s", source)
        return {
            "status": False,
            "message": "No face detected",
            "source": source,
            "faces": [],
            "attendance": None,
        }

    candidates = get_employee_candidates()
    if not candidates:
        logger.warning(
            "STEP-2 face_detection_ok but no_registered_employees source=%s",
            source,
        )
        return {
            "status": False,
            "message": "No registered employees found",
            "source": source,
            "faces": [],
            "attendance": None,
        }

    # ── Step 2: Face detection ───────────────────────────────────────────────
    logger.info(
        "STEP-2 face_detection_complete faces_detected=%d candidates_available=%d source=%s",
        len(faces), len(candidates), source,
    )

    results = []
    attendance = None
    any_match = False

    for face in faces:
        outcome = _build_face_result(
            face, candidates, threshold, source,
            camera_id=camera_id, camera_purpose=camera_purpose,
        )
        any_match = any_match or outcome["any_match"]
        if attendance is None and outcome["attendance"] is not None:
            attendance = outcome["attendance"]
        results.append(outcome["face_payload"])

    # ── Step 11: UI notification ──────────────────────────────────────────────
    matched_names = [r["employee_name"] for r in results if r.get("matched")]
    logger.info(
        "STEP-11 recognition_complete source=%s any_match=%s matched=%s "
        "attendance_recorded=%s",
        source, any_match, matched_names, attendance is not None,
    )

    return {
        "status": any_match,
        "message": "Recognition complete",
        "source": source,
        "faces": results,
        "attendance": attendance,
    }
