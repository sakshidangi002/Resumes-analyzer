"""Write and review anonymous doorway events without assigning identity."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.core.datetime_utils import get_ist_now
from app.services.attendance_event_service import business_date, to_naive_ist
from datetime import timedelta


def record(*, camera_id: str, purpose: str, event_time: datetime,
           track_id: Optional[int] = None, unknown_face_id: Optional[int] = None,
           crop_path: Optional[str] = None, quality_score: Optional[float] = None,
           match_score: Optional[float] = None, match_margin: Optional[float] = None) -> Optional[int]:
    """Record one anonymous crossing; MONITOR cameras are refused."""
    direction = (purpose or "").upper()
    if direction not in {"IN", "OUT"}:
        return None
    from app.db.session import SessionLocal
    from app.models.unknown_attendance_event import UnknownAttendanceEvent
    from app.models.unknown_face import UnknownFace

    when = to_naive_ist(event_time or get_ist_now())
    event_type = f"UNKNOWN_{direction}"
    with SessionLocal() as db:
        if unknown_face_id and not crop_path:
            face = db.query(UnknownFace).filter(UnknownFace.id == unknown_face_id).first()
            crop_path = face.crop_path if face else None
        # Idempotency protects against a camera retry and tracker fragmentation.
        recent = db.query(UnknownAttendanceEvent).filter(
            UnknownAttendanceEvent.camera_id == str(camera_id),
            UnknownAttendanceEvent.event_type == event_type,
            UnknownAttendanceEvent.event_time >= when - timedelta(seconds=20),
            UnknownAttendanceEvent.event_time <= when + timedelta(seconds=20),
        ).order_by(UnknownAttendanceEvent.event_time.desc()).first()
        if recent is not None and (when - recent.event_time).total_seconds() <= 20:
            return int(recent.id)
        row = UnknownAttendanceEvent(
            camera_id=str(camera_id), event_time=when,
            attendance_date=business_date(when), event_type=event_type,
            status="PENDING", unknown_face_id=unknown_face_id,
            track_id=track_id, crop_path=crop_path,
            quality_score=quality_score, match_score=match_score,
            match_margin=match_margin,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)


def list_events(*, status: str = "PENDING", limit: int = 100) -> list[dict]:
    from app.db.session import SessionLocal
    from app.models.unknown_attendance_event import UnknownAttendanceEvent
    with SessionLocal() as db:
        rows = db.query(UnknownAttendanceEvent).filter(
            UnknownAttendanceEvent.status == status.upper()
        ).order_by(UnknownAttendanceEvent.event_time.desc()).limit(limit).all()
        return [{"id": r.id, "camera_id": r.camera_id, "event_time": r.event_time,
                 "attendance_date": r.attendance_date, "event_type": r.event_type,
                 "status": r.status, "track_id": r.track_id,
                 "crop_path": r.crop_path, "quality_score": r.quality_score,
                 "match_score": r.match_score, "match_margin": r.match_margin,
                 "assigned_employee_id": r.assigned_employee_id} for r in rows]


def resolve(event_id: int, *, employee_id: Optional[int], reviewed_by: Optional[int]) -> dict:
    """Assign or dismiss an event.

    Assignment is an explicit HR action: it creates the corresponding normal
    IN/OUT event and enrolls the captured camera embedding. Automatic camera
    processing never performs either action for an unknown person.
    """
    from app.db.session import SessionLocal
    from app.models.unknown_attendance_event import UnknownAttendanceEvent
    from app.services.employee_face_service import enroll_embeddings
    from app.services.attendance_event_service import add_attendance_event
    from app.models.unknown_face import UnknownFace
    import numpy as np
    with SessionLocal() as db:
        row = db.query(UnknownAttendanceEvent).filter(UnknownAttendanceEvent.id == event_id).first()
        if row is None:
            raise ValueError("unknown attendance event not found")
        if employee_id is not None:
            if db.query(UnknownFace).filter(UnknownFace.id == row.unknown_face_id).first() is None:
                raise ValueError("event has no review crop/embedding")
            direction = "IN" if row.event_type.endswith("_IN") else "OUT"
            add_attendance_event(
                db, int(employee_id), event_time=row.event_time, event_type=direction,
                source="REVIEW", camera_id=row.camera_id,
                evidence={"match_score": row.match_score, "match_margin": row.match_margin,
                          "track_id": row.track_id, "snapshot_path": row.crop_path},
            )
            face = db.query(UnknownFace).filter(UnknownFace.id == row.unknown_face_id).first()
            vector = np.frombuffer(face.embedding, dtype=np.float32) if face and face.embedding else None
            if vector is not None and vector.size:
                enroll_embeddings(
                    employee_id=int(employee_id),
                    observations=[{"embedding": vector, "camera_id": row.camera_id,
                                  "quality_score": row.quality_score}],
                    source="cctv",
                )
        row.status = "ASSIGNED" if employee_id is not None else "IGNORED"
        linked_face = db.query(UnknownFace).filter(UnknownFace.id == row.unknown_face_id).first()
        if linked_face is not None:
            linked_face.status = "ASSIGNED" if employee_id is not None else "IGNORED"
            linked_face.assigned_employee_id = int(employee_id) if employee_id is not None else None
        row.assigned_employee_id = int(employee_id) if employee_id is not None else None
        row.reviewed_by = reviewed_by
        row.reviewed_at = get_ist_now()
        db.commit()
        return {"id": row.id, "status": row.status, "assigned_employee_id": row.assigned_employee_id}
