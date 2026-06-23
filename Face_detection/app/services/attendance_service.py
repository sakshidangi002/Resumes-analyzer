from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models.employee import Employee
from app.models.camera import Camera
from app.models.event import AttendanceEvent
from app.models.daily import DailyAttendance

def log_attendance_event(db: Session, employee_id: int, camera_id: int, confidence: float) -> AttendanceEvent | None:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        return None

    now = datetime.now(timezone.utc)
    today = now.date()
    
    event = AttendanceEvent(
        employee_id=employee_id,
        camera_id=camera_id,
        event_type=camera.camera_type,
        confidence=confidence,
        timestamp=now
    )
    db.add(event)

    daily = db.query(DailyAttendance).filter(
        DailyAttendance.employee_id == employee_id,
        DailyAttendance.date == today
    ).first()

    if not daily:
        daily = DailyAttendance(
            employee_id=employee_id,
            date=today,
        )
        db.add(daily)

    if camera.camera_type == "IN":
        if not daily.first_in:
            daily.first_in = now
    elif camera.camera_type == "OUT":
        daily.last_out = now
        if daily.first_in:
            first_in = daily.first_in
            if first_in.tzinfo is None:
                first_in = first_in.replace(tzinfo=timezone.utc)
            diff = daily.last_out - first_in
            daily.total_hours = round(diff.total_seconds() / 3600.0, 2)

    db.commit()
    db.refresh(event)
    return event


def log_webcam_attendance(db: Session, employee_id: int, confidence: float) -> DailyAttendance | None:
    """Log a webcam check-in directly to DailyAttendance (no camera record required).
    Returns the DailyAttendance if a new check-in was recorded, or None if already marked today.
    """
    now = datetime.now(timezone.utc)
    today = now.date()

    daily = db.query(DailyAttendance).filter(
        DailyAttendance.employee_id == employee_id,
        DailyAttendance.date == today
    ).first()

    if daily and daily.first_in:
        # Already checked in today — do not overwrite
        return None

    if not daily:
        daily = DailyAttendance(
            employee_id=employee_id,
            date=today,
        )
        db.add(daily)

    daily.first_in = now
    db.commit()
    db.refresh(daily)
    return daily
