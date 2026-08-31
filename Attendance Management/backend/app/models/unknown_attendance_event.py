"""Anonymous doorway events awaiting HR review.

These rows intentionally do not use the employee attendance state machine.  A
camera can prove that somebody crossed an IN/OUT doorway without proving who
that person is.
"""
from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String, Index
from sqlalchemy.orm import relationship

from app.core.datetime_utils import get_ist_now
from app.db.base_class import Base


class UnknownAttendanceEvent(Base):
    __tablename__ = "unknown_attendance_events"
    __table_args__ = (
        Index("ix_unknown_attendance_review", "status", "event_time"),
        Index("ix_unknown_attendance_camera_time", "camera_id", "event_time"),
    )

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    event_time = Column(DateTime, nullable=False, index=True)
    attendance_date = Column(Date, nullable=False, index=True)
    event_type = Column(String(16), nullable=False)  # UNKNOWN_IN / UNKNOWN_OUT
    status = Column(String(16), nullable=False, default="PENDING", index=True)
    unknown_face_id = Column(Integer, ForeignKey("unknown_faces.id", ondelete="SET NULL"), nullable=True, index=True)
    track_id = Column(Integer, nullable=True)
    crop_path = Column(String(300), nullable=True)
    quality_score = Column(Float, nullable=True)
    match_score = Column(Float, nullable=True)
    match_margin = Column(Float, nullable=True)
    assigned_employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True, index=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=get_ist_now)

    unknown_face = relationship("UnknownFace")
    assigned_employee = relationship("Employee")

