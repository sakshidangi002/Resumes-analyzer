"""Attendance records and correction requests."""
from datetime import datetime, date, time
from sqlalchemy import (
    Column, Integer, Date, Time, ForeignKey, Boolean, DateTime, Float, String,
    Numeric, UniqueConstraint,
)
from sqlalchemy.orm import relationship
from app.db.base_class import Base

from app.core.datetime_utils import get_ist_now


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    # One summary row per employee per day. Enforced in the DB because several
    # camera workers can recognise the same person concurrently and would
    # otherwise each insert their own row (inflating present-day counts).
    __table_args__ = (
        UniqueConstraint("employee_id", "date", name="uq_attendance_employee_date"),
    )

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    sign_in_time = Column(Time, nullable=True)
    sign_out_time = Column(Time, nullable=True)
    # A time entered by HR is authoritative: the camera can miss an employee
    # entirely, so a later detection must never overwrite what HR typed in.
    # Pinned per side, so the camera can still fill the side HR left blank.
    sign_in_manual = Column(Boolean, nullable=False, default=False, server_default="false")
    sign_out_manual = Column(Boolean, nullable=False, default=False, server_default="false")
    # Same for the break: if the camera missed the employee it saw no break
    # either, so HR must be able to enter one and have it deducted.
    break_manual = Column(Boolean, nullable=False, default=False, server_default="false")
    total_work_hours = Column(Numeric(5, 2), nullable=True)
    total_break_hours = Column(Numeric(5, 2), nullable=True)
    status = Column(String(20), nullable=False)  # PRESENT, ABSENT, HALF_DAY, ON_LEAVE
    is_late = Column(Boolean, default=False)
    is_early_exit = Column(Boolean, default=False)
    is_weekly_off = Column(Boolean, default=False)
    is_holiday = Column(Boolean, default=False)
    source = Column(String(20), default="SELF")  # SELF, AUTO, CORRECTION
    created_at = Column(DateTime, default=get_ist_now)
    updated_at = Column(DateTime, default=get_ist_now, onupdate=get_ist_now)

    employee = relationship("Employee", backref="attendance_records")


class AttendanceCorrectionRequest(Base):
    __tablename__ = "attendance_correction_requests"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    attendance_date = Column(Date, nullable=False)
    requested_sign_in_time = Column(Time, nullable=True)
    requested_sign_out_time = Column(Time, nullable=True)
    requested_status = Column(String(20), nullable=True)
    reason = Column(String(500), nullable=False)
    status = Column(String(20), default="PENDING")  # PENDING, APPROVED, REJECTED
    approver_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_at = Column(DateTime, nullable=True)
    rejection_reason = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=get_ist_now)

    employee = relationship("Employee", backref="attendance_correction_requests")


class AttendanceEvent(Base):
    __tablename__ = "attendance_events"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)
    attendance_record_id = Column(Integer, ForeignKey("attendance_records.id"), nullable=True, index=True)
    attendance_date = Column(Date, nullable=False, index=True)  # required NOT NULL column in DB
    event_time = Column(DateTime, nullable=False, index=True)
    event_type = Column(String(10), nullable=False)  # IN / OUT / BREAK_IN / BREAK_OUT
    source = Column(String(20), nullable=False, default="AUTO")  # AUTO / MANUAL / AUTO_CLOSE
    camera_id = Column(String(50), nullable=True)

    # ── Recognition evidence (camera events only) ────────────────────────────
    # Why a face match produced this row. Without it a disputed record cannot be
    # adjudicated, and recognition thresholds can only be tuned by anecdote.
    # All nullable: rows predating this, and manual/AUTO_CLOSE events, have no
    # evidence — NULL means "not applicable", not "scored zero".
    match_score = Column(Float, nullable=True)    # cosine similarity to the enrolled face
    match_margin = Column(Float, nullable=True)   # gap to the runner-up candidate
    track_id = Column(Integer, nullable=True)     # tracker id within the camera session
    snapshot_path = Column(String(300), nullable=True)  # relative to SNAPSHOT_ROOT

    created_at = Column(DateTime, default=get_ist_now)

    employee = relationship("Employee", backref="attendance_events")
    attendance_record = relationship("AttendanceRecord", backref="events")
