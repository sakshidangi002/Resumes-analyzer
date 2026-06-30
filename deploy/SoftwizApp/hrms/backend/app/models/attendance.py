"""Attendance records and correction requests."""
from datetime import datetime, date, time
from sqlalchemy import Column, Integer, Date, Time, ForeignKey, Boolean, DateTime, String, Numeric
from sqlalchemy.orm import relationship
from app.db.base_class import Base

from app.core.datetime_utils import get_ist_now


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    sign_in_time = Column(Time, nullable=True)
    sign_out_time = Column(Time, nullable=True)
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
    event_time = Column(DateTime, nullable=False, index=True)
    event_type = Column(String(10), nullable=False)  # IN / OUT
    source = Column(String(20), nullable=False, default="AUTO")  # AUTO / MANUAL
    camera_id = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=get_ist_now)

    employee = relationship("Employee", backref="attendance_events")
    attendance_record = relationship("AttendanceRecord", backref="events")
