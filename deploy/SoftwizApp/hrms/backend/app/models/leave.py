"""
Leave models. Leave cycle is April–March; no carry-forward to next FY.
Unused leaves expire at end of financial year.
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Date, ForeignKey, Numeric, Boolean, DateTime
from sqlalchemy.orm import relationship
from app.db.base_class import Base

from app.core.datetime_utils import get_ist_now


class LeaveType(Base):
    __tablename__ = "leave_types"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(20), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    is_paid = Column(Boolean, default=True)
    allow_half_day = Column(Boolean, default=False)


class LeaveAllocation(Base):
    """
    Per-employee, per–financial year leave allocation.
    No carry-forward: allocated_days and used_days only. Balance expires at FY end.
    """
    __tablename__ = "leave_allocations"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)
    financial_year_id = Column(Integer, ForeignKey("financial_years.id"), nullable=False, index=True)
    leave_type_id = Column(Integer, ForeignKey("leave_types.id"), nullable=False)
    allocated_days = Column(Numeric(5, 2), default=0)
    used_days = Column(Numeric(5, 2), default=0)

    employee = relationship("Employee", backref="leave_allocations")
    financial_year = relationship("FinancialYear", backref="leave_allocations")
    leave_type = relationship("LeaveType", backref="leave_allocations")


class LeaveRequest(Base):
    __tablename__ = "leave_requests"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)
    leave_type_id = Column(Integer, ForeignKey("leave_types.id"), nullable=False)
    start_date = Column(Date, nullable=False, index=True)
    end_date = Column(Date, nullable=False, index=True)
    is_half_day = Column(Boolean, default=False)
    reason = Column(String(500), nullable=True)
    status = Column(String(20), default="PENDING")  # PENDING, APPROVED, REJECTED, CANCELLED
    applied_at = Column(DateTime, default=get_ist_now)
    manager_approver_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    hr_approver_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_at = Column(DateTime, nullable=True)
    rejected_at = Column(DateTime, nullable=True)
    rejection_reason = Column(String(500), nullable=True)
    # Comment/response from approver (shown to employee for both approve/reject)
    response_comment = Column(String(500), nullable=True)

    employee = relationship("Employee", backref="leave_requests")
    leave_type = relationship("LeaveType", backref="leave_requests")
