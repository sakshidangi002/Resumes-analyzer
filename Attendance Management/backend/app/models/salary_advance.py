"""Salary advances taken by employees.

An advance is money paid to an employee ahead of salary. It is recovered in full
on the next payroll run (see payroll_service.run_payroll_for_period): the pending
amount is added to that month's deductions and the advance is marked DEDUCTED.
"""
from sqlalchemy import Column, Integer, Numeric, String, Date, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base_class import Base

from app.core.datetime_utils import get_ist_now


class SalaryAdvance(Base):
    __tablename__ = "salary_advances"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)
    amount = Column(Numeric(12, 2), nullable=False)
    date_taken = Column(Date, nullable=False)
    reason = Column(String(255), nullable=True)
    status = Column(String(20), nullable=False, default="PENDING", index=True)  # PENDING, DEDUCTED, CANCELLED
    # Which payroll period recovered this advance (set when deducted).
    deducted_period_id = Column(Integer, ForeignKey("payroll_periods.id"), nullable=True)
    deducted_at = Column(DateTime, nullable=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_by_name = Column(String(150), nullable=True)
    created_at = Column(DateTime, default=get_ist_now)

    employee = relationship("Employee", backref="salary_advances")
    deducted_period = relationship("PayrollPeriod")
