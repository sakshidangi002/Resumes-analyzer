"""Salary structure, payroll period, payslips. Salary calculated on 30-day month basis."""
from datetime import datetime, date
from sqlalchemy import Column, Integer, Date, ForeignKey, DateTime, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from app.db.base_class import Base

from app.core.datetime_utils import get_ist_now


class SalaryStructure(Base):
    __tablename__ = "salary_structures"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    basic = Column(Numeric(12, 2), default=0)
    hra = Column(Numeric(12, 2), default=0)
    medical = Column(Numeric(12, 2), default=0, nullable=False)
    travelling = Column(Numeric(12, 2), default=0, nullable=False)
    miscellaneous = Column(Numeric(12, 2), default=0, nullable=False)
    allowances = Column(Numeric(12, 2), default=0)
    deductions = Column(Numeric(12, 2), default=0)
    effective_from = Column(Date, nullable=False)
    effective_to = Column(Date, nullable=True)

    employee = relationship("Employee", backref="salary_structures")


class PayrollPeriod(Base):
    __tablename__ = "payroll_periods"

    id = Column(Integer, primary_key=True, index=True)
    month = Column(Integer, nullable=False)  # 1-12
    year = Column(Integer, nullable=False)
    status = Column(String(20), default="OPEN")  # OPEN, PROCESSED, LOCKED
    processed_at = Column(DateTime, nullable=True)

    payslips = relationship("Payslip", back_populates="payroll_period", cascade="all, delete-orphan")


class Payslip(Base):
    __tablename__ = "payslips"
    __table_args__ = (
        UniqueConstraint("employee_id", "payroll_period_id", name="uq_payslips_employee_period"),
    )

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    payroll_period_id = Column(Integer, ForeignKey("payroll_periods.id"), nullable=False)
    gross_salary = Column(Numeric(12, 2), default=0)
    total_earnings = Column(Numeric(12, 2), default=0)
    total_deductions = Column(Numeric(12, 2), default=0)
    net_salary = Column(Numeric(12, 2), default=0)
    paid_days = Column(Numeric(5, 2), default=0)
    lop_days = Column(Numeric(5, 2), default=0)
    component_breakdown = Column(Text, nullable=True)  # JSON string
    generated_at = Column(DateTime, default=get_ist_now)

    employee = relationship("Employee", backref="payslips")
    payroll_period = relationship("PayrollPeriod", back_populates="payslips")
