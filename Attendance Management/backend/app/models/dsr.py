"""Daily Status Report (DSR) - per-employee daily work log."""
from datetime import datetime, date
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from app.db.base_class import Base


class DailyStatusReport(Base):
    """One employee files one DSR per calendar day.

    Status lifecycle: DRAFT -> SUBMITTED. A DRAFT can be edited / deleted by
    its owner; a SUBMITTED DSR is read-only for the employee (Admin/HR can
    still edit if needed).
    """

    __tablename__ = "daily_status_reports"
    __table_args__ = (
        UniqueConstraint("employee_id", "report_date", name="uq_dsr_employee_date"),
    )

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(
        Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    report_date: date = Column(Date, nullable=False, index=True)

    project_work = Column(String(255), nullable=True)
    work_location = Column(String(50), nullable=True, default="Office")
    total_hours = Column(Numeric(5, 2), nullable=True)

    work_done = Column(Text, nullable=False)
    plan_for_tomorrow = Column(Text, nullable=True)

    status = Column(String(20), nullable=False, default="DRAFT", index=True)
    submitted_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("Employee", backref="daily_status_reports")
