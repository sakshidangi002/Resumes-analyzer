"""Employee position & salary increment history.

Append-only audit trail of promotions / salary increments. Rows are NEVER
updated or deleted in normal use — each promotion or increment adds one new
snapshot row. This table is a *display/history* record only; the live position
stays on `employees` and the live salary stays in `salary_structures` (the
payroll source of truth), so existing payroll logic is untouched.
"""
from sqlalchemy import Column, Integer, String, Date, DateTime, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from app.db.base_class import Base

from app.core.datetime_utils import get_ist_now


class EmployeePositionSalaryHistory(Base):
    __tablename__ = "employee_position_salary_history"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)

    # FK references (kept for joins) …
    designation_id = Column(Integer, ForeignKey("designations.id"), nullable=True)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=True)
    # … plus text snapshots so history survives if a designation/department is
    # later renamed or deleted.
    position_title = Column(String(150), nullable=True)
    department_name = Column(String(150), nullable=True)

    salary = Column(Numeric(12, 2), nullable=True)  # gross salary snapshot at this event
    effective_date = Column(Date, nullable=False, index=True)
    reason = Column(String(255), nullable=True)
    change_type = Column(String(30), nullable=False, default="UPDATE")  # INITIAL, PROMOTION, INCREMENT, UPDATE

    updated_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_by_name = Column(String(150), nullable=True)  # snapshot of who made the change
    created_at = Column(DateTime, default=get_ist_now)

    employee = relationship("Employee", backref="position_salary_history")
    designation = relationship("Designation")
    department = relationship("Department")
