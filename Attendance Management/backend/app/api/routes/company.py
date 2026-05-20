"""Company config and financial years (for leave/attendance config)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models import User, CompanyConfig, FinancialYear, Holiday
from app.api.deps import get_current_user, require_roles
from pydantic import BaseModel
from typing import Optional
from datetime import date

router = APIRouter(prefix="/company", tags=["company"])


class CompanyConfigResponse(BaseModel):
    id: int
    name: str
    default_working_days_per_month: int
    weekly_off_days: Optional[str]
    grace_time_minutes: int
    half_day_threshold_hours: int
    class Config:
        from_attributes = True


class FinancialYearResponse(BaseModel):
    id: int
    start_date: date
    end_date: date
    is_current: bool
    name: Optional[str]
    class Config:
        from_attributes = True


class CompanyStatsResponse(BaseModel):
    total_employees: int
    total_departments: int
    present_today: int
    on_leave_today: int
    late_today: int


@router.get("/config", response_model=CompanyConfigResponse | None)
def get_config(db: Session = Depends(get_db)):
    return db.query(CompanyConfig).first()


@router.get("/financial-years", response_model=list[FinancialYearResponse])
def list_financial_years(db: Session = Depends(get_db)):
    return db.query(FinancialYear).order_by(FinancialYear.start_date.desc()).all()


class HolidayResponse(BaseModel):
    id: int
    date: date
    name: str
    is_optional: bool
    class Config:
        from_attributes = True


@router.get("/holidays", response_model=list[HolidayResponse])
def list_holidays(db: Session = Depends(get_db)):
    return db.query(Holiday).filter(Holiday.date >= date.today()).order_by(Holiday.date.asc()).limit(5).all()


@router.get("/stats", response_model=CompanyStatsResponse)
def get_company_stats(db: Session = Depends(get_db)):
    from app.models import Employee, Department, AttendanceRecord, LeaveRequest
    from app.models.employee import EmploymentStatus
    from datetime import date as dt_date
    today = dt_date.today()

    active_status = EmploymentStatus.ACTIVE.value
    active_emp_ids_subq = db.query(Employee.id).filter(Employee.employment_status == active_status).subquery()

    total_employees = db.query(Employee).filter(Employee.employment_status == active_status).count()
    total_departments = db.query(Department).count()

    # Today's attendance (active employees only)
    attendance = (
        db.query(AttendanceRecord)
        .filter(AttendanceRecord.date == today)
        .filter(AttendanceRecord.employee_id.in_(active_emp_ids_subq))
        .all()
    )
    present_today = len([r for r in attendance if r.status in ['PRESENT', 'SHORT', 'HALF_DAY']])
    late_today = len([r for r in attendance if r.is_late])

    # Today's leave (active employees only)
    on_leave_today = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.status == 'APPROVED',
            LeaveRequest.start_date <= today,
            LeaveRequest.end_date >= today,
            LeaveRequest.employee_id.in_(active_emp_ids_subq),
        )
        .count()
    )

    return {
        "total_employees": total_employees,
        "total_departments": total_departments,
        "present_today": present_today,
        "on_leave_today": on_leave_today,
        "late_today": late_today
    }
