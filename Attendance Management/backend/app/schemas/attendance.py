from pydantic import BaseModel
from typing import Optional
from datetime import date, time, datetime


class AttendanceRecordCreate(BaseModel):
    employee_id: int
    date: date
    sign_in_time: Optional[time] = None
    sign_out_time: Optional[time] = None
    status: str = "PRESENT"


class AttendanceRecordResponse(BaseModel):
    id: int
    employee_id: int
    date: date
    sign_in_time: Optional[time] = None
    sign_out_time: Optional[time] = None
    total_work_hours: Optional[float] = None
    status: str
    is_late: bool
    is_early_exit: bool
    is_weekly_off: bool
    is_holiday: bool
    source: str
    created_at: datetime

    class Config:
        from_attributes = True


class AdminSetAttendance(BaseModel):
    """Used by HR/Admin to set or correct attendance in grid format."""
    employee_id: int
    date: date
    sign_in_time: Optional[time] = None
    sign_out_time: Optional[time] = None
    status: Optional[str] = None


class AttendanceCorrectionRequestCreate(BaseModel):
    attendance_date: date
    requested_sign_in_time: Optional[time] = None
    requested_sign_out_time: Optional[time] = None
    requested_status: Optional[str] = None
    reason: str


class AttendanceCorrectionRequestResponse(BaseModel):
    id: int
    employee_id: int
    attendance_date: date
    requested_sign_in_time: Optional[time] = None
    requested_sign_out_time: Optional[time] = None
    requested_status: Optional[str] = None
    reason: str
    status: str
    approver_id: Optional[int] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
