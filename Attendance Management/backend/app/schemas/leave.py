from pydantic import BaseModel, computed_field
from typing import Optional
from datetime import date, datetime
from decimal import Decimal


class LeaveTypeResponse(BaseModel):
    id: int
    code: str
    name: str
    is_paid: bool
    allow_half_day: bool
    class Config:
        from_attributes = True


class LeaveAllocationResponse(BaseModel):
    id: int
    employee_id: int
    financial_year_id: int
    leave_type_id: int
    allocated_days: Decimal
    used_days: Decimal

    @computed_field
    @property
    def balance_days(self) -> Decimal:
        return self.allocated_days - self.used_days

    class Config:
        from_attributes = True


class LeaveRequestCreate(BaseModel):
    leave_type_id: int
    start_date: date
    end_date: date
    is_half_day: bool = False
    reason: Optional[str] = None


class LeaveRequestResponse(BaseModel):
    id: int
    employee_id: int
    leave_type_id: int
    start_date: date
    end_date: date
    is_half_day: bool
    reason: Optional[str] = None
    status: str
    applied_at: datetime
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    response_comment: Optional[str] = None
    created_at: Optional[datetime] = None
    class Config:
        from_attributes = True


class LeaveApprovalRow(BaseModel):
    id: int
    employee_id: int
    employee_code: str
    employee_name: str
    leave_type_id: int
    leave_type_name: str
    start_date: date
    end_date: date
    is_half_day: bool
    reason: Optional[str] = None
    status: str
    applied_at: datetime
    requester_is_hr: bool = False
    rejection_reason: Optional[str] = None
    response_comment: Optional[str] = None

