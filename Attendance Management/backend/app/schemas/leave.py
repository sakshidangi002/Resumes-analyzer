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
    # Paid vs Unpaid (LWP) split decided at approval (Paid Leave policy).
    paid_days: Optional[Decimal] = None
    unpaid_days: Optional[Decimal] = None
    class Config:
        from_attributes = True


class PaidLeaveSummaryResponse(BaseModel):
    """Monthly-earned Paid-Leave figures for the 'My Leave' page."""
    employee_id: int
    financial_year_id: int
    annual_days: Decimal      # Annual Paid Leave (entitlement, e.g. 12)
    earned: Decimal           # Earned Till Date (accrued, future excluded)
    used_paid: Decimal        # Paid Leave Used
    remaining: Decimal        # Paid Leave Remaining
    unpaid_used: Decimal      # Unpaid Leave Used (LWP)
    balance: Decimal          # Current Leave Balance (= remaining paid)


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
    # Paid-Leave split preview (populated for Paid Leave requests only), shown to
    # HR while approving. None for non-PL leave types.
    pl_earned: Optional[Decimal] = None
    pl_used: Optional[Decimal] = None
    pl_remaining: Optional[Decimal] = None
    pl_requested: Optional[Decimal] = None
    pl_paid: Optional[Decimal] = None
    pl_unpaid: Optional[Decimal] = None

