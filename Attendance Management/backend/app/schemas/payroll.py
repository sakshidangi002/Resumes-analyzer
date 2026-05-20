from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime
from decimal import Decimal


class SalaryStructureCreate(BaseModel):
    employee_id: int
    basic: Decimal
    hra: Decimal
    medical: Decimal = Decimal("0")
    travelling: Decimal = Decimal("0")
    miscellaneous: Decimal = Decimal("0")
    allowances: Decimal = Decimal("0")
    deductions: Decimal = Decimal("0")
    effective_from: date


class SalaryStructureUpdate(BaseModel):
    basic: Optional[Decimal] = None
    hra: Optional[Decimal] = None
    allowances: Optional[Decimal] = None
    deductions: Optional[Decimal] = None
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None
    medical: Optional[Decimal] = None
    travelling: Optional[Decimal] = None
    miscellaneous: Optional[Decimal] = None


class SalaryStructureResponse(BaseModel):
    id: int
    employee_id: int
    basic: Decimal
    hra: Decimal
    medical: Decimal
    travelling: Decimal
    miscellaneous: Decimal
    allowances: Decimal
    deductions: Decimal
    effective_from: date
    effective_to: Optional[date] = None
    class Config:
        from_attributes = True


class PayrollPeriodResponse(BaseModel):
    id: int
    month: int
    year: int
    status: str
    processed_at: Optional[datetime] = None
    class Config:
        from_attributes = True


class PayrollPeriodUpdate(BaseModel):
    status: Optional[str] = None
    month: Optional[int] = None
    year: Optional[int] = None


class PayslipResponse(BaseModel):
    id: int
    employee_id: int
    payroll_period_id: int
    gross_salary: Decimal
    total_earnings: Decimal
    total_deductions: Decimal
    net_salary: Decimal
    paid_days: Decimal
    lop_days: Decimal
    component_breakdown: Optional[str] = None
    generated_at: datetime
    class Config:
        from_attributes = True


class PayslipCreate(BaseModel):
    employee_id: int
    gross_salary: Decimal
    total_earnings: Decimal
    total_deductions: Decimal
    net_salary: Decimal
    paid_days: Decimal
    lop_days: Decimal
    component_breakdown: Optional[str] = None


class PayslipUpdate(BaseModel):
    gross_salary: Optional[Decimal] = None
    total_earnings: Optional[Decimal] = None
    total_deductions: Optional[Decimal] = None
    net_salary: Optional[Decimal] = None
    paid_days: Optional[Decimal] = None
    lop_days: Optional[Decimal] = None
    component_breakdown: Optional[str] = None
