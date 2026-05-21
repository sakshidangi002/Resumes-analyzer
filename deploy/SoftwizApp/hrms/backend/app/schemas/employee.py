from pydantic import BaseModel, computed_field, ConfigDict
from typing import Optional, List
from datetime import date, datetime


class EmployeeRef(BaseModel):
    """Minimal employee info for nested reporting manager on profile responses."""

    id: int
    employee_code: str
    first_name: str
    last_name: str

    model_config = ConfigDict(from_attributes=True)


class DepartmentCreate(BaseModel):
    name: str
    code: Optional[str] = None


class DepartmentResponse(BaseModel):
    id: int
    name: str
    code: Optional[str] = None
    class Config:
        from_attributes = True


class DepartmentUpdate(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None


class DesignationCreate(BaseModel):
    title: str


class DesignationResponse(BaseModel):
    id: int
    title: str
    class Config:
        from_attributes = True


class DesignationUpdate(BaseModel):
    title: Optional[str] = None


class EmployeeBankDetailCreate(BaseModel):
    bank_name: str
    branch_name: Optional[str] = None
    account_holder_name: str
    account_number: str
    ifsc_code: str
    account_type: str = "Savings"


class EmployeeBankDetailResponse(BaseModel):
    id: int
    bank_name: str
    branch_name: Optional[str] = None
    account_holder_name: str
    account_number: str
    ifsc_code: str
    account_type: str
    class Config:
        from_attributes = True


class EmployeeCreate(BaseModel):
    employee_code: str
    first_name: str
    last_name: str
    official_email: str
    personal_email: Optional[str] = None
    phone: Optional[str] = None
    date_of_joining: date
    designation_id: Optional[int] = None
    department_id: Optional[int] = None
    employment_type: str = "Full-time"
    reporting_manager_id: Optional[int] = None
    employment_status: str = "Active"
    date_of_birth: Optional[date] = None
    date_of_marriage: Optional[date] = None
    marital_status: Optional[str] = None
    date_of_leaving: Optional[date] = None
    expected_working_hours: float = 9.0
    pan_number: Optional[str] = None
    aadhar_number: Optional[str] = None
    passport_number: Optional[str] = None
    passport_expiry_date: Optional[date] = None
    driving_license_number: Optional[str] = None
    driving_license_expiry_date: Optional[date] = None
    bank_details: Optional[EmployeeBankDetailCreate] = None


class EmployeeUpdate(BaseModel):
    employee_code: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    official_email: Optional[str] = None
    personal_email: Optional[str] = None
    phone: Optional[str] = None
    date_of_joining: Optional[date] = None
    designation_id: Optional[int] = None
    department_id: Optional[int] = None
    employment_type: Optional[str] = None
    reporting_manager_id: Optional[int] = None
    employment_status: Optional[str] = None
    date_of_birth: Optional[date] = None
    date_of_marriage: Optional[date] = None
    marital_status: Optional[str] = None
    date_of_leaving: Optional[date] = None
    expected_working_hours: Optional[float] = None
    pan_number: Optional[str] = None
    aadhar_number: Optional[str] = None
    passport_number: Optional[str] = None
    passport_expiry_date: Optional[date] = None
    driving_license_number: Optional[str] = None
    driving_license_expiry_date: Optional[date] = None


class EmployeeResponse(BaseModel):
    id: int
    employee_code: str
    first_name: str
    last_name: str
    official_email: str
    personal_email: Optional[str] = None
    phone: Optional[str] = None
    date_of_joining: date
    designation_id: Optional[int] = None
    department_id: Optional[int] = None
    employment_type: str
    reporting_manager_id: Optional[int] = None
    employment_status: str
    date_of_birth: Optional[date] = None
    date_of_marriage: Optional[date] = None
    marital_status: Optional[str] = None
    date_of_leaving: Optional[date] = None
    expected_working_hours: float
    pan_number: Optional[str] = None
    aadhar_number: Optional[str] = None
    passport_number: Optional[str] = None
    passport_expiry_date: Optional[date] = None
    driving_license_number: Optional[str] = None
    driving_license_expiry_date: Optional[date] = None
    created_at: datetime
    reporting_manager: Optional[EmployeeRef] = None

    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    model_config = ConfigDict(from_attributes=True)
