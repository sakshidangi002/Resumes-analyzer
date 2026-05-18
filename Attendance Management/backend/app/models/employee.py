"""Employee master, department, designation, bank details."""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Date, ForeignKey, Boolean, DateTime, Enum, Float
from sqlalchemy.orm import relationship
from app.db.base_class import Base
import enum


class EmploymentType(str, enum.Enum):
    FULL_TIME = "Full-time"
    INTERN = "Intern"
    CONTRACT = "Contract"


class EmploymentStatus(str, enum.Enum):
    ACTIVE = "Active"
    RESIGNED = "Resigned"
    TERMINATED = "Terminated"


class AccountType(str, enum.Enum):
    SAVINGS = "Savings"
    CURRENT = "Current"


class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    code = Column(String(20), unique=True, nullable=True)

    employees = relationship("Employee", back_populates="department")


class Designation(Base):
    __tablename__ = "designations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), nullable=False)

    employees = relationship("Employee", back_populates="designation")


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    employee_code = Column(String(50), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    official_email = Column(String(255), nullable=False)
    personal_email = Column(String(255), nullable=True)
    phone = Column(String(20), nullable=True)
    date_of_joining = Column(Date, nullable=False)
    designation_id = Column(Integer, ForeignKey("designations.id"), nullable=True)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=True)
    employment_type = Column(String(20), default=EmploymentType.FULL_TIME.value)
    reporting_manager_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    employment_status = Column(String(20), default=EmploymentStatus.ACTIVE.value)
    date_of_birth = Column(Date, nullable=True)
    date_of_leaving = Column(Date, nullable=True)
    expected_working_hours = Column(Float, default=9.0, nullable=False)
    # Important documents (metadata only; no file storage)
    pan_number = Column(String(50), nullable=True)
    aadhar_number = Column(String(50), nullable=True)
    passport_number = Column(String(50), nullable=True)
    passport_expiry_date = Column(Date, nullable=True)
    driving_license_number = Column(String(50), nullable=True)
    driving_license_expiry_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    designation = relationship("Designation", back_populates="employees")
    department = relationship("Department", back_populates="employees")
    reporting_manager = relationship("Employee", remote_side=[id], backref="reportees")
    bank_details = relationship("EmployeeBankDetail", back_populates="employee", uselist=False)

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    @property
    def work_anniversary(self):
        return self.date_of_joining  # Same date each year


class EmployeeBankDetail(Base):
    """Bank details for salary; access restricted to Admin/HR and payroll only."""
    __tablename__ = "employee_bank_details"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    bank_name = Column(String(100), nullable=False)
    branch_name = Column(String(100), nullable=True)
    account_holder_name = Column(String(100), nullable=False)
    account_number = Column(String(50), nullable=False)
    ifsc_code = Column(String(20), nullable=False)
    account_type = Column(String(20), default=AccountType.SAVINGS.value)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("Employee", back_populates="bank_details")
