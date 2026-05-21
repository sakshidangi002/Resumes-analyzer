"""Import all models for SQLAlchemy metadata and relationships."""
from app.db.base_class import Base
from app.models.employee import Employee, EmployeeBankDetail, Department, Designation
from app.models.user import User, Role, user_roles
from app.models.company import CompanyConfig, FinancialYear, Holiday
from app.models.attendance import AttendanceRecord, AttendanceCorrectionRequest
from app.models.leave import LeaveType, LeaveAllocation, LeaveRequest
from app.models.payroll import SalaryStructure, PayrollPeriod, Payslip
from app.models.audit import AuditLog
from app.models.email_log import EmailLog
from app.models.letter import LetterTemplate, LetterInstance, LetterReply
from app.models.event import Event
from app.models.in_app_notification import AppNotification
from app.models.onboarding import OnboardingTask

__all__ = [
    "Base",
    "User",
    "Role",
    "user_roles",
    "Employee",
    "EmployeeBankDetail",
    "Department",
    "Designation",
    "CompanyConfig",
    "FinancialYear",
    "Holiday",
    "AttendanceRecord",
    "AttendanceCorrectionRequest",
    "LeaveType",
    "LeaveAllocation",
    "LeaveRequest",
    "SalaryStructure",
    "PayrollPeriod",
    "Payslip",
    "LetterTemplate",
    "LetterInstance",
    "LetterReply",
    "AuditLog",
    "EmailLog",
    "Event",
    "AppNotification",
    "OnboardingTask",
]
