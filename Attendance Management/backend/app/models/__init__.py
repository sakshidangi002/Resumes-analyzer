"""Import all models for SQLAlchemy metadata and relationships."""
from app.db.base_class import Base
from app.models.employee import Employee, EmployeeBankDetail, Department, Designation
from app.models.user import User, Role, user_roles
from app.models.company import CompanyConfig, FinancialYear, Holiday
from app.models.attendance import AttendanceRecord, AttendanceCorrectionRequest, AttendanceEvent
from app.models.leave import LeaveType, LeaveAllocation, LeaveRequest
from app.models.payroll import SalaryStructure, PayrollPeriod, Payslip
from app.models.salary_advance import SalaryAdvance
from app.models.career_history import EmployeePositionSalaryHistory
from app.models.audit import AuditLog
from app.models.email_log import EmailLog
from app.models.letter import LetterTemplate, LetterInstance, LetterReply
from app.models.event import Event
from app.models.in_app_notification import AppNotification
from app.models.hr_query import HRQuery, HRQueryReply
from app.models.company_policy import CompanyPolicy
from app.models.interview_question import InterviewQuestion
from app.models.onboarding import OnboardingTask
from app.models.dsr import DailyStatusReport
from app.models.push_subscription import PushSubscription
from app.models.camera import CameraConfig
from app.models.body_embedding import BodyEmbedding
from app.models.employee_face import EmployeeFaceEmbedding
from app.models.unknown_face import UnknownFace
from app.models.unknown_attendance_event import UnknownAttendanceEvent

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
    "AttendanceEvent",
    "LeaveType",
    "LeaveAllocation",
    "LeaveRequest",
    "SalaryStructure",
    "PayrollPeriod",
    "Payslip",
    "SalaryAdvance",
    "EmployeePositionSalaryHistory",
    "LetterTemplate",
    "LetterInstance",
    "LetterReply",
    "AuditLog",
    "EmailLog",
    "Event",
    "AppNotification",
    "HRQuery",
    "HRQueryReply",
    "CompanyPolicy",
    "InterviewQuestion",
    "OnboardingTask",
    "DailyStatusReport",
    "PushSubscription",
    "CameraConfig",
    "BodyEmbedding",
    "EmployeeFaceEmbedding",
    "UnknownFace",
    "UnknownAttendanceEvent",
]
