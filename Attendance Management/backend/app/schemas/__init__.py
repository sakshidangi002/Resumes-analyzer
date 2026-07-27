from app.schemas.auth import Token, TokenPayload, LoginRequest
from app.schemas.user import UserCreate, UserResponse, UserWithRoles
from app.schemas.employee import (
    EmployeeCreate,
    EmployeeUpdate,
    EmployeeResponse,
    EmployeeBankDetailCreate,
    EmployeeBankDetailResponse,
    DepartmentCreate,
    DepartmentResponse,
    DesignationCreate,
    DesignationResponse,
)
from app.schemas.attendance import (
    AttendanceRecordCreate,
    AttendanceRecordResponse,
    AttendanceCorrectionRequestCreate,
    AttendanceCorrectionRequestResponse,
)
from app.schemas.leave import (
    LeaveTypeResponse,
    LeaveAllocationResponse,
    LeaveRequestCreate,
    LeaveRequestResponse,
)
from app.schemas.payroll import (
    SalaryStructureCreate,
    SalaryStructureResponse,
    PayslipResponse,
    PayrollPeriodResponse,
)
from app.schemas.letter import LetterTemplateCreate, LetterTemplateResponse, LetterInstanceResponse
from app.schemas.common import MessageResponse
