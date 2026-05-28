"""Employee master CRUD and bank details."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from app.db.session import get_db
from app.api.deps import is_employment_status_blocked
from app.models import (
    User,
    Employee,
    EmployeeBankDetail,
    Department,
    Designation,
    AttendanceRecord,
    AttendanceCorrectionRequest,
    SalaryStructure,
    Payslip,
    LeaveAllocation,
    LeaveRequest,
    Event,
)
from app.models.employee import EmploymentStatus
from app.schemas.employee import (
    EmployeeCreate,
    EmployeeUpdate,
    EmployeeResponse,
    EmployeeBankDetailCreate,
    EmployeeBankDetailResponse,
    DepartmentCreate,
    DepartmentUpdate,
    DepartmentResponse,
    DesignationCreate,
    DesignationUpdate,
    DesignationResponse,
)
from app.api.deps import get_current_user, require_roles

router = APIRouter()

def _ensure_default_departments(db: Session) -> None:
    """Create default departments if missing (idempotent)."""
    default_names = ["Frontend Developer", "Backend Developer", "HR", "SEO"]
    existing = {d.name.strip().lower() for d in db.query(Department).all()}
    created = False
    for name in default_names:
        if name.strip().lower() not in existing:
            db.add(Department(name=name))
            created = True
    if created:
        db.commit()


# ---------- Departments ----------
@router.get("/departments", response_model=list[DepartmentResponse])
def list_departments(db: Session = Depends(get_db)):
    return db.query(Department).all()


@router.post("/departments", response_model=DepartmentResponse)
def create_department(
    data: DepartmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    if data.code and db.query(Department).filter(Department.code == data.code).first():
        raise HTTPException(status_code=400, detail="Department code already exists")
    d = Department(name=data.name, code=data.code)
    db.add(d)
    db.commit()
    db.refresh(d)
    return d


@router.get("/departments/{department_id}", response_model=DepartmentResponse)
def get_department(
    department_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager"])),
):
    d = db.query(Department).filter(Department.id == department_id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Department not found")
    return d


@router.patch("/departments/{department_id}", response_model=DepartmentResponse)
def update_department(
    department_id: int,
    data: DepartmentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    d = db.query(Department).filter(Department.id == department_id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Department not found")
    # Enforce unique code on update as well
    if data.code is not None and data.code != d.code:
        if db.query(Department).filter(Department.code == data.code).first():
            raise HTTPException(status_code=400, detail="Department code already exists")
    if data.name is not None:
        d.name = data.name
    if data.code is not None:
        d.code = data.code
    db.commit()
    db.refresh(d)
    return d


@router.delete("/departments/{department_id}")
def delete_department(
    department_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    d = db.query(Department).filter(Department.id == department_id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Department not found")
    # If linked with employees, unlink first so employee's department becomes blank.
    db.query(Employee).filter(Employee.department_id == department_id).update(
        {Employee.department_id: None},
        synchronize_session=False,
    )
    db.delete(d)
    db.commit()
    return {"message": "Department deleted"}


# ---------- Designations ----------
@router.get("/designations", response_model=list[DesignationResponse])
def list_designations(db: Session = Depends(get_db)):
    return db.query(Designation).all()


@router.post("/designations", response_model=DesignationResponse)
def create_designation(
    data: DesignationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    des = Designation(title=data.title)
    db.add(des)
    db.commit()
    db.refresh(des)
    return des


@router.get("/designations/{designation_id}", response_model=DesignationResponse)
def get_designation(
    designation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager"])),
):
    des = db.query(Designation).filter(Designation.id == designation_id).first()
    if not des:
        raise HTTPException(status_code=404, detail="Designation not found")
    return des


@router.patch("/designations/{designation_id}", response_model=DesignationResponse)
def update_designation(
    designation_id: int,
    data: DesignationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    des = db.query(Designation).filter(Designation.id == designation_id).first()
    if not des:
        raise HTTPException(status_code=404, detail="Designation not found")
    if data.title is not None:
        des.title = data.title
    db.commit()
    db.refresh(des)
    return des


@router.delete("/designations/{designation_id}")
def delete_designation(
    designation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    des = db.query(Designation).filter(Designation.id == designation_id).first()
    if not des:
        raise HTTPException(status_code=404, detail="Designation not found")
    # If linked with employees, unlink first so employee's designation becomes blank.
    db.query(Employee).filter(Employee.designation_id == designation_id).update(
        {Employee.designation_id: None},
        synchronize_session=False,
    )
    db.delete(des)
    db.commit()
    return {"message": "Designation deleted"}


# ---------- Employees ----------
@router.get("", response_model=list[EmployeeResponse])
def list_employees(
    db: Session = Depends(get_db),
    department_id: int | None = Query(None),
    status: str | None = Query(None),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    q = db.query(Employee)
    if department_id is not None:
        q = q.filter(Employee.department_id == department_id)
    if status:
        q = q.filter(Employee.employment_status == status)
    # Employee role: only self
    if "Employee" in [r.name for r in current_user.roles] and "Manager" not in [r.name for r in current_user.roles] and "HR" not in [r.name for r in current_user.roles] and "Admin" not in [r.name for r in current_user.roles]:
        if current_user.employee_id:
            q = q.filter(Employee.id == current_user.employee_id)
        else:
            return []
    # Stable ordering to prevent row shifting after edits
    return q.order_by(Employee.id).all()


@router.get("/{employee_id}", response_model=EmployeeResponse)
def get_employee(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    emp = (
        db.query(Employee)
        .options(joinedload(Employee.reporting_manager))
        .filter(Employee.id == employee_id)
        .first()
    )
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    role_names = [r.name for r in current_user.roles]
    if "Employee" in role_names and "Manager" not in role_names and "HR" not in role_names and "Admin" not in role_names:
        if current_user.employee_id != employee_id:
            raise HTTPException(status_code=403, detail="Access denied")
    return emp


@router.post("", response_model=EmployeeResponse)
def create_employee(
    data: EmployeeCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    if db.query(Employee).filter(Employee.employee_code == data.employee_code).first():
        raise HTTPException(status_code=400, detail="Employee code already exists")
    emp = Employee(
        employee_code=data.employee_code,
        first_name=data.first_name,
        last_name=data.last_name,
        official_email=data.official_email,
        personal_email=data.personal_email,
        phone=data.phone,
        date_of_joining=data.date_of_joining,
        designation_id=data.designation_id,
        department_id=data.department_id,
        employment_type=data.employment_type,
        reporting_manager_id=data.reporting_manager_id,
        employment_status=data.employment_status,
        date_of_birth=data.date_of_birth,
        date_of_marriage=data.date_of_marriage,
        marital_status=data.marital_status,
        date_of_leaving=data.date_of_leaving,
        pan_number=data.pan_number,
        aadhar_number=data.aadhar_number,
        passport_number=data.passport_number,
        passport_expiry_date=data.passport_expiry_date,
        driving_license_number=data.driving_license_number,
        driving_license_expiry_date=data.driving_license_expiry_date,
    )
    db.add(emp)
    db.flush()
    if data.bank_details:
        b = EmployeeBankDetail(
            employee_id=emp.id,
            bank_name=data.bank_details.bank_name,
            account_holder_name=data.bank_details.account_holder_name,
            account_number=data.bank_details.account_number,
            ifsc_code=data.bank_details.ifsc_code,
            account_type=data.bank_details.account_type,
        )
        db.add(b)
    db.commit()
    db.refresh(emp)
    return emp


@router.patch("/{employee_id}", response_model=EmployeeResponse)
def update_employee(
    employee_id: int,
    data: EmployeeUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    patch = data.model_dump(exclude_unset=True)
    # Validate employee_code uniqueness when changing
    if "employee_code" in patch and patch["employee_code"] and patch["employee_code"] != emp.employee_code:
        if db.query(Employee).filter(Employee.employee_code == patch["employee_code"]).first():
            raise HTTPException(status_code=400, detail="Employee code already exists")
    previous_status = emp.employment_status
    for k, v in patch.items():
        setattr(emp, k, v)

    # When an Admin/HR marks an employee as Resigned or Terminated, deactivate
    # the linked user account so existing tokens / future logins are rejected.
    # If they are later reactivated to "Active", the linked user is reactivated
    # too so they can log in again without a manual fix in the Users panel.
    new_status = emp.employment_status
    if new_status != previous_status:
        linked_users = db.query(User).filter(User.employee_id == emp.id).all()
        if is_employment_status_blocked(new_status):
            for u in linked_users:
                u.is_active = False
        elif new_status == EmploymentStatus.ACTIVE.value:
            for u in linked_users:
                u.is_active = True

    db.commit()
    db.refresh(emp)
    return emp


# ---------- Bank details (restricted) ----------
@router.get("/{employee_id}/bank", response_model=EmployeeBankDetailResponse)
def get_employee_bank(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Employee"])),
):
    role_names = [r.name for r in current_user.roles]
    is_admin = "Admin" in role_names
    is_hr = "HR" in role_names
    is_employee = "Employee" in role_names
    # Employee can only view own bank details
    if is_employee and not (is_admin or is_hr):
        if current_user.employee_id != employee_id:
            raise HTTPException(status_code=403, detail="Access denied")
    b = db.query(EmployeeBankDetail).filter(
        EmployeeBankDetail.employee_id == employee_id,
        EmployeeBankDetail.is_active == True,
    ).first()
    if not b:
        raise HTTPException(status_code=404, detail="Bank details not found")
    return b


@router.delete("/{employee_id}")
def delete_employee(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    # Hard delete: explicitly remove related rows, then employee
    # Attendance records and correction requests
    db.query(AttendanceRecord).filter(AttendanceRecord.employee_id == employee_id).delete()
    db.query(AttendanceCorrectionRequest).filter(
        AttendanceCorrectionRequest.employee_id == employee_id
    ).delete()
    # Payroll & leave data
    db.query(SalaryStructure).filter(SalaryStructure.employee_id == employee_id).delete()
    db.query(Payslip).filter(Payslip.employee_id == employee_id).delete()
    db.query(LeaveAllocation).filter(LeaveAllocation.employee_id == employee_id).delete()
    db.query(LeaveRequest).filter(LeaveRequest.employee_id == employee_id).delete()
    # Calendar events linked to this employee (if any)
    db.query(Event).filter(Event.employee_id == employee_id).delete()
    # Bank details
    db.query(EmployeeBankDetail).filter(EmployeeBankDetail.employee_id == employee_id).delete()
    # Unlink or delete user account
    db.query(User).filter(User.employee_id == employee_id).update({"employee_id": None})

    db.delete(emp)
    db.commit()
    return {"message": "Employee deleted"}


@router.put("/{employee_id}/bank", response_model=EmployeeBankDetailResponse)
def update_employee_bank(
    employee_id: int,
    data: EmployeeBankDetailCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    # Guard against null/empty payloads so we return a clean error
    # instead of an IntegrityError from the database.
    if not (data.bank_name and data.bank_name.strip()):
        raise HTTPException(status_code=400, detail="Bank name is required")
    if not (data.account_holder_name and data.account_holder_name.strip()):
        raise HTTPException(status_code=400, detail="Account holder name is required")
    if not (data.account_number and data.account_number.strip()):
        raise HTTPException(status_code=400, detail="Account number is required")
    if not (data.ifsc_code and data.ifsc_code.strip()):
        raise HTTPException(status_code=400, detail="IFSC code is required")
    b = db.query(EmployeeBankDetail).filter(EmployeeBankDetail.employee_id == employee_id).first()
    if not b:
        b = EmployeeBankDetail(employee_id=employee_id)
        db.add(b)
    # IMPORTANT: set required non-null fields before any flush/commit.
    b.bank_name = data.bank_name.strip()
    b.branch_name = data.branch_name.strip() if data.branch_name else None
    b.account_holder_name = data.account_holder_name.strip()
    b.account_number = data.account_number.strip()
    b.ifsc_code = data.ifsc_code.strip()
    b.account_type = data.account_type
    db.commit()
    db.refresh(b)
    return b
