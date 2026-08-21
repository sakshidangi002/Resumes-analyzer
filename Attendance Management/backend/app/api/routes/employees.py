"""Employee master CRUD and bank details."""
from datetime import date, timedelta
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
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
    EmployeePositionSalaryHistory,
)
from app.models.employee import EmploymentStatus
from app.schemas.employee import (
    EmployeeCreate,
    StaffCreate,
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
    CareerHistoryCreate,
    CareerHistoryUpdate,
    CareerHistoryResponse,
    CareerHistoryBundle,
    CareerCurrentSnapshot,
)
from app.api.deps import require_roles
from app.core.pii import is_masked, mask_secret
from app.services.audit_service import log_audit
from app.services.payroll_service import get_salary_structure_for_date
from app.services.embedding_cache import invalidate_embedding_cache
from app.services.employee_face_service import (
    process_face_uploads,
    save_employee_photo,
    delete_employee_photos,
    enroll_embeddings,
    gallery_summary,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Sensitive-field masking
# ---------------------------------------------------------------------------
# Government IDs and bank account numbers are masked on every normal read. Full
# values are available only from the /sensitive and /bank/reveal endpoints,
# which are Admin/HR-only and write an audit row for each disclosure.
#
# Everyone who can already see the record still sees the masked tail, so the
# routine "is this the right person?" check keeps working without handing out
# the identifier itself.

_MASKED_EMPLOYEE_FIELDS = (
    "pan_number",
    "aadhar_number",
    "passport_number",
    "driving_license_number",
)


def _owns_record(current_user: User, employee_id: int) -> bool:
    """Employees always see their OWN identifiers unmasked -- they supplied
    them, so masking there protects nobody and just breaks the profile page."""
    return current_user.employee_id == employee_id


def _employee_response(emp: Employee, current_user: User) -> EmployeeResponse:
    """Serialise an Employee, masking identifiers unless it's the owner.

    Builds a Pydantic model first and masks THAT. Never mutate the ORM instance:
    SQLAlchemy would see the masked strings as pending changes and flush them
    into the database on the next commit.
    """
    resp = EmployeeResponse.model_validate(emp)
    if _owns_record(current_user, emp.id):
        return resp
    return resp.model_copy(
        update={f: mask_secret(getattr(resp, f)) for f in _MASKED_EMPLOYEE_FIELDS}
    )


def _bank_response(
    bank: EmployeeBankDetail, current_user: User
) -> EmployeeBankDetailResponse:
    resp = EmployeeBankDetailResponse.model_validate(bank)
    if _owns_record(current_user, bank.employee_id):
        return resp
    return resp.model_copy(update={"account_number": mask_secret(resp.account_number)})


def _drop_masked_writes(data: dict) -> dict:
    """Remove sensitive fields whose incoming value is one of our own masks.

    The edit form is seeded from a masked GET, so an unmodified save round-trips
    "•••• 1234" back to us. Writing that would destroy the real value.
    """
    return {
        key: value
        for key, value in data.items()
        if not (key in _MASKED_EMPLOYEE_FIELDS + ("account_number",) and is_masked(value))
    }

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
    page: int | None = Query(None, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """List employees. Pagination is OPT-IN: pass `page` to get one page.

    `page` deliberately has NO default. Every caller of this endpoint is a
    selector that needs the complete set -- attendance, payroll, payslips, leave
    allocation, calendar, user management. A default page size silently dropped
    every employee past the 50th from those lists, so payroll and attendance
    quietly skipped staff with nothing on screen to say so. Truncating data a
    caller did not ask to truncate is worse than the slow query it avoids.
    """
    # EmployeeResponse serialises `reporting_manager`, so without this the
    # response builder lazily loaded that relationship once per employee — an
    # N+1 on the endpoint that renders the whole directory.
    q = db.query(Employee).options(joinedload(Employee.reporting_manager))
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
    q = q.order_by(Employee.id)
    if page is not None:
        q = q.offset((page - 1) * page_size).limit(page_size)
    return [_employee_response(e, current_user) for e in q.all()]


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
    return _employee_response(emp, current_user)


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
        staff_type=(data.staff_type or "Employee"),
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
    invalidate_embedding_cache()
    return _employee_response(emp, current_user)


def _next_staff_code(db: Session) -> str:
    """Generate a unique STF#### code for a non-employee staff record."""
    existing = {
        c[0] for c in db.query(Employee.employee_code)
        .filter(Employee.employee_code.like("STF%")).all()
    }
    n = len(existing) + 1
    while f"STF{n:04d}" in existing:
        n += 1
    return f"STF{n:04d}"


@router.post("/staff", response_model=EmployeeResponse)
def create_staff(
    data: StaffCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Register a non-employee staff member (housekeeping, security, driver…).

    Only name + staff type are required; everything else is auto-filled. These
    people are recognised on camera but are NOT marked for attendance (that is
    enforced in the recognition layer by staff_type).
    """
    from datetime import date as _date

    code = _next_staff_code(db)
    emp = Employee(
        employee_code=code,
        staff_type=(data.staff_type or "Staff"),
        first_name=data.first_name.strip(),
        last_name=(data.last_name or "").strip(),
        official_email=f"{code.lower()}@staff.local",
        phone=data.phone,
        date_of_joining=_date.today(),
        employment_type="Contract",
        employment_status="Active",
        expected_working_hours=0.0,  # no work-hour expectation for support staff
    )
    db.add(emp)
    db.commit()
    db.refresh(emp)
    invalidate_embedding_cache()
    return _employee_response(emp, current_user)


@router.post("/register")
async def register_face_data(
    employee_id: int = Form(...),
    name: str = Form(...),
    department: str = Form(""),
    images: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    clean_name = name.strip()
    clean_department = department.strip()

    if not clean_name:
        raise HTTPException(status_code=400, detail="Employee name is required")
    if not 1 <= len(images) <= 10:
        raise HTTPException(status_code=400, detail="Upload between 1 and 10 face images")

    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail=f"Employee with ID {employee_id} not found in HRMS database.")

    prepared_images = await process_face_uploads(images)
    photo_path = save_employee_photo(
        employee_id, prepared_images[0]["bytes"], prepared_images[0]["filename"]
    )
    emp.photo_path = photo_path
    db.commit()

    # Store ALL enrolled embeddings (one row per photo/angle) — NOT averaged.
    # Recognition matches against the best of them, which handles different
    # angles/lighting far more accurately than a mean vector.
    #
    # `replace=True` because this endpoint is "register this employee's face
    # data", not "append to it": re-running it after a bad first enrolment must
    # actually supersede the bad vectors rather than leave them in the gallery
    # competing with the good ones. Superseded rows are soft-deleted, so the
    # audit trail of what the system believed when it wrote past attendance
    # survives. enroll_embeddings rebuilds employees.embedding and invalidates
    # the matcher cache.
    enroll_embeddings(
        employee_id=employee_id,
        observations=[
            {
                "embedding": item["embedding"],
                "aligned": item.get("aligned", True),
                "quality": item.get("quality"),
            }
            for item in prepared_images
        ],
        source="upload",
        replace=True,
    )
    db.refresh(emp)

    return {
        "message": "Employee registered successfully",
        "employee": {
            "id": employee_id,
            "name": clean_name,
            "department": clean_department,
            "photo_path": photo_path,
            "sample_count": int(emp.sample_count or 0),
        },
        "skipped": prepared_images[0].get("skipped") or [],
        "gallery": gallery_summary(employee_id),
    }


@router.get("/{employee_id}/face")
def get_face_status(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Report whether an employee has registered face data, and how many photos."""
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    # The gallery breakdown is what tells HR whether this employee is likely to
    # be recognised on a fixed ceiling camera: `has_camera_enrolment` false means
    # every vector came from a studio photo, which is a different distribution
    # from what the camera sees and is the most common cause of an employee who
    # "never gets recognised".
    return {
        "employee_id": employee_id,
        "registered": emp.embedding is not None,
        "sample_count": int(emp.sample_count or 0),
        "gallery": gallery_summary(employee_id),
    }


@router.delete("/{employee_id}/face")
def delete_face_data(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Clear an employee's registered face data so new photos can be enrolled.

    Removes the stored embedding, sample count, and saved photo files. The
    recognition cache is invalidated so live cameras stop matching this person
    until they are re-enrolled.
    """
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    emp.embedding = None
    emp.photo_path = None
    emp.sample_count = 0

    # Retire the provenance rows too, or the next enrolment would rebuild the
    # stack from vectors the operator believed they had just deleted. Soft
    # delete: an attendance row written last month was justified by these
    # embeddings, and destroying that record would make a disputed event
    # unauditable.
    from app.models.employee_face import EmployeeFaceEmbedding

    (
        db.query(EmployeeFaceEmbedding)
        .filter(
            EmployeeFaceEmbedding.employee_id == employee_id,
            EmployeeFaceEmbedding.active.is_(True),
        )
        .update({"active": False}, synchronize_session=False)
    )
    db.commit()
    delete_employee_photos(employee_id)
    invalidate_embedding_cache()
    return {"message": "Face data deleted", "employee_id": employee_id}


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
    # Discard any identifier that came back to us still masked -- that means the
    # editor never touched the field, and writing the mask would destroy the
    # stored value.
    patch = _drop_masked_writes(data.model_dump(exclude_unset=True))
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
    invalidate_embedding_cache()
    return _employee_response(emp, current_user)


# ---------- Position & Salary increment history (Feature 1) ----------
def _structure_gross(s: SalaryStructure | None) -> float | None:
    """Gross salary = sum of earning components (matches payroll_service)."""
    if s is None:
        return None
    return float(
        (s.basic or 0) + (s.hra or 0) + (s.medical or 0) + (s.travelling or 0)
        + (s.miscellaneous or 0) + (s.allowances or 0)
    )


def _can_view_employee(current_user: User, employee_id: int) -> bool:
    role_names = [r.name for r in current_user.roles]
    privileged = any(r in role_names for r in ("Admin", "HR", "Manager"))
    return privileged or current_user.employee_id == employee_id


@router.get("/{employee_id}/career-history", response_model=CareerHistoryBundle)
def get_career_history(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """Current position/salary + full chronological promotion & increment history."""
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    if not _can_view_employee(current_user, employee_id):
        raise HTTPException(status_code=403, detail="Access denied")

    rows = (
        db.query(EmployeePositionSalaryHistory)
        .filter(EmployeePositionSalaryHistory.employee_id == employee_id)
        .order_by(
            EmployeePositionSalaryHistory.effective_date.asc(),
            EmployeePositionSalaryHistory.id.asc(),
        )
        .all()
    )

    # Current live position (from employee master) + current salary (from the
    # salary structure effective today — the payroll source of truth).
    current_struct = get_salary_structure_for_date(db, employee_id, date.today())
    current = CareerCurrentSnapshot(
        position_title=emp.designation.title if emp.designation else None,
        department_name=emp.department.name if emp.department else None,
        salary=_structure_gross(current_struct),
        effective_date=current_struct.effective_from if current_struct else None,
    )
    return CareerHistoryBundle(current=current, history=rows)


@router.post("/{employee_id}/career-history", response_model=CareerHistoryResponse)
def add_career_history(
    employee_id: int,
    data: CareerHistoryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Record a promotion / salary increment.

    Writes an append-only history snapshot and (by default) updates the live
    position on the employee and creates a new salary structure so payroll picks
    up the new figure. Existing rows are never overwritten.
    """
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    # Resolve target designation/department (default: keep current).
    target_designation_id = data.designation_id if data.designation_id is not None else emp.designation_id
    target_department_id = data.department_id if data.department_id is not None else emp.department_id

    designation = (
        db.query(Designation).filter(Designation.id == target_designation_id).first()
        if target_designation_id else None
    )
    department = (
        db.query(Department).filter(Department.id == target_department_id).first()
        if target_department_id else None
    )

    # Resolve salary (default: keep current gross).
    current_struct = get_salary_structure_for_date(db, employee_id, data.effective_date)
    current_salary = _structure_gross(current_struct)
    new_salary = data.salary if data.salary is not None else current_salary

    # Derive change type if the caller didn't specify one.
    position_changed = data.designation_id is not None and data.designation_id != emp.designation_id
    salary_changed = data.salary is not None and data.salary != current_salary
    if data.change_type:
        change_type = data.change_type
    elif not db.query(EmployeePositionSalaryHistory).filter_by(employee_id=employee_id).first():
        change_type = "INITIAL"
    elif position_changed:
        change_type = "PROMOTION"
    elif salary_changed:
        change_type = "INCREMENT"
    else:
        change_type = "UPDATE"

    history = EmployeePositionSalaryHistory(
        employee_id=employee_id,
        designation_id=target_designation_id,
        department_id=target_department_id,
        position_title=designation.title if designation else None,
        department_name=department.name if department else None,
        salary=new_salary,
        effective_date=data.effective_date,
        reason=data.reason,
        change_type=change_type,
        updated_by_user_id=current_user.id,
        updated_by_name=current_user.username,
    )
    db.add(history)

    if data.apply_to_live:
        # Update live position on the employee master.
        if data.designation_id is not None:
            emp.designation_id = data.designation_id
        if data.department_id is not None:
            emp.department_id = data.department_id

        # If a salary was supplied, create a new salary structure so payroll uses
        # it. Close the previous open-ended structure at the day before this one
        # so the effective-dated timeline stays clean (payroll already picks the
        # latest effective_from, so this is a correctness nicety, not a fix).
        if data.salary is not None:
            prev = (
                db.query(SalaryStructure)
                .filter(
                    SalaryStructure.employee_id == employee_id,
                    SalaryStructure.effective_from < data.effective_date,
                    SalaryStructure.effective_to == None,  # noqa: E711
                )
                .order_by(SalaryStructure.effective_from.desc())
                .first()
            )
            if prev is not None:
                prev.effective_to = data.effective_date - timedelta(days=1)
            db.add(
                SalaryStructure(
                    employee_id=employee_id,
                    basic=data.salary,
                    hra=0, medical=0, travelling=0, miscellaneous=0,
                    allowances=0, deductions=0,
                    effective_from=data.effective_date,
                )
            )

    db.commit()
    db.refresh(history)
    return history


@router.patch("/{employee_id}/career-history/{history_id}", response_model=CareerHistoryResponse)
def update_career_history(
    employee_id: int,
    history_id: int,
    data: CareerHistoryUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Correct a history row in place. Audit-only: it does NOT change the
    employee's live designation or create/modify salary structures."""
    row = (
        db.query(EmployeePositionSalaryHistory)
        .filter(
            EmployeePositionSalaryHistory.id == history_id,
            EmployeePositionSalaryHistory.employee_id == employee_id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="History record not found")

    patch = data.model_dump(exclude_unset=True)
    if "designation_id" in patch:
        row.designation_id = patch["designation_id"]
        des = (
            db.query(Designation).filter(Designation.id == patch["designation_id"]).first()
            if patch["designation_id"] else None
        )
        row.position_title = des.title if des else None
    if "department_id" in patch:
        row.department_id = patch["department_id"]
        dep = (
            db.query(Department).filter(Department.id == patch["department_id"]).first()
            if patch["department_id"] else None
        )
        row.department_name = dep.name if dep else None
    for field in ("salary", "effective_date", "reason", "change_type"):
        if field in patch:
            setattr(row, field, patch[field])

    db.commit()
    db.refresh(row)
    return row


@router.delete("/{employee_id}/career-history/{history_id}")
def delete_career_history(
    employee_id: int,
    history_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Delete a history row (audit record only; live position/salary unaffected)."""
    row = (
        db.query(EmployeePositionSalaryHistory)
        .filter(
            EmployeePositionSalaryHistory.id == history_id,
            EmployeePositionSalaryHistory.employee_id == employee_id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="History record not found")
    db.delete(row)
    db.commit()
    return {"message": "Deleted"}


# ---------- Sensitive-value disclosure (Admin/HR, audited) ----------
@router.post("/{employee_id}/sensitive")
def reveal_employee_identifiers(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Return the UNMASKED government identifiers for one employee.

    Separate from GET /{id} on purpose: normal reads stay masked, and every
    disclosure lands in the audit log with who asked and for whom.
    """
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    log_audit(
        db, current_user.id, "PII_REVEALED", "Employee", str(employee_id),
        f"Viewed government identifiers for {emp.employee_code}",
    )
    return {f: getattr(emp, f) for f in _MASKED_EMPLOYEE_FIELDS}


@router.post("/{employee_id}/bank/reveal")
def reveal_employee_bank_account(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Return the UNMASKED bank account number. Audited, as above."""
    b = db.query(EmployeeBankDetail).filter(
        EmployeeBankDetail.employee_id == employee_id,
        EmployeeBankDetail.is_active == True,
    ).first()
    if not b:
        raise HTTPException(status_code=404, detail="Bank details not found")
    log_audit(
        db, current_user.id, "PII_REVEALED", "Employee", str(employee_id),
        "Viewed bank account number",
    )
    return {"account_number": b.account_number}


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
    return _bank_response(b, current_user)


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
    # Position/salary history
    db.query(EmployeePositionSalaryHistory).filter(
        EmployeePositionSalaryHistory.employee_id == employee_id
    ).delete()
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
    invalidate_embedding_cache()
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
    # The form is seeded from a masked GET. If the account number comes back
    # still masked the editor did not change it, so keep what is stored -- and
    # reject a masked value outright when there is nothing to keep.
    account_number_unchanged = is_masked(data.account_number)
    if account_number_unchanged and not b.account_number:
        raise HTTPException(
            status_code=400,
            detail="Account number is required. Reveal the stored value or enter it in full.",
        )
    # IMPORTANT: set required non-null fields before any flush/commit.
    b.bank_name = data.bank_name.strip()
    b.branch_name = data.branch_name.strip() if data.branch_name else None
    b.account_holder_name = data.account_holder_name.strip()
    if not account_number_unchanged:
        b.account_number = data.account_number.strip()
    b.ifsc_code = data.ifsc_code.strip()
    b.account_type = data.account_type
    db.commit()
    db.refresh(b)
    return _bank_response(b, current_user)






