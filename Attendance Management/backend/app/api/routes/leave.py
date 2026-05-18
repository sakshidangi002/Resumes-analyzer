"""Leave types, allocations, apply and approve."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import get_db
from app.models import User, LeaveType, LeaveAllocation, LeaveRequest, FinancialYear, Employee
from app.models.user import Role, user_roles
from app.schemas.leave import (
    LeaveTypeResponse,
    LeaveAllocationResponse,
    LeaveRequestCreate,
    LeaveRequestResponse,
    LeaveApprovalRow,
)
from app.api.deps import get_current_user, require_roles
from app.services.leave_service import (
    get_current_financial_year,
    get_leave_balance,
    allocate_leave_for_fy,
    apply_leave_request,
    approve_leave_request,
    ensure_default_allocations_for_employee,
    _count_leave_days,
)
from app.services.notification_service import notify_user_for_employee, notify_users_with_roles
from decimal import Decimal
from datetime import date as _date

router = APIRouter()

def _ensure_user_employee_link(db: Session, current_user: User) -> None:
    """
    If the logged-in user is not linked to an Employee, try to auto-link by official email.
    This helps HR/Admin accounts so "My Leave" can show their own balance.
    """
    if current_user.employee_id:
        return
    candidate_email = (current_user.official_email or "").strip()
    if not candidate_email:
        candidate_email = (current_user.username or "").strip()
    if "@" not in candidate_email:
        return
    emp = db.query(Employee).filter(Employee.official_email == candidate_email).first()
    if not emp:
        return
    # Avoid attaching `current_user` from a different Session into this request Session.
    # Update via this request's Session instead.
    db.query(User).filter(User.id == current_user.id).update({"employee_id": emp.id})
    db.commit()


def _employee_has_role(db: Session, employee_id: int, role_name: str) -> bool:
    # Find a user linked to this employee and check role assignment
    u = (
        db.query(User)
        .join(user_roles, User.id == user_roles.c.user_id)
        .join(Role, Role.id == user_roles.c.role_id)
        .filter(User.employee_id == employee_id, Role.name == role_name)
        .first()
    )
    return u is not None


def _ensure_default_leave_types(db: Session) -> None:
    """
    Ensure standard paid and unpaid leave types exist so that
    frontend mappings (Full Day, Half Day, Short, Unpaid/LOP) always work.
    """
    existing = db.query(LeaveType).all()
    codes = {lt.code for lt in existing}
    created = False

    if "PL" not in codes:
        db.add(
            LeaveType(
                code="PL",
                name="Paid Leave",
                is_paid=True,
                allow_half_day=True,
            )
        )
        created = True
    if "SL" not in codes:
        # Short Leave (2 hours) – tracked separately, treated as paid and can be combined logically as needed.
        db.add(
            LeaveType(
                code="SL",
                name="Short Leave (2 hours)",
                is_paid=True,
                allow_half_day=True,
            )
        )
        created = True
    if "UL" not in codes:
        db.add(
            LeaveType(
                code="UL",
                name="Unpaid Leave (LOP)",
                is_paid=False,
                allow_half_day=False,
            )
        )
        created = True

    if created:
        db.commit()


@router.get("/types", response_model=list[LeaveTypeResponse])
def list_leave_types(db: Session = Depends(get_db)):
    _ensure_default_leave_types(db)
    return db.query(LeaveType).all()


@router.get("/allocations", response_model=list[LeaveAllocationResponse])
def list_leave_allocations(
    employee_id: int | None = Query(None),
    financial_year_id: int | None = Query(None),
    month: int | None = Query(None, description="For Short Leave monthly balance; default current month"),
    year: int | None = Query(None, description="For Short Leave monthly balance; default current year"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """
    When employee_id is not provided: return only the current user's allocations (for "My Leave" balance),
    unless Admin/HR pass financial_year_id only (then return all allocations for that FY for management).
    When employee_id is provided: Admin/HR can pass it to view a specific employee's allocations.
    """
    q = db.query(LeaveAllocation)
    role_names = [r.name for r in current_user.roles]
    is_admin_or_hr = "Admin" in role_names or "HR" in role_names
    target_employee_id: int | None = None

    _ensure_user_employee_link(db, current_user)

    if employee_id is not None:
        if not is_admin_or_hr and current_user.employee_id != employee_id:
            return []
        q = q.filter(LeaveAllocation.employee_id == employee_id)
        target_employee_id = employee_id
    else:
        if is_admin_or_hr and financial_year_id is not None:
            q = q.filter(LeaveAllocation.financial_year_id == financial_year_id)
        else:
            if not current_user.employee_id:
                return []
            q = q.filter(LeaveAllocation.employee_id == current_user.employee_id)
            target_employee_id = current_user.employee_id

    if financial_year_id is not None:
        q = q.filter(LeaveAllocation.financial_year_id == financial_year_id)
        fy = db.query(FinancialYear).filter(FinancialYear.id == financial_year_id).first()
    else:
        fy = get_current_financial_year(db)
        # For single-employee views (My Leave, or Admin/HR viewing one employee),
        # default to current FY so the UI doesn't mix allocations across years.
        if target_employee_id is not None and fy:
            q = q.filter(LeaveAllocation.financial_year_id == fy.id)

    # Ensure defaults (Paid Leave yearly, Short Leave monthly) exist for single-employee views
    if target_employee_id is not None and fy:
        ensure_default_allocations_for_employee(db, target_employee_id, fy.id)

    allocs = q.order_by(LeaveAllocation.id).all()
    # Short Leave (SL): monthly 2 (or per allocation), not carried forward to next month.
    sl_type = db.query(LeaveType).filter(LeaveType.code == "SL").first()
    pl_type = db.query(LeaveType).filter(LeaveType.code == "PL").first()
    result = []
    
    from app.models.attendance import AttendanceRecord
    
    for a in allocs:
        if sl_type and a.leave_type_id == sl_type.id:
            m = month or _date.today().month
            y = year or _date.today().year
            start = _date(y, m, 1)
            end = _date(y + 1, 1, 1) if m == 12 else _date(y, m + 1, 1)
            n = (
                db.query(func.count(LeaveRequest.id))
                .filter(
                    LeaveRequest.employee_id == a.employee_id,
                    LeaveRequest.leave_type_id == sl_type.id,
                    LeaveRequest.status == "APPROVED",
                    LeaveRequest.start_date >= start,
                    LeaveRequest.start_date < end,
                )
                .scalar()
                or 0
            )
            
            from app.models.employee import Employee
            att_records = db.query(AttendanceRecord, Employee).join(Employee, Employee.id == AttendanceRecord.employee_id).filter(
                AttendanceRecord.employee_id == a.employee_id,
                AttendanceRecord.date >= start,
                AttendanceRecord.date < end,
            ).all()
            
            unrequested_sl = 0
            # To handle 1 HD = 2 SL accurately, we sort by date
            att_records_sorted = sorted(att_records, key=lambda x: x[0].date)
            sl_buffer_sim = 2
            
            for rec, emp in att_records_sorted:
                is_sl = rec.status == "SHORT"
                is_hd = rec.status == "HALF_DAY"
                
                if is_hd and sl_buffer_sim >= 2:
                    unrequested_sl += 2
                    sl_buffer_sim -= 2
                elif is_sl and sl_buffer_sim >= 1:
                    unrequested_sl += 1
                    sl_buffer_sim -= 1
                elif is_sl or is_hd:
                    # After buffer, we don't count it towards the "Paid SL" limit display 
                    # but it will be deducted from PL (dealt with in the next elif block)
                    pass

            result.append(
                LeaveAllocationResponse(
                    id=a.id,
                    employee_id=a.employee_id,
                    financial_year_id=a.financial_year_id,
                    leave_type_id=a.leave_type_id,
                    # allocated_days is monthly for SL
                    allocated_days=a.allocated_days,
                    used_days=Decimal(n) + Decimal(unrequested_sl),
                )
            )
        elif pl_type and a.leave_type_id == pl_type.id and fy:
            # Count attendance records marked as PAID_LEAVE or HALF_DAY directly by HR
            att_records = db.query(AttendanceRecord).filter(
                AttendanceRecord.employee_id == a.employee_id,
                AttendanceRecord.date >= fy.start_date,
                AttendanceRecord.date <= fy.end_date,
            ).order_by(AttendanceRecord.date.asc()).all()
            
            # To handle the "1 HD = 2 SL" conversion, we need to know the SL status per month
            # But for simplicity in the yearly PL view, we count all PAID_LEAVE and HALF_DAY.
            # We will SUBTRACT any HALF_DAY that was already covered by SL buffer in the SL block.
            
            unrequested_paid_leaves = Decimal("0")
            for rec in att_records:
                if rec.status == "PAID_LEAVE":
                    unrequested_paid_leaves += Decimal("1.0")
                elif rec.status == "HALF_DAY":
                    # Check if this HD was covered by SL buffer (2 SLs available in that month)
                    # For simplicity, we assume if HR marked it as HD and it's the first HD/SL of the month, 
                    # it might have been covered. However, to be 100% accurate, we check the month's records.
                    m_start = _date(rec.date.year, rec.date.month, 1)
                    m_end = (_date(rec.date.year, rec.date.month + 1, 1) if rec.date.month < 12 else _date(rec.date.year + 1, 1, 1))
                    
                    month_atts = [r for r in att_records if m_start <= r.date < m_end and r.date < rec.date]
                    month_sl_used = sum(1 for r in month_atts if r.status == "SHORT")
                    month_hd_used = sum(1 for r in month_atts if r.status == "HALF_DAY")
                    
                    buffer_left = 2 - (month_sl_used + (month_hd_used * 2))
                    if buffer_left >= 2:
                        # Covered by SL, don't deduct from PL
                        pass
                    else:
                        unrequested_paid_leaves += Decimal("0.5")
            
            result.append(
                LeaveAllocationResponse(
                    id=a.id,
                    employee_id=a.employee_id,
                    financial_year_id=a.financial_year_id,
                    leave_type_id=a.leave_type_id,
                    allocated_days=a.allocated_days,
                    used_days=a.used_days + unrequested_paid_leaves,
                )
            )
        else:
            result.append(LeaveAllocationResponse.model_validate(a))
    return result


@router.post("/allocations", response_model=LeaveAllocationResponse)
def create_allocation(
    employee_id: int,
    leave_type_id: int,
    allocated_days: Decimal,
    financial_year_id: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    fy = db.query(FinancialYear).filter(FinancialYear.id == financial_year_id).first() if financial_year_id else get_current_financial_year(db)
    if not fy:
        raise HTTPException(status_code=400, detail="Financial year required")
    try:
        return allocate_leave_for_fy(db, employee_id, fy.id, leave_type_id, allocated_days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/requests", response_model=LeaveRequestResponse)
def create_leave_request(
    data: LeaveRequestCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["HR", "Manager", "Employee"])),
):
    if not current_user.employee_id:
        raise HTTPException(status_code=400, detail="No employee linked")
    if not data.reason or not data.reason.strip():
        raise HTTPException(status_code=400, detail="Reason is required")
    try:
        req = apply_leave_request(
            db,
            current_user.employee_id,
            data.leave_type_id,
            data.start_date,
            data.end_date,
            data.is_half_day,
            data.reason,
        )
        emp = db.query(Employee).filter(Employee.id == current_user.employee_id).first()
        lt = db.query(LeaveType).filter(LeaveType.id == data.leave_type_id).first()
        name = (emp.full_name if emp else None) or "Employee"
        lt_name = (lt.name if lt else None) or "leave"
        requester_is_hr = _employee_has_role(db, current_user.employee_id, "HR")
        if requester_is_hr:
            notify_users_with_roles(
                db,
                ["Admin"],
                "New leave request (HR)",
                f"{name} ({lt_name}) {data.start_date} → {data.end_date}.",
                kind="LEAVE",
                link_path="/leave-approvals",
            )
        else:
            notify_users_with_roles(
                db,
                ["HR"],
                "New leave request",
                f"{name} requested {lt_name} ({data.start_date} → {data.end_date}).",
                kind="LEAVE",
                link_path="/leave-approvals",
            )
        return req
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/requests", response_model=list[LeaveRequestResponse])
def list_leave_requests(
    employee_id: int | None = Query(None),
    status: str | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """
    When employee_id is not provided: return only the current user's own requests (for "My Leave").
    When employee_id is provided: Admin/HR can pass it to view a specific employee's requests.
    """
    q = db.query(LeaveRequest)
    role_names = [r.name for r in current_user.roles]
    is_admin_or_hr = "Admin" in role_names or "HR" in role_names

    _ensure_user_employee_link(db, current_user)

    if employee_id is not None:
        if not is_admin_or_hr and current_user.employee_id != employee_id:
            return []  # Only Admin/HR can query another employee's requests
        q = q.filter(LeaveRequest.employee_id == employee_id)
    else:
        # "My Leave" – only the current user's own requests (Employee, HR, Manager, Admin)
        if not current_user.employee_id:
            return []
        q = q.filter(LeaveRequest.employee_id == current_user.employee_id)

    if status:
        q = q.filter(LeaveRequest.status == status)
    return q.order_by(LeaveRequest.applied_at.desc()).all()


@router.get("/approvals", response_model=list[LeaveApprovalRow])
def list_leave_approvals(
    status: str = Query("PENDING"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """
    HR sees employee leave requests (non-HR requesters) pending for approval.
    Admin can view all and approve HR's leave requests.
    """
    role_names = [r.name for r in current_user.roles]
    is_admin = "Admin" in role_names
    is_hr = "HR" in role_names

    from app.models import Role, user_roles, User
    from sqlalchemy import exists

    # Subquery to check if an employee is linked to a user with the 'HR' role
    hr_exists = exists().where(
        User.employee_id == Employee.id
    ).where(
        User.id == user_roles.c.user_id
    ).where(
        user_roles.c.role_id == Role.id
    ).where(
        Role.name == "HR"
    ).correlate(Employee)

    q = (
        db.query(LeaveRequest, Employee, LeaveType, hr_exists.label("requester_is_hr"))
        .join(Employee, Employee.id == LeaveRequest.employee_id)
        .join(LeaveType, LeaveType.id == LeaveRequest.leave_type_id)
    )
    if status:
        q = q.filter(LeaveRequest.status == status)
    rows = q.order_by(LeaveRequest.applied_at.desc()).all()

    result: list[LeaveApprovalRow] = []
    for req, emp, lt, is_hr_val in rows:
        if is_hr and not is_admin:
            # HR should not approve HR leave; HR approvals are for employees only
            if is_hr_val:
                continue
        result.append(
            LeaveApprovalRow(
                id=req.id,
                employee_id=emp.id,
                employee_code=emp.employee_code,
                employee_name=emp.full_name,
                leave_type_id=lt.id,
                leave_type_name=lt.name,
                start_date=req.start_date,
                end_date=req.end_date,
                is_half_day=req.is_half_day,
                reason=req.reason,
                status=req.status,
                applied_at=req.applied_at,
                requester_is_hr=is_hr_val,
                rejection_reason=req.rejection_reason,
                response_comment=req.response_comment,
            )
        )
    return result


@router.patch("/requests/{request_id}", response_model=LeaveRequestResponse)
def approve_reject_leave(
    request_id: int,
    approved: bool,
    comment: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    try:
        # Enforce approval rules:
        # - HR leave requests can only be approved by Admin.
        # - Employee leave requests can only be approved by HR (Admin can still view but not approve).
        req_row = db.query(LeaveRequest).filter(LeaveRequest.id == request_id).first()
        if not req_row:
            raise ValueError("Leave request not found or not pending")
        requester_is_hr = _employee_has_role(db, req_row.employee_id, "HR")
        role_names = [r.name for r in current_user.roles]
        is_admin = "Admin" in role_names
        is_hr = "HR" in role_names

        if requester_is_hr:
            # HR applying leave – only Admin may approve/reject
            if not is_admin:
                raise HTTPException(status_code=403, detail="Only Admin can approve HR leave requests")
        else:
            # Employee (non-HR) applying leave – only HR may approve/reject
            if not is_hr:
                raise HTTPException(status_code=403, detail="Only HR can approve employee leave requests")

        if not comment:
            raise HTTPException(status_code=400, detail="Comment is required")
        req = approve_leave_request(db, request_id, current_user.id, approved, comment)
        # Track HR/Admin approver in hr_approver_id
        req.hr_approver_id = current_user.id
        db.commit()
        db.refresh(req)
        status_word = "approved" if approved else "rejected"
        notify_user_for_employee(
            db,
            req.employee_id,
            f"Leave request {status_word}",
            f"Your leave request was {status_word}. Comment: {comment or '—'}",
            kind="LEAVE",
            link_path="/leave",
        )
        return req
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/requests/{request_id}")
def delete_leave_request(
    request_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Employee"])),
):
    """
    Delete a leave request. 
    - Employees can only delete their own PENDING requests (cancellation).
    - HR/Admin can hard delete any request.
    """
    req = db.query(LeaveRequest).filter(LeaveRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Leave request not found")
        
    role_names = [r.name for r in current_user.roles]
    is_admin_or_hr = "Admin" in role_names or "HR" in role_names

    if not is_admin_or_hr:
        if req.employee_id != current_user.employee_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this request")
        if req.status != "PENDING":
            raise HTTPException(status_code=400, detail="Only pending requests can be cancelled")

    if req.status == "APPROVED":
        from app.models.attendance import AttendanceRecord
        from datetime import timedelta
        
        # 1. Revert allocation
        lt = db.query(LeaveType).filter(LeaveType.id == req.leave_type_id).first()
        if lt and lt.code == "SL":
            days = Decimal("1")
        else:
            days = _count_leave_days(req.start_date, req.end_date, bool(req.is_half_day))
            
        fy = get_current_financial_year(db)
        if fy:
            alloc = db.query(LeaveAllocation).filter(
                LeaveAllocation.employee_id == req.employee_id,
                LeaveAllocation.financial_year_id == fy.id,
                LeaveAllocation.leave_type_id == req.leave_type_id,
            ).first()
            if alloc and not (lt and lt.code == "SL"):
                alloc.used_days = max(Decimal("0"), alloc.used_days - days)
                
        # 2. Revert Attendance
        d = req.start_date
        while d <= req.end_date:
            rec = db.query(AttendanceRecord).filter(
                AttendanceRecord.employee_id == req.employee_id,
                AttendanceRecord.date == d
            ).first()
            if rec and rec.status in ("ON_LEAVE", "PAID_LEAVE", "SHORT", "HALF_DAY"):
                if rec.sign_in_time is None and rec.sign_out_time is None:
                    db.delete(rec)
                else:
                    rec.status = "ABSENT"  # Can be recalculated later if punches exist
            d = d + timedelta(days=1)

    # 3. Cleanup Notifications
    try:
        from app.models import AppNotification
        db.query(AppNotification).filter(
            AppNotification.kind == "LEAVE",
            AppNotification.body.contains(f"{req.start_date} → {req.end_date}")
        ).delete(synchronize_session=False)
    except Exception:
        pass # Don't block deletion if notification cleanup fails
    
    db.delete(req)
    db.commit()
    return {"message": "Leave request deleted"}

@router.get("/balance")
def leave_balance(
    leave_type_id: int,
    employee_id: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    fy = get_current_financial_year(db)
    if not fy:
        raise HTTPException(status_code=400, detail="No financial year configured")
    eid = employee_id or current_user.employee_id
    if not eid:
        raise HTTPException(status_code=400, detail="Employee required")
    role_names = [r.name for r in current_user.roles]
    if "Employee" in role_names and eid != current_user.employee_id:
        raise HTTPException(status_code=403, detail="Access denied")
    bal = get_leave_balance(db, eid, leave_type_id, fy.id)
    return {"balance_days": float(bal), "financial_year_id": fy.id}
