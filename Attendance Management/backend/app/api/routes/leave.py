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
    PaidLeaveSummaryResponse,
)
from app.api.deps import require_roles
from app.services.leave_service import (
    get_current_financial_year,
    get_leave_balance,
    allocate_leave_for_fy,
    apply_leave_request,
    approve_leave_request,
    ensure_default_allocations_for_employee,
    count_hr_direct_paid_leave_days,
    compute_paid_leave_split,
    paid_leave_summary,
    _count_leave_days,
)
from app.services.audit_service import log_audit
from app.services.notification_service import notify_user_for_employee, notify_users_with_roles
from app.services.email_service import send_notification
from app.core.config import get_settings
from decimal import Decimal
from datetime import date as _date
import logging

logger = logging.getLogger(__name__)


def _hr_notification_recipients(db: Session, requester_is_hr: bool) -> list[str]:
    """
    Decide which addresses receive HR-bound emails (e.g. new leave request).

    Priority:
      1) `HR_NOTIFICATION_EMAIL` from settings/.env (comma-separated list of addresses).
         This is the recommended setup: a single mailbox (e.g. hr@company.com) gets
         every notification, so you don't have to keep DB role assignments in sync
         with who is allowed to receive HR mail.
      2) Fallback: all users in the DB with the HR role (or Admin when the
         requester themselves is HR).
    """
    settings = get_settings()
    configured = (settings.hr_notification_email or "").strip()
    if configured:
        addrs = [a.strip() for a in configured.replace(";", ",").split(",") if a.strip()]
        return addrs

    target_roles = ["Admin"] if requester_is_hr else ["HR"]
    return _emails_for_roles(db, target_roles)


def _emails_for_roles(db: Session, role_names: list[str]) -> list[str]:
    """Return distinct, non-empty email addresses for users that have ANY of the given roles."""
    users = (
        db.query(User)
        .join(user_roles, User.id == user_roles.c.user_id)
        .join(Role, Role.id == user_roles.c.role_id)
        .filter(Role.name.in_(role_names))
        .all()
    )
    # Resolve every linked employee in ONE query. This used to be a per-user
    # lookup inside the loop below, so notifying HR cost one round trip per HR
    # user on every leave application/approval.
    employee_ids = [u.employee_id for u in users if u.employee_id]
    emails_by_employee_id = {
        row[0]: row[1]
        for row in db.query(Employee.id, Employee.official_email)
        .filter(Employee.id.in_(employee_ids))
        .all()
    } if employee_ids else {}

    seen: set[str] = set()
    out: list[str] = []
    for u in users:
        candidates: list[str] = []
        if u.official_email:
            candidates.append(u.official_email.strip())
        if u.username and "@" in u.username:
            candidates.append(u.username.strip())
        if u.employee_id:
            emp_email = emails_by_employee_id.get(u.employee_id)
            if emp_email:
                candidates.append(emp_email.strip())
        for c in candidates:
            if c and c.lower() not in seen:
                seen.add(c.lower())
                out.append(c)
    return out


def _send_leave_apply_email_to_hr(
    db: Session,
    requester_is_hr: bool,
    employee,
    employee_name: str,
    leave_type_name: str,
    start_date,
    end_date,
    is_half_day: bool,
    reason: str,
    requester_email: str | None,
) -> None:
    """
    Best-effort: notify HR (or Admin when the requester is HR) that a leave
    was applied. Renders the dashboard-style summary email. Never raises —
    leave application must not fail because of an email problem.
    """
    try:
        recipients = _hr_notification_recipients(db, requester_is_hr)
        if not recipients:
            logger.info("No HR notification recipients configured; skipping leave email.")
            return

        half_text = " (Half day)" if is_half_day else ""
        reason_clean = (reason or "").strip().replace("\n", "<br>")
        subject = f"Leave request from {employee_name} — {leave_type_name}"

        employee_line = employee_name
        if requester_email:
            employee_line = (
                f'{employee_name} &lt;<a href="mailto:{requester_email}" '
                f'style="color:#1a73e8; text-decoration:none">{requester_email}</a>&gt;'
            )

        reply_hint = (
            f'<p style="margin-top:14px"><em>Hit <strong>Reply</strong> '
            f'to respond directly to {employee_name}.</em></p>'
            if requester_email
            else ""
        )

        body_html = f"""\
<div style="font-family: Arial, sans-serif; color:#111; line-height:1.55">
    <h2 style="margin:0 0 12px 0; color:#1a73e8;">New leave request</h2>
    <p><strong>Employee:</strong> {employee_line}</p>
    <p><strong>Leave type:</strong> {leave_type_name}{half_text}</p>
    <p><strong>From:</strong> {start_date} &nbsp; <strong>To:</strong> {end_date}</p>
    <p><strong>Reason:</strong><br>{reason_clean or '—'}</p>
    {reply_hint}
    <hr style="border:none; border-top:1px solid #eee; margin:18px 0">
    <p style="font-size:12px; color:#666">
        This request is pending in the HRMS. Open
        <strong>Leave Approvals</strong> in the portal to approve or reject it.
    </p>
</div>"""

        # Reply-To is always the employee so "Reply" composes directly to them,
        # regardless of how Gmail handles the From-header rewrite.
        for addr in recipients:
            ok, err = send_notification(
                db,
                addr,
                subject,
                body_html,
                template_code="LEAVE_APPLY",
                related_entity_type="LeaveRequest",
                from_email=requester_email or None,
                from_name=employee_name or None,
                reply_to=requester_email or None,
                reply_to_name=employee_name or None,
            )
            if not ok:
                logger.warning("Leave email to %s failed: %s", addr, err)
    except Exception:
        logger.exception("Unexpected error while sending leave-apply email")

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
def list_leave_types(
    page: int | None = Query(None, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Leave types. Pagination is opt-in -- every caller populates a dropdown
    and needs the complete list, so `page` has no default."""
    q = db.query(LeaveType).order_by(LeaveType.id)
    if page is not None:
        q = q.offset((page - 1) * page_size).limit(page_size)
    return q.all()


@router.get("/allocations", response_model=list[LeaveAllocationResponse])
def list_leave_allocations(
    employee_id: int | None = Query(None),
    financial_year_id: int | None = Query(None),
    month: int | None = Query(None, description="For Short Leave monthly balance; default current month"),
    year: int | None = Query(None, description="For Short Leave monthly balance; default current year"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
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

    if employee_id is not None:
        if not is_admin_or_hr and current_user.employee_id != employee_id:
            return []
        if is_admin_or_hr:
            target_emp = db.query(Employee).filter(Employee.id == employee_id).first()
            if not target_emp or (target_emp.employment_status or "Active") != "Active":
                return []
        q = q.filter(LeaveAllocation.employee_id == employee_id)
        target_employee_id = employee_id
    else:
        if is_admin_or_hr and financial_year_id is not None:
            q = q.join(Employee, Employee.id == LeaveAllocation.employee_id).filter(
                LeaveAllocation.financial_year_id == financial_year_id,
                Employee.employment_status == "Active",
            )
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

    allocs = (
        q.order_by(LeaveAllocation.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
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
            hr_direct = count_hr_direct_paid_leave_days(
                db,
                a.employee_id,
                fy.start_date,
                fy.end_date,
                pl_type.id,
            )
            result.append(
                LeaveAllocationResponse(
                    id=a.id,
                    employee_id=a.employee_id,
                    financial_year_id=a.financial_year_id,
                    leave_type_id=a.leave_type_id,
                    allocated_days=a.allocated_days,
                    used_days=a.used_days + hr_direct,
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
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp or (emp.employment_status or "Active") != "Active":
        raise HTTPException(status_code=400, detail="Inactive employees cannot receive leave allocations.")
    try:
        return allocate_leave_for_fy(db, employee_id, fy.id, leave_type_id, allocated_days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/allocations/{allocation_id}")
def delete_allocation(
    allocation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Remove a leave allocation row.

    Refuses once any of it has been consumed. `used_days` is the only record
    that those days were taken -- the approved requests were decremented against
    THIS row -- so deleting it would not "free up" the leave, it would erase the
    evidence that it was ever spent while the approved requests stayed on the
    books. Set the allocation to the used figure instead if the intent is to
    stop further leave being taken.

    Note for PL and SL specifically: `ensure_default_allocations_for_employee`
    recreates a missing default the next time the employee's balance is read, so
    deleting one of those returns it as a 0-day row rather than removing it for
    good. Deleting is therefore a way to reset a row, not to hide a leave type.
    """
    alloc = db.query(LeaveAllocation).filter(LeaveAllocation.id == allocation_id).first()
    if not alloc:
        raise HTTPException(status_code=404, detail="Leave allocation not found")

    used = Decimal(str(alloc.used_days or 0))
    if used > 0:
        lt = db.query(LeaveType).filter(LeaveType.id == alloc.leave_type_id).first()
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot delete: {used} day(s) of "
                f"{(lt.name if lt else 'this leave')} have already been used. "
                "Reduce the allocation instead."
            ),
        )

    emp = db.query(Employee).filter(Employee.id == alloc.employee_id).first()
    lt = db.query(LeaveType).filter(LeaveType.id == alloc.leave_type_id).first()
    # Captured before the delete -- afterwards there is nothing left to describe.
    details = (
        f"{(emp.full_name if emp else 'employee ' + str(alloc.employee_id))}: "
        f"{(lt.code if lt else 'leave')} allocation of {alloc.allocated_days} day(s) "
        f"deleted (FY {alloc.financial_year_id})"
    )

    db.delete(alloc)
    db.commit()

    log_audit(
        db,
        current_user.id,
        "LEAVE_ALLOCATION_DELETE",
        entity_type="LeaveAllocation",
        entity_id=str(allocation_id),
        details=details,
    )
    return {"detail": "Leave allocation deleted"}


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
                with_push=True,
                exclude_user_id=current_user.id,
            )
        else:
            notify_users_with_roles(
                db,
                ["HR"],
                "New leave request",
                f"{name} requested {lt_name} ({data.start_date} → {data.end_date}).",
                kind="LEAVE",
                link_path="/leave-approvals",
                with_push=True,
                exclude_user_id=current_user.id,
            )
        requester_email = (emp.official_email if emp else None) or (
            current_user.official_email if current_user else None
        )
        _send_leave_apply_email_to_hr(
            db,
            requester_is_hr=requester_is_hr,
            employee=emp,
            employee_name=name,
            leave_type_name=lt_name,
            start_date=data.start_date,
            end_date=data.end_date,
            is_half_day=data.is_half_day,
            reason=data.reason or "",
            requester_email=requester_email,
        )
        return req
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/requests", response_model=list[LeaveRequestResponse])
def list_leave_requests(
    employee_id: int | None = Query(None),
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
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
    return (
        q.order_by(LeaveRequest.applied_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )


@router.get("/approvals", response_model=list[LeaveApprovalRow])
def list_leave_approvals(
    status: str = Query("PENDING"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
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
    rows = (
        q.order_by(LeaveRequest.applied_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    fy = get_current_financial_year(db)
    result: list[LeaveApprovalRow] = []
    for req, emp, lt, is_hr_val in rows:
        if is_hr and not is_admin:
            # HR should not approve HR leave; HR approvals are for employees only
            if is_hr_val:
                continue

        # Paid-Leave split preview so HR sees Earned / Used / Remaining / Paid /
        # Unpaid before approving. Only meaningful for a still-PENDING full-day
        # PL request; for an already-decided one we surface what was stored.
        pl_fields: dict = {}
        if lt.code == "PL" and fy:
            if req.status == "PENDING" and not req.is_half_day:
                split = compute_paid_leave_split(db, req, fy)
                pl_fields = {
                    "pl_earned": split["earned"],
                    "pl_used": split["used"],
                    "pl_remaining": split["remaining"],
                    "pl_requested": split["requested"],
                    "pl_paid": split["paid"],
                    "pl_unpaid": split["unpaid"],
                }
            else:
                pl_fields = {
                    "pl_requested": _count_leave_days(
                        req.start_date, req.end_date, bool(req.is_half_day)
                    ),
                    "pl_paid": req.paid_days,
                    "pl_unpaid": req.unpaid_days,
                }

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
                **pl_fields,
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
            with_push=True,
            push_tag=f"leave-decision-{req.id}",
        )
        return req
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


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
        
        # 1. Revert allocation
        lt = db.query(LeaveType).filter(LeaveType.id == req.leave_type_id).first()
        if lt and lt.code == "SL":
            days = Decimal("1")
        else:
            days = _count_leave_days(req.start_date, req.end_date, bool(req.is_half_day))

        # Only the PAID portion was added to used_days at approval, so only revert
        # that. Legacy rows approved before the paid/unpaid split (both 0) fall
        # back to the full day count to preserve the old revert behaviour.
        paid_part = Decimal(str(req.paid_days or 0))
        unpaid_part = Decimal(str(req.unpaid_days or 0))
        revert_days = paid_part if (paid_part or unpaid_part) else days

        fy = get_current_financial_year(db)
        if fy:
            alloc = db.query(LeaveAllocation).filter(
                LeaveAllocation.employee_id == req.employee_id,
                LeaveAllocation.financial_year_id == fy.id,
                LeaveAllocation.leave_type_id == req.leave_type_id,
            ).first()
            if alloc and not (lt and lt.code == "SL"):
                alloc.used_days = max(Decimal("0"), alloc.used_days - revert_days)
                
        # 2. Revert Attendance
        # One range query instead of one query PER DAY -- a month-long leave
        # previously cost ~30 sequential round trips here.
        affected = db.query(AttendanceRecord).filter(
            AttendanceRecord.employee_id == req.employee_id,
            AttendanceRecord.date >= req.start_date,
            AttendanceRecord.date <= req.end_date,
        ).all()
        for rec in affected:
            if rec.status in ("ON_LEAVE", "PAID_LEAVE", "SHORT", "HALF_DAY"):
                if rec.sign_in_time is None and rec.sign_out_time is None:
                    db.delete(rec)
                else:
                    rec.status = "ABSENT"  # Can be recalculated later if punches exist

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


@router.get("/paid-leave-summary", response_model=PaidLeaveSummaryResponse)
def paid_leave_summary_endpoint(
    employee_id: int | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """Monthly-earned Paid-Leave figures for the 'My Leave' page (as of today).

    Employees may only view their own; Admin/HR may pass any employee_id.
    """
    fy = get_current_financial_year(db)
    if not fy:
        raise HTTPException(status_code=400, detail="No financial year configured")

    role_names = [r.name for r in current_user.roles]
    is_admin_or_hr = "Admin" in role_names or "HR" in role_names
    eid = employee_id if (employee_id and is_admin_or_hr) else current_user.employee_id
    if not eid:
        raise HTTPException(status_code=400, detail="Employee required")
    if not is_admin_or_hr and eid != current_user.employee_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Make sure the PL allocation exists so 'annual' reflects the real entitlement.
    ensure_default_allocations_for_employee(db, eid, fy.id)
    s = paid_leave_summary(db, eid, fy)
    return PaidLeaveSummaryResponse(
        employee_id=eid,
        financial_year_id=fy.id,
        annual_days=s["annual_days"],
        earned=s["earned"],
        used_paid=s["used_paid"],
        remaining=s["remaining"],
        unpaid_used=s["unpaid_used"],
        balance=s["balance"],
    )
