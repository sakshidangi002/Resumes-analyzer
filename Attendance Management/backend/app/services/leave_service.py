"""Leave: FY April–March, no carry-forward; allocation and request workflow."""
from datetime import date
from decimal import Decimal
from sqlalchemy.orm import Session
from app.models import LeaveType, LeaveAllocation, LeaveRequest, FinancialYear, Employee
from app.models.employee import EmploymentType


def get_current_financial_year(db: Session) -> FinancialYear | None:
    """
    Return the financial year covering today.

    If none exists, automatically create one using the standard
    April–March cycle so that leave can work out of the box.
    """
    today = date.today()
    fy = db.query(FinancialYear).filter(
        FinancialYear.start_date <= today,
        FinancialYear.end_date >= today,
    ).first()
    if fy:
        return fy

    # Auto-create current FY: April–March
    year = today.year
    if today.month >= 4:
        start = date(year, 4, 1)
        end = date(year + 1, 3, 31)
        name = f"{year}-{str((year + 1) % 100).zfill(2)}"
    else:
        start = date(year - 1, 4, 1)
        end = date(year, 3, 31)
        name = f"{year-1}-{str(year % 100).zfill(2)}"

    fy = FinancialYear(start_date=start, end_date=end, is_current=True, name=name)
    db.add(fy)
    db.commit()
    db.refresh(fy)
    return fy


def _approved_leave_dates(
    db: Session,
    employee_id: int,
    leave_type_id: int,
    fy_start: date,
    fy_end: date,
) -> set[date]:
    """Calendar dates covered by approved leave requests within the financial year."""
    from datetime import timedelta

    dates: set[date] = set()
    rows = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id == employee_id,
            LeaveRequest.leave_type_id == leave_type_id,
            LeaveRequest.status == "APPROVED",
            LeaveRequest.start_date <= fy_end,
            LeaveRequest.end_date >= fy_start,
        )
        .all()
    )
    for req in rows:
        d = max(req.start_date, fy_start)
        end = min(req.end_date, fy_end)
        while d <= end:
            dates.add(d)
            d += timedelta(days=1)
    return dates


def count_hr_direct_paid_leave_days(
    db: Session,
    employee_id: int,
    fy_start: date,
    fy_end: date,
    pl_leave_type_id: int,
) -> Decimal:
    """Count PAID_LEAVE / HALF_DAY attendance marked by HR, excluding approved-request dates.

    Approved leave already increments ``used_days`` and sets matching attendance
    rows — counting those attendance rows again would double the used total.

    Non-employee staff (housekeeping, security, …) are on a fixed salary with no
    daily-hours target, so their short days are not leave at all and must not be
    charged here. They typically hold a ZERO Paid-Leave allocation, so counting
    them produced a nonsensical "3.5 used of 0 allocated" and a negative balance.
    """
    from app.models.attendance import AttendanceRecord
    from app.core.staff_policy import is_fixed_salary_staff
    from app.models.employee import Employee

    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if employee is not None and is_fixed_salary_staff(employee):
        return Decimal("0")

    approved_dates = _approved_leave_dates(
        db, employee_id, pl_leave_type_id, fy_start, fy_end
    )
    att_records = (
        db.query(AttendanceRecord)
        .filter(
            AttendanceRecord.employee_id == employee_id,
            AttendanceRecord.date >= fy_start,
            AttendanceRecord.date <= fy_end,
        )
        .order_by(AttendanceRecord.date.asc())
        .all()
    )

    hr_direct = Decimal("0")
    for rec in att_records:
        if rec.date in approved_dates:
            continue
        if rec.status == "PAID_LEAVE":
            hr_direct += Decimal("1")
        elif rec.status == "HALF_DAY":
            m_start = date(rec.date.year, rec.date.month, 1)
            m_end = (
                date(rec.date.year, rec.date.month + 1, 1)
                if rec.date.month < 12
                else date(rec.date.year + 1, 1, 1)
            )
            month_atts = [
                r for r in att_records if m_start <= r.date < m_end and r.date < rec.date
            ]
            month_sl_used = sum(1 for r in month_atts if r.status == "SHORT")
            month_hd_used = sum(1 for r in month_atts if r.status == "HALF_DAY")
            buffer_left = 2 - (month_sl_used + (month_hd_used * 2))
            if buffer_left < 2:
                hr_direct += Decimal("0.5")
    return hr_direct


def get_leave_balance(db: Session, employee_id: int, leave_type_id: int, fy_id: int) -> Decimal:
    alloc = db.query(LeaveAllocation).filter(
        LeaveAllocation.employee_id == employee_id,
        LeaveAllocation.financial_year_id == fy_id,
        LeaveAllocation.leave_type_id == leave_type_id,
    ).first()
    if not alloc:
        return Decimal("0")
    
    lt = db.query(LeaveType).filter(LeaveType.id == leave_type_id).first()
    if lt and lt.code == "PL":
        fy = db.query(FinancialYear).filter(FinancialYear.id == fy_id).first()
        if fy:
            hr_direct = count_hr_direct_paid_leave_days(
                db, employee_id, fy.start_date, fy.end_date, leave_type_id
            )
            return alloc.allocated_days - alloc.used_days - hr_direct
    elif lt and lt.code == "SL":
        from app.models.attendance import AttendanceRecord
        from sqlalchemy import func
        from datetime import date as ddate
        m = ddate.today().month
        y = ddate.today().year
        start = ddate(y, m, 1)
        end = ddate(y + 1, 1, 1) if m == 12 else ddate(y, m + 1, 1)
        
        n_req = db.query(func.count(LeaveRequest.id)).filter(
            LeaveRequest.employee_id == employee_id,
            LeaveRequest.leave_type_id == leave_type_id,
            LeaveRequest.status == "APPROVED",
            LeaveRequest.start_date >= start,
            LeaveRequest.start_date < end,
        ).scalar() or 0
        
        from app.models.employee import Employee
        att_records = db.query(AttendanceRecord, Employee).join(Employee, Employee.id == AttendanceRecord.employee_id).filter(
            AttendanceRecord.employee_id == employee_id,
            AttendanceRecord.date >= start,
            AttendanceRecord.date < end,
        ).all()
        
        unrequested_sl = 0
        for rec, emp in att_records:
            if rec.status == "SHORT":
                unrequested_sl += 1
            elif rec.status == "HALF_DAY":
                # 1 Half Day = 2 Short Leaves
                unrequested_sl += 2
            elif rec.status == "PRESENT" and rec.total_work_hours is not None:
                worked = float(rec.total_work_hours)
                expected = float(emp.expected_working_hours or 9.0)
                if (expected - 2.0) <= worked < expected:
                    unrequested_sl += 1
        
        return alloc.allocated_days - Decimal(n_req) - Decimal(unrequested_sl)

    return alloc.allocated_days - alloc.used_days


def allocate_leave_for_fy(
    db: Session,
    employee_id: int,
    financial_year_id: int,
    leave_type_id: int,
    days: Decimal,
) -> LeaveAllocation:
    if days < 0:
        raise ValueError("Allocated days cannot be negative")
    alloc = db.query(LeaveAllocation).filter(
        LeaveAllocation.employee_id == employee_id,
        LeaveAllocation.financial_year_id == financial_year_id,
        LeaveAllocation.leave_type_id == leave_type_id,
    ).first()
    if alloc:
        if days < alloc.used_days:
            raise ValueError(
                f"Allocated days ({days}) cannot be less than already used days ({alloc.used_days})"
            )
        alloc.allocated_days = days
    else:
        alloc = LeaveAllocation(
            employee_id=employee_id,
            financial_year_id=financial_year_id,
            leave_type_id=leave_type_id,
            allocated_days=days,
            used_days=Decimal("0"),
        )
        db.add(alloc)
    db.commit()
    db.refresh(alloc)
    return alloc


def apply_leave_request(
    db: Session,
    employee_id: int,
    leave_type_id: int,
    start_date: date,
    end_date: date,
    is_half_day: bool = False,
    reason: str | None = None,
) -> LeaveRequest:
    fy = get_current_financial_year(db)
    if not fy:
        raise ValueError("No financial year configured")
    req = LeaveRequest(
        employee_id=employee_id,
        leave_type_id=leave_type_id,
        start_date=start_date,
        end_date=end_date,
        is_half_day=is_half_day,
        reason=reason,
        status="PENDING",
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    return req


def _count_leave_days(start: date, end: date, is_half_day: bool) -> Decimal:
    if start > end:
        return Decimal("0")
    if is_half_day and start == end:
        return Decimal("0.5")
    delta = (end - start).days + 1
    return Decimal(delta)


# ---------------------------------------------------------------------------
# Monthly-earned Paid Leave (12/year, accrued 1 per month, no future months)
# ---------------------------------------------------------------------------
def _pl_leave_type(db: Session) -> LeaveType | None:
    return db.query(LeaveType).filter(LeaveType.code == "PL").first()


def earned_paid_leave_as_of(
    db: Session,
    employee_id: int,
    fy: FinancialYear,
    as_of: date,
    annual_days: Decimal | None = None,
) -> Decimal:
    """Paid leave EARNED from the start of the leave year up to ``as_of``.

    Policy: 1 paid leave earned per month, accruing from the later of the
    financial-year start and the employee's joining month. Only the current and
    all PREVIOUS months (relative to ``as_of``) count — future months are never
    included. Capped at the annual entitlement (default 12).

    Example: FY starts April, ``as_of`` in July -> Apr,May,Jun,Jul = 4 earned.
    """
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    start = fy.start_date
    if emp and emp.date_of_joining and emp.date_of_joining > start:
        start = emp.date_of_joining
    # Clamp the cutoff inside the FY so a leave dated before/after the year can
    # never over- or under-count the accrual.
    cutoff = min(max(as_of, fy.start_date), fy.end_date)
    if cutoff < start:
        return Decimal("0")
    months = (cutoff.year - start.year) * 12 + (cutoff.month - start.month) + 1
    if annual_days is None:
        annual_days = Decimal("12")
    earned = Decimal(max(0, months))
    return min(earned, annual_days)


def paid_leave_summary(
    db: Session,
    employee_id: int,
    fy: FinancialYear,
    as_of: date | None = None,
    exclude_request_id: int | None = None,
) -> dict:
    """Monthly-earned Paid-Leave figures for one employee in one financial year.

    Returns: annual_days, earned, used_paid, remaining, unpaid_used, balance.
      * earned      – accrued up to ``as_of`` (default today), future excluded.
      * used_paid   – PAID portion of approved PL (allocation.used_days, which
                      now tracks only paid days) + HR-marked PAID_LEAVE days.
      * remaining   – max(0, earned - used_paid).
      * unpaid_used – Unpaid/LWP portion split off approved PL + approved
                      fully-unpaid (UL) leave, within the FY.
    """
    as_of = as_of or date.today()
    pl = _pl_leave_type(db)
    zero = {
        "annual_days": Decimal("0"), "earned": Decimal("0"), "used_paid": Decimal("0"),
        "remaining": Decimal("0"), "unpaid_used": Decimal("0"), "balance": Decimal("0"),
    }
    if not pl:
        return zero

    alloc = db.query(LeaveAllocation).filter(
        LeaveAllocation.employee_id == employee_id,
        LeaveAllocation.financial_year_id == fy.id,
        LeaveAllocation.leave_type_id == pl.id,
    ).first()
    annual = alloc.allocated_days if alloc else Decimal("12")
    earned = earned_paid_leave_as_of(db, employee_id, fy, as_of, annual)
    hr_direct = count_hr_direct_paid_leave_days(
        db, employee_id, fy.start_date, fy.end_date, pl.id
    )
    used_paid = (alloc.used_days if alloc else Decimal("0")) + hr_direct

    # Unpaid (LWP) days recorded this FY: the unpaid split of approved PL requests
    # plus any approved fully-unpaid leave. Optionally exclude one request (the
    # one currently being approved, so its own figures aren't double counted).
    unpaid_used = Decimal("0")
    approved = (
        db.query(LeaveRequest, LeaveType)
        .join(LeaveType, LeaveType.id == LeaveRequest.leave_type_id)
        .filter(
            LeaveRequest.employee_id == employee_id,
            LeaveRequest.status == "APPROVED",
            LeaveRequest.start_date <= fy.end_date,
            LeaveRequest.end_date >= fy.start_date,
        )
        .all()
    )
    for req, lt in approved:
        if exclude_request_id is not None and req.id == exclude_request_id:
            continue
        if lt.code == "PL":
            unpaid_used += Decimal(str(req.unpaid_days or 0))
        elif not lt.is_paid:
            unpaid_used += _count_leave_days(
                req.start_date, req.end_date, bool(req.is_half_day)
            )

    remaining = earned - used_paid
    if remaining < 0:
        remaining = Decimal("0")
    return {
        "annual_days": annual,
        "earned": earned,
        "used_paid": used_paid,
        "remaining": remaining,
        "unpaid_used": unpaid_used,
        "balance": remaining,
    }


def compute_paid_leave_split(db: Session, req: LeaveRequest, fy: FinancialYear) -> dict:
    """Split a Paid-Leave request into paid vs unpaid (LWP) days.

    Uses the balance EARNED as of the leave START date, and excludes this
    request from the "already used" figure. Returns earned, used, remaining,
    requested, paid, unpaid.
    """
    summary = paid_leave_summary(
        db, req.employee_id, fy, as_of=req.start_date, exclude_request_id=req.id
    )
    requested = _count_leave_days(req.start_date, req.end_date, bool(req.is_half_day))
    remaining = summary["remaining"]
    paid = min(requested, remaining)
    if paid < 0:
        paid = Decimal("0")
    unpaid = requested - paid
    return {
        "earned": summary["earned"],
        "used": summary["used_paid"],
        "remaining": remaining,
        "requested": requested,
        "paid": paid,
        "unpaid": unpaid,
    }


def approve_leave_request(
    db: Session,
    request_id: int,
    approver_id: int,
    approved: bool,
    comment: str | None = None,
) -> LeaveRequest:
    req = db.query(LeaveRequest).filter(LeaveRequest.id == request_id).first()
    if not req or req.status != "PENDING":
        raise ValueError("Leave request not found or not pending")
    from app.core.datetime_utils import get_ist_now
    if approved:
        req.status = "APPROVED"
        req.approved_at = get_ist_now()
        req.manager_approver_id = approver_id
        req.rejection_reason = None
        req.response_comment = comment
        # Short Leave (SL) is counted as 1 unit per request (one 2-hour leave)
        lt = db.query(LeaveType).filter(LeaveType.id == req.leave_type_id).first()
        if lt and lt.code == "SL":
            days = Decimal("1")
        else:
            days = _count_leave_days(req.start_date, req.end_date, req.is_half_day)
        fy = get_current_financial_year(db)

        # ── Monthly-earned Paid-Leave split ─────────────────────────────────
        # For a FULL-DAY Paid Leave request, only the balance EARNED so far
        # (accrued 1/month up to the leave start date) is paid; the excess
        # becomes unpaid Loss-Of-Pay. Half-day PL and other leave types keep
        # their own paid/unpaid nature (no change in behaviour).
        is_full_day_pl = bool(lt and lt.code == "PL" and not req.is_half_day)
        if is_full_day_pl and fy:
            split = compute_paid_leave_split(db, req, fy)
            paid_days = split["paid"]
            unpaid_days = split["unpaid"]
        elif lt and not lt.is_paid:
            paid_days = Decimal("0")
            unpaid_days = days
        else:
            paid_days = days
            unpaid_days = Decimal("0")
        req.paid_days = paid_days
        req.unpaid_days = unpaid_days

        if fy:
            alloc = db.query(LeaveAllocation).filter(
                LeaveAllocation.employee_id == req.employee_id,
                LeaveAllocation.financial_year_id == fy.id,
                LeaveAllocation.leave_type_id == req.leave_type_id,
            ).first()
            # Only the PAID portion consumes the yearly allocation. Short Leave is
            # monthly (computed per-month) and never accumulates in used_days.
            if alloc and not (lt and lt.code == "SL"):
                alloc.used_days = alloc.used_days + paid_days

        # Update Attendance Records for the leave period. The PAID portion is
        # marked PAID_LEAVE and the unpaid (LWP) portion ON_LEAVE, earliest
        # working days first — so payroll deducts only the unpaid days, and only
        # in the month they fall in.
        from app.services.attendance_service import get_or_create_attendance
        from datetime import timedelta
        paid_assigned = Decimal("0")
        d = req.start_date
        while d <= req.end_date:
            rec = get_or_create_attendance(db, req.employee_id, d)
            # Update status unless it's a weekend/holiday that shouldn't be overridden
            if rec.status not in ("WEEKLY_OFF", "HOLIDAY"):
                if lt and lt.code == "SL":
                    rec.status = "SHORT"
                elif req.is_half_day:
                    rec.status = "HALF_DAY"
                elif lt and lt.code == "PL":
                    if paid_assigned < paid_days:
                        rec.status = "PAID_LEAVE"
                        paid_assigned += Decimal("1")
                    else:
                        rec.status = "ON_LEAVE"  # unpaid LWP portion
                else:
                    rec.status = "ON_LEAVE"
            d = d + timedelta(days=1)
    else:
        req.status = "REJECTED"
        req.rejected_at = get_ist_now()
        req.rejection_reason = comment
        req.response_comment = comment
    db.commit()
    db.refresh(req)
    return req


def ensure_default_allocations_for_employee(db: Session, employee_id: int, fy_id: int) -> None:
    """
    Create default allocations if missing.
    - Paid Leave (PL): 12 per FY for Full-time, else 0 (can be edited in Leave Allocations).
    - Short Leave (SL): 2 per month for Full-time, else 0 (displayed monthly; not carried forward).
    """
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        return
    pl = db.query(LeaveType).filter(LeaveType.code == "PL").first()
    sl = db.query(LeaveType).filter(LeaveType.code == "SL").first()
    if not pl or not sl:
        return

    is_full_time = (emp.employment_type or "") == EmploymentType.FULL_TIME.value
    defaults = [
        (pl.id, Decimal("12") if is_full_time else Decimal("0")),
        (sl.id, Decimal("2") if is_full_time else Decimal("0")),
    ]
    for leave_type_id, allocated in defaults:
        existing = db.query(LeaveAllocation).filter(
            LeaveAllocation.employee_id == employee_id,
            LeaveAllocation.financial_year_id == fy_id,
            LeaveAllocation.leave_type_id == leave_type_id,
        ).first()
        if not existing:
            db.add(
                LeaveAllocation(
                    employee_id=employee_id,
                    financial_year_id=fy_id,
                    leave_type_id=leave_type_id,
                    allocated_days=allocated,
                    used_days=Decimal("0"),
                )
            )
    db.commit()
