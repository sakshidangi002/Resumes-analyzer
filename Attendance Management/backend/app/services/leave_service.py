"""Leave: FY April–March, no carry-forward; allocation and request workflow."""
from datetime import date, datetime
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
        from app.models.attendance import AttendanceRecord
        fy = db.query(FinancialYear).filter(FinancialYear.id == fy_id).first()
        if fy:
            from sqlalchemy import func
            unrequested_paid_leaves = db.query(func.count(AttendanceRecord.id)).filter(
                AttendanceRecord.employee_id == employee_id,
                AttendanceRecord.status == "PAID_LEAVE",
                AttendanceRecord.date >= fy.start_date,
                AttendanceRecord.date <= fy.end_date,
            ).scalar() or 0
            return alloc.allocated_days - alloc.used_days - Decimal(unrequested_paid_leaves)
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
        if fy:
            alloc = db.query(LeaveAllocation).filter(
                LeaveAllocation.employee_id == req.employee_id,
                LeaveAllocation.financial_year_id == fy.id,
                LeaveAllocation.leave_type_id == req.leave_type_id,
            ).first()
            # For monthly Short Leave we don't accumulate used_days in allocation; it's computed per-month.
            if alloc and not (lt and lt.code == "SL"):
                alloc.used_days = alloc.used_days + days
        
        # Update Attendance Records for the leave period
        from app.services.attendance_service import get_or_create_attendance
        from datetime import timedelta
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
                    rec.status = "PAID_LEAVE"
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
