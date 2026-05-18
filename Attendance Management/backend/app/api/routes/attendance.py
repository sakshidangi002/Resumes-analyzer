"""Attendance: HR-managed grid, sign-out helpers, correction requests."""
from datetime import date, time, datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models import User, AttendanceRecord, AttendanceCorrectionRequest
from app.schemas.attendance import (
    AttendanceRecordResponse,
    AttendanceCorrectionRequestCreate,
    AttendanceCorrectionRequestResponse,
    AdminSetAttendance,
)
from app.api.deps import get_current_user, require_roles
from app.services.attendance_service import (
    sign_in,
    sign_out,
    get_or_create_attendance,
    calculate_work_hours,
    apply_status_from_hours,
)

router = APIRouter()


@router.post("/sign-in", response_model=AttendanceRecordResponse)
def attendance_sign_in(
    d: date,
    sign_in_time: time,
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["HR"])),
):
    """Sign-in is now HR-only; employee_id must be provided."""
    if d > date.today():
        raise HTTPException(status_code=400, detail="Cannot record attendance for future dates")
    if d == date.today() and sign_in_time > datetime.now().time():
        raise HTTPException(status_code=400, detail="Cannot record future Sign-In time for today")
    rec = sign_in(db, employee_id, d, sign_in_time)
    return rec


@router.post("/sign-out", response_model=AttendanceRecordResponse)
def attendance_sign_out(
    d: date,
    sign_out_time: time,
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["HR"])),
):
    """Sign-out is now HR-only; employee_id must be provided."""
    if d > date.today():
        raise HTTPException(status_code=400, detail="Cannot record attendance for future dates")
    if d == date.today() and sign_out_time > datetime.now().time():
        raise HTTPException(status_code=400, detail="Cannot record future Sign-Out time for today")
    rec = sign_out(db, employee_id, d, sign_out_time)
    return rec


@router.get("", response_model=list[AttendanceRecordResponse])
def list_attendance(
    from_date: date = Query(...),
    to_date: date = Query(...),
    employee_id: int | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    role_names = [r.name for r in current_user.roles]
    eid = employee_id
    if "Employee" in role_names and "Manager" not in role_names and "HR" not in role_names and "Admin" not in role_names:
        eid = current_user.employee_id
        if not eid:
            return []
    q = db.query(AttendanceRecord).filter(
        AttendanceRecord.date >= from_date,
        AttendanceRecord.date <= to_date,
    )
    if eid is not None:
        q = q.filter(AttendanceRecord.employee_id == eid)
    return q.order_by(AttendanceRecord.date.desc()).all()


@router.put("/admin-set", response_model=AttendanceRecordResponse)
def admin_set_attendance(
    data: AdminSetAttendance,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["HR"])),
):
    """
    HR can set or correct attendance for any employee and past/current dates only,
    matching the weekly grid (time in, time out, hours, status).
    """
    if data.date > date.today():
        raise HTTPException(status_code=400, detail="Cannot record attendance for future dates")
    if data.date == date.today():
        now_t = datetime.now().time()
        
        # Check Sign-In
        if data.sign_in_time and data.sign_in_time > now_t:
            raise HTTPException(status_code=400, detail="Cannot record future Sign-In time for today")
            
        # Check Sign-Out (with AM/PM logic)
        if data.sign_out_time:
            actual_out = data.sign_out_time
            # If out is before in, we treat it as PM internally
            if data.sign_in_time and actual_out <= data.sign_in_time:
                if actual_out.hour < 12:
                    actual_out = time(actual_out.hour + 12, actual_out.minute, actual_out.second)
            
            if actual_out > now_t:
                raise HTTPException(status_code=400, detail="Cannot record future Sign-Out time for today")
    rec = get_or_create_attendance(db, data.employee_id, data.date)
    rec.sign_in_time = data.sign_in_time
    rec.sign_out_time = data.sign_out_time
    rec.total_work_hours = calculate_work_hours(rec.sign_in_time, rec.sign_out_time)
    # 1. System calculates automatic status based on times
    apply_status_from_hours(db, rec)
    
    # 2. Handle cases where no times are entered (WO/Holiday/Absent)
    if rec.sign_in_time is None and rec.sign_out_time is None:
        if rec.is_weekly_off:
            rec.status = "WEEKLY_OFF"
        elif rec.is_holiday:
            rec.status = "HOLIDAY"
        else:
            rec.status = "ABSENT"
    
    # 3. Final Override: If HR explicitly provided a status in the request, use it.
    # This ensures that if HR selects "Present", it STAYS as "Present".
    if data.status:
        rec.status = data.status
    rec.source = "ADMIN"
    db.commit()
    db.refresh(rec)
    return rec


@router.post("/correction-requests", response_model=AttendanceCorrectionRequestResponse)
def create_correction_request(
    data: AttendanceCorrectionRequestCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    if not current_user.employee_id:
        raise HTTPException(status_code=400, detail="No employee linked")
    req = AttendanceCorrectionRequest(
        employee_id=current_user.employee_id,
        attendance_date=data.attendance_date,
        requested_sign_in_time=data.requested_sign_in_time,
        requested_sign_out_time=data.requested_sign_out_time,
        requested_status=data.requested_status,
        reason=data.reason,
        status="PENDING",
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    return req


@router.get("/correction-requests", response_model=list[AttendanceCorrectionRequestResponse])
def list_correction_requests(
    employee_id: int | None = Query(None),
    status: str | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    q = db.query(AttendanceCorrectionRequest)
    role_names = [r.name for r in current_user.roles]
    if "Employee" in role_names and "Manager" not in role_names and "HR" not in role_names and "Admin" not in role_names:
        q = q.filter(AttendanceCorrectionRequest.employee_id == current_user.employee_id)
    elif employee_id is not None:
        q = q.filter(AttendanceCorrectionRequest.employee_id == employee_id)
    if status:
        q = q.filter(AttendanceCorrectionRequest.status == status)
    return q.order_by(AttendanceCorrectionRequest.created_at.desc()).all()


@router.patch("/correction-requests/{request_id}", response_model=AttendanceCorrectionRequestResponse)
def approve_correction_request(
    request_id: int,
    approved: bool,
    rejection_reason: str | None = None,
  db: Session = Depends(get_db),
  current_user: User = Depends(require_roles(["Admin", "HR", "Manager"])),
):
    req = db.query(AttendanceCorrectionRequest).filter(AttendanceCorrectionRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    if req.status != "PENDING":
        raise HTTPException(status_code=400, detail="Request already processed")
    from app.core.datetime_utils import get_ist_now
    req.status = "APPROVED" if approved else "REJECTED"
    req.approver_id = current_user.id
    req.approved_at = get_ist_now()
    req.rejection_reason = rejection_reason
    if approved:
        rec = get_or_create_attendance(db, req.employee_id, req.attendance_date)
        if req.requested_sign_in_time:
            rec.sign_in_time = req.requested_sign_in_time
        if req.requested_sign_out_time:
            rec.sign_out_time = req.requested_sign_out_time
        rec.total_work_hours = calculate_work_hours(rec.sign_in_time, rec.sign_out_time)
        if req.requested_status:
            rec.status = req.requested_status
        rec.source = "CORRECTION"
    db.commit()
    db.refresh(req)
    return req
