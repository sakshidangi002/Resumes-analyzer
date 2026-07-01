"""Attendance: HR-managed grid, sign-out helpers, correction requests, biometric events."""
from datetime import date, time, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import User, AttendanceRecord, AttendanceCorrectionRequest, Employee
from app.schemas.attendance import (
    AttendanceRecordResponse,
    AttendanceCorrectionRequestCreate,
    AttendanceCorrectionRequestResponse,
    AdminSetAttendance,
    AutoMarkAttendance,
    AttendanceEventCreate,
    AttendanceEventResponse,
    AttendanceDetailsResponse,
    DailyAttendanceReportRow,
)
from app.api.deps import get_current_user, require_roles
from app.services.attendance_service import (
    sign_in,
    sign_out,
    get_or_create_attendance,
    calculate_work_hours,
    apply_status_from_hours,
)
from app.services.attendance_event_service import (
    add_attendance_event,
    get_events_for_day,
    recalculate_attendance_summary,
    calculate_intervals_from_events,
    format_duration,
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


@router.post("/auto-mark", response_model=AttendanceRecordResponse)
def auto_mark_attendance(
    data: AutoMarkAttendance,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["HR", "Admin"])),
):
    """Create or update attendance from camera / face-recognition events."""
    if data.date > date.today():
        raise HTTPException(status_code=400, detail="Cannot record attendance for future dates")

    now_time = datetime.now().time()
    sign_in_time = data.sign_in_time
    sign_out_time = data.sign_out_time

    if data.date == date.today():
        if sign_in_time and sign_in_time > now_time:
            raise HTTPException(status_code=400, detail="Cannot record future Sign-In time for today")
        if sign_out_time and sign_out_time > now_time:
            raise HTTPException(status_code=400, detail="Cannot record future Sign-Out time for today")

    # Prefer event-based flow when no explicit times are supplied
    if sign_in_time is None and sign_out_time is None:
        event_time = datetime.combine(data.date, now_time)
        try:
            event, rec, action = add_attendance_event(
                db,
                data.employee_id,
                event_time=event_time,
                source="AUTO",
            )
            if action == "cooldown":
                rec = get_or_create_attendance(db, data.employee_id, data.date)
                db.refresh(rec)
                return rec
            return rec
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    rec = get_or_create_attendance(db, data.employee_id, data.date)

    if sign_in_time is not None:
        rec.sign_in_time = sign_in_time
        rec.is_late = False
    if sign_out_time is not None:
        rec.sign_out_time = sign_out_time

    if sign_in_time is not None and sign_out_time is None:
        rec.total_work_hours = None
        rec.status = "PRESENT"
        config = None
        try:
            from app.services.attendance_service import get_company_config
            config = get_company_config(db)
        except Exception:
            config = None
        grace_min = config.grace_time_minutes if config else 15
        standard_start = time(9, 0)
        t = datetime.combine(data.date, sign_in_time)
        s = datetime.combine(data.date, standard_start)
        rec.is_late = (t - s).total_seconds() > grace_min * 60
    else:
        rec.total_work_hours = calculate_work_hours(rec.sign_in_time, rec.sign_out_time)
        apply_status_from_hours(db, rec)

    if rec.sign_in_time is None and rec.sign_out_time is None:
        if rec.is_weekly_off:
            rec.status = "WEEKLY_OFF"
        elif rec.is_holiday:
            rec.status = "HOLIDAY"
        else:
            rec.status = "ABSENT"

    standard_end = time(18, 0)
    if rec.sign_out_time is not None:
        t = datetime.combine(data.date, rec.sign_out_time)
        s = datetime.combine(data.date, standard_end)
        rec.is_early_exit = t < s

    rec.source = "AUTO"
    db.commit()
    db.refresh(rec)
    return rec


@router.post("/events", response_model=AttendanceEventResponse)
def create_attendance_event(
    data: AttendanceEventCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Add a biometric-style IN/OUT attendance event."""
    try:
        event, _rec, action = add_attendance_event(
            db,
            data.employee_id,
            event_time=data.event_time,
            event_type=data.event_type,
            source=data.source or "MANUAL",
            camera_id=data.camera_id,
        )
        if action == "cooldown":
            raise HTTPException(
                status_code=409,
                detail="Duplicate detection ignored (60 second cooldown active)",
            )
        return event
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/events", response_model=list[AttendanceEventResponse])
def list_attendance_events(
    employee_id: int = Query(...),
    event_date: date = Query(..., alias="date"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    role_names = [r.name for r in current_user.roles]
    if (
        "Employee" in role_names
        and "Manager" not in role_names
        and "HR" not in role_names
        and "Admin" not in role_names
    ):
        if current_user.employee_id != employee_id:
            raise HTTPException(status_code=403, detail="Not allowed to view other employees' events")
    return get_events_for_day(db, employee_id, event_date)


@router.get("/details", response_model=AttendanceDetailsResponse)
def get_attendance_details(
    employee_id: int = Query(...),
    event_date: date = Query(..., alias="date"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    role_names = [r.name for r in current_user.roles]
    if (
        "Employee" in role_names
        and "Manager" not in role_names
        and "HR" not in role_names
        and "Admin" not in role_names
    ):
        if current_user.employee_id != employee_id:
            raise HTTPException(status_code=403, detail="Not allowed to view other employees' details")

    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    rec = recalculate_attendance_summary(db, employee_id, event_date)
    db.commit()
    db.refresh(rec)
    events = get_events_for_day(db, employee_id, event_date)

    return AttendanceDetailsResponse(
        employee_id=employee_id,
        employee_name=employee.full_name,
        date=event_date,
        events=events,
        sign_in_time=rec.sign_in_time,
        sign_out_time=rec.sign_out_time,
        total_work_hours=float(rec.total_work_hours) if rec.total_work_hours is not None else None,
        total_break_hours=float(rec.total_break_hours) if rec.total_break_hours is not None else None,
        status=rec.status,
        is_late=rec.is_late,
        is_early_exit=rec.is_early_exit,
    )


@router.post("/recalculate", response_model=AttendanceRecordResponse)
def recalculate_attendance(
    employee_id: int = Query(...),
    event_date: date = Query(..., alias="date"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    rec = recalculate_attendance_summary(db, employee_id, event_date)
    db.commit()
    db.refresh(rec)
    return rec


@router.get("/daily-report", response_model=list[DailyAttendanceReportRow])
def daily_attendance_report(
    report_date: date = Query(..., alias="date"),
    department_id: int | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager"])),
):
    q = db.query(Employee).filter(Employee.employment_status == "Active")
    if department_id is not None:
        q = q.filter(Employee.department_id == department_id)
    employees = q.order_by(Employee.employee_code.asc()).all()

    rows: list[DailyAttendanceReportRow] = []
    for emp in employees:
        rec = (
            db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.employee_id == emp.id,
                AttendanceRecord.date == report_date,
            )
            .first()
        )
        events = get_events_for_day(db, emp.id, report_date)
        if events and rec:
            rec = recalculate_attendance_summary(db, emp.id, report_date)

        rows.append(
            DailyAttendanceReportRow(
                employee_id=emp.id,
                employee_code=emp.employee_code or "",
                employee_name=emp.full_name,
                date=report_date,
                sign_in_time=rec.sign_in_time if rec else None,
                sign_out_time=rec.sign_out_time if rec else None,
                total_work_hours=float(rec.total_work_hours) if rec and rec.total_work_hours is not None else None,
                total_break_hours=float(rec.total_break_hours) if rec and rec.total_break_hours is not None else None,
                expected_working_hours=float(emp.expected_working_hours or 9.0),
                status=rec.status if rec else "ABSENT",
                is_late=rec.is_late if rec else False,
                is_early_exit=rec.is_early_exit if rec else False,
                event_count=len(events),
            )
        )
    db.commit()
    return rows


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

        if data.sign_in_time and data.sign_in_time > now_t:
            raise HTTPException(status_code=400, detail="Cannot record future Sign-In time for today")

        if data.sign_out_time:
            actual_out = data.sign_out_time
            if data.sign_in_time and actual_out <= data.sign_in_time:
                if actual_out.hour < 12:
                    actual_out = time(actual_out.hour + 12, actual_out.minute, actual_out.second)

            if actual_out > now_t:
                raise HTTPException(status_code=400, detail="Cannot record future Sign-Out time for today")
    rec = get_or_create_attendance(db, data.employee_id, data.date)
    rec.sign_in_time = data.sign_in_time
    rec.sign_out_time = data.sign_out_time
    rec.total_work_hours = calculate_work_hours(rec.sign_in_time, rec.sign_out_time)
    apply_status_from_hours(db, rec)

    if rec.sign_in_time is None and rec.sign_out_time is None:
        if rec.is_weekly_off:
            rec.status = "WEEKLY_OFF"
        elif rec.is_holiday:
            rec.status = "HOLIDAY"
        else:
            rec.status = "ABSENT"

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


@router.get("/today")
def get_today_attendance(
    department_id: int | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager"])),
):
    """Get today's attendance for all employees (or filtered by department)."""
    from app.core.datetime_utils import get_ist_now
    from app.services.attendance_event_service import get_events_for_day, recalculate_attendance_summary
    
    today = get_ist_now().date()
    
    q = db.query(Employee).filter(Employee.employment_status == "Active")
    if department_id is not None:
        q = q.filter(Employee.department_id == department_id)
    employees = q.order_by(Employee.employee_code.asc()).all()
    
    attendance_list = []
    for emp in employees:
        rec = (
            db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.employee_id == emp.id,
                AttendanceRecord.date == today,
            )
            .first()
        )
        
        if rec:
            rec = recalculate_attendance_summary(db, emp.id, today)
            events = get_events_for_day(db, emp.id, today)
        else:
            events = []
        
        attendance_list.append({
            "employee_id": emp.id,
            "employee_code": emp.employee_code,
            "employee_name": emp.full_name,
            "department": emp.department.name if emp.department else None,
            "status": rec.status if rec else "ABSENT",
            "sign_in_time": rec.sign_in_time.isoformat() if rec and rec.sign_in_time else None,
            "sign_out_time": rec.sign_out_time.isoformat() if rec and rec.sign_out_time else None,
            "total_work_hours": float(rec.total_work_hours) if rec and rec.total_work_hours else 0,
            "total_break_hours": float(rec.total_break_hours) if rec and rec.total_break_hours else 0,
            "is_late": rec.is_late if rec else False,
            "is_early_exit": rec.is_early_exit if rec else False,
            "event_count": len(events),
        })
    
    return attendance_list


@router.get("/live-status")
def get_live_attendance_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager"])),
):
    """Get real-time attendance status for all active employees today."""
    from app.core.datetime_utils import get_ist_now
    from app.services.attendance_event_service import get_latest_event_for_day
    
    today = get_ist_now().date()
    
    # Get all active employees
    employees = db.query(Employee).filter(Employee.employment_status == "Active").all()
    
    currently_working = []
    currently_outside = []
    checked_out = []
    absent = []
    last_recognition_events = []
    
    for emp in employees:
        rec = (
            db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.employee_id == emp.id,
                AttendanceRecord.date == today,
            )
            .first()
        )
        latest_event = get_latest_event_for_day(db, emp.id, today)
        
        # Determine current state
        current_state = "ABSENT"
        if latest_event:
            latest_type = latest_event.event_type.upper()
            if latest_type in {"IN", "BREAK_IN"}:
                current_state = "WORKING"
            elif latest_type in {"OUT", "BREAK_OUT"}:
                # Check if they have a check-in today
                if rec and rec.sign_in_time:
                    current_state = "OUTSIDE"
                else:
                    current_state = "CHECKED_OUT"
        
        employee_data = {
            "employee_id": emp.id,
            "employee_code": emp.employee_code,
            "employee_name": emp.full_name,
            "department": emp.department.name if emp.department else None,
            "status": rec.status if rec else "ABSENT",
            "sign_in_time": rec.sign_in_time.isoformat() if rec and rec.sign_in_time else None,
            "sign_out_time": rec.sign_out_time.isoformat() if rec and rec.sign_out_time else None,
            "total_work_hours": float(rec.total_work_hours) if rec and rec.total_work_hours else 0,
            "total_break_hours": float(rec.total_break_hours) if rec and rec.total_break_hours else 0,
            "last_event_type": latest_event.event_type if latest_event else None,
            "last_event_time": latest_event.event_time.isoformat() if latest_event else None,
            "current_state": current_state,
        }
        
        # Categorize
        if current_state == "WORKING":
            currently_working.append(employee_data)
        elif current_state == "OUTSIDE":
            currently_outside.append(employee_data)
        elif current_state == "CHECKED_OUT":
            checked_out.append(employee_data)
        else:
            absent.append(employee_data)
        
        # Add to last recognition events
        if latest_event:
            last_recognition_events.append({
                "employee_id": emp.id,
                "employee_name": emp.full_name,
                "event_type": latest_event.event_type,
                "event_time": latest_event.event_time.isoformat(),
                "camera_id": latest_event.camera_id,
            })
    
    # Sort last recognition events by time (most recent first)
    last_recognition_events.sort(key=lambda x: x["event_time"], reverse=True)
    last_recognition_events = last_recognition_events[:20]  # Last 20 events
    
    return {
        "currently_working": currently_working,
        "currently_outside": currently_outside,
        "checked_out": checked_out,
        "absent": absent,
        "last_recognition_events": last_recognition_events,
        "summary": {
            "total_employees": len(employees),
            "currently_working_count": len(currently_working),
            "currently_outside_count": len(currently_outside),
            "checked_out_count": len(checked_out),
            "absent_count": len(absent),
        },
    }


@router.get("/employee/{employee_id}/history")
def get_employee_attendance_history(
    employee_id: int,
    from_date: date = Query(...),
    to_date: date = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager"])),
):
    """Get detailed attendance history for an employee with events."""
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    from app.services.attendance_event_service import get_events_for_day, recalculate_attendance_summary
    
    history = []
    current_date = from_date
    while current_date <= to_date:
        rec = (
            db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.employee_id == employee_id,
                AttendanceRecord.date == current_date,
            )
            .first()
        )
        
        if rec:
            rec = recalculate_attendance_summary(db, employee_id, current_date)
            events = get_events_for_day(db, employee_id, current_date)
        else:
            events = []
        
        history.append({
            "date": current_date.isoformat(),
            "sign_in_time": rec.sign_in_time.isoformat() if rec and rec.sign_in_time else None,
            "sign_out_time": rec.sign_out_time.isoformat() if rec and rec.sign_out_time else None,
            "total_work_hours": float(rec.total_work_hours) if rec and rec.total_work_hours else 0,
            "total_break_hours": float(rec.total_break_hours) if rec and rec.total_break_hours else 0,
            "status": rec.status if rec else "ABSENT",
            "is_late": rec.is_late if rec else False,
            "is_early_exit": rec.is_early_exit if rec else False,
            "is_weekly_off": rec.is_weekly_off if rec else False,
            "is_holiday": rec.is_holiday if rec else False,
            "events": [
                {
                    "event_time": e.event_time.isoformat(),
                    "event_type": e.event_type,
                    "source": e.source,
                    "camera_id": e.camera_id,
                }
                for e in events
            ],
        })
        
        current_date = date.fromordinal(current_date.toordinal() + 1)
    
    return {
        "employee_id": employee_id,
        "employee_code": employee.employee_code,
        "employee_name": employee.full_name,
        "history": history,
    }


@router.get("/employee/{employee_id}/timeline")
def get_attendance_timeline(
    employee_id: int,
    attendance_date: date = Query(..., alias="date"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """Get detailed attendance timeline with work/break breakdown for a specific date."""
    role_names = [r.name for r in current_user.roles]
    if (
        "Employee" in role_names
        and "Manager" not in role_names
        and "HR" not in role_names
        and "Admin" not in role_names
    ):
        if current_user.employee_id != employee_id:
            raise HTTPException(status_code=403, detail="Not allowed to view other employees' timeline")

    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    rec = recalculate_attendance_summary(db, employee_id, attendance_date)
    events = get_events_for_day(db, employee_id, attendance_date)
    work_h, break_h, first_in, last_out, timeline = calculate_intervals_from_events(events)

    # Determine current status
    latest_event = events[-1] if events else None
    current_status = "ABSENT"
    if latest_event:
        latest_type = latest_event.event_type.upper()
        if latest_type in {"IN", "BREAK_IN"}:
            current_status = "CHECKED_IN"
        elif latest_type in {"OUT", "BREAK_OUT"}:
            current_status = "CHECKED_OUT"

    return {
        "employee_id": employee_id,
        "employee_code": employee.employee_code,
        "employee_name": employee.full_name,
        "date": attendance_date.isoformat(),
        "first_check_in": first_in.isoformat() if first_in else None,
        "final_check_out": last_out.isoformat() if last_out else None,
        "total_work_hours": float(work_h) if work_h else 0,
        "total_break_hours": float(break_h) if break_h else 0,
        "current_status": current_status,
        "status": rec.status,
        "timeline": timeline,
        "events": [
            {
                "event_time": e.event_time.isoformat(),
                "event_type": e.event_type,
                "source": e.source,
                "camera_id": e.camera_id,
            }
            for e in events
        ],
    }


@router.get("/employee/{employee_id}/monthly-summary")
def get_monthly_attendance_summary(
    employee_id: int,
    year: int = Query(...),
    month: int = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """Get monthly attendance summary with statistics."""
    role_names = [r.name for r in current_user.roles]
    if (
        "Employee" in role_names
        and "Manager" not in role_names
        and "HR" not in role_names
        and "Admin" not in role_names
    ):
        if current_user.employee_id != employee_id:
            raise HTTPException(status_code=403, detail="Not allowed to view other employees' summary")

    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    from calendar import monthrange
    from datetime import timedelta

    first_day = date(year, month, 1)
    last_day = date(year, month, monthrange(year, month)[1])

    records = (
        db.query(AttendanceRecord)
        .filter(
            AttendanceRecord.employee_id == employee_id,
            AttendanceRecord.date >= first_day,
            AttendanceRecord.date <= last_day,
        )
        .all()
    )

    total_present = 0
    total_absent = 0
    total_work_hours = 0
    total_break_hours = 0
    late_arrivals = 0
    early_exits = 0
    check_in_times = []
    check_out_times = []

    for rec in records:
        if rec.status in {"PRESENT", "SHORT", "HALF_DAY"}:
            total_present += 1
        elif rec.status == "ABSENT":
            total_absent += 1
        
        if rec.total_work_hours:
            total_work_hours += float(rec.total_work_hours)
        if rec.total_break_hours:
            total_break_hours += float(rec.total_break_hours)
        if rec.is_late:
            late_arrivals += 1
        if rec.is_early_exit:
            early_exits += 1
        if rec.sign_in_time:
            check_in_times.append(rec.sign_in_time)
        if rec.sign_out_time:
            check_out_times.append(rec.sign_out_time)

    avg_check_in = None
    avg_check_out = None
    if check_in_times:
        total_seconds = sum(t.hour * 3600 + t.minute * 60 + t.second for t in check_in_times)
        avg_seconds = total_seconds // len(check_in_times)
        avg_check_in = time(avg_seconds // 3600, (avg_seconds % 3600) // 60).isoformat()
    if check_out_times:
        total_seconds = sum(t.hour * 3600 + t.minute * 60 + t.second for t in check_out_times)
        avg_seconds = total_seconds // len(check_out_times)
        avg_check_out = time(avg_seconds // 3600, (avg_seconds % 3600) // 60).isoformat()

    total_working_days = len(records)
    attendance_percentage = (total_present / total_working_days * 100) if total_working_days > 0 else 0

    return {
        "employee_id": employee_id,
        "employee_code": employee.employee_code,
        "employee_name": employee.full_name,
        "year": year,
        "month": month,
        "total_present_days": total_present,
        "total_absent_days": total_absent,
        "total_work_hours": round(total_work_hours, 2),
        "total_break_hours": round(total_break_hours, 2),
        "average_check_in_time": avg_check_in,
        "average_check_out_time": avg_check_out,
        "late_arrivals": late_arrivals,
        "early_exits": early_exits,
        "attendance_percentage": round(attendance_percentage, 2),
    }


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
