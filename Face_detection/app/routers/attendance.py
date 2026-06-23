from __future__ import annotations

import csv
import io
from datetime import datetime, date

from fastapi import APIRouter, Form, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func

from app.config import get_settings
from app.db import get_db
from app.models.employee import Employee
from app.models.daily import DailyAttendance
from app.config import get_db_timezone

def now_local():
    return datetime.now(get_db_timezone())


def to_local(dt: datetime | None) -> datetime | None:
    """Convert a UTC (or naive-UTC) datetime to the configured local timezone."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # SQLite returns naive datetimes — assume UTC
        from datetime import timezone as _tz
        dt = dt.replace(tzinfo=_tz.utc)
    return dt.astimezone(get_db_timezone())

router = APIRouter(prefix="/api/attendance", tags=["attendance"])

def serialize_daily_attendance(daily: DailyAttendance, employee: Employee) -> dict:
    local_in  = to_local(daily.first_in)
    local_out = to_local(daily.last_out)
    return {
        "id": daily.id,
        "employee_id": employee.id,
        "employee_name": employee.name,
        "attendance_date": daily.date.isoformat(),
        "check_in_time":  local_in.strftime("%H:%M:%S")  if local_in  else None,
        "check_out_time": local_out.strftime("%H:%M:%S") if local_out else None,
        "total_hours": daily.total_hours
    }

def _build_query(db: Session, search: str | None, from_date: str | None, to_date: str | None):
    query = db.query(DailyAttendance, Employee).join(Employee, DailyAttendance.employee_id == Employee.id)
    
    if search and search.strip():
        query = query.filter(Employee.name.ilike(f"%{search.strip()}%"))
    if from_date:
        try:
            d = datetime.strptime(from_date.strip(), "%Y-%m-%d").date()
            query = query.filter(DailyAttendance.date >= d)
        except ValueError:
            pass
    if to_date:
        try:
            d = datetime.strptime(to_date.strip(), "%Y-%m-%d").date()
            query = query.filter(DailyAttendance.date <= d)
        except ValueError:
            pass
            
    return query

@router.get("")
def list_attendance(
    search: str | None = None,
    from_date: str | None = Query(None, alias="from"),
    to_date: str | None = Query(None, alias="to"),
    limit: int | None = None,
    offset: int = 0,
    db: Session = Depends(get_db)
) -> dict:
    page_size = limit or get_settings().attendance_page_size
    query = _build_query(db, search, from_date, to_date)
    
    total = query.count()
    results = query.order_by(DailyAttendance.date.desc(), DailyAttendance.first_in.desc(), DailyAttendance.id.desc()).offset(offset).limit(page_size).all()
    
    return {
        "attendance": [serialize_daily_attendance(daily, emp) for daily, emp in results],
        "total": total,
        "limit": page_size,
        "offset": offset,
    }

@router.get("/export.csv")
def export_attendance_csv(
    search: str | None = None,
    from_date: str | None = Query(None, alias="from"),
    to_date: str | None = Query(None, alias="to"),
    db: Session = Depends(get_db)
) -> StreamingResponse:
    query = _build_query(db, search, from_date, to_date)
    results = query.order_by(DailyAttendance.date.desc(), DailyAttendance.first_in.desc(), DailyAttendance.id.desc()).all()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Employee Name", "Date", "Check-In", "Check-Out", "Total Hours"])
    for daily, emp in results:
        serialized = serialize_daily_attendance(daily, emp)
        writer.writerow(
            [
                serialized["employee_name"],
                serialized["attendance_date"],
                serialized["check_in_time"] or "",
                serialized["check_out_time"] or "",
                serialized["total_hours"] or "0.0"
            ]
        )

    filename = f"attendance_{now_local().strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@router.post("/manual")
def create_manual_attendance(
    employee_id: int = Form(...),
    attendance_date: str = Form(...),
    check_in_time: str = Form(...),
    check_out_time: str | None = Form(None),
    db: Session = Depends(get_db)
) -> dict:
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    try:
        dt_date = datetime.strptime(attendance_date.strip(), "%Y-%m-%d").date()
        
        from datetime import timezone
        
        cin_time = datetime.strptime(check_in_time.strip(), "%H:%M:%S").time()
        first_in = datetime.combine(dt_date, cin_time).replace(tzinfo=timezone.utc)
        
        last_out = None
        if check_out_time:
            cout_time = datetime.strptime(check_out_time.strip(), "%H:%M:%S").time()
            last_out = datetime.combine(dt_date, cout_time).replace(tzinfo=timezone.utc)
            
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Use YYYY-MM-DD for date and HH:MM:SS for times",
        ) from exc

    existing = db.query(DailyAttendance).filter(
        DailyAttendance.employee_id == employee_id,
        DailyAttendance.date == dt_date
    ).first()
    
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="Attendance already exists for this employee on this date",
        )

    daily = DailyAttendance(
        employee_id=employee_id,
        date=dt_date,
        first_in=first_in,
        last_out=last_out,
    )
    if last_out:
        diff = last_out - first_in
        daily.total_hours = round(diff.total_seconds() / 3600.0, 2)
        
    db.add(daily)
    db.commit()
    db.refresh(daily)

    return {
        "message": "Manual attendance created",
        "attendance": serialize_daily_attendance(daily, employee),
    }

@router.post("/{attendance_id}/checkout")
def manual_checkout(attendance_id: int, db: Session = Depends(get_db)) -> dict:
    daily = db.query(DailyAttendance).filter(DailyAttendance.id == attendance_id).first()
    if daily is None:
        raise HTTPException(status_code=404, detail="Attendance record not found")

    employee = db.query(Employee).filter(Employee.id == daily.employee_id).first()
    
    from datetime import timezone as _tz
    daily.last_out = datetime.now(_tz.utc)

    if daily.first_in:
        first_in = daily.first_in
        if first_in.tzinfo is None:
            first_in = first_in.replace(tzinfo=_tz.utc)
        diff = daily.last_out - first_in
        daily.total_hours = round(diff.total_seconds() / 3600.0, 2)

    db.commit()
    db.refresh(daily)

    return {
        "message": "Check-out recorded",
        "attendance": serialize_daily_attendance(daily, employee),
    }

@router.delete("/{attendance_id}")
def delete_attendance(attendance_id: int, db: Session = Depends(get_db)) -> dict:
    daily = db.query(DailyAttendance).filter(DailyAttendance.id == attendance_id).first()
    if daily is None:
        raise HTTPException(status_code=404, detail="Attendance record not found")

    employee = db.query(Employee).filter(Employee.id == daily.employee_id).first()
    employee_name = employee.name if employee else "Unknown"
    date_str = daily.date.isoformat()
    cin_str = daily.first_in.strftime("%H:%M:%S") if daily.first_in else ""

    db.delete(daily)
    db.commit()

    return {
        "message": "Attendance deleted",
        "attendance_id": attendance_id,
        "employee_name": employee_name,
        "attendance_date": date_str,
        "check_in_time": cin_str,
    }

