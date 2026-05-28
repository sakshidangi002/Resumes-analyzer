"""Calendar: holidays, events, birthdays, anniversaries, reminders."""
from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models import User, Holiday, Event, Employee
from app.api.deps import get_current_user, require_roles
from app.services.notification_service import notify_user_for_employee

router = APIRouter()

class HolidayPayload(BaseModel):
    date: date
    name: str
    is_optional: bool = False
    financial_year_id: int | None = None


@router.get("/holidays")
def list_holidays(
    from_date: date = Query(None),
    to_date: date = Query(None),
    financial_year_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(Holiday)
    if from_date:
        q = q.filter(Holiday.date >= from_date)
    if to_date:
        q = q.filter(Holiday.date <= to_date)
    if financial_year_id:
        q = q.filter(Holiday.financial_year_id == financial_year_id)
    holidays = q.order_by(Holiday.date).all()
    return [{"id": h.id, "date": h.date.isoformat(), "name": h.name, "is_optional": h.is_optional} for h in holidays]


@router.post("/holidays")
def create_holiday(
    data: HolidayPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    existing = db.query(Holiday).filter(Holiday.date == data.date, Holiday.name == data.name).first()
    if existing:
        return {"id": existing.id, "date": existing.date.isoformat(), "name": existing.name, "is_optional": existing.is_optional}
    h = Holiday(
        date=data.date,
        name=data.name,
        is_optional=data.is_optional,
        financial_year_id=data.financial_year_id,
    )
    db.add(h)
    db.commit()
    db.refresh(h)
    return {"id": h.id, "date": h.date.isoformat(), "name": h.name, "is_optional": h.is_optional}


@router.delete("/holidays/{holiday_id}")
def delete_holiday(
    holiday_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    h = db.query(Holiday).filter(Holiday.id == holiday_id).first()
    if not h:
        raise HTTPException(status_code=404, detail="Holiday not found")
    db.delete(h)
    db.commit()
    return {"message": "Deleted"}


@router.get("/events")
def list_events(
    from_date: date = Query(None),
    to_date: date = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from sqlalchemy import func
    q = (
        db.query(Event, func.concat(Employee.first_name, " ", Employee.last_name).label("full_name"))
        .outerjoin(Employee, Employee.id == Event.employee_id)
    )
    if from_date:
        q = q.filter(Event.date >= from_date)
    if to_date:
        q = q.filter(Event.date <= to_date)
    
    events = q.order_by(Event.date).all()
    
    out = []
    for ev, emp_name in events:
        out.append({
            "id": ev.id,
            "title": ev.title,
            "date": ev.date.isoformat(),
            "event_type": ev.event_type,
            "description": ev.description,
            "employee_id": ev.employee_id,
            "employee_name": emp_name,
        })
    return out


class EventPayload(BaseModel):
    title: str
    date: date
    event_type: str
    description: str | None = None
    employee_id: int | None = None


@router.post("/events")
def create_event(
    data: EventPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["HR"])),
):
    ev = Event(
        title=data.title,
        date=data.date,
        event_type=data.event_type,
        description=data.description,
        employee_id=data.employee_id,
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    emp_name = None
    if ev.employee_id:
        emp = db.query(Employee).filter(Employee.id == ev.employee_id).first()
        emp_name = emp.full_name if emp else None
        notify_user_for_employee(
            db,
            ev.employee_id,
            f"New event: {ev.title}",
            (
                f"{ev.event_type} on {ev.date.strftime('%d %b %Y')}"
                + (f" — {ev.description.strip()[:200]}" if ev.description else "")
            ),
            kind="EVENT",
            link_path="/calendar",
            with_push=True,
            push_tag=f"event-{ev.id}",
        )
    return {
        "id": ev.id,
        "title": ev.title,
        "date": ev.date.isoformat(),
        "event_type": ev.event_type,
        "description": ev.description,
        "employee_id": ev.employee_id,
        "employee_name": emp_name,
    }


@router.patch("/events/{event_id}")
def update_event(
    event_id: int,
    data: EventPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["HR"])),
):
    ev = db.query(Event).filter(Event.id == event_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    ev.title = data.title
    ev.date = data.date
    ev.event_type = data.event_type
    ev.description = data.description
    ev.employee_id = data.employee_id
    db.commit()
    db.refresh(ev)
    emp_name = None
    if ev.employee_id:
        emp = db.query(Employee).filter(Employee.id == ev.employee_id).first()
        emp_name = emp.full_name if emp else None
    return {
        "id": ev.id,
        "title": ev.title,
        "date": ev.date.isoformat(),
        "event_type": ev.event_type,
        "description": ev.description,
        "employee_id": ev.employee_id,
        "employee_name": emp_name,
    }


@router.delete("/events/{event_id}")
def delete_event(
    event_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["HR"])),
):
    ev = db.query(Event).filter(Event.id == event_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    db.delete(ev)
    db.commit()
    return {"message": "Deleted"}


@router.get("/birthdays")
def list_birthdays(
    month: int = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    employees = db.query(Employee).filter(
        Employee.employment_status == "Active",
        Employee.date_of_birth.isnot(None),
    ).all()
    result = []
    for e in employees:
        if e.date_of_birth and e.date_of_birth.month == month:
            result.append({"employee_id": e.id, "name": e.full_name, "date": e.date_of_birth.isoformat()})
    return result


@router.get("/anniversaries")
def list_anniversaries(
    month: int = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    employees = db.query(Employee).filter(Employee.employment_status == "Active").all()
    result = []
    for e in employees:
        if e.date_of_joining.month == month:
            result.append({"employee_id": e.id, "name": e.full_name, "date_of_joining": e.date_of_joining.isoformat()})
    return result


@router.get("/marriage-anniversaries")
def list_marriage_anniversaries(
    month: int = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    employees = (
        db.query(Employee)
        .filter(
            Employee.employment_status == "Active",
            Employee.date_of_marriage.isnot(None),
        )
        .all()
    )
    result = []
    for e in employees:
        if e.date_of_marriage and e.date_of_marriage.month == month:
            result.append({
                "employee_id": e.id,
                "name": e.full_name,
                "date_of_marriage": e.date_of_marriage.isoformat(),
            })
    return result


@router.get("/reminders")
def list_reminders(
    for_date: date | None = Query(None, description="Start date for reminders"),
    days: int = Query(1, ge=1, le=14, description="Number of days to fetch reminders for"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Reminders for HR/Admin: birthdays, work anniversaries, and custom events.
    Batch-optimized to fetch all data for the range in a few queries instead of a loop.
    """
    from sqlalchemy import extract, func
    
    if for_date is None:
        for_date = date.today() + timedelta(days=1)
    
    end_date = for_date + timedelta(days=days - 1)
    
    # 1. Fetch all active employees to filter birthdays/anniversaries in memory (efficient for range checks)
    employees = db.query(Employee).filter(Employee.employment_status == "Active").all()
    
    # 2. Fetch all events in the date range with a join for employee names
    events_q = (
        db.query(Event, func.concat(Employee.first_name, " ", Employee.last_name).label("full_name"))
        .outerjoin(Employee, Employee.id == Event.employee_id)
        .filter(Event.date >= for_date, Event.date <= end_date)
        .order_by(Event.date, Event.title)
        .all()
    )
    
    # Group events by date
    events_by_date = {}
    for ev, emp_name in events_q:
        d_str = ev.date.isoformat()
        if d_str not in events_by_date:
            events_by_date[d_str] = []
        events_by_date[d_str].append({
            "id": ev.id,
            "title": ev.title,
            "date": d_str,
            "event_type": ev.event_type,
            "description": ev.description,
            "employee_id": ev.employee_id,
            "employee_name": emp_name,
        })

    results = []
    curr = for_date
    while curr <= end_date:
        curr_iso = curr.isoformat()
        
        # Filter birthdays for this specific day in the range
        birthdays = []
        for e in employees:
            if e.date_of_birth and e.date_of_birth.month == curr.month and e.date_of_birth.day == curr.day:
                birthdays.append({
                    "employee_id": e.id,
                    "employee_code": e.employee_code,
                    "name": e.full_name,
                    "date": curr_iso,
                })
        
        # Filter anniversaries for this specific day in the range
        work_anniversaries = []
        for e in employees:
            if e.date_of_joining and e.date_of_joining.month == curr.month and e.date_of_joining.day == curr.day:
                # Only count if joining year is before current year
                if e.date_of_joining.year < curr.year:
                    work_anniversaries.append({
                        "employee_id": e.id,
                        "employee_code": e.employee_code,
                        "name": e.full_name,
                        "date_of_joining": e.date_of_joining.isoformat(),
                        "date": curr_iso,
                        "years": curr.year - e.date_of_joining.year,
                    })

        # Filter marriage anniversaries for this specific day in the range
        marriage_anniversaries = []
        for e in employees:
            dom = getattr(e, "date_of_marriage", None)
            if dom and dom.month == curr.month and dom.day == curr.day:
                # Only count if marriage year is before current year
                if dom.year < curr.year:
                    marriage_anniversaries.append({
                        "employee_id": e.id,
                        "employee_code": e.employee_code,
                        "name": e.full_name,
                        "date_of_marriage": dom.isoformat(),
                        "date": curr_iso,
                        "years": curr.year - dom.year,
                    })

        day_events = events_by_date.get(curr_iso, [])

        results.append({
            "for_date": curr_iso,
            "birthdays": birthdays,
            "work_anniversaries": work_anniversaries,
            "marriage_anniversaries": marriage_anniversaries,
            "events": day_events,
            "total": (
                len(birthdays)
                + len(work_anniversaries)
                + len(marriage_anniversaries)
                + len(day_events)
            ),
        })
        curr += timedelta(days=1)
        
    return results if days > 1 else results[0]
