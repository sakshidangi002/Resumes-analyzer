"""Daily Status Report (DSR) endpoints.

Visibility:
- Every authenticated employee can list / create / edit their OWN DSRs.
- Admin & HR can view + edit any employee's DSR via `/all` and `/{id}`.
- A SUBMITTED DSR cannot be edited or deleted by its owner (only Admin/HR).
"""
from calendar import monthrange
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import extract
from sqlalchemy.orm import Session, joinedload

from app.api.deps import require_roles
from app.db.session import get_db
from app.models import DailyStatusReport, Employee, User
from app.schemas.dsr import DSRCreate, DSRResponse, DSRSummary, DSRUpdate
from app.services.notification_service import (
    create_notification,
    notify_users_with_roles,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# DSR-submitted fan-out (reporting manager + Admin/HR)
# ---------------------------------------------------------------------------

def _notify_dsr_submitted(
    db: Session,
    *,
    dsr: DailyStatusReport,
    submitter: User,
) -> None:
    """When an employee submits a DSR, ping their reporting manager and the
    Admin/HR group so they can review without polling the list.

    Best-effort: never raises. The DSR row itself is already saved.
    """
    try:
        emp_name = "An employee"
        if dsr.employee is not None:
            emp_name = (
                getattr(dsr.employee, "full_name", None)
                or getattr(dsr.employee, "first_name", None)
                or emp_name
            )
        date_label = dsr.report_date.strftime("%d %b %Y")
        title = f"{emp_name} submitted DSR for {date_label}"
        body = (dsr.work_done or "").strip()[:200] or None
        link_path = f"/dsr?dsr_id={dsr.id}"

        # 1) Reporting manager (if any). employees.reporting_manager_id points
        #    to another employee row; we then map employee -> user.
        if dsr.employee is not None and dsr.employee.reporting_manager_id:
            mgr_user = (
                db.query(User)
                .filter(User.employee_id == dsr.employee.reporting_manager_id)
                .first()
            )
            if mgr_user and mgr_user.id != submitter.id:
                create_notification(
                    db,
                    user_id=mgr_user.id,
                    title=title,
                    body=body,
                    kind="DSR_SUBMITTED",
                    link_path=link_path,
                )

        # 2) Admin/HR group (so HR gets the same instant feed, regardless of
        #    who manages whom).
        notify_users_with_roles(
            db,
            role_names=["Admin", "HR"],
            title=title,
            body=body,
            kind="DSR_SUBMITTED",
            link_path=link_path,
            exclude_user_id=submitter.id,
        )
    except Exception:
        # Notification is a side-effect — never let it 500 the actual submit.
        import logging

        logging.getLogger(__name__).exception(
            "DSR submit notification fan-out failed for dsr_id=%s", dsr.id
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_admin_or_hr(user: User) -> bool:
    names = {r.name for r in user.roles}
    return "Admin" in names or "HR" in names


def _serialize(dsr: DailyStatusReport) -> DSRResponse:
    emp_name = None
    emp_code = None
    designation = None
    if dsr.employee is not None:
        emp_name = getattr(dsr.employee, "full_name", None) or getattr(
            dsr.employee, "first_name", None
        )
        emp_code = getattr(dsr.employee, "employee_code", None)
        emp_designation = getattr(dsr.employee, "designation", None)
        designation = getattr(emp_designation, "title", None) if emp_designation else None
    return DSRResponse(
        id=dsr.id,
        employee_id=dsr.employee_id,
        employee_name=emp_name,
        employee_code=emp_code,
        designation=designation,
        report_date=dsr.report_date,
        project_work=dsr.project_work,
        work_location=dsr.work_location,
        total_hours=dsr.total_hours,
        work_done=dsr.work_done,
        plan_for_tomorrow=dsr.plan_for_tomorrow,
        status=dsr.status,  # type: ignore[arg-type]
        submitted_at=dsr.submitted_at,
        created_at=dsr.created_at,
        updated_at=dsr.updated_at,
    )


def _require_employee_id(user: User) -> int:
    if not user.employee_id:
        raise HTTPException(
            status_code=400,
            detail="Your account is not linked to an employee profile.",
        )
    return user.employee_id


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------

@router.get("/me", response_model=list[DSRResponse])
def my_dsrs(
    year: int | None = Query(None, ge=2000, le=2100),
    month: int | None = Query(None, ge=1, le=12),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """Logged-in user's own DSRs, newest first. Optional month/year filter."""
    employee_id = _require_employee_id(current_user)
    q = (
        db.query(DailyStatusReport)
        .options(joinedload(DailyStatusReport.employee))
        .filter(DailyStatusReport.employee_id == employee_id)
    )
    if year is not None:
        q = q.filter(extract("year", DailyStatusReport.report_date) == year)
    if month is not None:
        q = q.filter(extract("month", DailyStatusReport.report_date) == month)
    rows = q.order_by(DailyStatusReport.report_date.desc()).limit(limit).all()
    return [_serialize(r) for r in rows]


@router.get("/me/summary", response_model=DSRSummary)
def my_dsr_summary(
    year: int | None = Query(None, ge=2000, le=2100),
    month: int | None = Query(None, ge=1, le=12),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """Counts (Total / Submitted / Draft / Pending) for the given month."""
    employee_id = _require_employee_id(current_user)

    today = date.today()
    y = year or today.year
    m = month or today.month

    rows: list[DailyStatusReport] = (
        db.query(DailyStatusReport)
        .filter(
            DailyStatusReport.employee_id == employee_id,
            extract("year", DailyStatusReport.report_date) == y,
            extract("month", DailyStatusReport.report_date) == m,
        )
        .all()
    )

    submitted = sum(1 for r in rows if r.status == "SUBMITTED")
    draft = sum(1 for r in rows if r.status == "DRAFT")

    # "Pending" = working days in the month so far that have no DSR yet.
    # We count Mon-Fri up to today (inclusive) for the current month, or the
    # whole month for past months. This is a deliberately simple heuristic
    # (it ignores company holidays / approved leaves) so callers always see a
    # sensible number; clients can still hide the field if they prefer.
    days_in_month = monthrange(y, m)[1]
    last_day = days_in_month
    if y == today.year and m == today.month:
        last_day = today.day
    working_days = 0
    for d in range(1, last_day + 1):
        try:
            wk = date(y, m, d).weekday()
        except ValueError:
            continue
        if wk < 5:
            working_days += 1
    pending = max(0, working_days - submitted)

    return DSRSummary(
        year=y, month=m, total=len(rows), submitted=submitted, draft=draft, pending=pending
    )


@router.get("/me/today-status")
def my_dsr_today_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """Lightweight 'do I owe a DSR today?' check used by the 5 PM IST reminder
    banner on the frontend.

    Returns the IST date the server considered "today" so the client never has
    to reason about the user's laptop clock.
    """
    ist_today = (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()
    if not current_user.employee_id:
        return {
            "today_ist": ist_today.isoformat(),
            "has_dsr": False,
            "submitted": False,
            "dsr_id": None,
            "needs_dsr": False,
        }

    row = (
        db.query(DailyStatusReport)
        .filter(
            DailyStatusReport.employee_id == current_user.employee_id,
            DailyStatusReport.report_date == ist_today,
        )
        .first()
    )
    submitted = bool(row and row.status == "SUBMITTED")
    return {
        "today_ist": ist_today.isoformat(),
        "has_dsr": row is not None,
        "submitted": submitted,
        "dsr_id": row.id if row else None,
        "needs_dsr": not submitted,
    }


@router.get("/all", response_model=list[DSRResponse])
def list_all_dsrs(
    year: int | None = Query(None, ge=2000, le=2100),
    month: int | None = Query(None, ge=1, le=12),
    employee_id: int | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """List every employee's DSR. Restricted to Admin and HR roles only.

    Employees and Managers can only see their own DSRs via GET /dsr/me.
    """
    q = db.query(DailyStatusReport).options(joinedload(DailyStatusReport.employee))

    if year is not None:
        q = q.filter(extract("year", DailyStatusReport.report_date) == year)
    if month is not None:
        q = q.filter(extract("month", DailyStatusReport.report_date) == month)
    if employee_id is not None:
        q = q.filter(DailyStatusReport.employee_id == employee_id)
    if status:
        q = q.filter(DailyStatusReport.status == status.upper())
    rows = q.order_by(DailyStatusReport.report_date.desc()).limit(limit).all()
    return [_serialize(r) for r in rows]


@router.get("/{dsr_id}", response_model=DSRResponse)
def get_dsr(
    dsr_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    row = (
        db.query(DailyStatusReport)
        .options(joinedload(DailyStatusReport.employee))
        .filter(DailyStatusReport.id == dsr_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="DSR not found")

    if _is_admin_or_hr(current_user):
        return _serialize(row)

    # Owner can always see own DSR. Manager + Employee can only see their own.
    if row.employee_id == current_user.employee_id:
        return _serialize(row)

    raise HTTPException(status_code=403, detail="Access denied")


# ---------------------------------------------------------------------------
# Write endpoints
# ---------------------------------------------------------------------------

@router.post("/me", response_model=DSRResponse, status_code=201)
def create_my_dsr(
    data: DSRCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    employee_id = _require_employee_id(current_user)

    existing = (
        db.query(DailyStatusReport)
        .filter(
            DailyStatusReport.employee_id == employee_id,
            DailyStatusReport.report_date == data.report_date,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"A DSR for {data.report_date.isoformat()} already exists.",
        )

    row = DailyStatusReport(
        employee_id=employee_id,
        report_date=data.report_date,
        project_work=data.project_work,
        work_location=data.work_location or "Office",
        total_hours=data.total_hours,
        work_done=data.work_done,
        plan_for_tomorrow=data.plan_for_tomorrow,
        status=data.status,
        submitted_at=datetime.utcnow() if data.status == "SUBMITTED" else None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    # Re-load with employee eager-loaded so the response includes employee_name.
    row = (
        db.query(DailyStatusReport)
        .options(joinedload(DailyStatusReport.employee))
        .filter(DailyStatusReport.id == row.id)
        .first()
    )
    if row is not None and row.status == "SUBMITTED":
        _notify_dsr_submitted(db, dsr=row, submitter=current_user)
    return _serialize(row)  # type: ignore[arg-type]


@router.patch("/{dsr_id}", response_model=DSRResponse)
def update_dsr(
    dsr_id: int,
    data: DSRUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    row = (
        db.query(DailyStatusReport)
        .options(joinedload(DailyStatusReport.employee))
        .filter(DailyStatusReport.id == dsr_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="DSR not found")

    admin_hr = _is_admin_or_hr(current_user)
    if not admin_hr:
        if row.employee_id != current_user.employee_id:
            raise HTTPException(status_code=403, detail="Access denied")
        if row.status == "SUBMITTED":
            raise HTTPException(
                status_code=403,
                detail="Submitted DSRs are read-only. Ask HR if you need to edit it.",
            )

    if data.project_work is not None:
        row.project_work = data.project_work.strip() or None
    if data.work_location is not None:
        row.work_location = data.work_location.strip() or "Office"
    if data.total_hours is not None:
        row.total_hours = data.total_hours
    if data.work_done is not None:
        row.work_done = data.work_done.strip()
    if data.plan_for_tomorrow is not None:
        v = data.plan_for_tomorrow.strip()
        row.plan_for_tomorrow = v or None
    just_submitted = False
    if data.status is not None and data.status != row.status:
        if data.status == "SUBMITTED" and row.status != "SUBMITTED":
            just_submitted = True
        row.status = data.status
        row.submitted_at = datetime.utcnow() if data.status == "SUBMITTED" else None

    row.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(row)
    if just_submitted:
        _notify_dsr_submitted(db, dsr=row, submitter=current_user)
    return _serialize(row)


@router.delete("/{dsr_id}")
def delete_dsr(
    dsr_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    row = db.query(DailyStatusReport).filter(DailyStatusReport.id == dsr_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="DSR not found")
    admin_hr = _is_admin_or_hr(current_user)
    if not admin_hr:
        if row.employee_id != current_user.employee_id:
            raise HTTPException(status_code=403, detail="Access denied")
        if row.status == "SUBMITTED":
            raise HTTPException(
                status_code=403,
                detail="Submitted DSRs cannot be deleted. Contact HR if needed.",
            )
    db.delete(row)
    db.commit()
    return {"message": "deleted"}
