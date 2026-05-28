"""DSR reminder schedule API (separate router).

Mounted at ``/api/dsr-reminder`` so paths never collide with
``/api/dsr/{dsr_id}`` on servers that were started before
``/reminder-settings`` existed under the DSR router.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session, joinedload

from app.api.deps import require_roles
from app.db.session import get_db
from app.models import (
    AppNotification,
    CompanyConfig,
    DailyStatusReport,
    Employee,
    User,
)
from app.services.notification_service import create_notification
from app.services.reminder_settings import (
    DEFAULT_REMINDER_TIME,
    DEFAULT_REMINDER_WEEKDAYS,
    get_or_create_config,
    normalize_time,
    normalize_weekdays,
)

router = APIRouter()


_IST_OFFSET = timedelta(hours=5, minutes=30)


def _ist_today() -> date:
    return (datetime.utcnow() + _IST_OFFSET).date()


def _ist_day_utc_window(today: date) -> tuple[datetime, datetime]:
    """Map an IST calendar day to its naive-UTC [start, end) window so we can
    compare against ``AppNotification.created_at`` (which is naive UTC).
    """
    ist_midnight_naive = datetime.combine(today, datetime.min.time())
    start_utc = ist_midnight_naive - _IST_OFFSET
    return start_utc, start_utc + timedelta(days=1)


class ReminderSettingsResponse(BaseModel):
    enabled: bool
    time: str
    weekdays: list[str]
    current_ist: str


class ReminderSettingsUpdate(BaseModel):
    enabled: bool | None = None
    time: str | None = Field(default=None, description='IST clock time as "HH:MM"')
    weekdays: list[str] | None = None

    @field_validator("time")
    @classmethod
    def _check_time(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            return normalize_time(v)
        except ValueError as e:
            raise ValueError(str(e)) from e

    @field_validator("weekdays")
    @classmethod
    def _check_weekdays(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        try:
            normalized_csv = normalize_weekdays(",".join(v))
        except ValueError as e:
            raise ValueError(str(e)) from e
        return [d for d in normalized_csv.split(",") if d]


def _settings_response(cfg: CompanyConfig) -> ReminderSettingsResponse:
    ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    return ReminderSettingsResponse(
        enabled=bool(cfg.dsr_reminder_enabled),
        time=(cfg.dsr_reminder_time or DEFAULT_REMINDER_TIME),
        weekdays=[
            d
            for d in (cfg.dsr_reminder_weekdays or DEFAULT_REMINDER_WEEKDAYS).split(",")
            if d
        ],
        current_ist=ist_now.strftime("%Y-%m-%d %H:%M"),
    )


@router.get("/settings", response_model=ReminderSettingsResponse)
def get_reminder_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    cfg = get_or_create_config(db)
    return _settings_response(cfg)


@router.patch("/settings", response_model=ReminderSettingsResponse)
@router.put("/settings", response_model=ReminderSettingsResponse)
def update_reminder_settings(
    data: ReminderSettingsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    cfg = get_or_create_config(db)

    if data.enabled is not None:
        cfg.dsr_reminder_enabled = bool(data.enabled)
    if data.time is not None:
        cfg.dsr_reminder_time = data.time
    if data.weekdays is not None:
        if not data.weekdays:
            raise HTTPException(
                status_code=400,
                detail="At least one weekday must be enabled.",
            )
        cfg.dsr_reminder_weekdays = ",".join(data.weekdays)

    db.commit()
    db.refresh(cfg)
    return _settings_response(cfg)


class NotifyMeResponse(BaseModel):
    created: bool
    reason: str
    today_ist: str


@router.post("/notify-me", response_model=NotifyMeResponse)
def notify_me(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """Drop today's DSR reminder into the caller's in-app inbox.

    Idempotent per IST calendar day: a user can call this any number of times
    after 5 PM IST (banner mount, refresh, etc.) and we will create the bell
    row at most once. Skipped if today's DSR is already SUBMITTED so we don't
    nag people who finished their day.

    This is the frontend's belt-and-braces companion to the APScheduler 5 PM
    job in ``app.services.dsr_reminder``: if the scheduler missed the user
    (offline server, late login), the bell still lights up the moment they
    open the app after the reminder time.
    """
    today = _ist_today()
    window_start, window_end = _ist_day_utc_window(today)

    if current_user.employee_id:
        submitted = (
            db.query(DailyStatusReport.id)
            .filter(
                DailyStatusReport.employee_id == current_user.employee_id,
                DailyStatusReport.report_date == today,
                DailyStatusReport.status == "SUBMITTED",
            )
            .first()
        )
        if submitted:
            return NotifyMeResponse(
                created=False, reason="already_submitted", today_ist=today.isoformat()
            )

    existing = (
        db.query(AppNotification.id)
        .filter(
            AppNotification.user_id == current_user.id,
            AppNotification.kind == "DSR_REMINDER",
            AppNotification.created_at >= window_start,
            AppNotification.created_at < window_end,
        )
        .first()
    )
    if existing:
        return NotifyMeResponse(
            created=False, reason="already_notified", today_ist=today.isoformat()
        )

    date_label = today.strftime("%d %b %Y")
    title = "Submit your DSR before leaving"
    body = (
        f"Reminder: please submit your Daily Status Report for "
        f"{date_label} before signing off for the day."
    )
    n = AppNotification(
        user_id=current_user.id,
        title=title,
        body=body,
        kind="DSR_REMINDER",
        link_path="/dsr",
    )
    db.add(n)
    db.commit()

    # Best-effort Web Push so the OS-level popup fires the same moment the
    # bell badge lights up. Tagged per IST day so re-pushing replaces the
    # earlier device notification rather than stacking.
    try:
        from app.services.web_push_service import send_push_to_user

        send_push_to_user(
            db,
            user_id=current_user.id,
            title=title,
            body=body,
            url="/dsr",
            tag=f"dsr-reminder-{today.isoformat()}",
        )
    except Exception:
        # Push failure must never break the inbox flow.
        pass

    return NotifyMeResponse(
        created=True, reason="created", today_ist=today.isoformat()
    )


# ---------------------------------------------------------------------------
# Admin / HR: who hasn't submitted today's DSR + manual reminder
# ---------------------------------------------------------------------------

class PendingDsrEmployee(BaseModel):
    user_id: int
    employee_id: int
    employee_name: str
    employee_code: str | None = None
    department: str | None = None
    designation: str | None = None
    official_email: str | None = None
    has_draft: bool


class PendingDsrResponse(BaseModel):
    today_ist: str
    total_active_employees: int
    submitted: int
    pending: list[PendingDsrEmployee]


class ManualRemindRequest(BaseModel):
    """If ``user_ids`` is empty, every pending user receives the reminder."""

    user_ids: list[int] = Field(default_factory=list)


class ManualRemindResponse(BaseModel):
    today_ist: str
    notified: int
    skipped_submitted: int
    skipped_no_target: int


def _pending_today(db: Session, today: date) -> tuple[
    list[tuple[User, Employee, bool]],
    int,
    int,
]:
    """Return ``(pending_rows, total_active, submitted_count)`` for ``today``.

    ``pending_rows`` is a list of ``(user, employee, has_draft)`` tuples for
    every active user who has an employee profile but has NOT marked today's
    DSR as ``SUBMITTED``.
    """
    rows: list[tuple[User, Employee]] = (
        db.query(User, Employee)
        .join(Employee, Employee.id == User.employee_id)
        .options(
            joinedload(Employee.department),
            joinedload(Employee.designation),
        )
        .filter(User.is_active.is_(True), User.employee_id.isnot(None))
        .all()
    )
    total_active = len(rows)

    dsr_map: dict[int, DailyStatusReport] = {
        r.employee_id: r
        for r in db.query(DailyStatusReport)
        .filter(DailyStatusReport.report_date == today)
        .all()
    }
    submitted_count = sum(1 for r in dsr_map.values() if r.status == "SUBMITTED")

    pending: list[tuple[User, Employee, bool]] = []
    for user, emp in rows:
        dsr = dsr_map.get(emp.id)
        if dsr is not None and dsr.status == "SUBMITTED":
            continue
        pending.append((user, emp, bool(dsr and dsr.status == "DRAFT")))
    return pending, total_active, submitted_count


@router.get("/pending-today", response_model=PendingDsrResponse)
def list_pending_dsr_today(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Admin / HR view of employees who haven't submitted today's DSR."""
    today = _ist_today()
    pending, total_active, submitted_count = _pending_today(db, today)

    payload: list[PendingDsrEmployee] = []
    for user, emp, has_draft in pending:
        full_name = (
            getattr(emp, "full_name", None)
            or " ".join(
                p
                for p in (
                    getattr(emp, "first_name", "") or "",
                    getattr(emp, "last_name", "") or "",
                )
                if p
            ).strip()
            or user.username
        )
        dept = None
        if getattr(emp, "department", None) is not None:
            dept = getattr(emp.department, "name", None)
        desg = None
        if getattr(emp, "designation", None) is not None:
            desg = getattr(emp.designation, "title", None)
        # Fall back to department when designation is not set, so the
        # "Designation" column in the Pending Today UI always shows the
        # most relevant role label.
        desg_effective = desg or dept
        payload.append(
            PendingDsrEmployee(
                user_id=user.id,
                employee_id=emp.id,
                employee_name=full_name,
                employee_code=getattr(emp, "employee_code", None),
                department=dept,
                designation=desg_effective,
                official_email=(
                    getattr(emp, "official_email", None)
                    or getattr(emp, "personal_email", None)
                    or getattr(user, "official_email", None)
                ),
                has_draft=has_draft,
            )
        )

    return PendingDsrResponse(
        today_ist=today.isoformat(),
        total_active_employees=total_active,
        submitted=submitted_count,
        pending=payload,
    )


@router.post("/pending-today/remind", response_model=ManualRemindResponse)
def remind_pending_dsr_today(
    data: ManualRemindRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Send a manual DSR reminder right now to selected (or all pending) users.

    Reuses the same 3-channel delivery as the 5 PM job (in-app + email + Web
    Push) so the experience is identical to the scheduled reminder. Idempotent
    per IST day: if the user already received today's reminder we DO NOT
    create a second in-app row; we DO re-send email + push so the human gets
    a fresh nudge.
    """
    today = _ist_today()
    window_start, window_end = _ist_day_utc_window(today)
    pending, _total, _sub = _pending_today(db, today)
    pending_user_ids = {u.id for u, _e, _d in pending}

    if data.user_ids:
        target_user_ids = {uid for uid in data.user_ids if uid in pending_user_ids}
    else:
        target_user_ids = pending_user_ids

    if not target_user_ids:
        return ManualRemindResponse(
            today_ist=today.isoformat(),
            notified=0,
            skipped_submitted=0,
            skipped_no_target=1,
        )

    already_notified_user_ids: set[int] = {
        uid
        for (uid,) in (
            db.query(AppNotification.user_id)
            .filter(
                AppNotification.kind == "DSR_REMINDER",
                AppNotification.user_id.in_(target_user_ids),
                AppNotification.created_at >= window_start,
                AppNotification.created_at < window_end,
            )
            .all()
        )
    }

    date_label = today.strftime("%d %b %Y")
    title = "Submit your DSR before leaving"
    body = (
        f"Reminder from HR: please submit your Daily Status Report for "
        f"{date_label} before signing off."
    )

    targets = (
        db.query(User)
        .filter(User.id.in_(target_user_ids), User.is_active.is_(True))
        .all()
    )

    notified = 0
    for u in targets:
        if u.id in already_notified_user_ids:
            # Don't stack a second in-app row; just re-send email + push.
            pass
        else:
            create_notification(
                db,
                user_id=u.id,
                title=title,
                body=body,
                kind="DSR_REMINDER",
                link_path="/dsr",
                with_push=True,
                push_tag=f"dsr-reminder-{today.isoformat()}",
            )
        # Email (best-effort) — re-using the daily job's exact builder so the
        # template stays consistent.
        try:
            from app.services.dsr_reminder import _safe_send_email

            _safe_send_email(db, u, today)
        except Exception:
            pass
        notified += 1

    return ManualRemindResponse(
        today_ist=today.isoformat(),
        notified=notified,
        skipped_submitted=0,
        skipped_no_target=0,
    )
