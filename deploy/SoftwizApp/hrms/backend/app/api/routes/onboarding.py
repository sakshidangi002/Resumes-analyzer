"""HR-managed onboarding checklist per employee."""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models import User, OnboardingTask, AppNotification
from app.api.deps import require_roles
from app.schemas.onboarding import (
    OnboardingTaskCreate,
    OnboardingTaskUpdate,
    OnboardingTaskResponse,
)
from app.services.notification_service import notify_user_for_employee

router = APIRouter()


def _is_admin_or_hr(user: User) -> bool:
    names = {r.name for r in user.roles}
    return "Admin" in names or "HR" in names


@router.get("/me", response_model=list[OnboardingTaskResponse])
def my_onboarding_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    if not current_user.employee_id:
        return []
    return (
        db.query(OnboardingTask)
        .filter(OnboardingTask.employee_id == current_user.employee_id)
        .order_by(OnboardingTask.sort_order, OnboardingTask.created_at)
        .all()
    )


@router.get("/employee/{employee_id}", response_model=list[OnboardingTaskResponse])
def list_employee_tasks(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    return (
        db.query(OnboardingTask)
        .filter(OnboardingTask.employee_id == employee_id)
        .order_by(OnboardingTask.sort_order, OnboardingTask.created_at)
        .all()
    )


@router.get("/all", response_model=list[OnboardingTaskResponse])
def list_all_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    return (
        db.query(OnboardingTask)
        .order_by(OnboardingTask.created_at.desc())
        .all()
    )


@router.post("/tasks", response_model=OnboardingTaskResponse)
def create_task(
    data: OnboardingTaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    t = OnboardingTask(
        employee_id=data.employee_id,
        title=data.title.strip(),
        priority=data.priority,
        due_date=data.due_date,
        sort_order=data.sort_order,
    )
    db.add(t)
    db.commit()
    db.refresh(t)

    notify_user_for_employee(
        db,
        data.employee_id,
        "Task added",
        f"New task: {t.title}",
        kind="TASK_HUB",
        link_path="/onboarding",
    )
    return t


@router.patch("/tasks/{task_id}", response_model=OnboardingTaskResponse)
def update_task(
    task_id: int,
    data: OnboardingTaskUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    task = db.query(OnboardingTask).filter(OnboardingTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    admin_hr = _is_admin_or_hr(current_user)
    if not admin_hr:
        if not current_user.employee_id or task.employee_id != current_user.employee_id:
            raise HTTPException(status_code=403, detail="Access denied")
        if data.title is not None or data.sort_order is not None:
            raise HTTPException(status_code=403, detail="Only HR can edit task details")
        if data.is_completed is None:
            raise HTTPException(status_code=400, detail="Nothing to update")

    if data.title is not None and admin_hr:
        task.title = data.title.strip()
    if data.priority is not None and admin_hr:
        task.priority = data.priority
    if data.due_date is not None and admin_hr:
        task.due_date = data.due_date
    if data.sort_order is not None and admin_hr:
        task.sort_order = data.sort_order

    if data.is_completed is not None:
        from app.core.datetime_utils import get_ist_now
        task.is_completed = data.is_completed
        task.completed_at = get_ist_now() if data.is_completed else None

    db.commit()
    db.refresh(task)
    return task


@router.delete("/tasks/{task_id}")
def delete_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    task = db.query(OnboardingTask).filter(OnboardingTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    db.query(AppNotification).filter(
        AppNotification.kind.in_(["ONBOARDING", "TASK_HUB"]),
        AppNotification.user_id == task.employee_id,
        AppNotification.body.contains(task.title),
    ).delete(synchronize_session=False)

    db.delete(task)
    db.commit()
    return {"message": "deleted"}
