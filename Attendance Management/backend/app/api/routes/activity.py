"""In-app notifications (inbox)."""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models import User, AppNotification
from app.api.deps import get_current_user, require_roles
from app.schemas.activity import AppNotificationResponse

router = APIRouter()


@router.get("/notifications", response_model=list[AppNotificationResponse])
def list_notifications(
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    q = db.query(AppNotification).filter(AppNotification.user_id == current_user.id)
    if unread_only:
        q = q.filter(AppNotification.read_at.is_(None))
    return q.order_by(AppNotification.created_at.desc()).limit(limit).all()


@router.get("/notifications/unread-count")
def unread_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    n = (
        db.query(AppNotification)
        .filter(AppNotification.user_id == current_user.id, AppNotification.read_at.is_(None))
        .count()
    )
    return {"count": n}


@router.patch("/notifications/{notification_id}/read", response_model=AppNotificationResponse)
def mark_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    n = (
        db.query(AppNotification)
        .filter(AppNotification.id == notification_id, AppNotification.user_id == current_user.id)
        .first()
    )
    from app.core.datetime_utils import get_ist_now
    if not n:
        raise HTTPException(status_code=404, detail="Notification not found")
    if n.read_at is None:
        n.read_at = get_ist_now()
        db.commit()
        db.refresh(n)
    return n


@router.post("/notifications/read-all")
def mark_all_read(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    from app.core.datetime_utils import get_ist_now
    now = get_ist_now()
    rows = (
        db.query(AppNotification)
        .filter(AppNotification.user_id == current_user.id, AppNotification.read_at.is_(None))
        .all()
    )
    for n in rows:
        n.read_at = now
    db.commit()
    return {"message": "ok"}


@router.delete("/notifications/{notification_id}")
def delete_notification(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    n = (
        db.query(AppNotification)
        .filter(AppNotification.id == notification_id, AppNotification.user_id == current_user.id)
        .first()
    )
    if not n:
        raise HTTPException(status_code=404, detail="Notification not found")
    db.delete(n)
    db.commit()
    return {"message": "Notification deleted"}
