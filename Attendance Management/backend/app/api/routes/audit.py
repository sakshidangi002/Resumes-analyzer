"""Admin/HR audit-log access."""
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.db.session import get_db
from app.models import AuditLog, User

router = APIRouter()


@router.get("", tags=["audit"])
def list_audit_logs(
    limit: int = Query(200, ge=1, le=1000),
    action: str | None = Query(None),
    user_id: int | None = Query(None),
    from_date: datetime | None = Query(None),
    to_date: datetime | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    query = (
        db.query(AuditLog, User.username)
        .outerjoin(User, User.id == AuditLog.user_id)
        .order_by(AuditLog.created_at.desc())
    )
    if action:
        query = query.filter(AuditLog.action.ilike(f"%{action.strip()}%"))
    if user_id is not None:
        query = query.filter(AuditLog.user_id == user_id)
    if from_date:
        query = query.filter(AuditLog.created_at >= from_date)
    if to_date:
        query = query.filter(AuditLog.created_at <= to_date)

    return [
        {
            "id": entry.id,
            "user_id": entry.user_id,
            "username": username,
            "action": entry.action,
            "entity_type": entry.entity_type,
            "entity_id": entry.entity_id,
            "details": entry.details,
            "ip_address": entry.ip_address,
            "created_at": entry.created_at,
        }
        for entry, username in query.limit(limit).all()
    ]
