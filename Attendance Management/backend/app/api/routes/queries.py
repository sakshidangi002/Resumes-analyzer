"""Employee ↔ HR query system (Feature 2).

Employees raise queries; HR replies and moves them OPEN → PENDING → RESOLVED.
Deliberately simple (no realtime): plain request/reply with threaded history.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from app.db.session import get_db
from app.models import User, HRQuery, HRQueryReply
from app.schemas.hr_query import (
    HRQueryCreate,
    HRQueryResponse,
    HRQueryReplyCreate,
    HRQueryReplyResponse,
    HRQueryStatusUpdate,
)
from app.api.deps import require_roles

router = APIRouter()

VALID_STATUSES = {"OPEN", "PENDING", "RESOLVED"}


def _is_hr(current_user: User) -> bool:
    return any(r.name in ("Admin", "HR") for r in current_user.roles)


def _attach_names(q: HRQuery) -> HRQuery:
    """Populate the display-only employee_name field from the ORM relationship."""
    q.employee_name = q.employee.full_name if q.employee else None
    return q


@router.post("", response_model=HRQueryResponse)
def create_query(
    data: HRQueryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    if not current_user.employee_id:
        raise HTTPException(
            status_code=400,
            detail="Your account is not linked to an employee record.",
        )
    if not (data.subject and data.subject.strip()):
        raise HTTPException(status_code=400, detail="Subject is required")
    if not (data.message and data.message.strip()):
        raise HTTPException(status_code=400, detail="Message is required")

    q = HRQuery(
        employee_id=current_user.employee_id,
        subject=data.subject.strip(),
        message=data.message.strip(),
        category=(data.category or None),
        status="OPEN",
    )
    db.add(q)
    db.commit()
    db.refresh(q)
    return _attach_names(q)


@router.get("", response_model=list[HRQueryResponse])
def list_queries(
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    q = db.query(HRQuery).options(
        joinedload(HRQuery.employee), joinedload(HRQuery.replies)
    )
    # Non-HR users only see their own queries.
    if not _is_hr(current_user):
        if not current_user.employee_id:
            return []
        q = q.filter(HRQuery.employee_id == current_user.employee_id)
    if status:
        q = q.filter(HRQuery.status == status)
    rows = (
        q.order_by(HRQuery.updated_at.desc(), HRQuery.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return [_attach_names(r) for r in rows]


def _get_owned_or_hr(db: Session, current_user: User, query_id: int) -> HRQuery:
    q = db.query(HRQuery).filter(HRQuery.id == query_id).first()
    if not q:
        raise HTTPException(status_code=404, detail="Query not found")
    if not _is_hr(current_user) and q.employee_id != current_user.employee_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return q


@router.get("/{query_id}", response_model=HRQueryResponse)
def get_query(
    query_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    return _attach_names(_get_owned_or_hr(db, current_user, query_id))


@router.post("/{query_id}/replies", response_model=HRQueryReplyResponse)
def add_reply(
    query_id: int,
    data: HRQueryReplyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    q = _get_owned_or_hr(db, current_user, query_id)
    if not (data.message and data.message.strip()):
        raise HTTPException(status_code=400, detail="Reply message is required")

    reply = HRQueryReply(
        query_id=q.id,
        user_id=current_user.id,
        author_name=current_user.username,
        author_role="HR" if _is_hr(current_user) else "Employee",
        message=data.message.strip(),
    )
    db.add(reply)
    # Touch the parent so it re-sorts to the top of the list.
    q.updated_at = reply.created_at
    db.commit()
    db.refresh(reply)
    return reply


@router.patch("/{query_id}", response_model=HRQueryResponse)
def update_status(
    query_id: int,
    data: HRQueryStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    q = db.query(HRQuery).filter(HRQuery.id == query_id).first()
    if not q:
        raise HTTPException(status_code=404, detail="Query not found")
    if data.status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status")
    q.status = data.status
    db.commit()
    db.refresh(q)
    return _attach_names(q)
