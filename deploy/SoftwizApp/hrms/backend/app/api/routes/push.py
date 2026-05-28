"""Web Push (VAPID) subscription management.

Flow:
  1) Frontend calls GET /api/push/vapid-public-key to retrieve the public key.
  2) Frontend asks the browser for `Notification.permission`, registers
     `/sw.js`, and calls `pushManager.subscribe({ applicationServerKey })`.
  3) Frontend POSTs the resulting subscription (endpoint + keys) to /subscribe.
  4) When the user signs out / unregisters, frontend POSTs to /unsubscribe.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.core.config import get_settings
from app.db.session import get_db
from app.models import PushSubscription, User

router = APIRouter()


class PushKeys(BaseModel):
    p256dh: str = Field(min_length=1, max_length=255)
    auth: str = Field(min_length=1, max_length=255)


class PushSubscribeRequest(BaseModel):
    endpoint: str = Field(min_length=1)
    keys: PushKeys


class PushUnsubscribeRequest(BaseModel):
    endpoint: str = Field(min_length=1)


@router.get("/vapid-public-key")
def vapid_public_key():
    """Public key the browser needs to subscribe. Empty string if Web Push
    is not configured yet — the frontend treats that as "skip push setup"."""
    return {"key": get_settings().vapid_public_key or ""}


@router.post("/subscribe")
def subscribe(
    payload: PushSubscribeRequest,
    request: Request,
    db: Session = Depends(get_db),
    user_agent: str | None = Header(default=None, alias="User-Agent"),
    current_user: User = Depends(
        require_roles(["Admin", "HR", "Manager", "Employee"])
    ),
):
    ua = (user_agent or "")[:500]

    existing = (
        db.query(PushSubscription)
        .filter(
            PushSubscription.user_id == current_user.id,
            PushSubscription.endpoint == payload.endpoint,
        )
        .first()
    )
    if existing:
        existing.p256dh = payload.keys.p256dh
        existing.auth = payload.keys.auth
        existing.user_agent = ua or existing.user_agent
        existing.last_used_at = datetime.utcnow()
        db.commit()
        return {"ok": True, "subscription_id": existing.id, "updated": True}

    sub = PushSubscription(
        user_id=current_user.id,
        endpoint=payload.endpoint,
        p256dh=payload.keys.p256dh,
        auth=payload.keys.auth,
        user_agent=ua or None,
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return {"ok": True, "subscription_id": sub.id, "updated": False}


@router.post("/unsubscribe")
def unsubscribe(
    payload: PushUnsubscribeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        require_roles(["Admin", "HR", "Manager", "Employee"])
    ),
):
    deleted = (
        db.query(PushSubscription)
        .filter(
            PushSubscription.user_id == current_user.id,
            PushSubscription.endpoint == payload.endpoint,
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"ok": True, "deleted": int(deleted)}
