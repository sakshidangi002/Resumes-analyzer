"""Create in-app notifications for users.

Each helper also accepts ``with_push=True`` to *additionally* fire an OS-level
Web Push (desktop notification) via :mod:`app.services.web_push_service` to
every subscribed device the recipient has. Web Push failures never abort the
in-app row creation — the inbox row is the source of truth, the desktop popup
is best-effort.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.models import AppNotification, User
from app.models.user import Role, user_roles
from app.ws.manager import connection_manager

logger = logging.getLogger(__name__)


def _serialize_notification(n: AppNotification) -> dict:
    """Match the AppNotificationResponse schema used by /api/activity/notifications."""
    return {
        "id": n.id,
        "user_id": n.user_id,
        "title": n.title,
        "body": n.body,
        "kind": n.kind,
        "link_path": n.link_path,
        "read_at": n.read_at.isoformat() + "Z" if n.read_at else None,
        "created_at": (
            n.created_at.isoformat() + "Z" if n.created_at else None
        ),
    }


def _publish_live(user_id: int, n: AppNotification) -> None:
    """Best-effort WebSocket fan-out. Swallows every exception."""
    try:
        connection_manager.publish_sync(
            user_id,
            {"type": "notification", "data": _serialize_notification(n)},
        )
    except Exception:
        logger.exception(
            "WS publish_sync failed for user_id=%s notification_id=%s",
            user_id,
            n.id,
        )


def _send_push_best_effort(
    db: Session,
    user_id: int,
    title: str,
    body: str | None,
    link_path: str | None,
    push_tag: str | None,
) -> None:
    """Fire-and-forget Web Push. Swallows every exception."""
    try:
        from app.services.web_push_service import send_push_to_user

        send_push_to_user(
            db,
            user_id=user_id,
            title=title,
            body=(body or "").strip() or title,
            url=link_path or "/",
            tag=push_tag,
        )
    except Exception:
        logger.exception(
            "Web Push send failed for user_id=%s (in-app row already saved)",
            user_id,
        )


def create_notification(
    db: Session,
    user_id: int,
    title: str,
    body: str | None = None,
    kind: str = "GENERAL",
    link_path: str | None = None,
    *,
    with_push: bool = False,
    push_tag: str | None = None,
) -> AppNotification:
    n = AppNotification(
        user_id=user_id,
        title=title,
        body=body,
        kind=kind,
        link_path=link_path,
    )
    db.add(n)
    db.commit()
    db.refresh(n)

    _publish_live(user_id, n)

    if with_push:
        _send_push_best_effort(db, user_id, title, body, link_path, push_tag or kind)
    return n


def notify_user_for_employee(
    db: Session,
    employee_id: int,
    title: str,
    body: str | None = None,
    kind: str = "GENERAL",
    link_path: str | None = None,
    *,
    with_push: bool = False,
    push_tag: str | None = None,
) -> None:
    u = db.query(User).filter(User.employee_id == employee_id).first()
    if not u:
        return
    create_notification(
        db,
        u.id,
        title,
        body,
        kind,
        link_path,
        with_push=with_push,
        push_tag=push_tag,
    )


def notify_users_with_roles(
    db: Session,
    role_names: list[str],
    title: str,
    body: str | None = None,
    kind: str = "GENERAL",
    link_path: str | None = None,
    *,
    with_push: bool = False,
    push_tag: str | None = None,
    exclude_user_id: int | None = None,
) -> None:
    """Notify every active user holding any of ``role_names``.

    ``exclude_user_id`` is handy when the action's *requester* themselves
    holds the target role (e.g. an HR applying leave shouldn't see their own
    "new leave request" popup).
    """
    q = (
        db.query(User)
        .join(user_roles, User.id == user_roles.c.user_id)
        .join(Role, Role.id == user_roles.c.role_id)
        .filter(Role.name.in_(role_names))
        .distinct()
    )
    if exclude_user_id is not None:
        q = q.filter(User.id != exclude_user_id)
    users = q.all()
    if not users:
        return
    rows: list[AppNotification] = []
    for u in users:
        row = AppNotification(
            user_id=u.id,
            title=title,
            body=body,
            kind=kind,
            link_path=link_path,
        )
        db.add(row)
        rows.append(row)
    db.commit()
    for row in rows:
        try:
            db.refresh(row)
        except Exception:
            continue
        _publish_live(row.user_id, row)

    if with_push:
        for u in users:
            _send_push_best_effort(db, u.id, title, body, link_path, push_tag or kind)
