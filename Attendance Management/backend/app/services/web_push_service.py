"""Send Web Push (VAPID) notifications.

The push payload is small JSON the service worker (``public/sw.js``) parses
and turns into an OS-level notification, so this code DOES NOT need to know
anything about how it renders.

If VAPID keys are not configured, all sends are silently skipped. Subscriptions
that the push service rejects with HTTP 404/410 are deleted (the user
unsubscribed or wiped the browser); other errors are logged and swallowed.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Literal

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import PushSubscription

logger = logging.getLogger(__name__)


_DEAD_STATUSES = {404, 410}


def is_configured() -> bool:
    s = get_settings()
    return bool(s.vapid_private_key and s.vapid_public_key)


def _push_one(sub: PushSubscription, payload: dict) -> tuple[bool, int | None]:
    """Push to a single subscription. Returns (success, http_status_or_None)."""
    try:
        from pywebpush import webpush, WebPushException  # type: ignore
    except Exception:
        logger.warning(
            "pywebpush not installed — Web Push is disabled. "
            "Run `pip install pywebpush` to enable."
        )
        return False, None

    settings = get_settings()
    try:
        webpush(
            subscription_info={
                "endpoint": sub.endpoint,
                "keys": {"p256dh": sub.p256dh, "auth": sub.auth},
            },
            data=json.dumps(payload),
            vapid_private_key=settings.vapid_private_key,
            vapid_claims={"sub": settings.vapid_claim_email or "mailto:noreply@example.com"},
            ttl=60 * 60 * 6,  # 6h: if the user's browser is offline for longer, drop it
        )
        return True, 200
    except WebPushException as e:  # type: ignore[name-defined]
        status = getattr(getattr(e, "response", None), "status_code", None)
        return False, status
    except Exception:
        logger.exception("Unexpected error pushing to subscription id=%s", sub.id)
        return False, None


def send_push_to_user(
    db: Session,
    user_id: int,
    title: str,
    body: str,
    url: str = "/dsr",
    tag: str | None = None,
) -> Literal["sent", "none", "failed"]:
    """Push to every subscription this user has registered.

    Returns ``"sent"`` if at least one device received it, ``"none"`` if the
    user has no subscriptions, ``"failed"`` if every push attempt failed.
    """
    if not is_configured():
        return "none"

    subs: list[PushSubscription] = (
        db.query(PushSubscription).filter(PushSubscription.user_id == user_id).all()
    )
    if not subs:
        return "none"

    payload = {
        "title": title,
        "body": body,
        "url": url,
        "tag": tag or "softwiz-notification",
    }

    any_success = False
    dead: list[PushSubscription] = []
    for s in subs:
        ok, status = _push_one(s, payload)
        if ok:
            any_success = True
            s.last_used_at = datetime.utcnow()
        elif status in _DEAD_STATUSES:
            dead.append(s)
        else:
            logger.warning(
                "Web Push send failed user_id=%s sub_id=%s http=%s", user_id, s.id, status
            )

    if dead:
        for s in dead:
            db.delete(s)
        db.commit()
        logger.info(
            "Web Push: cleaned up %d dead subscription(s) for user_id=%s",
            len(dead),
            user_id,
        )

    return "sent" if any_success else "failed"


def send_dsr_reminder_push(
    db: Session, user_id: int, today: date
) -> Literal["sent", "none", "failed"]:
    """Convenience wrapper for the 5 PM IST DSR reminder job."""
    return send_push_to_user(
        db,
        user_id=user_id,
        title="Submit your DSR before leaving",
        body=(
            f"Reminder: please submit your Daily Status Report for "
            f"{today.strftime('%d %b %Y')} before signing off."
        ),
        url="/dsr",
        tag=f"dsr-reminder-{today.isoformat()}",
    )
