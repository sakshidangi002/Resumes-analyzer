"""Send notifications via SMTP; log to EmailLog."""
from typing import Optional, Sequence

from sqlalchemy.orm import Session

from app.core.email_smtp import send_email, Attachment
from app.models import EmailLog


def send_notification(
    db: Session,
    to_email: str,
    subject: str,
    body_html: str,
    template_code: str | None = None,
    related_entity_type: str | None = None,
    related_entity_id: str | None = None,
    from_email: str | None = None,
    from_name: str | None = None,
    attachments: Optional[Sequence[Attachment]] = None,
    reply_to: str | None = None,
    reply_to_name: str | None = None,
) -> tuple[bool, str | None]:
    success, err = send_email(
        to_email,
        subject,
        body_html,
        body_plain=None,
        from_email=from_email,
        from_name=from_name,
        attachments=attachments,
        reply_to=reply_to,
        reply_to_name=reply_to_name,
    )
    log = EmailLog(
        to_email=to_email,
        subject=subject,
        body=body_html,
        template_code=template_code,
        status="SENT" if success else "FAILED",
        related_entity_type=related_entity_type,
        related_entity_id=related_entity_id,
    )
    if not success and err:
        log.error_message = err
    if success:
        from app.core.datetime_utils import get_ist_now
        log.sent_at = get_ist_now()
    db.add(log)
    db.commit()
    return success, err
