"""
Email service using simple SMTP. All notifications and letters are sent via this.
"""
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import Optional, Sequence

from app.core.config import get_settings


# Attachment shape: (filename, content_bytes, mime_subtype)
# Example: ("offer.pdf", b"<bytes>", "pdf")  → produces application/pdf
Attachment = tuple[str, bytes, str]


def send_email(
    to_email: str,
    subject: str,
    body_html: str,
    body_plain: Optional[str] = None,
    from_email: Optional[str] = None,
    from_name: Optional[str] = None,
    attachments: Optional[Sequence[Attachment]] = None,
    reply_to: Optional[str] = None,
    reply_to_name: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Send a single email via SMTP. Uses app SMTP settings.
    Returns (success, error_message). If success=True, error_message=None.

    `attachments` is a sequence of (filename, bytes, mime_subtype) tuples.

    `reply_to` lets a notification keep the authenticated SMTP user as `From:`
    (required by Gmail / most providers) while making the actual employee
    address the one that opens up when the recipient hits "Reply".
    """
    settings = get_settings()
    if not settings.smtp_host or settings.smtp_host.strip().lower() in {"localhost", "127.0.0.1"}:
        return (
            False,
            "SMTP is not configured. Set SMTP_HOST/SMTP_USER/SMTP_PASSWORD (or a valid relay) in backend .env and restart the backend.",
        )

    # When we have file attachments, the top-level container must be 'mixed'
    # with an inner 'alternative' part holding plain + html bodies.
    if attachments:
        msg = MIMEMultipart("mixed")
        alt = MIMEMultipart("alternative")
        if body_plain:
            alt.attach(MIMEText(body_plain, "plain"))
        alt.attach(MIMEText(body_html, "html"))
        msg.attach(alt)
        for filename, content, mime_subtype in attachments:
            part = MIMEApplication(content, _subtype=mime_subtype or "octet-stream")
            part.add_header("Content-Disposition", "attachment", filename=filename)
            msg.attach(part)
    else:
        msg = MIMEMultipart("alternative")
        if body_plain:
            msg.attach(MIMEText(body_plain, "plain"))
        msg.attach(MIMEText(body_html, "html"))

    msg["Subject"] = subject
    sender_email = from_email or settings.smtp_from_email
    sender_name = from_name or settings.smtp_from_name
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["To"] = to_email
    if reply_to:
        msg["Reply-To"] = (
            f"{reply_to_name} <{reply_to}>" if reply_to_name else reply_to
        )

    # SMTP MAIL FROM envelope. Some providers (notably regular smtp.gmail.com)
    # require the envelope sender to match the authenticated user. We try the
    # caller's preferred sender first; if the server rejects it we fall back
    # to the authenticated user so the email still goes out.
    envelope_candidates: list[str] = [sender_email]
    fallback_sender = (settings.smtp_user or settings.smtp_from_email or "").strip()
    if fallback_sender and fallback_sender not in envelope_candidates:
        envelope_candidates.append(fallback_sender)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            if settings.smtp_use_tls:
                server.starttls()
            if settings.smtp_user and settings.smtp_password:
                server.login(settings.smtp_user, settings.smtp_password)
            last_envelope_err: Exception | None = None
            sent = False
            for envelope in envelope_candidates:
                try:
                    server.sendmail(envelope, to_email, msg.as_string())
                    sent = True
                    break
                except smtplib.SMTPSenderRefused as exc:
                    last_envelope_err = exc
                    continue
            if not sent:
                raise last_envelope_err or smtplib.SMTPException("Sender refused by server")
        return True, None
    except socket.gaierror:
        host = (settings.smtp_host or "").strip()
        return (
            False,
            f"SMTP_HOST '{host}' cannot be resolved (DNS). Set SMTP_HOST to a real SMTP server (e.g. smtp.gmail.com / smtp.office365.com) in backend .env and restart backend.",
        )
    except smtplib.SMTPAuthenticationError as e:
        return (
            False,
            f"SMTP authentication failed ({e.smtp_code}). For Gmail you must use a 16-character App Password, not your account password. See backend/.env SMTP_USER / SMTP_PASSWORD.",
        )
    except Exception as e:
        return False, str(e)
