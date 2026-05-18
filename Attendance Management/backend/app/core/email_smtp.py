"""
Email service using simple SMTP. All notifications and letters are sent via this.
"""
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from app.core.config import get_settings


def send_email(
    to_email: str,
    subject: str,
    body_html: str,
    body_plain: Optional[str] = None,
    from_email: Optional[str] = None,
    from_name: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Send a single email via SMTP. Uses app SMTP settings.
    Returns (success, error_message). If success=True, error_message=None.
    """
    settings = get_settings()
    if not settings.smtp_host or settings.smtp_host.strip().lower() in {"localhost", "127.0.0.1"}:
        return (
            False,
            "SMTP is not configured. Set SMTP_HOST/SMTP_USER/SMTP_PASSWORD (or a valid relay) in backend .env and restart the backend.",
        )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    sender_email = from_email or settings.smtp_from_email
    sender_name = from_name or settings.smtp_from_name
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["To"] = to_email

    if body_plain:
        msg.attach(MIMEText(body_plain, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            if settings.smtp_use_tls:
                server.starttls()
            if settings.smtp_user and settings.smtp_password:
                server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        return True, None
    except socket.gaierror:
        host = (settings.smtp_host or "").strip()
        return (
            False,
            f"SMTP_HOST '{host}' cannot be resolved (DNS). Set SMTP_HOST to a real SMTP server (e.g. smtp.gmail.com / smtp.office365.com) in backend .env and restart backend.",
        )
    except Exception as e:
        return False, str(e)
