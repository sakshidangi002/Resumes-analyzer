"""Daily 5 PM IST DSR reminder job.

For every active user with an employee record who hasn't yet submitted today's
DSR, this job delivers the reminder over THREE channels:

  1) In-app notification    -> always created (bell badge / inbox)
  2) Email (SMTP)           -> sent if the user has an email address
  3) Web Push               -> sent to every PushSubscription on file
                               (handled by app.services.web_push_service)

All three are idempotent per IST calendar day: a given user gets at most one
in-app notification per day, and Web Push uses a tag so OS-level popups
don't stack. Each channel is best-effort — failures are logged but never
abort the loop, because the bell badge alone is already useful.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models import AppNotification, DailyStatusReport, User

logger = logging.getLogger(__name__)


_IST_OFFSET = timedelta(hours=5, minutes=30)


def _today_ist() -> date:
    """Return the current date in IST (UTC + 5:30)."""
    return (datetime.utcnow() + _IST_OFFSET).date()


def _ist_day_utc_window(today: date) -> tuple[datetime, datetime]:
    """Map an IST date to its naive-UTC [start, end) window.

    ``AppNotification.created_at`` is stored as naive UTC (``datetime.utcnow``),
    so we can do plain ``>=`` / ``<`` comparisons inside this window.
    """
    ist_midnight_naive = datetime.combine(today, datetime.min.time())
    start_utc = ist_midnight_naive - _IST_OFFSET
    return start_utc, start_utc + timedelta(days=1)


def _employee_email(user: User) -> str | None:
    """Best-effort email lookup: prefer employee.official_email,
    then employee.personal_email, then user.official_email.
    """
    emp = getattr(user, "employee", None)
    if emp is not None:
        for attr in ("official_email", "personal_email"):
            val = getattr(emp, attr, None)
            if val and str(val).strip():
                return str(val).strip()
    val = getattr(user, "official_email", None)
    if val and str(val).strip():
        return str(val).strip()
    return None


def _display_name(user: User) -> str:
    emp = getattr(user, "employee", None)
    if emp is not None:
        fn = (getattr(emp, "first_name", None) or "").strip()
        ln = (getattr(emp, "last_name", None) or "").strip()
        full = f"{fn} {ln}".strip()
        if full:
            return full
    return (getattr(user, "username", None) or "").strip() or "there"


def _build_email_html(name: str, today: date) -> str:
    settings = get_settings()
    base_url = (settings.app_base_url or "").rstrip("/")
    dsr_url = f"{base_url}/dsr" if base_url else None
    date_label = today.strftime("%d %b %Y")

    cta_block = ""
    if dsr_url:
        cta_block = f"""
        <p style="margin: 22px 0">
          <a href="{dsr_url}"
             style="display:inline-block; background:#4f46e5; color:#fff;
                    padding:11px 22px; border-radius:8px; text-decoration:none;
                    font-weight:600; font-family: Arial, Helvetica, sans-serif;">
            Open DSR
          </a>
        </p>
        """

    return f"""<!doctype html>
<html>
  <body style="margin:0; padding:24px; background:#f8fafc;
               font-family: Arial, Helvetica, sans-serif; color:#0f172a;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
           style="max-width:560px; margin:0 auto; background:#fff;
                  border:1px solid #e2e8f0; border-radius:12px;
                  padding:28px 28px 22px;">
      <tr><td>
        <h2 style="margin:0 0 12px; font-size:1.15rem;">
          Submit your DSR before leaving
        </h2>
        <p style="margin:0 0 10px; line-height:1.55;">
          Hi {name},
        </p>
        <p style="margin:0 0 10px; line-height:1.55;">
          This is a friendly 5&nbsp;PM&nbsp;IST reminder to submit your
          <strong>Daily Status Report</strong> for <strong>{date_label}</strong>
          before signing off for the day.
        </p>
        {cta_block}
        <p style="margin:24px 0 0; color:#64748b; font-size:12px; line-height:1.5;">
          You are receiving this automated reminder because today's DSR has
          not been marked as <em>Submitted</em>.
          <br/>— Softwiz HRMS
        </p>
      </td></tr>
    </table>
  </body>
</html>
"""


def _safe_send_email(db: Session, user: User, today: date) -> str:
    """Send the reminder email. Returns a short outcome label for the summary
    (``"sent"`` / ``"no_email"`` / ``"failed"``). Never raises.
    """
    to_email = _employee_email(user)
    if not to_email:
        return "no_email"
    try:
        from app.services.email_service import send_notification
        ok, err = send_notification(
            db,
            to_email=to_email,
            subject=f"Reminder: submit your DSR for {today.strftime('%d %b %Y')}",
            body_html=_build_email_html(_display_name(user), today),
            template_code="DSR_REMINDER",
            related_entity_type="dsr_reminder",
            related_entity_id=today.isoformat(),
        )
        if ok:
            return "sent"
        logger.warning("DSR reminder email failed for user_id=%s: %s", user.id, err)
        return "failed"
    except Exception:
        logger.exception("DSR reminder email crashed for user_id=%s", user.id)
        return "failed"


def _safe_send_push(db: Session, user: User, today: date) -> str:
    """Send a Web Push notification to every device subscribed for ``user``.
    Returns ``"sent"`` / ``"none"`` / ``"failed"``. Never raises.
    """
    try:
        from app.services.web_push_service import send_dsr_reminder_push
        return send_dsr_reminder_push(db, user_id=user.id, today=today)
    except Exception:
        logger.exception("DSR reminder push crashed for user_id=%s", user.id)
        return "failed"


def send_dsr_reminders(db: Session | None = None) -> dict:
    """Deliver the 5 PM DSR reminder across all channels. Idempotent per IST day.

    Returns a summary dict (useful for logs / a debug endpoint).
    """
    owns_session = db is None
    if owns_session:
        db = SessionLocal()
    assert db is not None  # for type checkers

    try:
        today = _today_ist()
        window_start, window_end = _ist_day_utc_window(today)

        eligible_users: list[User] = (
            db.query(User)
            .filter(User.is_active.is_(True), User.employee_id.isnot(None))
            .all()
        )

        submitted_user_ids: set[int] = {
            uid
            for (uid,) in (
                db.query(User.id)
                .join(
                    DailyStatusReport,
                    DailyStatusReport.employee_id == User.employee_id,
                )
                .filter(
                    DailyStatusReport.report_date == today,
                    DailyStatusReport.status == "SUBMITTED",
                )
                .all()
            )
        }

        already_notified: set[int] = {
            uid
            for (uid,) in (
                db.query(AppNotification.user_id)
                .filter(
                    AppNotification.kind == "DSR_REMINDER",
                    AppNotification.created_at >= window_start,
                    AppNotification.created_at < window_end,
                )
                .all()
            )
        }

        counters = {
            "total_eligible_users": len(eligible_users),
            "in_app_created": 0,
            "in_app_skipped_already": 0,
            "skipped_submitted": 0,
            "email_sent": 0,
            "email_no_address": 0,
            "email_failed": 0,
            "push_sent": 0,
            "push_no_subscription": 0,
            "push_failed": 0,
        }

        date_label = today.strftime("%d %b %Y")
        body_text = (
            f"Reminder: please submit your Daily Status Report for {date_label} "
            "before signing off for the day."
        )

        new_rows: list[AppNotification] = []
        for u in eligible_users:
            if u.id in submitted_user_ids:
                counters["skipped_submitted"] += 1
                continue

            if u.id in already_notified:
                counters["in_app_skipped_already"] += 1
            else:
                row = AppNotification(
                    user_id=u.id,
                    title="Submit your DSR before leaving",
                    body=body_text,
                    kind="DSR_REMINDER",
                    link_path="/dsr",
                )
                db.add(row)
                # Track for live WS publish after commit.
                new_rows.append(row)
                counters["in_app_created"] += 1

            email_outcome = _safe_send_email(db, u, today)
            if email_outcome == "sent":
                counters["email_sent"] += 1
            elif email_outcome == "no_email":
                counters["email_no_address"] += 1
            else:
                counters["email_failed"] += 1

            push_outcome = _safe_send_push(db, u, today)
            if push_outcome == "sent":
                counters["push_sent"] += 1
            elif push_outcome == "none":
                counters["push_no_subscription"] += 1
            else:
                counters["push_failed"] += 1

        db.commit()

        # Live WebSocket fan-out for every just-created in-app row so any
        # open tab the user has shows the toast / bell badge instantly.
        # Best-effort: failures are swallowed inside ``publish_sync``.
        try:
            from app.services.notification_service import _publish_live  # local import to avoid cycles

            for row in new_rows:
                try:
                    db.refresh(row)
                except Exception:
                    continue
                _publish_live(row.user_id, row)
        except Exception:
            logger.exception("DSR reminder: live WS publish failed")

        summary = {"date_ist": today.isoformat(), **counters}
        logger.info("DSR 5 PM reminder run: %s", summary)
        return summary

    except Exception:
        if owns_session:
            try:
                db.rollback()
            except Exception:
                pass
        logger.exception("DSR 5 PM reminder run failed")
        raise
    finally:
        if owns_session:
            try:
                db.close()
            except Exception:
                pass
