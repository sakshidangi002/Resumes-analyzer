import imaplib
import json
import logging
import os
import re
import ssl
from dataclasses import dataclass
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser
from typing import Iterable, Optional


logger = logging.getLogger(__name__)

SUBJECT_REQUIRED_SUBSTRING = "[Action required] New application for"
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
_MIME_TO_EXT = {
    "application/pdf": ".pdf",
    "application/x-pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


@dataclass(frozen=True)
class ImapConfig:
    host: str
    username: str
    password: str
    mailbox: str = "INBOX"
    port: int = 993
    use_ssl: bool = True


def _decode_mime_header(value: str) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return str(value).strip()


def _safe_filename(name: str) -> str:
    """
    Make a filename safe for Windows/Linux, preserving extension.
    """
    name = (name or "").strip()
    if not name:
        return "attachment"
    name = name.replace("\x00", "")
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Prevent path traversal
    name = name.replace("..", ".")
    return name[:180]


def _load_processed_uids(path: str) -> set[str]:
    if not path:
        return set()
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {str(x) for x in data}
        if isinstance(data, dict) and isinstance(data.get("uids"), list):
            return {str(x) for x in data["uids"]}
    except Exception as exc:
        logger.warning("Could not read processed UID file %s: %s", path, exc)
    return set()


def _save_processed_uids(path: str, uids: set[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"uids": sorted(uids)}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_imap_config_from_env() -> ImapConfig:
    host = (os.getenv("IMAP_HOST") or "").strip()
    username = (os.getenv("IMAP_USER") or "").strip()
    password = (os.getenv("IMAP_PASSWORD") or "").strip()
    # Gmail app passwords are often entered with spaces; IMAP expects the raw token.
    password = password.replace(" ", "")
    mailbox = (os.getenv("IMAP_MAILBOX") or "INBOX").strip()
    port = int(os.getenv("IMAP_PORT") or "993")
    use_ssl = (os.getenv("IMAP_SSL") or "1").strip().lower() in {"1", "true", "yes"}

    if not host or not username or not password:
        raise ValueError("Missing IMAP configuration. Set IMAP_HOST, IMAP_USER, IMAP_PASSWORD.")

    return ImapConfig(
        host=host,
        username=username,
        password=password,
        mailbox=mailbox,
        port=port,
        use_ssl=use_ssl,
    )


def _connect_imap(cfg: ImapConfig) -> imaplib.IMAP4:
    """
    Connect and login. Raises imaplib.IMAP4.error for auth issues.
    """
    if cfg.use_ssl:
        context = ssl.create_default_context()
        client: imaplib.IMAP4 = imaplib.IMAP4_SSL(cfg.host, cfg.port, ssl_context=context)
    else:
        client = imaplib.IMAP4(cfg.host, cfg.port)
    client.login(cfg.username, cfg.password)
    return client


def _imap_ok(status: str) -> bool:
    return (status or "").upper() == "OK"


def _iter_attachments(msg: Message) -> Iterable[tuple[str, bytes, str]]:
    """
    Yield (filename, bytes, content_type) for attachments.
    """
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = _decode_mime_header(part.get_filename() or "")
        cd = part.get_content_disposition()
        ctype = (part.get_content_type() or "").lower().strip()
        
        # Accept if explicitly marked as attachment OR if it has a valid filename/extension
        if not filename and (cd != "attachment"):
            continue
            
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        yield filename, payload, ctype


def _subject_matches(subject: str) -> bool:
    """
    IMAP SUBJECT search is substring-based, but we enforce the requirement again here.
    """
    return SUBJECT_REQUIRED_SUBSTRING.lower() in (subject or "").lower()


def _download_matching_attachments(
    *,
    msg: Message,
    save_dir: str,
) -> list[str]:
    saved: list[str] = []
    os.makedirs(save_dir, exist_ok=True)

    att_i = 0
    for filename, content, content_type in _iter_attachments(msg):
        att_i += 1
        safe = _safe_filename(filename) or "attachment"
        base, ext = os.path.splitext(safe)
        ext = (ext or "").lower()
        inferred = _MIME_TO_EXT.get((content_type or "").lower().strip(), "")
        if ext not in ALLOWED_EXTENSIONS:
            # Try inference from MIME type (common for Indeed/Gmail attachments)
            if inferred in ALLOWED_EXTENSIONS:
                ext = inferred
                safe = f"{base or 'attachment'}{ext}"
            else:
                continue
        if not content:
            continue
        # If we still have no usable name, generate one.
        if not safe or safe == "attachment":
            safe = f"resume_attachment_{att_i}{ext or ''}".strip()
        out_path = os.path.join(save_dir, safe)

        # Avoid overwriting: add numeric suffix if exists
        if os.path.exists(out_path):
            base, ext2 = os.path.splitext(safe)
            for i in range(2, 200):
                cand = os.path.join(save_dir, f"{base} ({i}){ext2}")
                if not os.path.exists(cand):
                    out_path = cand
                    break

        try:
            with open(out_path, "wb") as f:
                f.write(content)
            saved.append(os.path.basename(out_path))
        except OSError as exc:
            logger.warning("Failed to save attachment %s: %s", out_path, exc)

    return saved


def extract_indeed_view_urls(msg: Message) -> list[str]:
    """
    Extract Indeed "View resume" URLs from an email (HTML + plain text).
    This is best-effort and intentionally conservative.
    """
    urls: list[str] = []

    # Normalize to the ONLY allowed endpoint:
    # https://employers.indeed.com/candidates/resume?...
    try:
        from services.indeed_resume_downloader import normalize_indeed_resume_url
    except Exception:  # pragma: no cover
        normalize_indeed_resume_url = None  # type: ignore

    def _add_from_text(blob: str):
        if not blob:
            return
        for m in re.finditer(r"(https?://[^\s<>\"']+)", blob, flags=re.I):
            u = m.group(1).strip().rstrip(").,;")
            if "indeed" not in u.lower():
                continue
            # Heuristics: prefer resume/application links
            if any(k in u.lower() for k in ("resume", "application", "view", "candidate")):
                if normalize_indeed_resume_url:
                    norm = normalize_indeed_resume_url(u)
                    if norm:
                        urls.append(norm)
                else:
                    urls.append(u)

    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = (part.get_content_type() or "").lower().strip()
        if ctype not in {"text/plain", "text/html"}:
            continue
        try:
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="ignore")
        except Exception:
            continue
        _add_from_text(text)

    # de-dupe
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out[:10]


def fetch_resumes_via_imap(
    *,
    cfg: ImapConfig,
    save_dir: str,
    processed_uids_path: str,
    subject_substring: str = SUBJECT_REQUIRED_SUBSTRING,
    max_to_process: int = 50,
    ignore_processed: bool = False,
    unread_only: bool = False,
    mark_seen: bool = False,
) -> dict:
    """
    Fetch emails from IMAP that match subject_substring and download only .pdf/.docx attachments.
    Dedup by IMAP UID stored in processed_uids_path.

    When unread_only is True, only UNSEEN messages are searched (useful for Gmail automation on
    new Indeed application alerts). Default False keeps legacy behavior for generic inbox fetch.

    When mark_seen is True, messages that are successfully ingested into the returned ``downloaded``
    list are flagged \\Seen on the server.
    """
    if not save_dir:
        raise ValueError("save_dir is required")
    if max_to_process <= 0:
        max_to_process = 1

    processed = set() if ignore_processed else _load_processed_uids(processed_uids_path)
    downloaded_total: list[dict] = []
    matched_uids: list[str] = []

    client: Optional[imaplib.IMAP4] = None
    try:
        client = _connect_imap(cfg)
        status, _ = client.select(cfg.mailbox, readonly=False)
        if not _imap_ok(status):
            raise RuntimeError(f"Could not select mailbox: {cfg.mailbox}")

        # IMAP SEARCH uses substring match for SUBJECT. We still verify in parsed header.
        if unread_only:
            logger.info("IMAP SEARCH: UNSEEN + SUBJECT contains %r", subject_substring[:48])
            status, data = client.uid("SEARCH", None, "UNSEEN", "SUBJECT", f"\"{subject_substring}\"")
        else:
            logger.info("IMAP SEARCH: SUBJECT contains %r (including read mail)", subject_substring[:48])
            status, data = client.uid("SEARCH", None, "SUBJECT", f"\"{subject_substring}\"")
        if not _imap_ok(status):
            raise RuntimeError("IMAP SEARCH failed")

        raw = (data[0] or b"").decode(errors="ignore").strip()
        uids = [u for u in raw.split() if u]
        logger.info("IMAP SEARCH returned %d UID(s)", len(uids))

        for uid in uids:
            if (not ignore_processed) and (uid in processed):
                continue
            matched_uids.append(uid)
            if len(matched_uids) > max_to_process:
                break

        for uid in matched_uids:
            status, msg_data = client.uid("FETCH", uid, "(RFC822)")
            if not _imap_ok(status) or not msg_data or not msg_data[0]:
                processed.add(uid)
                continue

            try:
                raw_bytes = msg_data[0][1]
                msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
            except Exception as exc:
                logger.warning("Corrupt email uid=%s: %s", uid, exc)
                processed.add(uid)
                continue

            subject = _decode_mime_header(str(msg.get("Subject", "")))
            sender = _decode_mime_header(str(msg.get("From", "")))

            # Extra safety: only process emails that truly match the subject condition.
            if subject_substring.lower() not in subject.lower():
                processed.add(uid)
                continue
            if not _subject_matches(subject):
                processed.add(uid)
                continue

            saved_files = _download_matching_attachments(msg=msg, save_dir=save_dir)
            indeed_urls = extract_indeed_view_urls(msg)

            logger.info("IMAP matched email subject=%r from=%r saved=%s", subject, sender, saved_files)
            downloaded_total.append(
                {
                    "uid": uid,
                    "subject": subject,
                    "from": sender,
                    "downloaded_files": saved_files,
                    "indeed_view_urls": indeed_urls,
                }
            )
            if mark_seen:
                try:
                    st, _ = client.uid("STORE", uid, "+FLAGS", r"(\Seen)")
                    if _imap_ok(st):
                        logger.info("Marked uid=%s as \\Seen", uid)
                    else:
                        logger.warning("STORE \\Seen failed for uid=%s status=%r", uid, st)
                except Exception as exc:
                    logger.warning("Could not mark uid=%s read: %s", uid, exc)
            # Mark processed unless explicitly bypassing cache
            if not ignore_processed:
                processed.add(uid)

        if not ignore_processed:
            _save_processed_uids(processed_uids_path, processed)

        return {
            "mailbox": cfg.mailbox,
            "searched_subject": subject_substring,
            "unread_only": bool(unread_only),
            "mark_seen": bool(mark_seen),
            "matched_email_count": len(uids),
            "processed_new_count": len(matched_uids),
            "downloaded": downloaded_total,
        }
    except imaplib.IMAP4.error as exc:
        # invalid login / auth issues
        raise PermissionError(f"IMAP login failed: {exc}") from exc
    finally:
        try:
            if client is not None:
                client.logout()
        except Exception:
            pass

