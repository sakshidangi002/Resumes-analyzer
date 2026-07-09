"""
Intelligent Email Resume Importer
=================================

Fetches emails from a configurable mailbox (Gmail label / Outlook folder /
dedicated careers@ inbox / generic IMAP), decides which attachments are really
resumes (ignoring invoices, agreements, reports, brochures, meeting notes…),
and imports the valid ones through the EXISTING resume upload pipeline.

Pipeline (mirrors the required architecture)::

    Inbox → read new emails → find attachments → Layer-1 filter → text extract
          → Layer-3 resume validation (confidence) → duplicate check
          → existing upload pipeline → candidate DB

Design principles
-----------------
* Reuses the existing services only — NO second resume parser:
    - text extraction     : api._extract_text_from_bytes
    - extraction + store  : api.upload_resume (via process_resume_file_with_existing_pipeline)
    - duplicate detection : built into upload_resume (returns status="duplicate")
* Content is the source of truth. Subject/filename NEVER accept or reject on
  their own — they only influence processing priority (Layer 2).
* Idempotent: processed message UIDs are persisted; handled emails are never
  re-imported.
* Resilient: every attachment is handled in isolation; one failure (corrupt /
  password-protected / image-only PDF, parse/network error) never stops the run.

Configuration (env)
--------------------
IMAP_HOST / IMAP_USER / IMAP_PASSWORD / IMAP_MAILBOX (Layer 0 source; e.g. a
Gmail label "Resumes"), plus EMAIL_IMPORT_* below (all thresholds/sizes tunable).
"""
from __future__ import annotations

import datetime
import logging
import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy.orm import Session

# email_service only depends on the stdlib (imaplib), so importing it at module
# load is safe. The shared upload processor is imported LAZILY at call time so
# this module never drags in the heavier Indeed/Playwright chain just to load.
try:  # package vs script import (matches the rest of the backend)
    from backend.email_service import (
        EmailAttachment, ImapConfig, fetch_new_emails,
        load_imap_config_from_env, _load_processed_uids, _save_processed_uids,
        _safe_filename,
    )
except ImportError:  # pragma: no cover
    from email_service import (  # type: ignore
        EmailAttachment, ImapConfig, fetch_new_emails,
        load_imap_config_from_env, _load_processed_uids, _save_processed_uids,
        _safe_filename,
    )

logger = logging.getLogger(__name__)


async def _import_file_via_upload(
    *, request: Any, db: Session, upload_resume_callable: Callable[..., Awaitable[list]], file_path: str,
) -> dict:
    """Import one file through the EXISTING upload pipeline (extraction + storage
    + duplicate detection). Prefers the shared
    ``process_resume_file_with_existing_pipeline`` (reuse); if its module isn't
    importable in this environment, falls back to a minimal adapter that still
    calls the SAME ``upload_resume`` — so behaviour is identical to a manual HR
    upload either way. No parsing logic is duplicated here.
    """
    proc = None
    try:
        try:
            from backend.indeed_resume_pipeline import process_resume_file_with_existing_pipeline as proc
        except ImportError:
            from indeed_resume_pipeline import process_resume_file_with_existing_pipeline as proc  # type: ignore
    except Exception:
        proc = None

    if proc is not None:
        return await proc(
            request=request, db=db, file_path=file_path,
            upload_resume_callable=upload_resume_callable,
        )

    # Minimal fallback — identical effect to uploading this file by hand.
    import io
    from fastapi import UploadFile
    with open(file_path, "rb") as fh:
        data = fh.read()
    uf = UploadFile(filename=os.path.basename(file_path), file=io.BytesIO(data))
    out = await upload_resume_callable(request=request, files=[uf], db=db)
    first = out[0] if isinstance(out, list) and out else out
    return first if isinstance(first, dict) else {"status": "error", "message": "unexpected upload response"}


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------
IMPORTED = "IMPORTED"
DUPLICATE = "DUPLICATE"
NEEDS_REVIEW = "NEEDS_REVIEW"
IGNORED = "IGNORED"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, "").strip() or default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: list[str]) -> list[str]:
    v = (os.getenv(name) or "").strip()
    if not v:
        return list(default)
    return [x.strip().lower() for x in v.split(",") if x.strip()]


@dataclass
class EmailImportConfig:
    """All Layer-1/2/3 knobs — everything is env-overridable."""
    save_dir: str
    review_dir: str
    processed_uids_path: str

    # Layer 1 — attachment filtering
    supported_ext: set[str] = field(default_factory=lambda: {".pdf", ".docx"})
    min_size_bytes: int = 15 * 1024          # skip tiny (signatures/logos)
    max_size_bytes: int = 10 * 1024 * 1024   # skip huge
    skip_filename_keywords: list[str] = field(default_factory=lambda: [
        "invoice", "bill", "receipt", "logo", "signature", "banner",
        "payment", "purchase_order", "purchase-order", "quotation", "po_",
    ])

    # Subject-based detection (Layer 0 without a label): when set, only emails
    # whose SUBJECT contains one of these keywords are fetched from the mailbox.
    # This lets the importer scan the plain INBOX (no manual labelling) yet stay
    # cheap. Content validation still decides each attachment. Empty = fetch all.
    subject_keywords: list[str] = field(default_factory=lambda: [
        "resume", "cv", "curriculum vitae", "application", "applicant",
        "applying", "apply", "candidate", "hiring", "vacancy", "vacancies",
        "position", "opening", "openings", "opportunity", "opportunities",
        "internship", "intern", "developer", "engineer", "job", "jobs",
        "career", "careers", "recruitment", "recruiter", "seeking",
        "interested", "joiner", "notice period", "fresher", "experienced",
        "designer", "tester", "analyst", "consultant", "specialist",
        "associate", "executive", "manager", "myself", "for the role",
        "resume attached",
        # Indeed application alerts: "[Action required] New application for ...".
        "action required", "new application",
    ])

    # Layer 2 — metadata scoring (priority ONLY, never accept/reject)
    positive_keywords: list[str] = field(default_factory=lambda: [
        "resume", "cv", "curriculum vitae", "application", "applying",
        "candidate", "hiring", "vacancy", "position", "attached resume",
        "please find attached", "job",
    ])
    negative_keywords: list[str] = field(default_factory=lambda: [
        "invoice", "payment", "quotation", "purchase order", "meeting", "report",
    ])

    # Layer 3 — content validation thresholds (0..100)
    auto_threshold: int = 70    # >= -> import automatically
    review_threshold: int = 40  # 40..69 -> needs review; <40 -> ignore
    min_text_chars: int = 40    # below this = image-only/empty -> needs review

    # Fetch
    unread_only: bool = True
    mark_seen: bool = True
    max_emails: int = 50

    # Indeed browser-automation download (resume behind a login-gated Indeed URL)
    indeed_temp_dir: str = ""        # where the browser saves the file before import
    indeed_failed_dir: str = ""      # kept copies of downloads that failed to process
    indeed_headless: bool = True     # set false to watch/debug the browser
    indeed_timeout_ms: int = 90_000  # per-navigation / download timeout
    indeed_max_retries: int = 2      # outer retries around the whole download
    # Date floor (IMAP SINCE, e.g. "01-Jan-2026"). Only mail on/after this date is
    # fetched from the server. Defaults to Jan 1 of the current year so a huge
    # historical inbox (6000+ old mails) is never scanned. None = no floor.
    since_date: Optional[str] = None


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _resolve_since_date() -> Optional[str]:
    """Resolve EMAIL_IMPORT_SINCE into an IMAP date string ('01-Jan-2026').

    Accepts: empty/unset -> Jan 1 of the current year (default);
    a bare 4-digit year '2026' -> 01-Jan-2026; an ISO date '2026-03-15';
    or an already-formatted IMAP date '15-Mar-2026' (passed through).
    'all'/'0'/'none' -> None (no date floor; scan everything).
    """
    raw = (os.getenv("EMAIL_IMPORT_SINCE", "") or "").strip()
    if raw.lower() in {"all", "0", "none", "false"}:
        return None
    if not raw:
        return datetime.date(datetime.date.today().year, 1, 1).strftime("%d-%b-%Y")
    if raw.isdigit() and len(raw) == 4:      # a year
        return f"01-Jan-{raw}"
    try:                                      # ISO date
        d = datetime.date.fromisoformat(raw)
        return d.strftime("%d-%b-%Y")
    except ValueError:
        pass
    # Assume it is already an IMAP date like 15-Mar-2026.
    return raw


def load_config_from_env(base_dir: str) -> EmailImportConfig:
    data_dir = os.getenv("EMAIL_IMPORT_SAVE_DIR", "").strip() or os.path.join(base_dir, "data", "email_inbox")
    review_dir = os.getenv("EMAIL_IMPORT_REVIEW_DIR", "").strip() or os.path.join(base_dir, "data", "email_needs_review")
    uids_path = os.getenv("EMAIL_IMPORT_PROCESSED_UIDS", "").strip() or os.path.join(base_dir, "data", "email_processed_uids.json")
    ext = {("." + e.lstrip(".")).lower() for e in _env_csv("EMAIL_IMPORT_SUPPORTED_EXT", ["pdf", "docx"])}
    default_subjects = EmailImportConfig.__dataclass_fields__["subject_keywords"].default_factory()
    return EmailImportConfig(
        save_dir=data_dir,
        review_dir=review_dir,
        processed_uids_path=uids_path,
        supported_ext=ext,
        subject_keywords=_env_csv("EMAIL_IMPORT_SUBJECT_KEYWORDS", default_subjects),
        min_size_bytes=_env_int("EMAIL_IMPORT_MIN_SIZE_KB", 15) * 1024,
        max_size_bytes=_env_int("EMAIL_IMPORT_MAX_SIZE_MB", 10) * 1024 * 1024,
        skip_filename_keywords=_env_csv("EMAIL_IMPORT_SKIP_FILENAMES", EmailImportConfig.__dataclass_fields__["skip_filename_keywords"].default_factory()),
        auto_threshold=_env_int("EMAIL_IMPORT_AUTO_THRESHOLD", 70),
        review_threshold=_env_int("EMAIL_IMPORT_REVIEW_THRESHOLD", 40),
        min_text_chars=_env_int("EMAIL_IMPORT_MIN_TEXT_CHARS", 40),
        unread_only=_env_bool("EMAIL_IMPORT_UNREAD_ONLY", True),
        mark_seen=_env_bool("EMAIL_IMPORT_MARK_SEEN", True),
        max_emails=_env_int("EMAIL_IMPORT_MAX_EMAILS", 50),
        since_date=_resolve_since_date(),
        indeed_temp_dir=os.getenv("EMAIL_IMPORT_INDEED_TEMP", "").strip() or os.path.join(base_dir, "temp", "resumes"),
        indeed_failed_dir=os.getenv("EMAIL_IMPORT_INDEED_FAILED", "").strip() or os.path.join(base_dir, "data", "failed_downloads"),
        indeed_headless=_env_bool("EMAIL_IMPORT_INDEED_HEADLESS", True),
        indeed_timeout_ms=_env_int("EMAIL_IMPORT_INDEED_TIMEOUT_MS", 90_000),
        indeed_max_retries=_env_int("EMAIL_IMPORT_INDEED_MAX_RETRIES", 2),
    )


# ---------------------------------------------------------------------------
# Layer 1 — attachment filtering (fast, metadata only)
# ---------------------------------------------------------------------------
def attachment_filter(att: EmailAttachment, cfg: EmailImportConfig) -> tuple[bool, str]:
    """Return (keep, reason). Cheap gate so we don't extract junk. Filename
    keywords can only SKIP obvious non-resumes — never accept."""
    name = (att.filename or "").lower()
    ext = os.path.splitext(name)[1].lower()

    if att.disposition == "inline":
        return False, "inline_attachment"
    if ext not in cfg.supported_ext:
        return False, f"unsupported_type:{ext or 'none'}"
    if att.size < cfg.min_size_bytes:
        return False, f"too_small:{att.size}B"
    if att.size > cfg.max_size_bytes:
        return False, f"too_large:{att.size}B"
    for kw in cfg.skip_filename_keywords:
        if kw in name:
            return False, f"blacklisted_filename:{kw}"
    return True, "passed_filter"


# ---------------------------------------------------------------------------
# Layer 2 — email metadata score (priority only)
# ---------------------------------------------------------------------------
def email_priority_score(subject: str, body: str, cfg: EmailImportConfig) -> int:
    """A small +/- score used ONLY to order processing. Never gates import."""
    blob = f"{subject or ''}\n{body or ''}".lower()
    score = 0
    for kw in cfg.positive_keywords:
        if kw in blob:
            score += 2
    for kw in cfg.negative_keywords:
        if kw in blob:
            score -= 2
    return score


# ---------------------------------------------------------------------------
# Layer 3 — content validation + resume confidence (the primary decision)
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3}[\s-]?\d{3,4})")

_SECTION_KEYWORDS = {
    "skills": ["skills", "technical skills", "technologies", "tech stack", "core competencies"],
    "experience": ["experience", "work experience", "employment", "work history", "professional experience"],
    "education": ["education", "academic", "qualification", "b.tech", "b.e.", "bachelor", "master", "mca", "bca"],
    "projects": ["projects", "project experience"],
    "certifications": ["certification", "certificate", "certified"],
    "summary": ["professional summary", "career objective", "objective", "profile summary", "about me"],
}

# Strong markers of NON-resume business documents. Used to reject invoices/etc.
# even when they happen to contain an email or phone number.
_NON_RESUME_MARKERS = [
    "invoice", "tax invoice", "purchase order", "gst", "amount due", "total due",
    "bill to", "ship to", "quotation", "payment terms", "balance due", "subtotal",
    "minutes of meeting", "agenda", "agreement", "terms and conditions", "sla",
]

# Strong markers of a COMPANY profile / marketing deck / brochure. These read
# like a resume (they list technologies, "services", clients) but describe a
# COMPANY, not a candidate. Two or more of these => reject, so a "Company Profile"
# pitch deck is never imported as a candidate.
_COMPANY_MARKERS = [
    "company profile", "company deck", "technology partner", "b2b platform",
    "our services", "our clients", "our clientele", "on-demand developer",
    "on demand developer", "hiring solutions", "staff augmentation",
    "why choose us", "our expertise", "our mission", "we provide", "our team of",
    "our portfolio", "trusted by", "years of experience delivering", "pvt. ltd",
    "private limited", "we help companies", "our solutions",
]


def _looks_like_name_line(line: str) -> bool:
    """Very light heuristic: a top-of-document line that reads like a person's name."""
    s = line.strip()
    if not s or len(s) > 40 or any(c.isdigit() for c in s) or "@" in s:
        return False
    words = s.split()
    if not (2 <= len(words) <= 4):
        return False
    return all(w[:1].isupper() and w.isalpha() for w in words)


def resume_confidence(text: str) -> tuple[int, dict]:
    """Score 0..100 that a document is a resume, from its TEXT only.

    +20 name, +20 email, +20 phone, +15 skills, +15 experience, +10 education.
    Strong non-resume markers (invoice/agreement/report) with few resume
    sections drive the score down so business documents are rejected.
    """
    signals: dict[str, Any] = {}
    if not text:
        return 0, {"empty": True}
    low = text.lower()

    has_email = bool(_EMAIL_RE.search(text))
    has_phone = bool(_PHONE_RE.search(text))
    section_hits = {k: any(kw in low for kw in kws) for k, kws in _SECTION_KEYWORDS.items()}
    has_name = any(_looks_like_name_line(ln) for ln in text.splitlines()[:8])

    score = 0
    if has_name:
        score += 20
    if has_email:
        score += 20
    if has_phone:
        score += 20
    if section_hits["skills"]:
        score += 15
    if section_hits["experience"]:
        score += 15
    if section_hits["education"]:
        score += 10

    n_sections = sum(1 for v in section_hits.values() if v)
    non_resume_hits = [m for m in _NON_RESUME_MARKERS if m in low]
    # A business document (invoice/agreement/report) with < 2 resume sections is
    # not a resume even if it carries a contact — cap it into the IGNORE band.
    if non_resume_hits and n_sections < 2:
        score = min(score, 30)

    # A COMPANY profile / marketing deck also lists tech & "services" and can
    # score high, but it's not a candidate. Two+ company markers => reject,
    # regardless of section count.
    company_hits = [m for m in _COMPANY_MARKERS if m in low]
    if len(company_hits) >= 2:
        score = min(score, 25)

    signals.update(
        has_name=has_name, has_email=has_email, has_phone=has_phone,
        sections=section_hits, section_count=n_sections,
        non_resume_markers=non_resume_hits, company_markers=company_hits,
        text_len=len(text),
    )
    return max(0, min(100, score)), signals


def decide(text: Optional[str], ext: str, cfg: EmailImportConfig) -> tuple[int, str, str, dict]:
    """Return (confidence, decision, reason, signals).

    decision ∈ {IMPORTED-candidate, NEEDS_REVIEW, IGNORED}. (IMPORTED here means
    "eligible to import"; the final IMPORTED/DUPLICATE label comes from the
    upload result.)
    """
    if text is None:
        # extraction raised → corrupt / password-protected / unreadable
        return 0, NEEDS_REVIEW, "text_extraction_failed", {}
    if len(text.strip()) < cfg.min_text_chars:
        # PDF with (almost) no extractable text → likely scanned/image-only
        return 0, NEEDS_REVIEW, "no_extractable_text_maybe_scanned", {"text_len": len(text)}

    conf, signals = resume_confidence(text)
    if conf >= cfg.auto_threshold:
        return conf, "IMPORT", f"confidence>={cfg.auto_threshold}", signals
    if conf >= cfg.review_threshold:
        return conf, NEEDS_REVIEW, f"confidence in [{cfg.review_threshold},{cfg.auto_threshold})", signals
    return conf, IGNORED, f"confidence<{cfg.review_threshold}", signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_bytes(directory: str, filename: str, content: bytes) -> str:
    os.makedirs(directory, exist_ok=True)
    safe = _safe_filename(filename) or "attachment"
    out = os.path.join(directory, safe)
    if os.path.exists(out):
        base, ex = os.path.splitext(safe)
        for i in range(2, 500):
            cand = os.path.join(directory, f"{base} ({i}){ex}")
            if not os.path.exists(cand):
                out = cand
                break
    with open(out, "wb") as fh:
        fh.write(content)
    return out


def _get_text_extractor() -> Callable[[bytes, str, str], str]:
    """The existing PDF/DOCX text extractor from the upload pipeline (reused, not duplicated)."""
    try:
        from backend.api import _extract_text_from_bytes  # type: ignore
    except ImportError:  # pragma: no cover
        from api import _extract_text_from_bytes  # type: ignore
    return _extract_text_from_bytes


def _base_dir() -> str:
    try:
        from backend.api import BASE_DIR  # type: ignore
    except ImportError:  # pragma: no cover
        from api import BASE_DIR  # type: ignore
    return BASE_DIR


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
async def _import_indeed_resume(
    *, request: Any, db: Session, upload_resume_callable: Callable[..., Awaitable[list]],
    indeed_urls: list[str], cfg: EmailImportConfig, extract_text, base: str, entry_base: dict,
) -> Optional[dict]:
    """Download a resume from an Indeed 'View resume' link (the resume is behind
    a login-required Indeed URL, not an attachment) and import it through the
    same validate + upload path. Requires a saved Indeed browser session
    (storage_state.json). Returns a decision log entry, or None if no usable link.
    """
    entry = dict(entry_base)
    entry["attachment"] = "indeed_resume"
    uid = entry.get("uid") or entry.get("message_id") or "?"

    try:
        from services.indeed_resume_downloader import (
            choose_best_resume_url, download_resume_with_playwright, _load_storage_state_path,
        )
    except Exception:
        entry.update(decision=NEEDS_REVIEW, reason="indeed_downloader_unavailable", confidence=0)
        return entry

    # ── Step 1: resolve the resume link ────────────────────────────────────
    url = choose_best_resume_url(indeed_urls)
    if not url:
        logger.info("INDEED uid=%s: no usable resume link in email", uid)
        return None
    logger.info("INDEED uid=%s: resume link = %s", uid, url[:120])

    # ── Step 2: require a saved login session (else nothing can be viewed) ──
    try:
        state_path = _load_storage_state_path()
    except Exception:
        state_path = os.getenv("INDEED_STORAGE_STATE", "").strip()
    if not state_path or not os.path.isfile(state_path):
        logger.warning("INDEED uid=%s: no saved login (storage_state) — flagged for setup", uid)
        entry.update(decision=NEEDS_REVIEW, reason="indeed_login_not_set_up", confidence=0)
        return entry

    # ── Step 3: download (with outer retries around the browser) ───────────
    temp_dir = cfg.indeed_temp_dir or os.path.join(base, "temp", "resumes")
    os.makedirs(temp_dir, exist_ok=True)
    dl = None
    last_err = ""
    for attempt_no in range(1, max(1, cfg.indeed_max_retries) + 1):
        logger.info("INDEED uid=%s: browser download attempt %d/%d (headless=%s)",
                    uid, attempt_no, cfg.indeed_max_retries, cfg.indeed_headless)
        try:
            dl = await download_resume_with_playwright(
                resume_url=url, save_dir=temp_dir, storage_state_path=state_path,
                headless=cfg.indeed_headless, timeout_ms=cfg.indeed_timeout_ms, prefix=f"indeed_uid{uid}",
            )
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            logger.warning("INDEED uid=%s: download raised: %s", uid, last_err)
            dl = None
            continue
        if getattr(dl, "ok", False):
            break
        last_err = str(getattr(dl, "error", "") or "unknown")
        logger.warning("INDEED uid=%s: download attempt %d failed: %s", uid, attempt_no, last_err[:160])

    fpath = str(getattr(dl, "file_path", "") or "") if dl else ""

    # ── Step 4: validate the downloaded file (exists, non-empty, right ext) ─
    def _valid(p: str) -> bool:
        return bool(p) and os.path.isfile(p) and os.path.getsize(p) > 0 and p.lower().endswith((".pdf", ".docx"))

    if not (dl and getattr(dl, "ok", False) and _valid(fpath)):
        logger.error("INDEED uid=%s: download failed/invalid — %s", uid, (last_err or "no file")[:160])
        entry.update(decision=NEEDS_REVIEW, reason=f"indeed_download_failed: {last_err}"[:150], confidence=0)
        return entry
    logger.info("INDEED uid=%s: downloaded %s (%d bytes)", uid, os.path.basename(fpath), os.path.getsize(fpath))

    # ── Step 5: hand the file to the EXISTING upload/extraction pipeline ────
    entry["attachment"] = os.path.basename(fpath)
    ext = os.path.splitext(fpath)[1].lower()
    try:
        with open(fpath, "rb") as fh:
            text = extract_text(fh.read(), ext, base)
    except Exception:
        text = None
    conf, decision, reason, _ = decide(text, ext, cfg)
    entry["confidence"] = conf

    processed_ok = False
    if decision == "IMPORT":
        logger.info("INDEED uid=%s: importing via existing pipeline", uid)
        result = await _import_file_via_upload(
            request=request, db=db, upload_resume_callable=upload_resume_callable, file_path=fpath,
        )
        status = (result or {}).get("status")
        if status == "duplicate":
            entry.update(decision=DUPLICATE, reason="duplicate_candidate")
            processed_ok = True
            logger.info("INDEED uid=%s: duplicate candidate", uid)
        elif status == "success":
            entry.update(decision=IMPORTED, reason="imported_from_indeed", resume_id=result.get("resume_id"))
            processed_ok = True
            logger.info("INDEED uid=%s: imported successfully (resume_id=%s)", uid, result.get("resume_id"))
        else:
            entry.update(decision=NEEDS_REVIEW, reason=f"upload_{status}")
            logger.error("INDEED uid=%s: upload returned status=%s", uid, status)
    elif decision == NEEDS_REVIEW:
        entry.update(decision=NEEDS_REVIEW, reason=reason)
    else:
        entry.update(decision=IGNORED, reason=reason)
        processed_ok = True  # a deliberate ignore is a clean outcome, not a failure

    # ── Step 6: cleanup — delete temp on success, keep failures for triage ──
    try:
        if processed_ok:
            os.remove(fpath)
            logger.info("INDEED uid=%s: cleanup removed temp file", uid)
        else:
            failed_dir = cfg.indeed_failed_dir or os.path.join(base, "data", "failed_downloads")
            os.makedirs(failed_dir, exist_ok=True)
            dest = os.path.join(failed_dir, os.path.basename(fpath))
            os.replace(fpath, dest)
            logger.warning("INDEED uid=%s: kept failed download for triage -> %s", uid, dest)
    except Exception as exc:
        logger.warning("INDEED uid=%s: cleanup issue: %s", uid, exc)

    return entry


async def import_resumes_from_email(
    *,
    request: Any,
    db: Session,
    upload_resume_callable: Callable[..., Awaitable[list]],
    cfg: Optional[EmailImportConfig] = None,
) -> dict:
    """Full Layer-0..3 email → resume import run. Returns a structured summary.

    ``request`` only needs ``request.app.state.executor`` (may be None); the
    scheduler passes a lightweight shim. ``upload_resume_callable`` is the real
    ``api.upload_resume`` so imports behave exactly like an HR manual upload
    (same extraction, storage and duplicate detection).
    """
    base = _base_dir()
    cfg = cfg or load_config_from_env(base)
    extract_text = _get_text_extractor()

    try:
        imap_cfg: ImapConfig = load_imap_config_from_env()
    except ValueError as exc:
        logger.error("Email import: IMAP not configured: %s", exc)
        return {"ok": False, "error": str(exc), "processed_emails": 0, "results": []}

    processed = _load_processed_uids(cfg.processed_uids_path)

    try:
        fetched = fetch_new_emails(
            cfg=imap_cfg,
            already_processed=processed,
            unread_only=cfg.unread_only,
            mark_seen=cfg.mark_seen,
            max_to_process=cfg.max_emails,
            subject_keywords=cfg.subject_keywords,
            since_date=cfg.since_date,
        )
        logger.info("Email import: date floor (SINCE) = %s", cfg.since_date or "none")
    except PermissionError as exc:
        logger.error("Email import: %s", exc)
        return {"ok": False, "error": str(exc), "processed_emails": 0, "results": []}
    except Exception as exc:  # network / mailbox errors — never crash the caller
        logger.exception("Email import: fetch failed")
        return {"ok": False, "error": f"fetch_failed: {exc}", "processed_emails": 0, "results": []}

    emails = fetched.get("emails") or []
    # Layer 2: process higher-priority (application-looking) emails first.
    emails.sort(key=lambda e: email_priority_score(e.get("subject", ""), e.get("body_text", ""), cfg), reverse=True)

    counts = {IMPORTED: 0, DUPLICATE: 0, NEEDS_REVIEW: 0, IGNORED: 0}
    results: list[dict] = []
    handled_uids: list[str] = []

    for em in emails:
        uid = str(em.get("uid") or "")
        msg_id = str(em.get("message_id") or uid)
        subject = str(em.get("subject") or "")
        sender = str(em.get("from") or "")
        atts: list[EmailAttachment] = em.get("attachments") or []
        priority = email_priority_score(subject, em.get("body_text", ""), cfg)

        att_logs: list[dict] = []
        for att in atts:
            entry = {
                "message_id": msg_id, "subject": subject, "from": sender,
                "attachment": att.filename, "priority": priority,
                "confidence": None, "decision": None, "reason": None,
            }
            try:
                # ── Layer 1 ────────────────────────────────────────────────
                keep, reason = attachment_filter(att, cfg)
                if not keep:
                    entry.update(decision=IGNORED, reason=reason, confidence=0)
                    counts[IGNORED] += 1
                    att_logs.append(entry)
                    continue

                # Save once; reused for text extraction and (if imported) upload.
                saved_path = _save_bytes(cfg.save_dir, att.filename, att.content)
                ext = os.path.splitext(saved_path)[1].lower()

                # ── Layer 3: text extraction (reuses the upload parser) ─────
                try:
                    text: Optional[str] = extract_text(att.content, ext, base)
                except Exception as exc:  # corrupt / password-protected / unreadable
                    logger.warning("Email import: text extraction failed for %s: %s", att.filename, exc)
                    text = None

                conf, decision, reason, _signals = decide(text, ext, cfg)
                entry.update(confidence=conf)

                if decision == "IMPORT":
                    # ── Duplicate check + import via the EXISTING pipeline ──
                    result = await _import_file_via_upload(
                        request=request, db=db,
                        upload_resume_callable=upload_resume_callable,
                        file_path=saved_path,
                    )
                    status = (result or {}).get("status")
                    if status == "duplicate":
                        entry.update(decision=DUPLICATE, reason="duplicate_candidate")
                        counts[DUPLICATE] += 1
                    elif status == "success":
                        entry.update(decision=IMPORTED, reason="imported")
                        entry["resume_id"] = result.get("resume_id")
                        counts[IMPORTED] += 1
                    else:
                        # Upload itself failed (e.g. unsupported .doc, parse error).
                        entry.update(decision=NEEDS_REVIEW, reason=f"upload_{status}: {result.get('message', '')}"[:200])
                        _move_to_review(saved_path, cfg)
                        counts[NEEDS_REVIEW] += 1
                elif decision == NEEDS_REVIEW:
                    entry.update(decision=NEEDS_REVIEW, reason=reason)
                    _move_to_review(saved_path, cfg)
                    counts[NEEDS_REVIEW] += 1
                else:  # IGNORED
                    entry.update(decision=IGNORED, reason=reason)
                    _safe_remove(saved_path)
                    counts[IGNORED] += 1
            except Exception as exc:  # isolate per-attachment failures
                logger.exception("Email import: error handling attachment %s", att.filename)
                entry.update(decision=NEEDS_REVIEW, reason=f"error: {exc}"[:200])
                counts[NEEDS_REVIEW] += 1

            logger.info(
                "EMAIL-IMPORT msg=%s from=%r att=%r conf=%s decision=%s reason=%s",
                msg_id, sender, att.filename, entry["confidence"], entry["decision"], entry["reason"],
            )
            att_logs.append(entry)

        # ── Indeed application alerts ──────────────────────────────────────
        # If no importable attachment was found but the email carries an Indeed
        # "View resume" link, download that resume (browser session) and import.
        imported_any = any(a.get("decision") in (IMPORTED, DUPLICATE) for a in att_logs)
        indeed_urls = em.get("indeed_urls") or []
        if not imported_any and indeed_urls:
            try:
                ind = await _import_indeed_resume(
                    request=request, db=db, upload_resume_callable=upload_resume_callable,
                    indeed_urls=indeed_urls, cfg=cfg, extract_text=extract_text, base=base,
                    entry_base={"uid": uid, "message_id": msg_id, "subject": subject, "from": sender,
                                "priority": priority, "confidence": None, "decision": None, "reason": None},
                )
            except Exception as exc:
                logger.exception("Email import: Indeed handling failed for %s", msg_id)
                ind = None
            if ind:
                counts[ind["decision"]] = counts.get(ind["decision"], 0) + 1
                att_logs.append(ind)
                logger.info(
                    "EMAIL-IMPORT (indeed) msg=%s decision=%s reason=%s",
                    msg_id, ind["decision"], ind["reason"],
                )

        results.append({"uid": uid, "message_id": msg_id, "subject": subject, "from": sender, "attachments": att_logs})
        handled_uids.append(uid)

    # Idempotency: record handled UIDs only after processing them.
    if handled_uids:
        _save_processed_uids(cfg.processed_uids_path, processed | set(handled_uids))

    summary = {
        "ok": True,
        "mailbox": fetched.get("mailbox"),
        "processed_emails": len(emails),
        "counts": counts,
        "results": results,
    }
    logger.info("EMAIL-IMPORT run complete: %s", counts)
    return summary


def _move_to_review(path: str, cfg: EmailImportConfig) -> None:
    try:
        os.makedirs(cfg.review_dir, exist_ok=True)
        dest = os.path.join(cfg.review_dir, os.path.basename(path))
        if os.path.abspath(dest) != os.path.abspath(path):
            os.replace(path, dest)
    except Exception:
        logger.warning("Email import: could not move %s to review dir", path)


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


def make_shim_request(executor: Any = None) -> Any:
    """Minimal request-like object for background/scheduler runs (upload_resume
    only reads request.app.state.executor)."""
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(executor=executor)))
