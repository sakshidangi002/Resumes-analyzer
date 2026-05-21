from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from fastapi import Request, UploadFile
from sqlalchemy.orm import Session

try:
    from backend.email_service import fetch_resumes_via_imap, load_imap_config_from_env
    from services.indeed_resume_downloader import (
        choose_best_resume_url,
        download_resume_with_playwright,
        extract_urls_from_email_message,
    )
except ImportError:  # pragma: no cover
    from email_service import fetch_resumes_via_imap, load_imap_config_from_env  # type: ignore
    from services.indeed_resume_downloader import (  # type: ignore
        choose_best_resume_url,
        download_resume_with_playwright,
        extract_urls_from_email_message,
    )

try:
    from email import policy
    from email.parser import BytesParser

    import imaplib
    import ssl
except Exception:  # pragma: no cover
    imaplib = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndeedFetchOptions:
    save_dir: str
    processed_uids_path: str
    ignore_processed: bool = False
    headless: bool = True
    playwright_timeout_ms: int = 90_000
    unread_only: bool = True
    mark_seen: bool = False
    max_to_process: int = 50


def _fetch_rfc822_bytes(cfg, uid: str) -> bytes:
    if imaplib is None:
        raise RuntimeError("imaplib not available")
    context = ssl.create_default_context()
    client = imaplib.IMAP4_SSL(cfg.host, cfg.port, ssl_context=context) if cfg.use_ssl else imaplib.IMAP4(cfg.host, cfg.port)
    try:
        client.login(cfg.username, cfg.password)
        status, _ = client.select(cfg.mailbox, readonly=True)
        if (status or "").upper() != "OK":
            raise RuntimeError(f"Could not select mailbox: {cfg.mailbox}")
        status, msg_data = client.uid("FETCH", uid, "(RFC822)")
        if (status or "").upper() != "OK" or not msg_data or not msg_data[0]:
            raise RuntimeError("IMAP FETCH failed")
        return msg_data[0][1]
    finally:
        try:
            client.logout()
        except Exception:
            pass


def shape_candidate_record(resume_dict: dict) -> dict[str, Any]:
    """Narrow DB/API resume dict to the pipeline API shape."""
    skills = resume_dict.get("skills") or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    exp_summary = (resume_dict.get("experience_summary") or "").strip()
    years = resume_dict.get("experience_years")
    experience = exp_summary
    if not experience and years is not None:
        experience = f"{years} years"
    return {
        "name": resume_dict.get("name") or "",
        "email": resume_dict.get("email") or "",
        "phone": resume_dict.get("phone") or "",
        "skills": skills,
        "experience": experience,
    }


def enrich_upload_with_candidate(db: Session, upload_out: dict) -> Optional[dict[str, Any]]:
    """Attach normalized candidate fields using the persisted resume row (success or duplicate)."""
    if not isinstance(upload_out, dict):
        return None
    rid = upload_out.get("resume_id") or upload_out.get("existing_id")
    status = upload_out.get("status")
    if status not in {"success", "duplicate"} or not rid:
        return None
    try:
        from backend.api import ResumeDB, _resume_to_dict
    except ImportError:
        from api import ResumeDB, _resume_to_dict  # type: ignore
    try:
        uid = UUID(str(rid))
    except Exception:
        return None
    row = db.query(ResumeDB).filter(ResumeDB.id == uid).first()
    if not row:
        return None
    return shape_candidate_record(_resume_to_dict(row))


async def process_resume_file_with_existing_pipeline(
    *,
    request: Request,
    db: Session,
    file_path: str,
    upload_resume_callable: Callable[..., Awaitable[list]],
) -> dict:
    if not file_path or not os.path.isfile(file_path):
        return {"status": "error", "message": "File not found", "file": file_path}
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in {".pdf", ".docx"}:
        return {"status": "skipped", "message": f"Not a resume file: {ext}", "file": file_path}

    logger.info("process_resume: ingesting %s", file_path)
    with open(file_path, "rb") as fh:
        data = fh.read()
    uf = UploadFile(filename=os.path.basename(file_path), file=__import__("io").BytesIO(data))
    out = await upload_resume_callable(request=request, files=[uf], db=db)
    first = out[0] if out else {}
    result = first if isinstance(first, dict) else {"status": "error", "message": "Unexpected upload response", "file": file_path}
    cand = enrich_upload_with_candidate(db, result)
    if cand:
        result["candidate"] = cand
    return result


async def process_resume(
    *,
    request: Request,
    db: Session,
    file_path: str,
    upload_resume_callable: Callable[..., Awaitable[list]],
) -> dict:
    """Alias for :func:`process_resume_file_with_existing_pipeline` (calls existing analyzer via upload)."""
    return await process_resume_file_with_existing_pipeline(
        request=request,
        db=db,
        file_path=file_path,
        upload_resume_callable=upload_resume_callable,
    )


async def fetch_and_process_indeed_emails(
    *,
    request: Request,
    db: Session,
    opts: IndeedFetchOptions,
    auto_extract: bool,
    upload_resume_callable: Callable[..., Awaitable[list]],
) -> dict:
    cfg = load_imap_config_from_env()
    logger.info(
        "Indeed pipeline: starting IMAP fetch save_dir=%r unread_only=%s mark_seen=%s",
        opts.save_dir,
        opts.unread_only,
        opts.mark_seen,
    )
    imap_result = fetch_resumes_via_imap(
        cfg=cfg,
        save_dir=opts.save_dir,
        processed_uids_path=opts.processed_uids_path,
        ignore_processed=bool(opts.ignore_processed),
        unread_only=bool(opts.unread_only),
        mark_seen=bool(opts.mark_seen),
        max_to_process=int(opts.max_to_process),
    )

    results: list[dict] = []

    for item in imap_result.get("downloaded") or []:
        uid = str(item.get("uid") or "")
        subject = str(item.get("subject") or "")
        sender = str(item.get("from") or "")
        prefix = f"uid{uid}" if uid else "uidunknown"

        logger.info("Indeed pipeline: uid=%s subject=%r", uid, subject)

        row: dict[str, Any] = {"uid": uid, "subject": subject, "from": sender, "steps": []}

        saved_attachments = [os.path.join(opts.save_dir, x) for x in (item.get("downloaded_files") or []) if x]
        if saved_attachments:
            row["steps"].append({"stage": "attachments", "files": saved_attachments})
            if auto_extract:
                extracts: list[dict] = []
                for p in saved_attachments:
                    extracts.append(
                        await process_resume_file_with_existing_pipeline(
                            request=request,
                            db=db,
                            file_path=p,
                            upload_resume_callable=upload_resume_callable,
                        )
                    )
                row["extract"] = extracts
                row["extracted_data"] = [e.get("candidate") for e in extracts if isinstance(e, dict)]
            results.append(row)
            continue

        try:
            raw = _fetch_rfc822_bytes(cfg, uid)
            msg = BytesParser(policy=policy.default).parsebytes(raw)
            urls = extract_urls_from_email_message(msg)
        except Exception as exc:
            logger.warning("Could not re-fetch email uid=%s for HTML parsing: %s", uid, exc)
            urls = []

        # Always include URLs already extracted by the IMAP reader (plain text + html heuristics).
        # This fixes cases where HTML is missing/stripped but the reader captured the link.
        urls = list(urls or []) + [str(u) for u in (item.get("indeed_view_urls") or []) if u]

        chosen = choose_best_resume_url(urls)
        row["steps"].append({"stage": "urls", "urls": urls, "chosen": chosen})
        if not chosen:
            row["steps"].append({"stage": "skip", "reason": "no_resume_link"})
            results.append(row)
            continue

        # Playwright-only download. IMPORTANT: downloader receives ONLY normalized resume URLs.
        dl = await download_resume_with_playwright(
            resume_url=chosen,
            save_dir=opts.save_dir,
            headless=bool(opts.headless),
            timeout_ms=int(opts.playwright_timeout_ms),
            prefix=prefix,
        )
        row["download"] = {
            "ok": dl.ok,
            "mode": dl.mode,
            "url": dl.url,
            "file_path": dl.file_path,
            "error": dl.error,
            "html_debug_path": dl.html_debug_path,
            "screenshot_debug_path": getattr(dl, "screenshot_debug_path", ""),
        }
        fpath = str(dl.file_path or "")

        if dl.ok and fpath.lower().endswith((".pdf", ".docx")) and auto_extract:
            row["extract"] = [
                await process_resume_file_with_existing_pipeline(
                    request=request,
                    db=db,
                    file_path=fpath,
                    upload_resume_callable=upload_resume_callable,
                )
            ]
            ex = row["extract"]
            row["extracted_data"] = [ex[0].get("candidate")] if ex and isinstance(ex[0], dict) else []
        results.append(row)

    return {"ok": True, **imap_result, "pipeline": results}


# Backward-compatible no-op alias (older callers imported this symbol).
download_resume_from_link_public_best_effort = None
