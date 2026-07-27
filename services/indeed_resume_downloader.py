"""
Indeed email → resume link → Playwright download → local file.

Constraints:
- Do NOT use requests/HTTP for Indeed resume downloads.
- MUST use Playwright (Chromium) with a logged-in storage state.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger(__name__)


INDEED_ACCOUNT_BLOCKLIST = ("account.indeed.com",)
INDEED_PREFERRED_SUBSTRINGS = ("employers.indeed.com/candidates",)
INDEED_REQUIRED_SUBSTRING = "employers.indeed.com/candidates/resume"

# Shared browser fingerprint — MUST be identical between the login-setup script
# and the downloader. Cloudflare binds its `cf_clearance` cookie to the exact
# user-agent (and IP); if the login captures clearance under one UA and the
# downloader replays it under another, Cloudflare rejects it and we get blocked
# despite being logged in. Keep these in ONE place so they can never drift.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
BROWSER_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
]


def browser_context_kwargs(accept_downloads: bool = True) -> dict:
    """Context settings shared by the login setup and the downloader."""
    return {
        "accept_downloads": accept_downloads,
        "locale": "en-US",
        "timezone_id": "Asia/Kolkata",
        "viewport": {"width": 1365, "height": 768},
        "user_agent": BROWSER_USER_AGENT,
    }


def normalize_indeed_resume_url(url: str) -> Optional[str]:
    """
    Normalize extracted Indeed URLs into a strict resume endpoint.

    Rules:
    1) If URL contains account.indeed.com AND continue= → decode continue param.
    2) If decoded URL contains /candidates/view → convert to /candidates/resume.
    3) Return ONLY if final URL contains employers.indeed.com/candidates/resume, else None.

    Logging:
    - original_url
    - decoded_url
    - final_url
    """
    original = (url or "").strip()
    if not original:
        return None

    # If the URL was copied from JSON/PowerShell, it may contain escape sequences.
    # Example: "\u0026" for "&" or a trailing "\u0027"/"'" quote.
    original = original.replace("\\u0026", "&").replace("\\u0027", "'")
    original = original.strip().strip('"').strip()
    if original.endswith("'"):
        original = original[:-1].rstrip()

    decoded = original
    try:
        low = original.lower()
        if ("account.indeed.com" in low) and ("continue=" in low):
            parsed = urlparse(original)
            qs = parse_qs(parsed.query or "")
            cont = (qs.get("continue") or [None])[0]
            if cont:
                decoded = unquote(cont).strip()
    except Exception:
        decoded = original

    final = decoded.strip()
    try:
        # Ensure correct endpoint
        final = re.sub(r"/candidates/view(\b|/|\?)", r"/candidates/resume\1", final, flags=re.I)
    except Exception:
        pass

    logger.info("Indeed URL normalize: original_url=%r", (original[:500] if original else ""))
    logger.info("Indeed URL normalize: decoded_url=%r", (decoded[:500] if decoded else ""))
    logger.info("Indeed URL normalize: final_url=%r", (final[:500] if final else ""))

    if INDEED_REQUIRED_SUBSTRING in (final or "").lower():
        return final
    return None


def extract_real_url(url: str) -> str:
    """
    If Indeed link contains `switch/confirm`, unwrap the real destination from `continue=`.
    Otherwise return the cleaned URL.
    """
    u = (url or "").strip()
    if not u:
        return ""
    try:
        if "switch/confirm" not in u.lower():
            return u
        parsed = urlparse(u)
        qs = parse_qs(parsed.query or "")
        cont = (qs.get("continue") or [None])[0]
        return (unquote(cont) if cont else u).strip()
    except Exception:
        return u


def _is_blocked_url(url: str) -> bool:
    low = (url or "").lower()
    return any(bad in low for bad in INDEED_ACCOUNT_BLOCKLIST)


def _score_url(url: str) -> int:
    """
    Prefer employer candidate URLs. Lower score = better.
    """
    low = (url or "").lower()
    if any(p in low for p in INDEED_PREFERRED_SUBSTRINGS):
        return 0
    if "indeed" in low:
        return 10
    return 100


def choose_best_resume_url(urls: Iterable[str]) -> Optional[str]:
    """
    Filters and chooses the best candidate resume URL.
    - normalize all extracted URLs first (raw account.indeed.com is never used)
    - prefer employers.indeed.com/candidates links
    """
    cleaned: list[str] = []
    for u in urls or []:
        norm = normalize_indeed_resume_url(str(u))
        if not norm:
            continue
        cleaned.append(norm)
    if not cleaned:
        return None
    cleaned.sort(key=_score_url)
    return cleaned[0]


def extract_urls_from_email_html(html: str) -> list[str]:
    """
    Extract ALL URLs from email HTML.
    - Primarily collects <a href>.
    - Also collects visible URLs in the HTML text as a fallback.
    """
    if not html:
        return []
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("beautifulsoup4 is required. Install: pip install beautifulsoup4") from exc

    urls: list[str] = []
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a"):
        try:
            href = (a.get("href") or "").strip()
            if href:
                urls.append(href)
        except Exception:
            continue

    # Fallback: raw URL scan across HTML
    for m in re.finditer(r"(https?://[^\s<>\"']+)", html, flags=re.I):
        urls.append(m.group(1).strip().rstrip(").,;"))

    # De-dupe, keep order
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        u = (u or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def extract_urls_from_email_message(msg) -> list[str]:
    """
    Extract ALL URLs from HTML parts of an email.message.Message.
    """
    urls: list[str] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = (part.get_content_type() or "").lower().strip()
        if ctype != "text/html":
            continue
        try:
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            html = payload.decode(charset, errors="ignore")
        except Exception:
            continue
        urls.extend(extract_urls_from_email_html(html))
    return urls


@dataclass(frozen=True)
class PlaywrightDownloadResult:
    ok: bool
    file_path: str
    mode: str
    url: str
    error: str = ""
    html_debug_path: str = ""
    screenshot_debug_path: str = ""


def _safe_basename(name: str) -> str:
    s = (name or "").strip()
    s = s.replace("\x00", "")
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:180] or "resume"


async def _save_html_debug(*, page, save_dir: str, prefix: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    try:
        html = await page.content()
    except Exception:
        html = ""
    fname = _safe_basename(f"{prefix}_{uuid.uuid4().hex[:10]}_page.html")
    out = os.path.join(save_dir, fname)
    with open(out, "w", encoding="utf-8", errors="ignore") as f:
        f.write(html or "")
    return out


async def _save_screenshot_debug(*, page, save_dir: str, prefix: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    fname = _safe_basename(f"{prefix}_{uuid.uuid4().hex[:10]}_page.png")
    out = os.path.join(save_dir, fname)
    try:
        await page.screenshot(path=out, full_page=True)
        return out
    except Exception:
        return ""


async def _looks_blocked(*, page) -> bool:
    """
    Indeed often serves an anti-bot page with title containing 'Blocked'.
    """
    try:
        title = (await page.title()) or ""
    except Exception:
        title = ""
    low = title.lower()
    if "blocked" in low:
        return True
    if "access denied" in low:
        return True
    if "captcha" in low:
        return True
    try:
        url = getattr(page, "url", "") or ""
    except Exception:
        url = ""
    if "blocked" in (url or "").lower():
        return True
    if "captcha" in (url or "").lower():
        return True

    # Body heuristic (Cloudflare/Indeed WAF pages)
    try:
        html = await page.content()
    except Exception:
        html = ""
    low_html = (html or "").lower()
    if "request blocked" in low_html or "you have been blocked" in low_html:
        return True
    if "cloudflare" in low_html and ("waf_block" in low_html or "cf-box-container" in low_html):
        return True
    return False


async def download_resume_with_playwright(
    *,
    resume_url: str,
    save_dir: str,
    storage_state_path: str = "",
    headless: bool = True,
    timeout_ms: int = 90_000,
    prefix: str = "indeed",
) -> PlaywrightDownloadResult:
    """
    Mandatory browser automation:
    - Launch Chromium
    - Use storage_state.json for logged-in session (optional)
    - Visit resume URL
    - Click download ("Download" / "Download Resume")
    - Save to /uploads/resumes/{unique_id}.pdf (or suggested file extension)
    """
    # IMPORTANT: downloader receives ONLY normalized resume URLs.
    url = normalize_indeed_resume_url(resume_url) or ""
    if not url:
        return PlaywrightDownloadResult(ok=False, file_path="", mode="playwright", url=resume_url, error="empty_url")
    os.makedirs(save_dir, exist_ok=True)

    try:
        from playwright.async_api import async_playwright
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Playwright is not installed. Run `pip install playwright` and `python -m playwright install chromium`."
        ) from exc

    storage_state_path = (storage_state_path or "").strip().strip('"')
    storage_state_ok = bool(storage_state_path and os.path.isfile(storage_state_path))

    async def attempt(*, headless_mode: bool, use_storage_state: bool) -> PlaywrightDownloadResult:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=headless_mode,
                args=list(BROWSER_LAUNCH_ARGS),
            )
            context_kwargs = browser_context_kwargs(accept_downloads=True)
            if use_storage_state and storage_state_ok:
                context_kwargs["storage_state"] = storage_state_path
            context = await browser.new_context(**context_kwargs)
            page = await context.new_page()

            # ── Direct-download capture (Bonus) ────────────────────────────
            # Prefer a direct file response over clicking the button. Indeed may
            # stream the resume as a PDF/DOCX response (content-type or
            # Content-Disposition: attachment) during page load. Capturing it
            # avoids depending on the button's DOM and is faster/more reliable.
            # Uses the SAME authenticated context, so no separate auth needed.
            captured: dict = {"bytes": None, "ext": ""}

            async def _on_response(resp):
                if captured["bytes"] is not None:
                    return
                try:
                    headers = await resp.all_headers()
                except Exception:
                    return
                ctype = (headers.get("content-type") or "").lower()
                disp = (headers.get("content-disposition") or "").lower()
                ext = ""
                if "application/pdf" in ctype or ".pdf" in disp:
                    ext = ".pdf"
                elif "wordprocessingml" in ctype or "msword" in ctype or ".docx" in disp:
                    ext = ".docx"
                elif "attachment" in disp:
                    ext = ".pdf"
                if not ext:
                    return
                try:
                    body = await resp.body()
                except Exception:
                    return
                if body and len(body) > 0:
                    captured["bytes"] = body
                    captured["ext"] = ext
                    logger.info("Playwright: captured direct %s response (%d bytes) from %s",
                                ext, len(body), (resp.url or "")[:120])

            page.on("response", lambda r: asyncio.ensure_future(_on_response(r)))

            async def _save_captured() -> str:
                fname = _safe_basename(f"{uuid.uuid4().hex}{captured['ext']}")
                out_path = os.path.join(save_dir, fname)
                with open(out_path, "wb") as f:
                    f.write(captured["bytes"])
                logger.info("Playwright: saved direct-download resume -> %s", out_path)
                return out_path

            try:
                # Reduce basic bot signals (best-effort; not a full stealth plugin).
                try:
                    await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
                except Exception:
                    pass

                logger.info(
                    "Playwright: goto url=%r headless=%s storage_state=%s",
                    url[:200],
                    bool(headless_mode),
                    "on" if (use_storage_state and storage_state_ok) else "off",
                )
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    await page.wait_for_load_state("networkidle", timeout=20_000)
                except Exception:
                    pass

                # Small human-ish interaction: scroll to trigger lazy UI.
                try:
                    await page.mouse.move(200, 180)
                    await page.mouse.wheel(0, 600)
                    await page.wait_for_timeout(350)
                except Exception:
                    pass

                # BONUS fast-path: if the page already streamed the resume file
                # directly (no click needed), save it and return immediately.
                if captured["bytes"] is not None:
                    out_path = await _save_captured()
                    return PlaywrightDownloadResult(ok=True, file_path=out_path, mode="direct", url=page.url or url)

                if await _looks_blocked(page=page):
                    html_path = await _save_html_debug(page=page, save_dir=save_dir, prefix=prefix)
                    png_path = await _save_screenshot_debug(page=page, save_dir=save_dir, prefix=prefix)
                    t = ""
                    try:
                        t = await page.title()
                    except Exception:
                        t = ""
                    return PlaywrightDownloadResult(
                        ok=False,
                        file_path="",
                        mode="playwright",
                        url=getattr(page, "url", "") or url,
                        error=f"Blocked by Indeed. title={t!r}",
                        html_debug_path=html_path,
                        screenshot_debug_path=png_path,
                    )

                # Flexible selectors (including menu / overflow cases)
                download_btn = page.get_by_role("button", name=re.compile(r"download( resume)?", re.I))
                if await download_btn.count() == 0:
                    download_btn = page.get_by_role("link", name=re.compile(r"download( resume)?", re.I))
                if await download_btn.count() == 0:
                    download_btn = page.locator("button:has-text('Download'), a:has-text('Download')")
                if await download_btn.count() == 0:
                    # Sometimes hidden under an actions menu
                    menu = page.get_by_role("button", name=re.compile(r"(more|actions|options)", re.I))
                    if await menu.count() > 0:
                        try:
                            await menu.first.click(timeout=5_000)
                            await page.wait_for_timeout(250)
                        except Exception:
                            pass
                    download_btn = page.locator("text=/\\bDownload\\b/i")

                if await download_btn.count() == 0:
                    title = ""
                    try:
                        title = await page.title()
                    except Exception:
                        pass
                    html_path = await _save_html_debug(page=page, save_dir=save_dir, prefix=prefix)
                    png_path = await _save_screenshot_debug(page=page, save_dir=save_dir, prefix=prefix)
                    err = f"Download button not found. url={page.url!r} title={title!r}"
                    return PlaywrightDownloadResult(
                        ok=False,
                        file_path="",
                        mode="playwright",
                        url=page.url or url,
                        error=err,
                        html_debug_path=html_path,
                        screenshot_debug_path=png_path,
                    )

                try:
                    async with page.expect_download(timeout=timeout_ms) as dl_info:
                        await download_btn.first.click(timeout=timeout_ms)
                    dl = await dl_info.value
                except Exception:
                    # The click may have triggered an in-page fetch instead of a
                    # browser download — fall back to the captured direct response.
                    await page.wait_for_timeout(1500)
                    if captured["bytes"] is not None:
                        out_path = await _save_captured()
                        return PlaywrightDownloadResult(ok=True, file_path=out_path, mode="direct", url=page.url or url)
                    raise

                suggested = (dl.suggested_filename or "").strip()
                ext = os.path.splitext(suggested)[1].lower() if suggested else ""
                if ext not in {".pdf", ".docx"}:
                    ext = ".pdf"

                fname = _safe_basename(f"{uuid.uuid4().hex}{ext}")
                out_path = os.path.join(save_dir, fname)
                await dl.save_as(out_path)
                logger.info("Playwright: download saved -> %s", out_path)
                return PlaywrightDownloadResult(ok=True, file_path=out_path, mode="playwright", url=page.url or url)
            except Exception as exc:
                html_path = ""
                png_path = ""
                try:
                    html_path = await _save_html_debug(page=page, save_dir=save_dir, prefix=prefix)
                except Exception:
                    pass
                try:
                    png_path = await _save_screenshot_debug(page=page, save_dir=save_dir, prefix=prefix)
                except Exception:
                    pass
                return PlaywrightDownloadResult(
                    ok=False,
                    file_path="",
                    mode="playwright",
                    url=getattr(page, "url", "") or url,
                    error=str(exc),
                    html_debug_path=html_path,
                    screenshot_debug_path=png_path,
                )
            finally:
                try:
                    await context.close()
                except Exception:
                    pass
                try:
                    await browser.close()
                except Exception:
                    pass

    # Retry strategy, best shot first. Cloudflare + Indeed's employer login are
    # both far more permissive to a HEADED browser carrying the logged-in session
    # (which also holds the cf_clearance cookie captured during login), so try
    # that first when a storage state exists; fall back through weaker combos.
    attempts: list[tuple[bool, bool]] = []
    if storage_state_ok:
        attempts.append((False, True))   # headed + login  (best)
        attempts.append((True, True))    # headless + login
    attempts.append((False, False))      # headed, no login
    attempts.append((headless, False))   # as-requested, no login

    last: Optional[PlaywrightDownloadResult] = None
    seen: set = set()
    for headless_mode, use_ss in attempts:
        key = (headless_mode, use_ss)
        if key in seen:
            continue
        seen.add(key)
        res = await attempt(headless_mode=headless_mode, use_storage_state=use_ss)
        if res.ok:
            return res
        # Keep the most informative failure (one with debug artifacts).
        if last is None or (res.html_debug_path or res.screenshot_debug_path):
            last = res
    return last or PlaywrightDownloadResult(
        ok=False, file_path="", mode="playwright", url=url, error="all_attempts_failed"
    )


def _load_storage_state_path() -> str:
    # Prefer explicit env; fallback to conventional repo root filename.
    p = (os.getenv("INDEED_STORAGE_STATE") or "").strip().strip('"')
    if p:
        return p
    return os.path.join(os.getcwd(), "storage_state.json")


async def process_indeed_email(
    *,
    uid: str,
    subject: str,
    sender: str,
    msg,
    save_dir: str,
    headless: bool,
    timeout_ms: int,
) -> dict:
    """
    Process a single Indeed email message:
    - extract URLs from HTML
    - choose best resume URL
    - download via Playwright using storage_state.json
    """
    logger.info("Indeed email: uid=%s subject=%r", uid, subject)
    urls = extract_urls_from_email_message(msg)
    chosen = choose_best_resume_url(urls)
    logger.info("Indeed email: uid=%s extracted_urls=%d chosen=%r", uid, len(urls), chosen)
    if not chosen:
        return {
            "uid": uid,
            "subject": subject,
            "from": sender,
            "ok": False,
            "stage": "link_extract",
            "error": "no_suitable_resume_url",
            "urls": urls,
        }

    # Optional: if present, it improves reliability on employer-only pages.
    storage_state_path = _load_storage_state_path()
    dl = await download_resume_with_playwright(
        resume_url=chosen,
        save_dir=save_dir,
        storage_state_path=storage_state_path if os.path.isfile(storage_state_path) else "",
        headless=headless,
        timeout_ms=timeout_ms,
        prefix=f"uid{uid}" if uid else "uidunknown",
    )
    return {
        "uid": uid,
        "subject": subject,
        "from": sender,
        "urls": urls,
        "chosen_url": chosen,
        "download": {
            "ok": dl.ok,
            "mode": dl.mode,
            "url": dl.url,
            "file_path": dl.file_path,
            "error": dl.error,
            "html_debug_path": dl.html_debug_path,
        },
    }

