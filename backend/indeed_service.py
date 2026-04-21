import logging
import os
import re
from urllib.parse import parse_qs, unquote, urlparse
from dataclasses import dataclass
from typing import Optional

import requests


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndeedConfig:
    """
    Indeed download automation relies on an authenticated browser context.

    Provide a Playwright storage state JSON (cookies/localStorage) created after logging into Indeed.
    """

    storage_state_path: str
    headless: bool = True


def load_indeed_config_from_env() -> IndeedConfig:
    storage = (os.getenv("INDEED_STORAGE_STATE") or "").strip().strip('"')
    if not storage:
        raise ValueError(
            "Missing INDEED_STORAGE_STATE. Provide a Playwright storage state JSON path "
            "with an authenticated Indeed session."
        )
    if not os.path.isfile(storage):
        raise ValueError(f"INDEED_STORAGE_STATE not found: {storage}")
    headless = (os.getenv("INDEED_HEADLESS") or "1").strip().lower() in {"1", "true", "yes"}
    return IndeedConfig(storage_state_path=storage, headless=headless)


def _safe_filename(name: str) -> str:
    s = (name or "").strip()
    s = s.replace("\x00", "")
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("..", ".")
    return (s[:180] or "resume.pdf")


def _unwrap_indeed_continue_url(url: str) -> str:
    """
    Indeed application emails often use a wrapper URL like:
      https://account.indeed.com/o/myaccess/switch/confirm?...&continue=<encoded>
    The real resume/application page is in the `continue` query param.
    """
    try:
        u = (url or "").strip()
        if not u:
            return u
        parsed = urlparse(u)
        qs = parse_qs(parsed.query or "")
        cont = qs.get("continue", [])
        if cont:
            return unquote(cont[0])
        return u
    except Exception:
        return (url or "").strip()


async def _raise_if_indeed_auth_or_blocked(page) -> None:
    """
    Indeed commonly redirects automated browsers to a login or 'Blocked' page.
    Detect that early and raise a helpful error so the caller can switch to cookie mode.
    """
    try:
        url = getattr(page, "url", "") or ""
    except Exception:
        url = ""
    title = ""
    try:
        title = await page.title()
    except Exception:
        title = ""

    low_url = url.lower()
    low_title = (title or "").lower()
    if "secure.indeed.com/auth" in low_url or "/auth" in low_url:
        raise PermissionError(
            f"Indeed requires employer login to access this resume page. url={url!r} title={title!r}. "
            "Create INDEED_STORAGE_STATE (one-time) and retry, and if still blocked set INDEED_HEADLESS=0."
        )
    if "blocked" in low_title:
        raise PermissionError(
            f"Indeed blocked automated access. url={url!r} title={title!r}. "
            "Try INDEED_HEADLESS=0 (headed mode) and use INDEED_STORAGE_STATE."
        )


def download_public_resume_from_url(*, url: str, save_dir: str, suggested_filename: str = "resume.pdf") -> str:
    """
    Download a resume from a publicly accessible URL (no login/cookies required).
    Follows redirects and saves PDF/DOCX based on content-type or URL extension.
    """
    if not url:
        raise ValueError("url is required")
    os.makedirs(save_dir, exist_ok=True)

    r = requests.get(
        url,
        allow_redirects=True,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (ResumeAnalyzer)"},
    )
    r.raise_for_status()

    ctype = (r.headers.get("content-type") or "").lower()
    cd = r.headers.get("content-disposition") or ""

    ext = ""
    if "application/pdf" in ctype:
        ext = ".pdf"
    elif "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in ctype:
        ext = ".docx"
    else:
        # fallback to URL extension
        m = re.search(r"(?i)\.(pdf|docx)(?:$|\\?)", r.url)
        if m:
            ext = f".{m.group(1).lower()}"

    if ext not in {".pdf", ".docx"}:
        # If HTML, try to discover a direct download link from the page.
        if "text/html" in ctype:
            html = r.text or ""
            # Try to find direct links to PDF/DOCX.
            for m3 in re.finditer(r'href=\"(https?://[^\"]+)\"', html, flags=re.I):
                u2 = m3.group(1)
                if any(x in u2.lower() for x in ("download", "resume")) and re.search(r"(?i)\\.(pdf|docx)(?:$|\\?)", u2):
                    return download_public_resume_from_url(url=u2, save_dir=save_dir, suggested_filename=suggested_filename)
        raise RuntimeError(f"URL did not return a PDF/DOCX. content-type={ctype!r} final_url={r.url!r}")

    # Try filename from Content-Disposition
    fname = ""
    m2 = re.search(r'filename\\*=UTF-8\\x27\\x27([^;]+)|filename=\"?([^\";]+)\"?', cd, flags=re.I)
    if m2:
        fname = (m2.group(1) or m2.group(2) or "").strip()
    if not fname:
        fname = suggested_filename
    if not fname.lower().endswith(ext):
        fname = os.path.splitext(fname)[0] + ext
    fname = _safe_filename(fname)

    out_path = os.path.join(save_dir, fname)
    if os.path.exists(out_path):
        base, ext2 = os.path.splitext(fname)
        for i in range(2, 200):
            cand = os.path.join(save_dir, f"{base} ({i}){ext2}")
            if not os.path.exists(cand):
                out_path = cand
                fname = os.path.basename(cand)
                break

    with open(out_path, "wb") as f:
        f.write(r.content)
    logger.info("Public resume downloaded to %s", out_path)
    return fname


def download_indeed_resume_from_view_url(
    *,
    view_url: str,
    save_dir: str,
    cfg: IndeedConfig,
    suggested_filename: Optional[str] = None,
) -> str:
    """
    Open an Indeed "View resume" URL and click the "Download" button.
    Returns the saved filename (basename).
    """
    if not view_url or "indeed" not in view_url.lower():
        raise ValueError("view_url must be an Indeed URL")
    os.makedirs(save_dir, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Playwright is not installed. Run `pip install playwright` and `python -m playwright install chromium`."
        ) from exc

    if not os.path.isfile(cfg.storage_state_path):
        raise FileNotFoundError(f"INDEED_STORAGE_STATE not found: {cfg.storage_state_path}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=cfg.headless)
        context = browser.new_context(accept_downloads=True, storage_state=cfg.storage_state_path)
        page = context.new_page()
        try:
            page.goto(view_url, wait_until="networkidle", timeout=60000)

            # The resume page in your screenshot has a "Download" button.
            # Try a few robust selectors.
            download_btn = (
                page.get_by_role("button", name=re.compile(r"^download$", re.I))
                .or_(page.get_by_role("link", name=re.compile(r"^download$", re.I)))
            )

            # If not visible, try any element containing "Download"
            if download_btn.count() == 0:
                download_btn = page.locator("text=/\\bDownload\\b/i")

            with page.expect_download(timeout=60000) as dl_info:
                download_btn.first.click()

            dl = dl_info.value
            fname = suggested_filename or dl.suggested_filename or "resume.pdf"
            fname = _safe_filename(fname)
            out_path = os.path.join(save_dir, fname)

            # Avoid overwrite
            if os.path.exists(out_path):
                base, ext = os.path.splitext(fname)
                for i in range(2, 200):
                    cand = os.path.join(save_dir, f"{base} ({i}){ext}")
                    if not os.path.exists(cand):
                        out_path = cand
                        fname = os.path.basename(cand)
                        break

            dl.save_as(out_path)
            logger.info("Indeed resume downloaded to %s", out_path)
            return fname
        finally:
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass


async def download_resume_via_playwright_public_async(
    *,
    view_url: str,
    save_dir: str,
    headless: bool = True,
    suggested_filename: str = "resume.pdf",
) -> str:
    """
    Download a resume from an Indeed resume page that is accessible WITHOUT login.
    Uses Playwright as a real browser (avoids 403 seen with plain HTTP requests).
    """
    if not view_url:
        raise ValueError("view_url is required")
    os.makedirs(save_dir, exist_ok=True)

    try:
        from playwright.async_api import async_playwright
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Playwright is not installed. Run `pip install playwright` and `python -m playwright install chromium`."
        ) from exc

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            accept_downloads=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        )
        page = await context.new_page()
        try:
            target = _unwrap_indeed_continue_url(view_url)
            await page.goto(target, wait_until="domcontentloaded", timeout=60000)
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            await _raise_if_indeed_auth_or_blocked(page)

            download_btn = page.get_by_role("button", name=re.compile(r"download", re.I))
            if await download_btn.count() == 0:
                download_btn = page.get_by_role("link", name=re.compile(r"download", re.I))
            if await download_btn.count() == 0:
                download_btn = page.locator("button:has-text('Download'), a:has-text('Download')")
            if await download_btn.count() == 0:
                download_btn = page.locator("text=/\\bDownload\\b/i")

            if await download_btn.count() == 0:
                title = ""
                try:
                    title = await page.title()
                except Exception:
                    pass
                raise RuntimeError(f"Download button not found. url={page.url!r} title={title!r}")

            async with page.expect_download(timeout=90000) as dl_info:
                await download_btn.first.click(timeout=60000)
            dl = await dl_info.value

            fname = dl.suggested_filename or suggested_filename
            fname = _safe_filename(fname)
            out_path = os.path.join(save_dir, fname)
            if os.path.exists(out_path):
                base, ext = os.path.splitext(fname)
                for i in range(2, 200):
                    cand = os.path.join(save_dir, f"{base} ({i}){ext}")
                    if not os.path.exists(cand):
                        out_path = cand
                        fname = os.path.basename(cand)
                        break
            await dl.save_as(out_path)
            logger.info("Playwright public resume downloaded to %s", out_path)
            return fname
        finally:
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass


async def download_indeed_resume_from_view_url_async(
    *,
    view_url: str,
    save_dir: str,
    storage_state_path: str,
    headless: bool = True,
    suggested_filename: Optional[str] = None,
) -> str:
    """
    Async version: Open an Indeed "View resume" URL and click "Download" using an authenticated session.
    """
    if not view_url or "indeed" not in view_url.lower():
        raise ValueError("view_url must be an Indeed URL")
    if not storage_state_path or not os.path.isfile(storage_state_path):
        raise FileNotFoundError(f"INDEED_STORAGE_STATE not found: {storage_state_path}")
    os.makedirs(save_dir, exist_ok=True)

    try:
        from playwright.async_api import async_playwright
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Playwright is not installed. Run `pip install playwright` and `python -m playwright install chromium`."
        ) from exc

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            accept_downloads=True,
            storage_state=storage_state_path,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        )
        page = await context.new_page()
        try:
            target = _unwrap_indeed_continue_url(view_url)
            await page.goto(target, wait_until="domcontentloaded", timeout=60000)
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            await _raise_if_indeed_auth_or_blocked(page)

            download_btn = page.get_by_role("button", name=re.compile(r"download", re.I))
            if await download_btn.count() == 0:
                download_btn = page.get_by_role("link", name=re.compile(r"download", re.I))
            if await download_btn.count() == 0:
                download_btn = page.locator("button:has-text('Download'), a:has-text('Download')")
            if await download_btn.count() == 0:
                download_btn = page.locator("text=/\\bDownload\\b/i")

            if await download_btn.count() == 0:
                title = ""
                try:
                    title = await page.title()
                except Exception:
                    pass
                raise RuntimeError(f"Download button not found. url={page.url!r} title={title!r}")

            async with page.expect_download(timeout=90000) as dl_info:
                await download_btn.first.click(timeout=60000)
            dl = await dl_info.value
            fname = suggested_filename or dl.suggested_filename or "resume.pdf"
            fname = _safe_filename(fname)
            out_path = os.path.join(save_dir, fname)
            if os.path.exists(out_path):
                base, ext = os.path.splitext(fname)
                for i in range(2, 200):
                    cand = os.path.join(save_dir, f"{base} ({i}){ext}")
                    if not os.path.exists(cand):
                        out_path = cand
                        fname = os.path.basename(cand)
                        break
            await dl.save_as(out_path)
            logger.info("Indeed resume downloaded to %s", out_path)
            return fname
        finally:
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass

