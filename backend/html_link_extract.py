from __future__ import annotations

import re
from email.message import Message
from typing import Iterable


RESUME_ISH = re.compile(r"(?i)resume|download|view\s+resume|see\s+resume|application")


def _dedupe(urls: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        u = (u or "").strip()
        if not u:
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def extract_resume_links_from_html(html: str) -> list[str]:
    """
    Requirement: parse HTML with BeautifulSoup and find anchors containing "resume" text.
    """
    if not html:
        return []
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("beautifulsoup4 is required. Install: pip install beautifulsoup4") from exc

    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    for a in soup.find_all("a"):
        try:
            href = (a.get("href") or "").strip()
            text = a.get_text(" ", strip=True) or ""
            if not href:
                continue
            if "indeed" not in href.lower():
                continue
            if not RESUME_ISH.search(text) and not RESUME_ISH.search(href):
                continue
            urls.append(href)
        except Exception:
            continue
    return _dedupe(urls)


def extract_resume_links_from_message(msg: Message) -> list[str]:
    """
    Best-effort: scan HTML parts for resume-ish anchors.
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
        urls.extend(extract_resume_links_from_html(html))
    return _dedupe(urls)
