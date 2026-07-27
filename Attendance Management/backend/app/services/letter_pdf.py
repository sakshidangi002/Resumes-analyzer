"""
Render letter HTML to PDF bytes using fpdf2 (pure Python, no native deps).

Used by the letters routes to attach a real PDF copy of the letter to outbound emails.

Notes:
- fpdf2's built-in `write_html()` handles a useful subset of HTML/CSS (headings,
  paragraphs, bold/italic/underline, lists, simple tables, line breaks, images),
  which is good enough for the letter templates in this app.
- If `write_html()` raises (unsupported markup), we fall back to flattening the
  HTML to plain text via BeautifulSoup and emitting that as a multi-line PDF so
  the email still gets *some* attachment.
"""
from __future__ import annotations

from io import BytesIO
import html as _html
import logging
import re

logger = logging.getLogger(__name__)


def _wrap_html_for_pdf(body_html: str) -> str:
    """Wrap the letter body in a minimal styled document for fpdf2."""
    # fpdf2's HTML renderer supports a small CSS subset. We keep this minimal
    # because it's run through that renderer, not a real browser engine.
    return f"""<!DOCTYPE html>
<html>
<body>
{body_html or ''}
</body>
</html>"""


def _html_to_plain_text(body_html: str) -> str:
    """Strip HTML to readable plain text as a last-resort fallback."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(body_html or "", "html.parser")
        for br in soup.find_all("br"):
            br.replace_with("\n")
        text = soup.get_text(separator="\n")
    except Exception:
        text = re.sub(r"<[^>]+>", " ", body_html or "")
        text = _html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def html_to_pdf_bytes(body_html: str, subject: str | None = None) -> bytes:
    """
    Convert an HTML fragment to PDF bytes. Returns b'' on failure; callers should
    handle that gracefully (e.g. send the email without an attachment).
    """
    try:
        from fpdf import FPDF
    except Exception as exc:  # pragma: no cover
        logger.warning("fpdf2 not available: %s", exc)
        return b""

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_font("Helvetica", size=11)

    if subject:
        try:
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.multi_cell(0, 8, subject)
            pdf.set_font("Helvetica", size=11)
            pdf.ln(2)
        except Exception:
            pass

    rendered = False
    try:
        pdf.write_html(_wrap_html_for_pdf(body_html or ""))
        rendered = True
    except Exception as exc:
        logger.warning("fpdf2 HTML render failed (%s); falling back to plain text", exc)

    if not rendered:
        plain = _html_to_plain_text(body_html or "")
        if not plain:
            plain = "(empty letter)"
        try:
            pdf.set_font("Helvetica", size=11)
            for paragraph in plain.split("\n"):
                pdf.multi_cell(0, 6, paragraph or " ")
        except Exception as exc:
            logger.exception("fpdf2 plain-text render failed: %s", exc)
            return b""

    try:
        out = pdf.output(dest="S")
    except Exception as exc:  # pragma: no cover
        logger.exception("fpdf2 output() failed: %s", exc)
        return b""

    # fpdf2 returns either bytes or bytearray depending on version.
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    if isinstance(out, str):
        return out.encode("latin-1", errors="ignore")
    try:
        return bytes(out)
    except Exception:
        return b""


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_pdf_filename(*parts: str | None) -> str:
    """Build a filesystem/email-safe PDF filename from arbitrary string parts."""
    cleaned: list[str] = []
    for p in parts:
        if not p:
            continue
        s = _FILENAME_SAFE.sub("_", str(p).strip())
        s = s.strip("._-")
        if s:
            cleaned.append(s)
    base = "_".join(cleaned) if cleaned else "letter"
    return f"{base[:80]}.pdf"
