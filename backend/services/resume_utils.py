"""Pure resume-domain helpers shared by API, imports, and tests."""

import json
import re
from collections.abc import Iterable


def normalize_email(email: str | None) -> str:
    return (email or "").strip().lower()


def normalize_phone(phone: str | None) -> str:
    digits = re.sub(r"\D+", "", phone or "")
    for country_code in ("91", "1"):
        if len(digits) > 10 and digits.startswith(country_code):
            return digits[len(country_code):]
    return digits


def normalize_name(name: str | None) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def parse_text_list(value: object) -> list[str]:
    """Parse JSON-list, comma-separated, newline-separated, or iterable input."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in re.split(r"[\n,]", text) if part.strip()]


def dump_text_list(values: Iterable[object] | str | None) -> str | None:
    """Serialize a de-duplicated list for legacy text columns."""
    if values is None:
        return None
    items = parse_text_list(values)
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return json.dumps(unique, ensure_ascii=False) if unique else None


def sanitize_embedding_text(value: str | None) -> str:
    text = (value or "").replace("\r", "\n")
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        symbols = sum(not (char.isalnum() or char.isspace()) for char in stripped)
        if len(stripped) > 12 and symbols / max(1, len(stripped)) > 0.45:
            continue
        lines.append(stripped)
    return re.sub(r"\s+", " ", "\n".join(lines)).strip()
