"""Shared formatting, so every skill states a figure the same way.

Small, but worth centralising: an assistant that writes "Rs 123000" in one
answer and "₹1,23,000" in the next reads as two different systems.
"""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Any, Iterable, Sequence


def to_float(value: Any) -> float:
    """Decimal/None/str -> float, at the boundary where figures leave the ORM."""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def inr(value: Any) -> str:
    """Indian digit grouping: ₹1,23,456, not ₹123,456.

    Grouping is three digits then twos, so `f"{v:,}"` is wrong for this currency.
    """
    amount = round(to_float(value))
    sign = "-" if amount < 0 else ""
    digits = str(abs(amount))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        digits = f"{head},{tail}"
    return f"{sign}₹{digits}"


def days(value: Any) -> str:
    """Half days are real, but 4.0 should still read as "4"."""
    number = to_float(value)
    return str(int(number)) if number == int(number) else f"{number:g}"


def pretty_date(value: date | None) -> str:
    return value.strftime("%d %b %Y") if value else "—"


def join_names(names: Sequence[str], limit: int = 8) -> str:
    """"Priya Sharma, Amit Rao and 3 others"."""
    clean = [n for n in names if n]
    if not clean:
        return ""
    if len(clean) > limit:
        return f"{', '.join(clean[:limit])} and {len(clean) - limit} other(s)"
    if len(clean) == 1:
        return clean[0]
    return f"{', '.join(clean[:-1])} and {clean[-1]}"


def bullet_list(items: Iterable[str], limit: int = 10) -> str:
    """A short vertical list. Chat bubbles render newlines, tables badly."""
    rows = [i for i in items if i]
    shown = rows[:limit]
    text = "\n".join(f"• {row}" for row in shown)
    if len(rows) > limit:
        text += f"\n• …and {len(rows) - limit} more"
    return text


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """"1 day" / "3 days", written once so no skill has to think about it."""
    word = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count} {word}"


def sentence(parts: Iterable[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def capitalize_first(text: str) -> str:
    """Upper-case the first letter and leave the rest alone.

    `str.capitalize()` lower-cases everything after the first character, which
    turns "Priya Sharma's" into "Priya sharma's" — wrong, and wrong about a
    person's name specifically.
    """
    return text[:1].upper() + text[1:] if text else text
