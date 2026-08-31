"""`request_analyzer` — turn free text into structured, checkable entities.

Deterministic on purpose. Period parsing is a solved problem in regex and a
1.1B-parameter local model gets it wrong often enough to produce confidently
incorrect attendance figures, which is worse than not answering. The LLM is used
later, for phrasing only.

All "now" reasoning goes through `get_ist_now()`. The HRMS treats IST as the
business timezone everywhere (there is a dedicated test for that contract), and
"last month" resolved in UTC is wrong for 5.5 hours of every day.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from app.assistant.runtime import RunContext
from app.core.datetime_utils import get_ist_now

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_FIRST_PERSON = re.compile(r"\b(my|mine|me|i|i'm|myself)\b", re.I)
_THIRD_PERSON = re.compile(r"\b(his|her|their|hers|theirs|he|she|they)\b", re.I)
_TEAM = re.compile(r"\b(my team|team's|teams|reportees|direct reports|my reports)\b", re.I)

#: "for Priya", "of Priya Sharma", "is Priya", "was Priya"
_NAME_AFTER = re.compile(
    r"\b(?:for|of|about|is|was|does|did)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)
#: "EMP001", "emp-12"
_EMP_CODE = re.compile(r"\b([A-Za-z]{2,5}[-_]?\d{2,6})\b")

#: Granularities the attendance tools cannot serve. Recognised explicitly so the
#: user gets "I can do months" instead of a silently wrong monthly answer.
_UNSUPPORTED_PERIOD = re.compile(
    r"\b(last|this|past|previous)\s+(quarter|week|fortnight|year)\b|"
    r"\b(quarterly|weekly|ytd|year to date)\b",
    re.I,
)


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    index = (year * 12 + (month - 1)) + delta
    return index // 12, index % 12 + 1


def _parse_period(text: str, today) -> dict[str, Any]:
    """Resolve the month/year the question is about.

    Returns `period_assumed=True` when nothing was stated and we defaulted to the
    current month — the response generator says so out loud, so the user can tell
    an assumption from a fact.
    """
    low = text.lower()

    if _UNSUPPORTED_PERIOD.search(low):
        return {"period_supported": False}

    # "last month" / "previous month"
    if re.search(r"\b(last|previous|prev)\s+month\b", low):
        year, month = _shift_month(today.year, today.month, -1)
        return {"month": month, "year": year, "period_label": "last month",
                "period_assumed": False, "period_supported": True}

    if re.search(r"\b(this|current)\s+month\b", low):
        return {"month": today.month, "year": today.year, "period_label": "this month",
                "period_assumed": False, "period_supported": True}

    # "N months ago"
    ago = re.search(r"\b(\d{1,2})\s+months?\s+ago\b", low)
    if ago:
        year, month = _shift_month(today.year, today.month, -int(ago.group(1)))
        return {"month": month, "year": year, "period_label": f"{ago.group(1)} months ago",
                "period_assumed": False, "period_supported": True}

    # "in January", "for March 2025"
    for name, number in _MONTHS.items():
        if re.search(rf"\b{name}\b", low):
            year_match = re.search(rf"\b{name}\b[^\d]{{0,6}}(20\d{{2}})", low)
            if year_match:
                year = int(year_match.group(1))
            else:
                # Bare month name means the most recent occurrence of it, not a
                # future month: "how was my attendance in December" asked in
                # March means last December.
                year = today.year if number <= today.month else today.year - 1
            return {"month": number, "year": year,
                    "period_label": f"{name.capitalize()} {year}",
                    "period_assumed": False, "period_supported": True}

    # "03/2025" or "2025-03"
    numeric = re.search(r"\b(0?[1-9]|1[0-2])[/-](20\d{2})\b", low)
    if numeric:
        return {"month": int(numeric.group(1)), "year": int(numeric.group(2)),
                "period_label": f"{numeric.group(1)}/{numeric.group(2)}",
                "period_assumed": False, "period_supported": True}
    iso = re.search(r"\b(20\d{2})-(0?[1-9]|1[0-2])\b", low)
    if iso:
        return {"month": int(iso.group(2)), "year": int(iso.group(1)),
                "period_label": f"{iso.group(2)}/{iso.group(1)}",
                "period_assumed": False, "period_supported": True}

    return {"month": today.month, "year": today.year, "period_label": "this month",
            "period_assumed": True, "period_supported": True}


def _parse_subject(text: str) -> dict[str, Any]:
    """Who is the question about? Answers 'self' vs 'someone else' vs 'team'.

    This only reads *intent* from the text. It grants nothing: `scope_resolver`
    decides what the actor is actually allowed to see.
    """
    if _TEAM.search(text):
        return {"subject_hint": "team", "target_name": None, "target_code": None}

    code = _EMP_CODE.search(text)
    name = _NAME_AFTER.search(text)

    if code or name:
        return {
            "subject_hint": "other",
            "target_name": name.group(1) if name else None,
            "target_code": code.group(1) if code else None,
        }
    if _THIRD_PERSON.search(text) and not _FIRST_PERSON.search(text):
        return {"subject_hint": "other", "target_name": None, "target_code": None}
    return {"subject_hint": "self", "target_name": None, "target_code": None}


def request_analyzer(state: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
    raw = (state.get("message") or "").strip()
    normalized = re.sub(r"\s+", " ", raw)
    today = get_ist_now().date()

    entities: dict[str, Any] = {}
    entities.update(_parse_period(normalized, today))
    entities.update(_parse_subject(normalized))

    return {
        "normalized": normalized,
        "entities": entities,
        "_trace": {
            "message_len": len(normalized),
            "period": entities.get("period_label"),
            "period_assumed": entities.get("period_assumed"),
            "period_supported": entities.get("period_supported"),
            "subject_hint": entities.get("subject_hint"),
        },
    }
