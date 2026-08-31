"""`intent_router` — decide which HRM capability answers this question.

Deterministic first (Rule 11/12). Weighted keyword scoring resolves the large
majority of HRM questions, costs no tokens and is trivially testable. The local
TinyLlama is consulted *only* when the rules are genuinely ambiguous, and its
answer is constrained to a fixed label set and discarded if it doesn't match —
a weak model is allowed to break a tie, never to invent a route.

Intents for capabilities that are not built yet (leave, payroll) are still
recognised. Naming the capability in the fallback ("I can't do leave yet") is a
far better answer than a generic shrug, and it means phase 2 only has to add
nodes, not re-teach the router.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Mapping

from app.assistant.runtime import RunContext
from app.assistant.state import (
    INTENT_ATTENDANCE,
    INTENT_LEAVE,
    INTENT_PAYROLL,
    INTENT_UNKNOWN,
)

logger = logging.getLogger(__name__)

#: term -> weight. Strong terms are unambiguous domain markers; weak terms only
#: break ties. Ordered longest-first at match time so "leave balance" doesn't
#: also score "balance".
_SIGNALS: dict[str, dict[str, float]] = {
    INTENT_ATTENDANCE: {
        "attendance": 3.0, "absent": 3.0, "present": 2.0, "late": 2.5,
        "half day": 2.5, "half-day": 2.5, "working days": 2.0, "check in": 2.0,
        "check-in": 2.0, "checkin": 2.0, "check out": 2.0, "check-out": 2.0,
        "sign in": 1.5, "sign out": 1.5, "punch": 2.0, "attendance percentage": 3.0,
        "days did i work": 2.5, "how many days": 1.5, "weekly off": 1.5,
        "holiday": 1.0, "leave days": -1.0,  # negative: belongs to the leave domain
    },
    INTENT_LEAVE: {
        "leave balance": 4.0, "leave": 2.5, "casual leave": 3.0, "sick leave": 3.0,
        "paid leave": 3.0, "cl": 1.0, "sl": 1.0, "pl": 1.0, "vacation": 2.5,
        "time off": 2.5, "apply leave": 3.5, "leave request": 3.0,
        "leaves do i have": 4.0, "leave applied": 3.0,
    },
    INTENT_PAYROLL: {
        "payslip": 4.0, "pay slip": 4.0, "salary": 3.0, "payroll": 3.0,
        "ctc": 2.5, "net pay": 3.0, "gross": 2.0, "deduction": 2.5, "pf": 1.5,
        "tds": 2.0, "bonus": 2.0, "increment": 2.0, "paid this month": 2.0,
        "salary slip": 4.0,
    },
}

_LLM_LABELS = {INTENT_ATTENDANCE, INTENT_LEAVE, INTENT_PAYROLL, INTENT_UNKNOWN}

#: Below this the rules are guessing, so ask the model (if one is wired up).
_AMBIGUITY_THRESHOLD = 1.5
#: Below this even the winner is noise — treat as unknown.
_MIN_SCORE = 1.0


def score_intents(text: str) -> dict[str, float]:
    """Pure function: text -> per-intent score. Exposed for tests and tuning."""
    low = f" {text.lower()} "
    scores = {intent: 0.0 for intent in _SIGNALS}
    for intent, terms in _SIGNALS.items():
        for term, weight in terms.items():
            # Word-boundary match so "pl" doesn't fire inside "please".
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low):
                scores[intent] += weight
    return scores


def _llm_tiebreak(text: str, ctx: RunContext) -> str | None:
    """One constrained classification call. Returns None on any doubt."""
    if ctx.llm is None:
        return None
    try:
        ctx.counters["llm_calls"] += 1
        raw = ctx.llm(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify the HR question into exactly one label: "
                        "attendance, leave, payroll, unknown. "
                        "Reply with the single label word and nothing else."
                    ),
                },
                {"role": "user", "content": text[:300]},
            ],
            max_new_tokens=6,
        )
    except Exception:  # noqa: BLE001 - a weak local model failing is expected
        logger.warning("assistant.router_llm_failed run_id=%s", ctx.run_id, exc_info=True)
        return None

    label = (raw or "").strip().lower().split()[0] if (raw or "").strip() else ""
    label = label.strip(".,:;\"'")
    return label if label in _LLM_LABELS else None


def intent_router(state: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
    # The recovery node can pin a route after a validation failure; honour it
    # rather than re-deriving the same wrong answer.
    forced = state.get("forced_intent")
    if forced:
        return {
            "intent": forced,
            "intent_confidence": 1.0,
            "intent_method": "recovery_override",
            "forced_intent": None,
            "_trace": {"intent": forced, "method": "recovery_override"},
        }

    text = state.get("normalized") or state.get("message") or ""
    scores = score_intents(text)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_intent, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - runner_up

    method = "rule"
    if top_score < _MIN_SCORE:
        top_intent, confidence = INTENT_UNKNOWN, 0.0
    elif margin < _AMBIGUITY_THRESHOLD:
        guess = _llm_tiebreak(text, ctx)
        if guess:
            top_intent, method = guess, "llm"
        confidence = round(min(1.0, margin / _AMBIGUITY_THRESHOLD) * 0.6, 2)
    else:
        confidence = round(min(1.0, 0.6 + margin / 10), 2)

    return {
        "intent": top_intent,
        "intent_confidence": confidence,
        "intent_method": method,
        "_trace": {
            "intent": top_intent,
            "confidence": confidence,
            "method": method,
            "scores": {k: round(v, 2) for k, v in scores.items() if v},
        },
    }
