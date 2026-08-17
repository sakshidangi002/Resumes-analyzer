"""`response_generator` — the single exit point that produces user-facing text.

Every terminal path funnels through here so there is exactly one place that
decides what a user sees, and exactly one place to audit for leakage.

The numbers are rendered by a deterministic template. The LLM is offered the
chance to phrase it more naturally, but its output is **verified before use**:
if the rewrite drops or changes any of the figures, it is discarded and the
template stands. A local 1.1B model will sometimes produce fluent, wrong
arithmetic, and a payroll-adjacent assistant that occasionally misreports
absence counts is worse than one that always sounds a bit flat.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Mapping

from app.assistant import failures
from app.assistant.runtime import RunContext

logger = logging.getLogger(__name__)

_DENIAL_TEXT = {
    "role": "You don't have permission to view another employee's attendance.",
    "no_employee_record": (
        "Your account isn't linked to an employee record, so I don't have "
        "attendance to show. Please contact HR."
    ),
}


def _render_summary(summary: dict, entities: dict, extras: dict) -> str:
    period = entities.get("period_label") or f"{summary.get('month')}/{summary.get('year')}"
    parts = [
        f"For {period} you were present {summary['present']} day(s) "
        f"out of {summary['working_days']} working day(s)."
    ]
    if summary.get("absent"):
        parts.append(f"Absent: {summary['absent']} day(s).")
    if summary.get("leave"):
        parts.append(f"On approved leave: {summary['leave']} day(s).")
    if summary.get("half_day"):
        parts.append(f"Half days: {summary['half_day']}.")
    late = (extras.get("late_mark_count") or {}).get("late_days")
    if late:
        parts.append(f"Late marks: {late}.")
    parts.append(f"Attendance: {summary['attendance_percentage']}%.")
    if entities.get("period_assumed"):
        parts.append("(I assumed you meant the current month.)")
    return " ".join(parts)


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _polish(template: str, question: str, ctx: RunContext) -> str:
    """Optional LLM rewrite, accepted only if it preserves every figure."""
    if ctx.llm is None:
        return template
    try:
        ctx.counters["llm_calls"] += 1
        candidate = ctx.llm(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the HR answer in one or two friendly sentences. "
                        "Keep every number exactly as given. Add no new facts."
                    ),
                },
                {"role": "user", "content": f"Question: {question}\nAnswer: {template}"},
            ],
            max_new_tokens=120,
        )
    except Exception:  # noqa: BLE001 - phrasing is a nicety, never a failure
        logger.warning("assistant.polish_failed run_id=%s", ctx.run_id, exc_info=True)
        return template

    candidate = (candidate or "").strip()
    if not candidate:
        return template
    # Verification: the rewrite may not introduce or lose a figure.
    if _numbers_in(candidate) != _numbers_in(template):
        logger.info("assistant.polish_rejected run_id=%s reason=number_drift", ctx.run_id)
        return template
    return candidate


def response_generator(state: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
    status = state.get("status")
    entities = state.get("entities") or {}
    results = state.get("tool_results") or {}

    # --- refusals ----------------------------------------------------------
    if status == "denied":
        reason = state.get("denial_reason") or "role"
        return {
            "response": _DENIAL_TEXT.get(reason, failures.user_message(failures.AUTHZ_FAILURE)),
            "data": None,
            "_trace": {"outcome": "denied", "reason": reason},
        }

    # --- needs one more fact ----------------------------------------------
    if status == "needs_input":
        question = state.get("clarify_question") or (state.get("error") or {}).get(
            "message"
        ) or "Could you give me a bit more detail?"
        return {
            "response": question,
            "data": None,
            "_trace": {"outcome": "needs_input"},
        }

    # --- capability not built yet -----------------------------------------
    if status == "unsupported":
        message = (state.get("error") or {}).get("message") or failures.user_message(
            failures.UNSUPPORTED
        )
        return {
            "response": message,
            "data": {"suggested_action": "raise_hr_query"},
            "_trace": {"outcome": "unsupported"},
        }

    # --- failed, after diagnosis and any recovery --------------------------
    if status == "failed":
        kind = (state.get("error") or {}).get("kind") or failures.VALIDATION_FAILURE
        text = failures.user_message(kind)
        summary = results.get("monthly_summary")
        # Degrade honestly: if the figures survived but validation didn't, say
        # so rather than either hiding them or presenting them as verified.
        if summary and kind == failures.VALIDATION_FAILURE:
            text = (
                "I found your attendance figures but couldn't fully verify them, "
                "so please double-check on the Attendance page."
            )
        return {
            "response": text,
            "data": {"suggested_action": "raise_hr_query"},
            "_trace": {"outcome": "failed", "kind": kind},
        }

    # --- success -----------------------------------------------------------
    summary = results.get("monthly_summary") or {}
    template = _render_summary(summary, entities, results)
    text = _polish(template, state.get("normalized") or "", ctx)
    return {
        "status": "ok",
        "response": text,
        "data": {
            "intent": state.get("intent"),
            "period": {
                "month": summary.get("month"),
                "year": summary.get("year"),
                "label": entities.get("period_label"),
                "assumed": bool(entities.get("period_assumed")),
            },
            "summary": summary,
            "late_days": (results.get("late_mark_count") or {}).get("late_days"),
            "scope": state.get("scope"),
        },
        "_trace": {"outcome": "ok", "polished": text != template},
    }
