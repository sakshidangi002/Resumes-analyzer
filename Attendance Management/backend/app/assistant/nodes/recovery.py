"""`recovery_node` — carry out the disposition chosen by `failure_classifier`.

This is the only node that can send execution *backwards*, and it is the reason
the graph has a cycle at all. It pays for that by decrementing the budget and
incrementing the iteration counter on every pass, so the loop is monotonic: each
trip strictly reduces the remaining allowance, and `failure_classifier` refuses
to authorise another once either is spent.

Nothing here retries "harder" — there is no exponential backoff, because a
user is waiting on an HTTP request and the transient failures worth retrying
(a dropped connection to the remote Postgres) succeed immediately or not at all.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from app.assistant.nodes.diagnosis import CLARIFY, RETRY, REROUTE
from app.assistant.runtime import RunContext

logger = logging.getLogger(__name__)

_CLARIFY_QUESTIONS = {
    "assumed_period_failed": "Which month did you mean? For example: 'last month' or 'March 2025'.",
}


def recovery_node(state: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
    action = state.get("recovery_action")
    iteration = int(state.get("iteration", 0) or 0)
    budget = int(state.get("retry_budget", 0) or 0)

    if action == RETRY:
        ctx.counters["retries"] += 1
        logger.info(
            "assistant.retry run_id=%s iteration=%d budget_left=%d",
            ctx.run_id, iteration + 1, budget - 1,
        )
        # Clear the previous attempt's residue so the fan-out starts clean;
        # keeping stale tool_errors would make the validator re-fail instantly.
        return {
            "iteration": iteration + 1,
            "retry_budget": budget - 1,
            "error": None,
            "tool_errors": {},
            "validation": None,
            "status": "running",
            "_trace": {"action": RETRY, "iteration": iteration + 1},
        }

    if action == REROUTE:
        # Reserved for phase 2: with leave/payroll nodes present the router has
        # a genuine second choice. Today it degrades rather than pretending.
        return {
            "iteration": iteration + 1,
            "status": "failed",
            "_trace": {"action": REROUTE, "note": "no alternate route in phase 1"},
        }

    if action == CLARIFY:
        error = state.get("error") or {}
        question = _CLARIFY_QUESTIONS.get(
            state.get("recovery_reason") or "",
            error.get("message") or "Could you give me a bit more detail?",
        )
        return {
            "status": "needs_input",
            "clarify_question": question,
            "_trace": {"action": CLARIFY},
        }

    return {"status": "failed", "_trace": {"action": "degrade"}}
