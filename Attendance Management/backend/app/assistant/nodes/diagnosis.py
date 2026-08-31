"""`failure_classifier` — diagnose a failure and choose a disposition.

Separated from `recovery_node` deliberately: diagnosis is a pure decision over
state (trivially unit-testable, no side effects), recovery is the mutation that
carries it out. Keeping them apart is what makes "did we decide to retry?"
assertable without running a retry.

The four dispositions are the graph's complete set of failure edges:

``retry``     transient only, and only while budget, iteration count and failure
              novelty all permit it
``clarify``   the user can supply the missing fact — one question, then stop
``reroute``   a different route might work (kept for phase 2, when leave/payroll
              nodes give the router somewhere else to go)
``degrade``   answer honestly with what we have, or explain the refusal

The novelty check is the important one. A retry budget alone still lets a
deterministic failure consume every attempt; refusing to repeat a failure
*signature* stops the loop the first time it proves it is not transient.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from app.assistant import failures
from app.assistant.runtime import RunContext
from app.assistant.state import MAX_ITERATIONS

logger = logging.getLogger(__name__)

RETRY = "retry"
CLARIFY = "clarify"
REROUTE = "reroute"
DEGRADE = "degrade"


def failure_classifier(state: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
    error = state.get("error") or {}
    kind = error.get("kind") or failures.VALIDATION_FAILURE
    history = list(state.get("failure_history") or [])
    iteration = int(state.get("iteration", 0) or 0)
    budget = int(state.get("retry_budget", 0) or 0)

    validation = state.get("validation") or {}
    detail = ",".join(f["code"] for f in validation.get("failures", [])) or error.get("node", "")
    sig = failures.signature(kind, detail)
    repeated = sig in history

    if repeated:
        action, reason = DEGRADE, "repeated_failure"
    elif iteration >= MAX_ITERATIONS:
        action, reason = DEGRADE, "max_iterations"
    elif kind in failures.TERMINAL:
        action, reason = DEGRADE, "terminal_kind"
    elif kind in failures.CLARIFIABLE:
        action, reason = CLARIFY, "clarifiable_kind"
    elif kind in failures.RETRYABLE:
        if budget > 0:
            action, reason = RETRY, "transient_with_budget"
        else:
            action, reason = DEGRADE, "retry_budget_exhausted"
    elif kind in failures.REROUTABLE and (state.get("entities") or {}).get("period_assumed"):
        # We guessed the month and the numbers didn't hold up. Asking which month
        # is far more likely to help than running the identical query again.
        action, reason = CLARIFY, "assumed_period_failed"
    else:
        action, reason = DEGRADE, "not_recoverable"

    logger.info(
        "assistant.failure_classified run_id=%s kind=%s action=%s reason=%s "
        "iteration=%d budget=%d repeated=%s",
        ctx.run_id, kind, action, reason, iteration, budget, repeated,
    )

    return {
        "recovery_action": action,
        "recovery_reason": reason,
        "failure_history": history + [sig],
        "_trace": {
            "kind": kind,
            "action": action,
            "reason": reason,
            "iteration": iteration,
            "budget": budget,
            "repeated": repeated,
        },
    }
