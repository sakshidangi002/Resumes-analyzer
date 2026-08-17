"""`report` — the single terminal node. Sets the final status and stop reason.

Every path into END goes through here, so there is exactly one place that decides
what "the run finished like this" means, and a stop is never ambiguous: there is
always a `stop_reason` naming which condition ended it.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from graph_engine.context import EngineContext
from graph_engine.nodes.failure_analysis import ACTION_STOP
from graph_engine.state import (
    STOP_FIX_BUDGET,
    STOP_GOAL_ACHIEVED,
    STOP_MAX_ITERATIONS,
    STOP_NO_BUGS,
    STOP_REPEATED_FAILURE,
    STOP_TIMEOUT,
    STOP_UNFIXABLE,
    STOP_VERIFICATION_FAILED,
)

logger = logging.getLogger(__name__)


def _resolve_stop(state: Mapping[str, Any], ctx: EngineContext) -> tuple[str, str]:
    """(status, stop_reason) — most specific cause first."""
    if state.get("stop_reason"):
        return state.get("status") or "stopped", state["stop_reason"]

    error = state.get("error") or {}
    if error.get("kind") == "timeout" or ctx.expired():
        return "stopped", STOP_TIMEOUT
    # A node that raised must never be reported as a completed run. This is
    # checked before everything else because the state it left behind is
    # partial, and reading "no bugs found" out of it would be a lie.
    if error.get("kind"):
        return "failed", error["kind"]

    analysis = state.get("failure_analysis") or {}
    if analysis.get("repeated"):
        return "stopped", STOP_REPEATED_FAILURE
    if analysis.get("next_action") == ACTION_STOP:
        return "stopped", STOP_UNFIXABLE

    exhausted_iterations = int(state.get("iteration", 0) or 0) >= ctx.config.max_iterations
    exhausted_budget = int(state.get("fix_budget", 0) or 0) <= 0

    verification = state.get("verification")
    if verification is not None:
        if verification.get("goal_achieved"):
            return "done", STOP_GOAL_ACHIEVED
        # Running out of allowance is a *stop*, not a failure: the work done so
        # far is valid, there was simply not enough budget to finish it. Calling
        # that "failed" would make a partial-but-correct run indistinguishable
        # from one that produced something wrong.
        if exhausted_iterations:
            return "stopped", STOP_MAX_ITERATIONS
        if exhausted_budget:
            return "stopped", STOP_FIX_BUDGET
        return "failed", STOP_VERIFICATION_FAILED

    if exhausted_iterations:
        return "stopped", STOP_MAX_ITERATIONS
    if exhausted_budget:
        return "stopped", STOP_FIX_BUDGET
    if not (state.get("bugs") or []):
        return "done", STOP_NO_BUGS
    return "stopped", "unspecified"


def report(state: Mapping[str, Any], ctx: EngineContext) -> Mapping[str, Any]:
    status, stop_reason = _resolve_stop(state, ctx)
    verification = state.get("verification") or {}

    logger.info(
        "graph_engine.report status=%s stop_reason=%s achieved=%s",
        status, stop_reason, verification.get("goal_achieved"),
    )
    return {
        "status": status,
        "stop_reason": stop_reason,
        "_trace": {
            "status": status,
            "stop_reason": stop_reason,
            "goal_achieved": verification.get("goal_achieved"),
        },
    }
