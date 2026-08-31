"""Shared graph state.

Kept small on purpose: **no file contents ever enter state**. Nodes carry paths
and structured findings; anything that needs the bytes reads them through the
repository tool. Two reasons — state is logged, and a review over a dozen modules
would otherwise put megabytes of source into every trace record.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

Status = Literal["running", "done", "stopped", "failed"]

#: Why a run ended. Recorded so a stop is never ambiguous.
STOP_GOAL_ACHIEVED = "goal_achieved"
STOP_NO_BUGS = "no_actionable_bugs"
STOP_MAX_ITERATIONS = "max_iterations"
STOP_FIX_BUDGET = "fix_budget_exhausted"
STOP_REPEATED_FAILURE = "repeated_failure"
STOP_UNFIXABLE = "failure_not_auto_fixable"
STOP_UNSAFE = "unsafe_precondition"
STOP_TIMEOUT = "timeout"
STOP_VERIFICATION_FAILED = "verification_failed"


class EngineState(TypedDict, total=False):
    # ---- goal (written once) ---------------------------------------------
    goal: str
    scope: str
    #: Repo-relative paths in scope. Populated by the review node.
    files: list[str]

    # ---- review ----------------------------------------------------------
    #: {file, line, code, severity, problem, reason, recommended_action, source}
    review_findings: list[dict]

    # ---- bug analysis ----------------------------------------------------
    #: Findings triaged as real, with root_cause / priority / auto_fixable.
    bugs: list[dict]
    #: Findings rejected as false positives, with the reason. Kept because
    #: "we looked and decided not to" is a result, not an absence of one.
    dismissed: list[dict]

    # ---- fix -------------------------------------------------------------
    #: {file, line, strategy, applied: bool, reason}
    fixes: list[dict]
    changed_files: list[str]

    # ---- test ------------------------------------------------------------
    #: {command, exit_code, passed, failed, duration_s, failures: [...], stage}
    test_results: Optional[dict]
    #: Failures present *before* any change — never counted against the run.
    baseline_failures: list[str]

    # ---- failure analysis ------------------------------------------------
    #: {failure_type, root_cause, affected_files, next_action}
    failure_analysis: Optional[dict]

    # ---- verification ----------------------------------------------------
    #: {goal_achieved, criteria: [{name, ok, evidence}], summary}
    verification: Optional[dict]

    # ---- control ---------------------------------------------------------
    iteration: int
    fix_budget: int
    #: Copied from config so routing predicates stay pure functions of state.
    max_iterations: int
    apply_fixes: bool
    #: Signatures of failures already seen. A repeat stops the loop.
    failure_history: list[str]
    errors: list[dict]
    status: Status
    stop_reason: Optional[str]


def new_state(
    goal: str,
    scope: str,
    fix_budget: int,
    max_iterations: int = 5,
    apply_fixes: bool = False,
) -> EngineState:
    return {
        "goal": goal,
        "scope": scope,
        "files": [],
        "review_findings": [],
        "bugs": [],
        "dismissed": [],
        "fixes": [],
        "changed_files": [],
        "test_results": None,
        "baseline_failures": [],
        "failure_analysis": None,
        "verification": None,
        "iteration": 0,
        "fix_budget": fix_budget,
        "max_iterations": max_iterations,
        "apply_fixes": apply_fixes,
        "failure_history": [],
        "errors": [],
        "status": "running",
        "stop_reason": None,
    }


def summarize(state: EngineState) -> dict[str, Any]:
    """Log-safe projection. Counts and paths only — no source, no test output."""
    tests = state.get("test_results") or {}
    return {
        "scope": state.get("scope"),
        "files": len(state.get("files") or []),
        "findings": len(state.get("review_findings") or []),
        "bugs": len(state.get("bugs") or []),
        "dismissed": len(state.get("dismissed") or []),
        "fixes_applied": sum(1 for f in (state.get("fixes") or []) if f.get("applied")),
        "changed_files": list(state.get("changed_files") or []),
        "tests": {
            "stage": tests.get("stage"),
            "exit_code": tests.get("exit_code"),
            "passed": tests.get("passed"),
            "failed": tests.get("failed"),
        },
        "iteration": state.get("iteration"),
        "fix_budget": state.get("fix_budget"),
        "status": state.get("status"),
        "stop_reason": state.get("stop_reason"),
    }
