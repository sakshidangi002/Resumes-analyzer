"""Conditional-edge predicates: pure functions of state, no side effects.

Kept out of the nodes so every routing decision is assertable on its own — a test
proves "a failing test goes to failure analysis" without running pytest.

Both cycles in the graph are closed here, and both are guarded by the same two
state fields (`iteration` vs `max_iterations`, and `fix_budget`), so there is one
place to check that the loops terminate.
"""
from __future__ import annotations

from typing import Any, Mapping

from graph_engine.nodes.failure_analysis import ACTION_FIX, ACTION_IGNORE


def route_after_preflight(state: Mapping[str, Any]) -> str:
    """A failed safety precondition never reaches the code."""
    return "stopped" if state.get("status") in ("stopped", "failed") else "ok"


def route_after_triage(state: Mapping[str, Any]) -> str:
    """No actionable bug means there is nothing to fix — go straight to verify."""
    actionable = [b for b in (state.get("bugs") or []) if b.get("auto_fixable")]
    return "bugs" if actionable else "none"


def route_after_fix(state: Mapping[str, Any]) -> str:
    """Only run tests when something actually changed.

    A dry run, or a pass where every candidate was skipped, has nothing for the
    suite to detect; running it would burn 30 seconds to re-measure the baseline.
    """
    return "test" if (state.get("changed_files") or []) else "verify"


def route_after_test(state: Mapping[str, Any]) -> str:
    results = state.get("test_results") or {}
    return "pass" if results.get("ok") else "fail"


def route_after_failure_analysis(state: Mapping[str, Any]) -> str:
    """The loop-closing edge. `fix` is the only label that goes backwards."""
    analysis = state.get("failure_analysis") or {}
    action = analysis.get("next_action")

    if action == ACTION_IGNORE:
        return "verify"
    if action != ACTION_FIX:
        return "stop"

    # Budget and iteration are checked here as well as in the classifier: the
    # classifier decides whether a fix is *sensible*, this decides whether the
    # run is still *allowed* another one.
    if int(state.get("iteration", 0) or 0) >= int(state.get("max_iterations", 5) or 5):
        return "stop"
    if int(state.get("fix_budget", 0) or 0) <= 0:
        return "stop"
    return "fix"


def route_after_verify(state: Mapping[str, Any]) -> str:
    """Second cycle: unaddressed actionable bugs get another fix pass.

    The fix node handles a bounded number of bugs per pass, so a scope with many
    findings legitimately needs more than one. It terminates because every pass
    records an outcome for the bugs it took, shrinking `unaddressed` — and
    because `iteration` increments regardless.
    """
    verification = state.get("verification") or {}
    if verification.get("goal_achieved"):
        return "done"

    # A dry run has nothing left to attempt: the fix node already recorded a
    # decline for every actionable bug, so looping would burn iterations
    # re-declining the same work.
    if not state.get("apply_fixes"):
        return "done"

    if int(state.get("iteration", 0) or 0) >= int(state.get("max_iterations", 5) or 5):
        return "done"
    if int(state.get("fix_budget", 0) or 0) <= 0:
        return "done"

    attempted = {(f["file"], f["line"]) for f in (state.get("fixes") or [])}
    unaddressed = [
        b for b in (state.get("bugs") or [])
        if b.get("auto_fixable") and (b["file"], b["line"]) not in attempted
    ]
    return "retry" if unaddressed else "done"
