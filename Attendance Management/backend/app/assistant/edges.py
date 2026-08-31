"""Conditional-edge predicates.

Pure functions of state, kept out of the nodes so routing is assertable on its
own: a test can prove "a denied scope goes straight to the response" without
constructing a graph or a database.

Each returns a label that `add_conditional_edges` maps to a node name. The
runtime raises if a predicate ever returns a label the mapping doesn't cover,
so an unhandled state can't silently fall through.
"""
from __future__ import annotations

from typing import Any, Mapping

from app.assistant.nodes.diagnosis import RETRY
from app.assistant.state import INTENT_ATTENDANCE


def route_after_scope(state: Mapping[str, Any]) -> str:
    """Security gate outcome. Anything but `ok` skips every domain node."""
    status = state.get("status")
    if status == "denied":
        return "denied"
    if status == "needs_input":
        return "needs_input"
    if status == "unsupported":
        return "unsupported"
    return "ok"


def route_by_intent(state: Mapping[str, Any]) -> str:
    """Domain dispatch. Phase 2 adds `leave` / `payroll` labels here."""
    if state.get("intent") == INTENT_ATTENDANCE:
        return INTENT_ATTENDANCE
    return "fallback"


def route_after_validation(state: Mapping[str, Any]) -> str:
    validation = state.get("validation") or {}
    return "pass" if validation.get("ok") else "fail"


def route_after_recovery(state: Mapping[str, Any]) -> str:
    """The loop-closing edge — the only one that can go backwards.

    Reads `recovery_action` rather than re-deriving the decision, so the
    classifier stays the single authority on whether another attempt is allowed.
    """
    return "retry" if state.get("recovery_action") == RETRY else "respond"
