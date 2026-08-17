"""Shared graph state for the HRM assistant.

Design rules this state obeys:

* **Small.** Only what a downstream node actually reads. No ORM objects, no
  full attendance record lists, no `User`. Nodes that need the database use
  `RunContext.db`, not state.
* **Safe to log.** Everything here is either an id, an enum-ish string, a count
  or an aggregate. There is no name, email, phone or salary figure in state, so
  the trace can be logged without leaking employee PII (Step 11).
* **Authorisation is a value, not a hope.** `target_employee_id` is written
  once, by `scope_resolver`, from the authenticated actor — never from the user
  message and never from the LLM. Domain nodes read it and cannot widen it.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

# Set by scope_resolver; domain nodes must use it verbatim.
Scope = Literal["self", "other", "team"]
Status = Literal["running", "ok", "denied", "needs_input", "failed", "unsupported"]

#: Intents the graph can currently serve. `unknown` is a first-class outcome,
#: not an error: it routes to the fallback node, which offers to file an HR query.
INTENT_ATTENDANCE = "attendance"
INTENT_UNKNOWN = "unknown"
#: Reserved for phase 2 — the router already recognises them so that the
#: fallback message can name the capability instead of shrugging.
INTENT_LEAVE = "leave"
INTENT_PAYROLL = "payroll"


class AssistantState(TypedDict, total=False):
    """The single dict every node reads from and returns partial updates to."""

    # ---- input, written once by the route ---------------------------------
    message: str
    actor_user_id: int
    actor_employee_id: Optional[int]
    actor_roles: list[str]

    # ---- request_analyzer -------------------------------------------------
    normalized: str
    entities: dict  # {"month": int, "year": int, "period_label": str, "target_name": str|None}

    # ---- intent_router ----------------------------------------------------
    intent: str
    intent_confidence: float
    intent_method: str  # "rule" | "llm" | "recovery_override"

    # ---- scope_resolver (security gate) -----------------------------------
    scope: Scope
    target_employee_id: Optional[int]
    denial_reason: Optional[str]

    # ---- domain nodes -----------------------------------------------------
    #: {tool_name: plain-dict result}. Aggregates only, never raw rows.
    tool_results: dict
    #: {tool_name: {"kind": FailureKind, "message": str}}
    tool_errors: dict

    # ---- result_validator -------------------------------------------------
    validation: Optional[dict]  # {"ok": bool, "failures": [{"code","detail"}]}

    # ---- control / loop engineering ---------------------------------------
    status: Status
    error: Optional[dict]  # {"kind","message","node"}
    #: Signatures of failures already seen this run. Repeating one stops the loop
    #: instead of burning the retry budget on a deterministic failure.
    failure_history: list[str]
    iteration: int
    retry_budget: int
    #: Written by failure_classifier, consumed by recovery_node and the
    #: post-recovery conditional edge: retry | clarify | reroute | degrade.
    recovery_action: Optional[str]
    recovery_reason: Optional[str]
    #: Set by recovery to pin the router's next choice; cleared once honoured.
    forced_intent: Optional[str]

    # ---- output -----------------------------------------------------------
    response: Optional[str]
    data: Optional[dict]  # structured payload the React UI can render
    clarify_question: Optional[str]


#: Hard ceilings. `MAX_ITERATIONS` bounds the recovery loop; the runtime's own
#: `max_steps` is a separate, lower-level backstop.
MAX_ITERATIONS = 2
DEFAULT_RETRY_BUDGET = 2
DEFAULT_TIMEOUT_SECONDS = 20.0


def new_state(
    *,
    message: str,
    actor_user_id: int,
    actor_employee_id: Optional[int],
    actor_roles: list[str],
) -> AssistantState:
    """Build the initial state. The only place input enters the graph."""
    return {
        "message": message,
        "actor_user_id": actor_user_id,
        "actor_employee_id": actor_employee_id,
        "actor_roles": list(actor_roles),
        "normalized": "",
        "entities": {},
        "intent": INTENT_UNKNOWN,
        "intent_confidence": 0.0,
        "intent_method": "rule",
        "scope": "self",
        "target_employee_id": None,
        "denial_reason": None,
        "tool_results": {},
        "tool_errors": {},
        "validation": None,
        "status": "running",
        "error": None,
        "failure_history": [],
        "iteration": 0,
        "retry_budget": DEFAULT_RETRY_BUDGET,
        "recovery_action": None,
        "recovery_reason": None,
        "forced_intent": None,
        "response": None,
        "data": None,
        "clarify_question": None,
    }


def redacted(state: AssistantState) -> dict:
    """A log-safe projection of state.

    The user's raw message can contain anything they typed, so it is reduced to
    a length. Everything else in state is already non-identifying.
    """
    return {
        "actor_user_id": state.get("actor_user_id"),
        "message_len": len(state.get("message") or ""),
        "intent": state.get("intent"),
        "intent_confidence": state.get("intent_confidence"),
        "intent_method": state.get("intent_method"),
        "scope": state.get("scope"),
        "has_target": state.get("target_employee_id") is not None,
        "status": state.get("status"),
        "iteration": state.get("iteration"),
        "tools_ok": sorted((state.get("tool_results") or {}).keys()),
        "tools_failed": sorted((state.get("tool_errors") or {}).keys()),
        "error_kind": (state.get("error") or {}).get("kind"),
    }
