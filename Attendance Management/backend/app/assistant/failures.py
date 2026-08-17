"""Failure taxonomy for the assistant graph.

Step 8 of the brief: *do not blindly retry every failure*. A retry only helps
when the failure is transient. Everything else needs a different edge — ask the
user, re-route, or degrade honestly. This module is the single place that decides
which is which, so the recovery node stays a dumb executor of that decision.
"""
from __future__ import annotations

from typing import Any

# --- kinds -----------------------------------------------------------------
INVALID_INPUT = "invalid_input"          # user asked something unparseable
AUTHZ_FAILURE = "authz_failure"          # authenticated, but not allowed
MISSING_ENTITY = "missing_entity"        # need one more fact from the user
MISSING_DATA = "missing_data"            # HRMS simply has no record
TOOL_FAILURE = "tool_failure"            # a service call blew up
TRANSIENT = "transient"                  # DB blip / lock timeout — retryable
TIMEOUT = "timeout"
LLM_FAILURE = "llm_failure"
VALIDATION_FAILURE = "validation_failure"
BUSINESS_RULE = "business_rule"          # e.g. period before date of joining
UNSUPPORTED = "unsupported"              # capability not built yet
NODE_EXCEPTION = "node_exception"
GRAPH_STEP_LIMIT = "graph_step_limit"

#: The ONLY kinds a retry can plausibly fix.
RETRYABLE = frozenset({TRANSIENT, TIMEOUT})

#: Kinds that mean "ask the user one question and stop".
CLARIFIABLE = frozenset({MISSING_ENTITY, INVALID_INPUT})

#: Kinds where a different route might succeed.
REROUTABLE = frozenset({VALIDATION_FAILURE})

#: Never retried, never re-routed — answering would be a security or honesty bug.
TERMINAL = frozenset(
    {AUTHZ_FAILURE, BUSINESS_RULE, UNSUPPORTED, MISSING_DATA, GRAPH_STEP_LIMIT}
)

#: Messages shown to the user. Deliberately generic for security-relevant kinds:
#: an authorisation failure must not reveal whether the employee exists.
USER_MESSAGE = {
    INVALID_INPUT: "I couldn't understand that request. Could you rephrase it?",
    AUTHZ_FAILURE: "You don't have permission to view that information.",
    MISSING_ENTITY: "I need a little more detail to answer that.",
    MISSING_DATA: "I couldn't find any HRMS records for that period.",
    TOOL_FAILURE: "I couldn't reach the attendance records just now. Please try again.",
    TRANSIENT: "I couldn't reach the attendance records just now. Please try again.",
    TIMEOUT: "That took too long to look up. Please try again.",
    LLM_FAILURE: "I found your data but couldn't phrase a summary. Here are the figures.",
    VALIDATION_FAILURE: "I couldn't verify the figures well enough to report them.",
    BUSINESS_RULE: "That period isn't valid for your record.",
    UNSUPPORTED: "I can't help with that yet.",
    NODE_EXCEPTION: "Something went wrong while answering. Please try again.",
    GRAPH_STEP_LIMIT: "Something went wrong while answering. Please try again.",
}


def classify_exception(exc: BaseException) -> str:
    """Map a raised exception to a failure kind.

    Connection-level SQLAlchemy errors are transient (the HRMS database is
    remote — a dropped connection is a normal, retryable event). A programming
    error in a query is not, and retrying it just wastes the budget.
    """
    name = type(exc).__name__
    if name in ("OperationalError", "InterfaceError", "DisconnectionError", "TimeoutError"):
        return TRANSIENT
    if name in ("DataError", "ProgrammingError", "IntegrityError", "StatementError"):
        return TOOL_FAILURE
    if isinstance(exc, ValueError):
        return INVALID_INPUT
    if isinstance(exc, PermissionError):
        return AUTHZ_FAILURE
    return TOOL_FAILURE


def signature(kind: str, detail: Any = "") -> str:
    """A stable id for "this exact failure".

    The recovery loop stops when a signature repeats: retrying a failure that
    already recurred is how a bounded loop turns into an unbounded one.
    """
    return f"{kind}:{str(detail)[:80]}"


def user_message(kind: str) -> str:
    return USER_MESSAGE.get(kind, USER_MESSAGE[NODE_EXCEPTION])
