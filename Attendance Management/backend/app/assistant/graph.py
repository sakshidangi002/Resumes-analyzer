"""Graph assembly and the single public entry point, `run_assistant`.

Graph shape (see `docs/assistant-graph.md` for the rendered diagram)::

    request_analyzer -> scope_resolver -+-> intent_router -+-> attendance_fetch (parallel)
                                        |                  |          |
                                        |                  |          v
                                        |                  |   result_validator
                                        |                  |     |          |
                                        |                  |   pass       fail
                                        |                  |     |          v
                                        |                  |     |   failure_classifier
                                        |                  |     |          |
                                        |                  |     |     recovery_node
                                        |                  |     |      |        |
                                        |                  |     |   retry    respond
                                        |                  |     |      |        |
                                        |                  | (back to attendance_fetch)
                                        |                  v     v               v
                                        +--------------> response_generator -> END

Two deliberate departures from the brief's starting sketch:

* **The security gate precedes routing.** Denials must not depend on which
  domain node happens to run, and an unauthorised request should do no work.
* **Retry re-enters `attendance_fetch`, not the router.** The failures worth
  retrying are transient tool failures; the route was already correct, so
  re-running analysis and classification would burn time (and possibly an LLM
  call) to reach the identical decision.

The compiled graph is a module-level singleton. It is immutable and holds no
per-request state — everything mutable lives in the state dict and `RunContext`.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from app.assistant import edges
from app.assistant.nodes.analyzer import request_analyzer
from app.assistant.nodes.attendance import ATTENDANCE_BRANCHES
from app.assistant.nodes.diagnosis import failure_classifier
from app.assistant.nodes.recovery import recovery_node
from app.assistant.nodes.response import response_generator
from app.assistant.nodes.router import intent_router
from app.assistant.nodes.scope import scope_resolver
from app.assistant.nodes.terminal import fallback_node
from app.assistant.nodes.validator import result_validator
from app.assistant.runtime import END, CompiledGraph, RunContext, StateGraph
from app.assistant.state import (
    DEFAULT_TIMEOUT_SECONDS,
    INTENT_ATTENDANCE,
    new_state,
    redacted,
)

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Construct (but don't compile) the graph. Exposed so tests and the docs
    generator can inspect the topology without executing anything."""
    g = StateGraph()

    g.add_node("request_analyzer", request_analyzer)
    g.add_node("scope_resolver", scope_resolver)
    g.add_node("intent_router", intent_router)
    g.add_parallel_node("attendance_fetch", ATTENDANCE_BRANCHES, max_workers=3)
    g.add_node("result_validator", result_validator)
    g.add_node("failure_classifier", failure_classifier)
    g.add_node("recovery_node", recovery_node)
    g.add_node("fallback_node", fallback_node)
    g.add_node("response_generator", response_generator)

    g.set_entry("request_analyzer")
    # Any node that raises lands in the response generator, which reports the
    # failure honestly instead of returning a 500 to the chat UI.
    g.set_error_node("response_generator")

    g.add_edge("request_analyzer", "scope_resolver")
    g.add_conditional_edges(
        "scope_resolver",
        edges.route_after_scope,
        {
            "ok": "intent_router",
            "denied": "response_generator",
            "needs_input": "response_generator",
            "unsupported": "response_generator",
        },
    )
    g.add_conditional_edges(
        "intent_router",
        edges.route_by_intent,
        {INTENT_ATTENDANCE: "attendance_fetch", "fallback": "fallback_node"},
    )
    g.add_edge("attendance_fetch", "result_validator")
    g.add_conditional_edges(
        "result_validator",
        edges.route_after_validation,
        {"pass": "response_generator", "fail": "failure_classifier"},
    )
    g.add_edge("failure_classifier", "recovery_node")
    g.add_conditional_edges(
        "recovery_node",
        edges.route_after_recovery,
        {"retry": "attendance_fetch", "respond": "response_generator"},
    )
    g.add_edge("fallback_node", "response_generator")
    g.add_edge("response_generator", END)
    return g


_compiled: Optional[CompiledGraph] = None


def get_compiled_graph() -> CompiledGraph:
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile(max_steps=24)
    return _compiled


def run_assistant(
    *,
    message: str,
    actor_user_id: int,
    actor_employee_id: Optional[int],
    actor_roles: list[str],
    db: Any,
    db_factory: Any = None,
    llm: Any = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Run one turn. The route's only contact with the graph.

    Returns the final state plus a `trace` list. The caller decides how much of
    the trace to expose; the route only returns it to Admin/HR.
    """
    run_id = uuid.uuid4().hex[:12]
    ctx = RunContext(
        run_id=run_id,
        db=db,
        db_factory=db_factory,
        llm=llm,
        deadline=time.monotonic() + timeout_seconds,
    )
    state = new_state(
        message=message,
        actor_user_id=actor_user_id,
        actor_employee_id=actor_employee_id,
        actor_roles=actor_roles,
    )

    started = time.monotonic()
    final = get_compiled_graph().invoke(state, ctx)
    elapsed_ms = (time.monotonic() - started) * 1000

    # One structured line per run, plus one per node. Never the message text,
    # never a name, never a figure — `redacted()` guarantees that shape.
    logger.info(
        "assistant.run run_id=%s duration_ms=%.1f nodes=%d llm_calls=%d "
        "tool_calls=%d retries=%d state=%s",
        run_id, elapsed_ms, len(ctx.traces), ctx.counters["llm_calls"],
        ctx.counters["tool_calls"], ctx.counters["retries"], redacted(final),
    )
    for trace in ctx.traces:
        logger.info("assistant.node %s", trace.as_dict())

    final["run_id"] = run_id
    final["trace"] = [t.as_dict() for t in ctx.traces]
    final["metrics"] = {
        "duration_ms": round(elapsed_ms, 1),
        "node_count": len(ctx.traces),
        **ctx.counters,
    }
    return final
