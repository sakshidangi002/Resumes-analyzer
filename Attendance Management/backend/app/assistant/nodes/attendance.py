"""`attendance_node` — a parallel fan-out over three independent reads.

The three tools hit different tables with no ordering dependency between them
(attendance records, the employee row, the late flags), so they run
concurrently and join. This is the one place parallelism is justified: they are
all *reads*, and the slowest determines the wall clock instead of the sum.

Threading rule: a SQLAlchemy `Session` is not thread-safe, and the request-scoped
one belongs to the FastAPI dependency. Each branch therefore mints its own
Session from `ctx.db_factory` and closes it. Sharing `ctx.db` across the pool
would be a real corruption bug, not a style preference.

Branches catch their own exceptions so they can classify the failure properly
(transient vs. permanent) instead of letting the runtime label everything a
generic tool failure.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Mapping

from app.assistant import failures
from app.assistant.runtime import RunContext
from app.assistant.tools import attendance_tools

logger = logging.getLogger(__name__)


@contextmanager
def _branch_session(ctx: RunContext):
    """A Session this branch owns, or the request's own when running single-threaded."""
    if ctx.db_factory is None:
        yield ctx.db
        return
    session = ctx.db_factory()
    try:
        yield session
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001 - closing must never mask the real error
            logger.warning("assistant.branch_session_close_failed run_id=%s", ctx.run_id)


def _make_branch(name: str):
    """Wrap a tool as a graph branch: own session, own failure classification.

    The tool is resolved from the module at *call* time rather than captured at
    import time, so the graph is a singleton yet tests can still substitute a
    tool without rebuilding it.
    """

    def branch(state: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
        fn: Callable[..., dict] = getattr(attendance_tools, name)
        entities = state.get("entities") or {}
        employee_id = state.get("target_employee_id")
        month, year = entities.get("month"), entities.get("year")

        # Defence in depth: the validator and the gate both cover this, but a
        # tool must never run unscoped even if a future edge reaches it wrongly.
        if not employee_id or not month or not year:
            return {
                "tool_errors": {
                    name: {
                        "kind": failures.MISSING_ENTITY,
                        "message": "missing employee or period",
                    }
                }
            }

        try:
            ctx.counters["tool_calls"] += 1
            with _branch_session(ctx) as db:
                result = fn(db, employee_id=employee_id, month=month, year=year)
            return {"tool_results": {name: result}, "_trace": {"status": "ok", "tool": name}}
        except Exception as exc:  # noqa: BLE001 - classified, not swallowed
            kind = failures.classify_exception(exc)
            logger.warning(
                "assistant.tool_failed run_id=%s tool=%s kind=%s exc=%s",
                ctx.run_id, name, kind, type(exc).__name__,
            )
            return {
                "tool_errors": {name: {"kind": kind, "message": type(exc).__name__}},
                # Surfaced to the runtime so this branch's trace reads "error"
                # rather than "ok" — it returned normally, but it produced nothing.
                "_trace": {"status": "error", "tool": name, "kind": kind,
                           "error": type(exc).__name__},
            }

    branch.__name__ = f"attendance_{name}"
    return branch


#: The fan-out. Registered with `add_parallel_node`, so all three execute
#: concurrently and their partial updates merge into `tool_results`/`tool_errors`.
ATTENDANCE_BRANCHES = {
    "monthly_summary": _make_branch("monthly_summary"),
    "employee_period_context": _make_branch("employee_period_context"),
    "late_mark_count": _make_branch("late_mark_count"),
}
