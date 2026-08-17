"""A small, dependency-free state-graph runtime.

Why not LangGraph: the HRMS backend and the Resume Analyzer share one virtualenv,
and `requirements.txt` pins a deliberately "COMPATIBLE SET" of
`langchain==0.1.20` / `langchain-core==0.1.52` for the NuExtract extraction
chain. Current LangGraph needs `langchain-core>=0.2.43`, so installing it would
force an upgrade that breaks resume extraction. This module provides the graph
primitives the assistant actually needs — nodes, edges, conditional edges,
parallel fan-out, bounded loops and per-node tracing — in ~300 lines with no new
dependencies. The API deliberately mirrors LangGraph's (`add_node`,
`add_edge`, `add_conditional_edges`, `compile`, `invoke`, `END`) so that
swapping in the real thing later is an import change, not a rewrite.

Contract for a node function::

    def my_node(state: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
        return {"some_state_key": value}      # PARTIAL update, not full state

Returning a partial update (rather than the whole state) is what keeps nodes
independently testable: a test can call the function with a hand-built dict and
assert on the handful of keys it owns.
"""
from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)

END = "__end__"

NodeFn = Callable[[Mapping[str, Any], "RunContext"], Mapping[str, Any]]
RouterFn = Callable[[Mapping[str, Any]], str]


class GraphError(RuntimeError):
    """Raised for graph *construction* mistakes (unknown node, bad edge)."""


class ParallelMergeConflict(GraphError):
    """Two parallel branches wrote the same non-mergeable state key.

    This is a graph-design bug, not a runtime condition: parallel branches must
    write to disjoint keys (or to dicts, which are merged key-wise). Failing
    loudly here beats silently letting whichever thread finished last win.
    """


@dataclass
class NodeTrace:
    """One node execution. The unit of graph observability (Step 10)."""

    run_id: str
    seq: int
    node: str
    status: str  # ok | error | skipped
    started_at: float
    ended_at: float
    duration_ms: float
    iteration: int = 0
    meta: dict = field(default_factory=dict)
    error: Optional[str] = None
    branch_of: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "seq": self.seq,
            "node": self.node,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "iteration": self.iteration,
            "meta": self.meta,
            "error": self.error,
            "branch_of": self.branch_of,
        }


@dataclass
class RunContext:
    """Everything a node needs that must NOT live in graph state.

    Database sessions, the LLM handle and the clock are deliberately kept out of
    state: state should stay small, inspectable and safe to log, and a SQLAlchemy
    Session in a shared dict is a threading hazard once parallel branches run.
    """

    run_id: str
    db: Any = None
    #: Called with no args to mint a *new* Session for a parallel branch.
    #: Branches must never share the request-scoped `db` across threads.
    db_factory: Optional[Callable[[], Any]] = None
    llm: Optional[Callable[..., str]] = None
    deadline: Optional[float] = None  # monotonic timestamp
    traces: list[NodeTrace] = field(default_factory=list)
    counters: dict = field(default_factory=lambda: {"llm_calls": 0, "tool_calls": 0, "retries": 0})

    def time_left(self) -> Optional[float]:
        if self.deadline is None:
            return None
        return self.deadline - time.monotonic()

    def expired(self) -> bool:
        left = self.time_left()
        return left is not None and left <= 0


def _merge_into(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> None:
    """Apply a node's partial update. Dict values merge key-wise, others replace."""
    for key, value in update.items():
        if key.startswith("_"):  # runtime-private keys (e.g. _trace) never persist
            continue
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged = dict(current)
            merged.update(value)
            base[key] = merged
        else:
            base[key] = value


class _ParallelNode:
    """Marker node that runs several branch functions concurrently."""

    def __init__(self, branches: Mapping[str, NodeFn], max_workers: int = 4):
        self.branches = dict(branches)
        self.max_workers = max_workers


class StateGraph:
    """Builder. Mirrors the LangGraph surface the assistant uses."""

    def __init__(self) -> None:
        self._nodes: dict[str, Any] = {}
        self._edges: dict[str, str] = {}
        self._conditional: dict[str, tuple[RouterFn, dict[str, str]]] = {}
        self._entry: Optional[str] = None
        self._error_node: Optional[str] = None

    # -- construction -------------------------------------------------------
    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        if name in self._nodes:
            raise GraphError(f"duplicate node: {name}")
        if name == END:
            raise GraphError("'__end__' is reserved")
        self._nodes[name] = fn
        return self

    def add_parallel_node(
        self, name: str, branches: Mapping[str, NodeFn], max_workers: int = 4
    ) -> "StateGraph":
        """Fan out to independent branches, then join.

        Every branch receives the same read-only state snapshot and returns a
        partial update. Branches must write disjoint keys — see
        `ParallelMergeConflict`. Use this ONLY for genuinely independent work;
        for HRMS that means concurrent *reads*, never writes.
        """
        if not branches:
            raise GraphError(f"parallel node {name} has no branches")
        if name in self._nodes:
            raise GraphError(f"duplicate node: {name}")
        self._nodes[name] = _ParallelNode(branches, max_workers=max_workers)
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        if src in self._conditional:
            raise GraphError(f"{src} already has conditional edges")
        self._edges[src] = dst
        return self

    def add_conditional_edges(
        self, src: str, router: RouterFn, mapping: Mapping[str, str]
    ) -> "StateGraph":
        if src in self._edges:
            raise GraphError(f"{src} already has a static edge")
        self._conditional[src] = (router, dict(mapping))
        return self

    def set_entry(self, name: str) -> "StateGraph":
        self._entry = name
        return self

    def set_error_node(self, name: str) -> "StateGraph":
        """Node to jump to when a node raises. Gives every node an implicit
        failure edge without wrapping each one in try/except."""
        self._error_node = name
        return self

    # -- validation + compile ----------------------------------------------
    def _validate(self) -> None:
        if not self._entry:
            raise GraphError("no entry node set")
        known = set(self._nodes) | {END}
        if self._entry not in self._nodes:
            raise GraphError(f"entry node {self._entry!r} is not registered")
        if self._error_node and self._error_node not in self._nodes:
            raise GraphError(f"error node {self._error_node!r} is not registered")
        for src, dst in self._edges.items():
            if src not in self._nodes:
                raise GraphError(f"edge from unknown node {src!r}")
            if dst not in known:
                raise GraphError(f"edge {src} -> unknown node {dst!r}")
        for src, (_, mapping) in self._conditional.items():
            if src not in self._nodes:
                raise GraphError(f"conditional edge from unknown node {src!r}")
            for label, dst in mapping.items():
                if dst not in known:
                    raise GraphError(f"conditional edge {src}[{label}] -> unknown node {dst!r}")
        # Every node must have somewhere to go, or the run would dead-end.
        for name in self._nodes:
            if name not in self._edges and name not in self._conditional:
                raise GraphError(f"node {name!r} has no outgoing edge (use END explicitly)")

    def compile(self, max_steps: int = 40) -> "CompiledGraph":
        self._validate()
        return CompiledGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditional=dict(self._conditional),
            entry=self._entry,  # type: ignore[arg-type]
            error_node=self._error_node,
            max_steps=max_steps,
        )

    def mermaid(self) -> str:
        """Emit the graph as Mermaid so docs can never drift from the code."""
        lines = ["flowchart TD"]
        lines.append(f"    START([start]) --> {self._entry}")
        for src, dst in sorted(self._edges.items()):
            target = "END([end])" if dst == END else dst
            lines.append(f"    {src} --> {target}")
        for src, (_, mapping) in sorted(self._conditional.items()):
            for label, dst in sorted(mapping.items()):
                target = "END([end])" if dst == END else dst
                lines.append(f"    {src} -->|{label}| {target}")
        return "\n".join(lines)


class CompiledGraph:
    """Executable graph. Immutable; safe to build once and reuse per request."""

    def __init__(
        self,
        nodes: dict,
        edges: dict,
        conditional: dict,
        entry: str,
        error_node: Optional[str],
        max_steps: int,
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional
        self._entry = entry
        self._error_node = error_node
        self._max_steps = max_steps

    @property
    def node_names(self) -> list[str]:
        return list(self._nodes)

    def invoke(self, state: Mapping[str, Any], ctx: Optional[RunContext] = None) -> dict:
        """Run to END and return the final state.

        `max_steps` is the hard backstop against a cyclic graph looping forever.
        It is intentionally separate from the assistant's own `iteration` /
        `retry_budget` fields: those express *business* stopping rules, this one
        is the engine refusing to spin regardless of what the nodes believe.
        """
        ctx = ctx or RunContext(run_id=uuid.uuid4().hex[:12])
        working: dict = dict(state)
        current = self._entry
        seq = 0

        while current != END:
            if seq >= self._max_steps:
                logger.error(
                    "graph.step_limit run_id=%s node=%s steps=%d", ctx.run_id, current, seq
                )
                working["error"] = {
                    "kind": "graph_step_limit",
                    "message": f"Graph exceeded {self._max_steps} steps.",
                    "node": current,
                }
                working["status"] = "failed"
                current = self._error_node if self._error_node and not working.get(
                    "_in_error_node"
                ) else END
                if current == END:
                    break
                working["_in_error_node"] = True
                continue

            if ctx.expired():
                working["error"] = {
                    "kind": "timeout",
                    "message": "The assistant took too long to answer.",
                    "node": current,
                }
                working["status"] = "failed"
                if self._error_node and not working.get("_in_error_node"):
                    working["_in_error_node"] = True
                    current = self._error_node
                    continue
                break

            seq += 1
            node = self._nodes[current]
            started = time.monotonic()
            iteration = int(working.get("iteration", 0) or 0)

            try:
                if isinstance(node, _ParallelNode):
                    update = self._run_parallel(current, node, working, ctx, seq, iteration)
                else:
                    update = node(working, ctx) or {}
            except Exception as exc:  # noqa: BLE001 - the graph's failure edge
                ended = time.monotonic()
                ctx.traces.append(
                    NodeTrace(
                        run_id=ctx.run_id,
                        seq=seq,
                        node=current,
                        status="error",
                        started_at=started,
                        ended_at=ended,
                        duration_ms=(ended - started) * 1000,
                        iteration=iteration,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                logger.exception(
                    "graph.node_raised run_id=%s node=%s", ctx.run_id, current
                )
                working["error"] = {
                    "kind": "node_exception",
                    "message": "An internal step failed.",
                    "node": current,
                    "exception": type(exc).__name__,
                }
                working["status"] = "failed"
                if self._error_node and current != self._error_node:
                    current = self._error_node
                    continue
                break

            ended = time.monotonic()
            meta = dict(update.get("_trace") or {}) if isinstance(update, Mapping) else {}
            ctx.traces.append(
                NodeTrace(
                    run_id=ctx.run_id,
                    seq=seq,
                    node=current,
                    status="ok",
                    started_at=started,
                    ended_at=ended,
                    duration_ms=(ended - started) * 1000,
                    iteration=iteration,
                    meta=meta,
                )
            )
            _merge_into(working, update)

            # Edge selection can fail too (a predicate returning a label the
            # mapping doesn't cover). That is a graph bug, but it must not 500
            # the chat endpoint — route it down the same failure edge.
            try:
                current = self._next(current, working)
            except GraphError as exc:
                logger.error("graph.bad_route run_id=%s node=%s: %s", ctx.run_id, current, exc)
                working["error"] = {
                    "kind": "node_exception",
                    "message": "An internal step failed.",
                    "node": current,
                    "exception": GraphError.__name__,
                }
                working["status"] = "failed"
                if self._error_node and current != self._error_node:
                    current = self._error_node
                    continue
                break

        working.pop("_in_error_node", None)
        return working

    # -- internals ----------------------------------------------------------
    def _next(self, current: str, state: Mapping[str, Any]) -> str:
        if current in self._edges:
            return self._edges[current]
        router, mapping = self._conditional[current]
        label = router(state)
        if label not in mapping:
            raise GraphError(
                f"router for {current!r} returned {label!r}, "
                f"which is not one of {sorted(mapping)}"
            )
        return mapping[label]

    def _run_parallel(
        self,
        name: str,
        node: _ParallelNode,
        state: Mapping[str, Any],
        ctx: RunContext,
        seq: int,
        iteration: int,
    ) -> dict:
        """Run branches concurrently and merge disjoint partial updates."""
        snapshot = dict(state)  # branches get a read-only copy; no shared mutation
        results: dict[str, Mapping[str, Any]] = {}
        errors: dict[str, str] = {}

        def _run(branch_name: str, fn: NodeFn) -> None:
            started = time.monotonic()
            meta: dict = {}
            try:
                update = fn(snapshot, ctx) or {}
                results[branch_name] = update
                meta = dict(update.get("_trace") or {}) if isinstance(update, Mapping) else {}
                # A branch that caught its own tool failure returns normally. Its
                # trace must still read as a failure, or the log says "ok" for a
                # step that produced nothing.
                if meta.get("status") == "error":
                    status, err = "error", meta.get("error")
                else:
                    status, err = "ok", None
            except Exception as exc:  # noqa: BLE001 - recorded, then surfaced below
                errors[branch_name] = f"{type(exc).__name__}: {exc}"
                status, err = "error", errors[branch_name]
                logger.exception(
                    "graph.branch_raised run_id=%s node=%s branch=%s",
                    ctx.run_id, name, branch_name,
                )
            ended = time.monotonic()
            ctx.traces.append(
                NodeTrace(
                    run_id=ctx.run_id,
                    seq=seq,
                    node=f"{name}:{branch_name}",
                    status=status,
                    started_at=started,
                    ended_at=ended,
                    duration_ms=(ended - started) * 1000,
                    iteration=iteration,
                    meta=meta,
                    error=err,
                    branch_of=name,
                )
            )

        with ThreadPoolExecutor(
            max_workers=min(node.max_workers, len(node.branches)),
            thread_name_prefix=f"graph-{name}",
        ) as pool:
            list(pool.map(lambda kv: _run(*kv), node.branches.items()))

        merged: dict = {}
        owner: dict[str, str] = {}
        for branch_name, update in results.items():
            for key, value in update.items():
                if key.startswith("_"):
                    continue
                if key in merged:
                    if isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key] = {**merged[key], **value}
                        continue
                    raise ParallelMergeConflict(
                        f"branches {owner[key]!r} and {branch_name!r} both wrote "
                        f"non-mergeable state key {key!r} in parallel node {name!r}"
                    )
                merged[key] = value
                owner[key] = branch_name

        # A branch failure is data, not a crash: the validator decides whether a
        # partial result set is still answerable.
        if errors:
            merged.setdefault("tool_errors", {})
            merged["tool_errors"] = {
                **merged["tool_errors"],
                **{b: {"kind": "tool_failure", "message": msg} for b, msg in errors.items()},
            }
        return merged
