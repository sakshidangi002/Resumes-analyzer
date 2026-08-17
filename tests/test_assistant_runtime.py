"""Graph runtime primitives.

These test the engine, not the HRM assistant: nodes here are toys. If the engine
is wrong, every node-level test above it is testing on sand.
"""
import time

import pytest

from app.assistant.runtime import (
    END,
    GraphError,
    ParallelMergeConflict,
    RunContext,
    StateGraph,
)


def _ctx():
    return RunContext(run_id="test")


def test_linear_graph_runs_nodes_in_order():
    g = StateGraph()
    g.add_node("a", lambda s, c: {"seen": s.get("seen", []) + ["a"]})
    g.add_node("b", lambda s, c: {"seen": s.get("seen", []) + ["b"]})
    g.set_entry("a").add_edge("a", "b").add_edge("b", END)

    final = g.compile().invoke({"seen": []}, _ctx())
    assert final["seen"] == ["a", "b"]


def test_partial_updates_merge_dicts_key_wise():
    """A node returning {"d": {...}} must not clobber other keys in d."""
    g = StateGraph()
    g.add_node("a", lambda s, c: {"d": {"x": 1}})
    g.add_node("b", lambda s, c: {"d": {"y": 2}})
    g.set_entry("a").add_edge("a", "b").add_edge("b", END)

    assert g.compile().invoke({"d": {}}, _ctx())["d"] == {"x": 1, "y": 2}


def test_conditional_edges_pick_branch_by_label():
    g = StateGraph()
    g.add_node("start", lambda s, c: {})
    g.add_node("left", lambda s, c: {"took": "left"})
    g.add_node("right", lambda s, c: {"took": "right"})
    g.set_entry("start")
    g.add_conditional_edges("start", lambda s: s["choice"], {"l": "left", "r": "right"})
    g.add_edge("left", END)
    g.add_edge("right", END)
    compiled = g.compile()

    assert compiled.invoke({"choice": "l"}, _ctx())["took"] == "left"
    assert compiled.invoke({"choice": "r"}, _ctx())["took"] == "right"


def test_router_returning_unmapped_label_is_an_error():
    g = StateGraph()
    g.add_node("start", lambda s, c: {})
    g.add_node("left", lambda s, c: {})
    g.set_entry("start")
    g.add_conditional_edges("start", lambda s: "nope", {"l": "left"})
    g.add_edge("left", END)

    # Raised inside a node execution -> caught by the runtime, surfaced as error
    # state rather than escaping to the caller.
    final = g.compile().invoke({}, _ctx())
    assert final["status"] == "failed"


def test_parallel_branches_run_concurrently_and_merge():
    def slow(tag):
        def _fn(s, c):
            time.sleep(0.15)
            return {"tool_results": {tag: tag}}
        return _fn

    g = StateGraph()
    g.add_parallel_node("fan", {"a": slow("a"), "b": slow("b"), "c": slow("c")})
    g.set_entry("fan").add_edge("fan", END)

    started = time.monotonic()
    final = g.compile().invoke({"tool_results": {}}, _ctx())
    elapsed = time.monotonic() - started

    assert final["tool_results"] == {"a": "a", "b": "b", "c": "c"}
    # Three 0.15s sleeps sequentially would be 0.45s; concurrently ~0.15s.
    assert elapsed < 0.40, f"branches did not run concurrently (took {elapsed:.2f}s)"


def test_parallel_branch_failure_becomes_tool_error_not_a_crash():
    def boom(s, c):
        raise RuntimeError("branch exploded")

    g = StateGraph()
    g.add_parallel_node("fan", {"ok": lambda s, c: {"tool_results": {"ok": 1}}, "bad": boom})
    g.set_entry("fan").add_edge("fan", END)

    final = g.compile().invoke({}, _ctx())
    assert final["tool_results"] == {"ok": 1}
    assert "bad" in final["tool_errors"]


def test_parallel_conflicting_non_dict_writes_are_rejected():
    g = StateGraph()
    g.add_parallel_node("fan", {"a": lambda s, c: {"x": 1}, "b": lambda s, c: {"x": 2}})
    g.set_entry("fan").add_edge("fan", END)

    # The conflict raises inside the node, so it surfaces as a failed run.
    final = g.compile().invoke({}, _ctx())
    assert final["status"] == "failed"
    assert final["error"]["exception"] == ParallelMergeConflict.__name__


def test_cycle_is_bounded_by_max_steps():
    """An unconditional cycle must terminate, not hang."""
    g = StateGraph()
    g.add_node("a", lambda s, c: {"n": s.get("n", 0) + 1})
    g.add_node("b", lambda s, c: {})
    g.set_entry("a").add_edge("a", "b").add_edge("b", "a")

    final = g.compile(max_steps=10).invoke({}, _ctx())
    assert final["status"] == "failed"
    assert final["error"]["kind"] == "graph_step_limit"


def test_node_exception_routes_to_error_node():
    def boom(s, c):
        raise ValueError("nope")

    g = StateGraph()
    g.add_node("a", boom)
    g.add_node("handler", lambda s, c: {"handled": True})
    g.set_entry("a").set_error_node("handler")
    g.add_edge("a", END).add_edge("handler", END)

    final = g.compile().invoke({}, _ctx())
    assert final["handled"] is True
    assert final["error"]["kind"] == "node_exception"


def test_deadline_stops_the_run():
    g = StateGraph()
    g.add_node("a", lambda s, c: {})
    g.set_entry("a").add_edge("a", END)

    ctx = RunContext(run_id="t", deadline=time.monotonic() - 1)  # already expired
    final = g.compile().invoke({}, ctx)
    assert final["error"]["kind"] == "timeout"


def test_traces_capture_every_node_execution():
    g = StateGraph()
    g.add_node("a", lambda s, c: {"_trace": {"note": "hello"}})
    g.add_node("b", lambda s, c: {})
    g.set_entry("a").add_edge("a", "b").add_edge("b", END)

    ctx = _ctx()
    final = g.compile().invoke({}, ctx)

    assert [t.node for t in ctx.traces] == ["a", "b"]
    assert ctx.traces[0].meta == {"note": "hello"}
    assert all(t.status == "ok" for t in ctx.traces)
    # _trace is runtime-private and must never leak into state.
    assert "_trace" not in final


def test_construction_errors_are_caught_at_compile_time():
    g = StateGraph()
    g.add_node("a", lambda s, c: {})
    g.set_entry("a").add_edge("a", "missing")
    with pytest.raises(GraphError):
        g.compile()

    g2 = StateGraph()
    g2.add_node("a", lambda s, c: {})
    g2.set_entry("a")  # no outgoing edge
    with pytest.raises(GraphError):
        g2.compile()
