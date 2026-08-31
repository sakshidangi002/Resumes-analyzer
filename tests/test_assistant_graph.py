"""Graph-level tests: whole runs, asserted on the path taken and the outcome.

External dependencies are substituted at the tool boundary — no database, no
network, no LLM. Everything between the entry node and the response generator is
the real implementation.
"""
from datetime import date

import pytest

from app.assistant.graph import build_graph, run_assistant
from app.assistant.nodes import analyzer as analyzer_mod
from app.assistant.tools import attendance_tools


class _OperationalError(Exception):
    """Named to match what `failures.classify_exception` treats as transient."""


_OperationalError.__name__ = "OperationalError"


GOOD_SUMMARY = {
    "month": 2, "year": 2026, "total_calendar_days": 28, "working_days": 20,
    "present": 18, "half_day": 0, "leave": 1, "absent": 1,
    "holiday": 1, "weekly_off": 7, "attendance_percentage": 94.7,
}
GOOD_CONTEXT = {
    "employee_found": True, "date_of_joining": "2020-01-01",
    "is_active": True, "joined_after_period": False,
}


@pytest.fixture(autouse=True)
def frozen_today(monkeypatch):
    class _Now:
        @staticmethod
        def date():
            return date(2026, 3, 15)

    monkeypatch.setattr(analyzer_mod, "get_ist_now", lambda: _Now())


@pytest.fixture
def stub_tools(monkeypatch):
    """Replace the three attendance tools; return a call log."""
    calls = {"monthly_summary": 0, "employee_period_context": 0, "late_mark_count": 0}

    def _install(summary_fn=None, context_fn=None, late_fn=None):
        def _wrap(name, fn):
            def _inner(db, **kwargs):
                calls[name] += 1
                return fn(db, **kwargs)
            monkeypatch.setattr(attendance_tools, name, _inner)

        _wrap("monthly_summary", summary_fn or (lambda db, **kw: dict(GOOD_SUMMARY)))
        _wrap("employee_period_context", context_fn or (lambda db, **kw: dict(GOOD_CONTEXT)))
        _wrap("late_mark_count", late_fn or (lambda db, **kw: {"late_days": 2}))
        return calls

    return _install


def _run(message, *, roles=("Employee",), employee_id=5):
    return run_assistant(
        message=message,
        actor_user_id=1,
        actor_employee_id=employee_id,
        actor_roles=list(roles),
        db=object(),          # branches fall back to ctx.db when db_factory is None
        db_factory=None,
        llm=None,
        timeout_seconds=10,
    )


def _nodes(result):
    return [t["node"] for t in result["trace"]]


# --------------------------------------------------------------------------
# 1-4: routing
# --------------------------------------------------------------------------
def test_attendance_question_reaches_the_attendance_node(stub_tools):
    stub_tools()
    result = _run("How many days was I absent last month?")

    assert result["status"] == "ok"
    assert result["intent"] == "attendance"
    assert "attendance_fetch:monthly_summary" in _nodes(result)
    assert "result_validator" in _nodes(result)
    assert result["data"]["period"] == {
        "month": 2, "year": 2026, "label": "last month", "assumed": False,
    }
    assert "18 day(s)" in result["response"]


def test_leave_question_falls_back_and_names_the_capability(stub_tools):
    calls = stub_tools()
    result = _run("How many leaves do I have?")

    assert result["status"] == "unsupported"
    assert "fallback_node" in _nodes(result)
    assert "leave" in result["response"].lower()
    assert calls["monthly_summary"] == 0, "fallback must not hit attendance tools"


def test_payroll_question_falls_back(stub_tools):
    stub_tools()
    result = _run("Show my payslip.")
    assert result["status"] == "unsupported"
    assert "payroll" in result["response"].lower()


def test_unknown_question_gets_the_capability_hint(stub_tools):
    stub_tools()
    result = _run("What is the office wifi password?")
    assert result["status"] == "unsupported"
    assert "attendance" in result["response"].lower()


# --------------------------------------------------------------------------
# 5: authorisation short-circuits the whole graph
# --------------------------------------------------------------------------
def test_employee_asking_about_a_colleague_never_reaches_a_tool(stub_tools):
    calls = stub_tools()
    result = _run("How many days was Priya Sharma absent last month?", roles=("Employee",))

    assert result["status"] == "denied"
    assert calls["monthly_summary"] == 0
    visited = _nodes(result)
    assert "intent_router" not in visited, "denial must precede routing"
    assert not any(n.startswith("attendance_fetch") for n in visited)
    assert result["data"] is None


# --------------------------------------------------------------------------
# 6-7: the recovery loop
# --------------------------------------------------------------------------
def test_transient_failure_is_retried_and_then_succeeds(stub_tools):
    state = {"n": 0}

    def flaky(db, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise _OperationalError("connection reset")
        return dict(GOOD_SUMMARY)

    stub_tools(summary_fn=flaky)
    result = _run("attendance last month")

    assert result["status"] == "ok"
    assert result["iteration"] == 1
    assert result["metrics"]["retries"] == 1
    assert state["n"] == 2
    assert "recovery_node" in _nodes(result)


def test_a_persistent_failure_stops_instead_of_looping(stub_tools):
    attempts = {"n": 0}

    def always_fails(db, **kwargs):
        attempts["n"] += 1
        raise _OperationalError("still down")

    stub_tools(summary_fn=always_fails)
    result = _run("attendance last month")

    assert result["status"] == "failed"
    # One initial attempt + at most the retry budget. The identical failure
    # signature recurring is what ends it, not exhaustion of the budget.
    assert attempts["n"] <= 3, f"loop ran {attempts['n']} times"
    assert result["iteration"] <= 2
    assert result["response"]


def test_validation_failure_degrades_honestly(stub_tools):
    """Internally inconsistent data must not be reported as verified fact."""
    bad = dict(GOOD_SUMMARY, working_days=25)  # breaks the calendar identity
    stub_tools(summary_fn=lambda db, **kw: bad)
    result = _run("attendance last month")

    assert result["status"] == "failed"
    assert result["validation"]["ok"] is False
    assert "failure_classifier" in _nodes(result)
    assert "couldn't fully verify" in result["response"]


def test_business_rule_failure_is_never_retried(stub_tools):
    calls = stub_tools(
        context_fn=lambda db, **kw: dict(GOOD_CONTEXT, joined_after_period=True)
    )
    result = _run("attendance last month")

    assert result["status"] == "failed"
    assert calls["monthly_summary"] == 1, "a business-rule failure must not re-fetch"
    assert result["metrics"]["retries"] == 0


# --------------------------------------------------------------------------
# 8-10: parallelism, degradation, observability
# --------------------------------------------------------------------------
def test_all_three_branches_execute_in_the_fan_out(stub_tools):
    calls = stub_tools()
    result = _run("attendance last month")

    assert calls == {"monthly_summary": 1, "employee_period_context": 1, "late_mark_count": 1}
    branch_nodes = [n for n in _nodes(result) if n.startswith("attendance_fetch:")]
    assert len(branch_nodes) == 3
    assert result["data"]["late_days"] == 2


def test_a_failed_enrichment_branch_still_answers(stub_tools):
    stub_tools(late_fn=lambda db, **kw: (_ for _ in ()).throw(_OperationalError("boom")))
    result = _run("attendance last month")

    assert result["status"] == "ok", "a non-critical branch must not fail the answer"
    assert "late_mark_count" in result["tool_errors"]
    assert "Late marks" not in result["response"]


def test_run_emits_a_complete_trace_and_metrics(stub_tools):
    stub_tools()
    result = _run("attendance last month")

    assert result["run_id"]
    assert result["metrics"]["node_count"] == len(result["trace"])
    assert result["metrics"]["tool_calls"] == 3
    assert result["metrics"]["llm_calls"] == 0
    for entry in result["trace"]:
        assert {"run_id", "seq", "node", "status", "duration_ms"} <= set(entry)


def test_trace_contains_no_employee_identifying_text(stub_tools):
    """The trace is logged; it must carry decisions, not personal data."""
    stub_tools()
    result = _run("How many days was I absent last month?")
    blob = str(result["trace"]).lower()
    assert "priya" not in blob
    assert "absent last month" not in blob, "the raw message must not appear in traces"


# --------------------------------------------------------------------------
# topology
# --------------------------------------------------------------------------
def test_graph_compiles_and_every_edge_resolves():
    build_graph().compile()  # raises GraphError on any dangling edge


def test_mermaid_reflects_the_real_topology():
    diagram = build_graph().mermaid()
    assert "request_analyzer --> scope_resolver" in diagram
    assert "recovery_node -->|retry| attendance_fetch" in diagram
    assert "result_validator -->|fail| failure_classifier" in diagram
