"""Node-level unit tests. No database, no graph, no LLM — one function at a time."""
from datetime import date

import pytest

from app.assistant import failures
from app.assistant.nodes import analyzer as analyzer_mod
from app.assistant.nodes.analyzer import request_analyzer
from app.assistant.nodes.diagnosis import CLARIFY, DEGRADE, RETRY, failure_classifier
from app.assistant.nodes.recovery import recovery_node
from app.assistant.nodes.response import response_generator
from app.assistant.nodes.router import intent_router, score_intents
from app.assistant.nodes.terminal import fallback_node
from app.assistant.nodes.validator import result_validator
from app.assistant.runtime import RunContext
from app.assistant.state import INTENT_ATTENDANCE, INTENT_LEAVE, INTENT_PAYROLL, INTENT_UNKNOWN


@pytest.fixture
def ctx():
    return RunContext(run_id="test")


@pytest.fixture
def frozen_today(monkeypatch):
    """Pin 'now' to 2026-03-15 IST so period parsing is deterministic."""
    class _Now:
        @staticmethod
        def date():
            return date(2026, 3, 15)

    monkeypatch.setattr(analyzer_mod, "get_ist_now", lambda: _Now())
    return date(2026, 3, 15)


# --------------------------------------------------------------------------
# request_analyzer
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message,expected_month,expected_year",
    [
        ("How many days was I absent last month?", 2, 2026),
        ("What is my attendance this month?", 3, 2026),
        ("my attendance in January", 1, 2026),
        ("attendance for December", 12, 2025),      # bare future month -> last year
        ("attendance for March 2025", 3, 2025),
        ("attendance 02/2026", 2, 2026),
        ("3 months ago attendance", 12, 2025),
    ],
)
def test_analyzer_resolves_period(message, expected_month, expected_year, ctx, frozen_today):
    out = request_analyzer({"message": message}, ctx)
    assert out["entities"]["month"] == expected_month
    assert out["entities"]["year"] == expected_year
    assert out["entities"]["period_assumed"] is False


def test_analyzer_defaults_to_current_month_and_flags_the_assumption(ctx, frozen_today):
    out = request_analyzer({"message": "how many days was I absent?"}, ctx)
    assert (out["entities"]["month"], out["entities"]["year"]) == (3, 2026)
    assert out["entities"]["period_assumed"] is True


def test_analyzer_marks_unsupported_period_granularity(ctx, frozen_today):
    out = request_analyzer({"message": "my attendance last quarter"}, ctx)
    assert out["entities"]["period_supported"] is False


@pytest.mark.parametrize(
    "message,hint",
    [
        ("how many days was I absent", "self"),
        ("what is my attendance", "self"),
        ("attendance for Priya Sharma", "other"),
        ("show attendance for EMP014", "other"),
        ("how many days was she absent", "other"),
        ("my team's attendance", "team"),
    ],
)
def test_analyzer_detects_subject(message, hint, ctx, frozen_today):
    assert request_analyzer({"message": message}, ctx)["entities"]["subject_hint"] == hint


# --------------------------------------------------------------------------
# intent_router
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message,expected",
    [
        ("How many days was I absent last month?", INTENT_ATTENDANCE),
        ("what is my attendance percentage", INTENT_ATTENDANCE),
        ("how many late marks do I have", INTENT_ATTENDANCE),
        ("How many leaves do I have?", INTENT_LEAVE),
        ("what is my leave balance", INTENT_LEAVE),
        ("Show my payslip.", INTENT_PAYROLL),
        ("why was my salary lower", INTENT_PAYROLL),
        ("what is the wifi password", INTENT_UNKNOWN),
    ],
)
def test_router_classifies_deterministically(message, expected, ctx):
    out = intent_router({"normalized": message}, ctx)
    assert out["intent"] == expected
    assert out["intent_method"] == "rule"   # no LLM consulted
    assert ctx.counters["llm_calls"] == 0


def test_router_scores_are_inspectable():
    scores = score_intents("how many days was I absent last month")
    assert scores[INTENT_ATTENDANCE] > scores[INTENT_LEAVE]


def test_router_honours_a_recovery_override(ctx):
    out = intent_router({"normalized": "anything", "forced_intent": INTENT_LEAVE}, ctx)
    assert out["intent"] == INTENT_LEAVE
    assert out["intent_method"] == "recovery_override"
    assert out["forced_intent"] is None  # cleared, so it can't loop


def test_router_llm_tiebreak_is_rejected_when_off_label(ctx):
    ctx.llm = lambda messages, max_new_tokens=6: "definitely attendance i think"
    # Ambiguous input -> tiebreak attempted -> answer isn't a bare label -> ignored.
    out = intent_router({"normalized": "leave holiday"}, ctx)
    assert out["intent_method"] == "rule"


# --------------------------------------------------------------------------
# result_validator
# --------------------------------------------------------------------------
def _summary(**overrides):
    base = {
        "month": 2, "year": 2026, "total_calendar_days": 28, "working_days": 20,
        "present": 18, "half_day": 0, "leave": 1, "absent": 1,
        "holiday": 1, "weekly_off": 7, "attendance_percentage": 94.7,
    }
    base.update(overrides)
    return base


def test_validator_passes_a_consistent_summary(ctx):
    out = result_validator({"tool_results": {"monthly_summary": _summary()}}, ctx)
    assert out["validation"]["ok"] is True
    assert "error" not in out


def test_validator_fails_when_the_critical_tool_is_missing(ctx):
    out = result_validator(
        {"tool_results": {}, "tool_errors": {"monthly_summary": {"kind": failures.TRANSIENT}}},
        ctx,
    )
    assert out["validation"]["ok"] is False
    assert out["error"]["kind"] == failures.TRANSIENT   # propagates for the retry decision


def test_validator_catches_arithmetic_inconsistency(ctx):
    # working_days must equal total - holiday - weekly_off
    out = result_validator({"tool_results": {"monthly_summary": _summary(working_days=25)}}, ctx)
    assert out["validation"]["ok"] is False
    assert "working_days_mismatch" in [f["code"] for f in out["validation"]["failures"]]


def test_validator_catches_bucket_overflow(ctx):
    out = result_validator({"tool_results": {"monthly_summary": _summary(present=30)}}, ctx)
    assert "bucket_overflow" in [f["code"] for f in out["validation"]["failures"]]


def test_validator_rejects_a_period_before_the_joining_date(ctx):
    out = result_validator(
        {
            "tool_results": {
                "monthly_summary": _summary(),
                "employee_period_context": {
                    "employee_found": True, "joined_after_period": True,
                    "date_of_joining": "2026-06-01",
                },
            }
        },
        ctx,
    )
    assert out["error"]["kind"] == failures.BUSINESS_RULE


def test_validator_tolerates_a_failed_enrichment_branch(ctx):
    """late_mark_count failing must not fail the answer."""
    out = result_validator(
        {
            "tool_results": {"monthly_summary": _summary()},
            "tool_errors": {"late_mark_count": {"kind": failures.TOOL_FAILURE}},
        },
        ctx,
    )
    assert out["validation"]["ok"] is True


# --------------------------------------------------------------------------
# failure_classifier + recovery_node  (loop engineering)
# --------------------------------------------------------------------------
def test_transient_failure_with_budget_is_retried(ctx):
    out = failure_classifier(
        {"error": {"kind": failures.TRANSIENT}, "iteration": 0, "retry_budget": 2}, ctx
    )
    assert out["recovery_action"] == RETRY


def test_transient_failure_without_budget_degrades(ctx):
    out = failure_classifier(
        {"error": {"kind": failures.TRANSIENT}, "iteration": 0, "retry_budget": 0}, ctx
    )
    assert out["recovery_action"] == DEGRADE
    assert out["_trace"]["reason"] == "retry_budget_exhausted"


def test_a_repeated_failure_signature_stops_the_loop(ctx):
    state = {
        "error": {"kind": failures.TRANSIENT, "node": "attendance_fetch"},
        "iteration": 1,
        "retry_budget": 5,          # budget remains, but the failure already recurred
        "failure_history": [failures.signature(failures.TRANSIENT, "attendance_fetch")],
    }
    out = failure_classifier(state, ctx)
    assert out["recovery_action"] == DEGRADE
    assert out["_trace"]["reason"] == "repeated_failure"


def test_max_iterations_stops_the_loop(ctx):
    out = failure_classifier(
        {"error": {"kind": failures.TRANSIENT}, "iteration": 2, "retry_budget": 5}, ctx
    )
    assert out["recovery_action"] == DEGRADE
    assert out["_trace"]["reason"] == "max_iterations"


def test_terminal_kinds_are_never_retried(ctx):
    for kind in (failures.AUTHZ_FAILURE, failures.BUSINESS_RULE, failures.UNSUPPORTED):
        out = failure_classifier({"error": {"kind": kind}, "iteration": 0, "retry_budget": 5}, ctx)
        assert out["recovery_action"] == DEGRADE, kind


def test_missing_entity_asks_the_user(ctx):
    out = failure_classifier(
        {"error": {"kind": failures.MISSING_ENTITY}, "iteration": 0, "retry_budget": 5}, ctx
    )
    assert out["recovery_action"] == CLARIFY


def test_recovery_retry_consumes_budget_and_clears_residue(ctx):
    out = recovery_node(
        {
            "recovery_action": RETRY,
            "iteration": 0,
            "retry_budget": 2,
            "error": {"kind": failures.TRANSIENT},
            "tool_errors": {"monthly_summary": {"kind": failures.TRANSIENT}},
        },
        ctx,
    )
    assert (out["iteration"], out["retry_budget"]) == (1, 1)
    assert out["error"] is None and out["tool_errors"] == {}
    assert ctx.counters["retries"] == 1


def test_recovery_degrade_marks_the_run_failed(ctx):
    assert recovery_node({"recovery_action": DEGRADE}, ctx)["status"] == "failed"


# --------------------------------------------------------------------------
# response_generator
# --------------------------------------------------------------------------
def test_response_renders_the_summary_deterministically(ctx):
    out = response_generator(
        {
            "status": "running",
            "entities": {"period_label": "last month", "period_assumed": False},
            "tool_results": {"monthly_summary": _summary(), "late_mark_count": {"late_days": 3}},
        },
        ctx,
    )
    assert out["status"] == "ok"
    assert "18 day(s)" in out["response"] and "94.7%" in out["response"]
    assert "Late marks: 3" in out["response"]
    assert out["data"]["summary"]["present"] == 18


def test_response_states_an_assumed_period(ctx):
    out = response_generator(
        {
            "status": "running",
            "entities": {"period_label": "this month", "period_assumed": True},
            "tool_results": {"monthly_summary": _summary()},
        },
        ctx,
    )
    assert "assumed" in out["response"].lower()


def test_response_discards_an_llm_rewrite_that_changes_the_numbers(ctx):
    ctx.llm = lambda messages, max_new_tokens=120: "You were present 25 days. Great work!"
    out = response_generator(
        {
            "status": "running",
            "entities": {"period_label": "last month"},
            "tool_results": {"monthly_summary": _summary()},
        },
        ctx,
    )
    assert "25 days" not in out["response"]
    assert "18 day(s)" in out["response"]      # template survived
    assert out["_trace"]["polished"] is False


def test_response_accepts_a_faithful_llm_rewrite(ctx):
    # Reworded, but every figure preserved — this is the case polish exists for.
    faithful = (
        "You were present for 18 of 20 working days last month, with 1 absence "
        "and 1 day of approved leave, giving 94.7% attendance."
    )
    ctx.llm = lambda messages, max_new_tokens=120: faithful
    out = response_generator(
        {
            "status": "running",
            "entities": {"period_label": "last month"},
            "tool_results": {"monthly_summary": _summary()},
        },
        ctx,
    )
    assert out["_trace"]["polished"] is True


def test_response_for_denied_scope_reveals_nothing(ctx):
    out = response_generator({"status": "denied", "denial_reason": "role"}, ctx)
    assert "permission" in out["response"].lower()
    assert out["data"] is None


def test_fallback_names_a_recognised_but_unbuilt_capability(ctx):
    out = fallback_node({"intent": INTENT_PAYROLL}, ctx)
    assert out["status"] == "unsupported"
    assert "payroll" in out["error"]["message"].lower()
