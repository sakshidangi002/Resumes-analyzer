"""Node-level tests. Each node called directly with a hand-built state.

No filesystem, no subprocess: the repository and test runner are fakes, so a
failing assertion points at node logic rather than at the environment.
"""
import pytest

from graph_engine.config import EngineConfig
from graph_engine.context import EngineContext
from graph_engine.nodes.bug_analysis import bug_analysis
from graph_engine.nodes.failure_analysis import (
    COLLECTION_ERROR,
    CONFIGURATION,
    DEPENDENCY,
    IMPLEMENTATION_BUG,
    PRE_EXISTING,
    TIMEOUT,
    failure_analysis,
    signature,
)
from graph_engine.nodes.fix import fix
from graph_engine.nodes.report import report
from graph_engine.nodes.review import review
from graph_engine.nodes.testing import run_tests, select_targets
from graph_engine.nodes.verify import verify
from graph_engine.tools import ast_checks


class FakeRepo:
    def __init__(self, files=None, contents=None):
        self._files = files or []
        self._contents = contents or {}
        self.written = {}

    def list_python_files(self, scope=None):
        if scope == "tests":
            return [f for f in self._files if f.startswith("tests/")]
        return [f for f in self._files if not f.startswith("tests/")]

    def list_source_files(self, scope=None, suffixes=None):
        return self.list_python_files(scope)

    def read(self, rel):
        return self._contents.get(rel, "")

    def write(self, rel, content, reason):
        self.written[rel] = content


class FakeTests:
    """Returns a queued result per call, so a run can fail then pass."""

    __test__ = False

    def __init__(self, *results):
        self._results = list(results)
        self.calls = []

    def run(self, targets, *, stage):
        self.calls.append((tuple(targets), stage))
        result = self._results.pop(0) if self._results else _result(ok=True)
        return {**result, "stage": stage}


def _result(*, ok=True, failures=(), passed=100, timed_out=False, tail=""):
    return {
        "stage": "targeted", "command": "pytest", "exit_code": 0 if ok else 1,
        "timed_out": timed_out, "ok": ok, "passed": passed,
        "failed": len(failures), "skipped": 0, "duration_s": 1.0,
        "failures": [{"nodeid": n, "message": m} for n, m in failures],
        "tail": tail,
    }


def _ctx(repo=None, tests=None, **config_kwargs):
    config = EngineConfig(goal="g", scope="pkg", **config_kwargs)
    return EngineContext(run_id="t", config=config, repo=repo or FakeRepo(), tests=tests)


# ==========================================================================
# review
# ==========================================================================
def test_review_reports_findings_and_writes_nothing(monkeypatch):
    monkeypatch.setattr("graph_engine.nodes.review.linters.run_ruff", lambda *a, **k: [])
    repo = FakeRepo(files=["pkg/a.py"], contents={"pkg/a.py": "if x == None:\n    pass\n"})
    out = review({}, _ctx(repo))

    assert out["files"] == ["pkg/a.py"]
    assert any(f["code"] == ast_checks.CMP_NONE for f in out["review_findings"])
    assert repo.written == {}, "the review node must never modify anything"


def test_review_sorts_high_severity_first(monkeypatch):
    monkeypatch.setattr(
        "graph_engine.nodes.review.linters.run_ruff",
        lambda *a, **k: [
            {"file": "pkg/a.py", "line": 9, "code": "F401", "severity": "medium",
             "problem": "unused", "source": "ruff", "ruff_fixable": True},
            {"file": "pkg/a.py", "line": 2, "code": "F821", "severity": "high",
             "problem": "undefined", "source": "ruff", "ruff_fixable": False},
        ],
    )
    out = review({}, _ctx(FakeRepo(files=["pkg/a.py"], contents={"pkg/a.py": ""})))
    assert [f["severity"] for f in out["review_findings"]][:2] == ["high", "medium"]


def test_review_counts_ast_findings_without_going_negative(monkeypatch):
    """Deriving the AST count from `len(findings) - ruff_count` went negative
    once dedupe removed more overlaps than the AST checks contributed."""
    duplicate = {"file": "pkg/a.py", "line": 1, "code": "GE001", "severity": "low",
                 "problem": "dup", "source": "ruff", "ruff_fixable": False}
    monkeypatch.setattr(
        "graph_engine.nodes.review.linters.run_ruff",
        lambda *a, **k: [duplicate, {**duplicate, "line": 2}, {**duplicate, "line": 3}],
    )
    # The AST check reports the same (file, line, code) as the first ruff finding.
    repo = FakeRepo(files=["pkg/a.py"], contents={"pkg/a.py": "if x == None:\n    pass\n"})
    out = review({}, _ctx(repo))

    assert out["_trace"]["ast_findings"] >= 0
    assert out["_trace"]["ruff_findings"] == 3
    assert out["_trace"]["deduped"] >= 0
    assert len(out["review_findings"]) == 3 + out["_trace"]["ast_findings"] - out["_trace"]["deduped"]


def test_review_handles_an_empty_scope(monkeypatch):
    monkeypatch.setattr("graph_engine.nodes.review.linters.run_ruff", lambda *a, **k: [])
    out = review({}, _ctx(FakeRepo(files=[])))
    assert out["files"] == [] and out["review_findings"] == []


# ==========================================================================
# bug analysis (triage)
# ==========================================================================
def _finding(**kw):
    base = {"file": "pkg/a.py", "line": 1, "code": "F401", "severity": "medium",
            "problem": "p", "source": "ruff", "ruff_fixable": True}
    base.update(kw)
    return base


def test_sqlalchemy_none_comparison_is_dismissed_not_fixed():
    """The rule that stops the engine breaking every soft-delete query."""
    state = {"review_findings": [
        _finding(code=ast_checks.CMP_NONE, in_query_context=True, severity="low")
    ]}
    out = bug_analysis(state, _ctx())

    assert out["bugs"] == []
    assert len(out["dismissed"]) == 1
    assert "SQLAlchemy" in out["dismissed"][0]["dismiss_reason"]


def test_plain_none_comparison_is_a_real_auto_fixable_bug():
    state = {"review_findings": [
        _finding(code=ast_checks.CMP_NONE, in_query_context=False, severity="low")
    ]}
    out = bug_analysis(state, _ctx())

    assert len(out["bugs"]) == 1
    assert out["bugs"][0]["auto_fixable"] is True
    assert out["bugs"][0]["fix_strategy"] == "none_comparison"


_B008_MSG = ("Do not perform function call `{}` in argument defaults; instead, "
             "perform the call within the function, or read the default from a "
             "module-level singleton variable")


@pytest.mark.parametrize("callable_name", ["Depends", "Query", "Path", "Body", "Security"])
def test_fastapi_argument_default_calls_are_dismissed(callable_name):
    """B008 fires on every FastAPI route. Rewriting those would break the app."""
    out = bug_analysis(
        {"review_findings": [_finding(code="B008", problem=_B008_MSG.format(callable_name))]},
        _ctx(),
    )
    assert out["bugs"] == []
    assert "FastAPI" in out["dismissed"][0]["dismiss_reason"]


@pytest.mark.parametrize("code", ["B033", "F541", "B007", "B905", "F401"])
def test_codes_with_a_verified_ruff_fix_are_auto_fixable(code):
    out = bug_analysis({"review_findings": [_finding(code=code)]}, _ctx())
    assert out["bugs"][0]["auto_fixable"] is True
    assert out["bugs"][0]["fix_strategy"] == "ruff_fix"


def test_b904_is_fixable_only_as_from_e_never_as_from_none():
    """`from e` preserves the cause and is mechanical. `from None` deliberately
    hides it, so that spelling is never inferred -- the strategy has no branch
    that can produce it."""
    out = bug_analysis({"review_findings": [_finding(code="B904")]}, _ctx())
    assert out["bugs"][0]["fix_strategy"] == "raise_from"

    # Behavioural proof: the only cause it can emit is the handler's bound name.
    plan = ast_checks.raise_from_plan(_RAISE_BOUND, 4)
    assert plan["provable"] is True and plan["name"] == "exc"


def test_this_apps_own_dependency_factory_is_dismissed():
    """require_roles() is used exactly like Depends() in every HRM route."""
    out = bug_analysis(
        {"review_findings": [_finding(code="B008", problem=_B008_MSG.format("require_roles"))]},
        _ctx(),
    )
    assert out["bugs"] == []


def test_a_genuine_mutable_default_call_is_not_dismissed():
    """B008 on a real offender must survive triage."""
    out = bug_analysis(
        {"review_findings": [_finding(code="B008", problem=_B008_MSG.format("time.time"))]},
        _ctx(),
    )
    assert len(out["bugs"]) == 1
    assert out["dismissed"] == []


def test_unused_import_in_package_init_is_dismissed_as_a_reexport():
    out = bug_analysis({"review_findings": [_finding(file="pkg/__init__.py")]}, _ctx())
    assert out["bugs"] == []
    assert "re-export" in out["dismissed"][0]["dismiss_reason"]


@pytest.mark.parametrize(
    "path",
    ["app/services/payroll_service.py", "app/core/security.py",
     "app/api/deps.py", "app/services/attendance_service.py"],
)
def test_behaviour_changing_fixes_in_sensitive_areas_are_refused(path):
    """A fix that could alter runtime behaviour never runs in payroll/auth code."""
    out = bug_analysis(
        {"review_findings": [_finding(file=path, code=ast_checks.CMP_NONE,
                                      in_query_context=False)]},
        _ctx(),
    )
    bug = out["bugs"][0]
    assert bug["auto_fixable"] is False
    assert "sensitive" in bug["not_fixable_reason"]


@pytest.mark.parametrize("code", ["F401", "F541", "B033"])
def test_import_hygiene_is_allowed_even_in_sensitive_areas(code):
    """Removing an unused import cannot change behaviour, so the sensitive-area
    guard would only leave dead code lying around in the files that matter most."""
    out = bug_analysis(
        {"review_findings": [_finding(file="app/services/payroll_service.py", code=code)]},
        _ctx(),
    )
    assert out["bugs"][0]["auto_fixable"] is True


def test_findings_without_a_strategy_are_reported_but_not_fixable():
    out = bug_analysis({"review_findings": [_finding(code="F821", severity="high")]}, _ctx())
    assert out["bugs"][0]["auto_fixable"] is False
    assert out["bugs"][0]["root_cause"]


def test_already_fixed_bugs_are_not_retriaged():
    """Otherwise the drain loop would keep proposing the same fix forever."""
    state = {
        "review_findings": [_finding()],
        "fixes": [{"file": "pkg/a.py", "line": 1, "code": "F401", "applied": True}],
    }
    assert bug_analysis(state, _ctx())["bugs"] == []


# ==========================================================================
# fix
# ==========================================================================
def _bug(**kw):
    base = {"file": "pkg/a.py", "line": 1, "code": ast_checks.CMP_NONE,
            "severity": "low", "auto_fixable": True, "fix_strategy": "none_comparison"}
    base.update(kw)
    return base


def test_dry_run_changes_nothing(monkeypatch):
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    repo = FakeRepo(contents={"pkg/a.py": "if x == None:\n"})
    out = fix({"bugs": [_bug()], "fix_budget": 3}, _ctx(repo, apply_fixes=False))

    assert repo.written == {}
    assert out["fixes"][0]["applied"] is False
    assert "dry run" in out["fixes"][0]["reason"]


def test_none_comparison_fix_rewrites_only_the_reported_line(monkeypatch):
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    source = "a = 1\nif x == None:\n    pass\nif y == None:\n    pass\n"
    repo = FakeRepo(contents={"pkg/a.py": source})
    out = fix({"bugs": [_bug(line=2)], "fix_budget": 3}, _ctx(repo, apply_fixes=True))

    written = repo.written["pkg/a.py"]
    assert "if x is None:" in written
    assert "if y == None:" in written, "only the reported line may change"
    assert out["fixes"][0]["applied"] is True
    assert out["changed_files"] == ["pkg/a.py"]


def test_fix_refuses_a_file_with_uncommitted_modifications(monkeypatch):
    monkeypatch.setattr(
        "graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {"pkg/a.py": "M"}
    )
    repo = FakeRepo(contents={"pkg/a.py": "if x == None:\n"})
    out = fix({"bugs": [_bug()], "fix_budget": 3}, _ctx(repo, apply_fixes=True))

    assert repo.written == {}
    assert "uncommitted" in out["fixes"][0]["reason"]


def test_fix_stops_when_the_budget_is_spent(monkeypatch):
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    repo = FakeRepo(contents={"pkg/a.py": "if x == None:\n"})
    out = fix({"bugs": [_bug()], "fix_budget": 0}, _ctx(repo, apply_fixes=True))

    assert repo.written == {}
    assert "budget" in out["fixes"][0]["reason"]


def test_fix_rejects_a_patch_that_does_not_parse(monkeypatch):
    """A broken edit is rolled back here, not discovered by the test suite."""
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    monkeypatch.setattr(
        "graph_engine.nodes.fix._STRATEGIES",
        {"none_comparison": lambda source, bug, ctx: "def broken(:\n"},
    )
    repo = FakeRepo(contents={"pkg/a.py": "if x == None:\n"})
    out = fix({"bugs": [_bug()], "fix_budget": 3}, _ctx(repo, apply_fixes=True))

    assert repo.written == {}
    assert "invalid syntax" in out["fixes"][0]["reason"]


def test_a_whole_file_ruff_fix_resolves_its_siblings(monkeypatch):
    """`ruff --fix` rewrites the whole file, removing every instance of the rule.
    The other findings in that file are fixed, not failed."""
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    monkeypatch.setattr(
        "graph_engine.nodes.fix._STRATEGIES",
        {"ruff_fix": lambda source, bug, ctx: "import sys\n"},
    )
    bugs = [
        _bug(line=2, code="F401", fix_strategy="ruff_fix"),
        _bug(line=7, code="F401", fix_strategy="ruff_fix"),   # same file, same rule
    ]
    repo = FakeRepo(contents={"pkg/a.py": "import os\nimport sys\n"})
    out = fix({"bugs": bugs, "fix_budget": 3}, _ctx(repo, apply_fixes=True))

    assert all(f["applied"] for f in out["fixes"]), "the sibling was fixed too"
    assert not any(f["outcome"] == "error" for f in out["fixes"])
    assert "whole-file" in out["fixes"][1]["reason"]
    assert out["fix_budget"] == 2, "one write, one unit of budget"


def test_no_available_fix_is_a_decline_not_an_engine_error(monkeypatch):
    from graph_engine.nodes.fix import FixUnavailable

    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    def _none_available(source, bug, ctx):
        raise FixUnavailable("ruff has no remaining F401 fix for this file")
    monkeypatch.setattr("graph_engine.nodes.fix._STRATEGIES", {"ruff_fix": _none_available})

    out = fix(
        {"bugs": [_bug(code="F401", fix_strategy="ruff_fix")], "fix_budget": 3},
        _ctx(FakeRepo(contents={"pkg/a.py": "x = 1\n"}), apply_fixes=True),
    )
    assert out["fixes"][0]["outcome"] == "declined"


def test_fix_increments_the_iteration_counter(monkeypatch):
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    out = fix({"bugs": [], "fix_budget": 3, "iteration": 2}, _ctx(FakeRepo()))
    assert out["iteration"] == 3


# ==========================================================================
# test node
# ==========================================================================
def test_target_selection_matches_changed_modules_by_keyword():
    all_tests = ["tests/test_assistant_graph.py", "tests/test_payroll.py", "tests/test_misc.py"]
    targets = select_targets(all_tests, ["app/assistant/nodes/scope.py"], "app/assistant")
    assert targets == ["tests/test_assistant_graph.py"]


def test_target_selection_falls_back_to_the_whole_suite():
    """Never silently test nothing."""
    assert select_targets(["tests/test_a.py"], ["app/zzz/qqq.py"], "app/zzz") == ["tests"]


def test_regression_runs_only_after_targeted_passes():
    runner = FakeTests(_result(ok=True), _result(ok=True))
    ctx = _ctx(FakeRepo(files=["tests/test_assistant_graph.py"]), runner)
    out = run_tests({"changed_files": ["app/assistant/nodes/scope.py"], "scope": "app/assistant"}, ctx)

    assert [stage for _, stage in runner.calls] == ["targeted", "regression"]
    assert out["test_results"]["stage"] == "regression"


def test_regression_is_skipped_when_targeted_fails():
    runner = FakeTests(_result(ok=False, failures=[("tests/test_assistant_graph.py::t", "boom")]))
    ctx = _ctx(FakeRepo(files=["tests/test_assistant_graph.py"]), runner)
    run_tests({"changed_files": ["app/assistant/nodes/scope.py"], "scope": "app/assistant"}, ctx)

    assert [stage for _, stage in runner.calls] == ["targeted"]


def test_regression_can_be_skipped_by_configuration():
    runner = FakeTests(_result(ok=True))
    ctx = _ctx(FakeRepo(files=["tests/test_assistant_graph.py"]), runner, skip_regression=True)
    run_tests({"changed_files": ["app/assistant/nodes/scope.py"], "scope": "app/assistant"}, ctx)
    assert len(runner.calls) == 1


# ==========================================================================
# failure analysis
# ==========================================================================
def test_pre_existing_failures_are_ignored_not_fixed():
    """The repo's own seat-anchoring failure must never be blamed on a run."""
    state = {
        "test_results": _result(ok=False, failures=[("tests/test_seat.py::test_default", "assert")]),
        "baseline_failures": ["tests/test_seat.py::test_default"],
        "changed_files": ["pkg/a.py"],
    }
    out = failure_analysis(state, _ctx())
    assert out["failure_analysis"]["failure_type"] == PRE_EXISTING
    assert out["failure_analysis"]["next_action"] == "ignore"


def test_a_new_failure_in_a_changed_file_routes_to_fix():
    state = {
        "test_results": _result(ok=False, failures=[("pkg/a.py::test_thing", "AssertionError")]),
        "baseline_failures": [],
        "changed_files": ["pkg/a.py"],
    }
    out = failure_analysis(state, _ctx())
    assert out["failure_analysis"]["failure_type"] == IMPLEMENTATION_BUG
    assert out["failure_analysis"]["next_action"] == "fix"


@pytest.mark.parametrize(
    "tail,expected",
    [
        ("ModuleNotFoundError: No module named 'foo'", DEPENDENCY),
        ("SECRET_KEY missing environment", CONFIGURATION),
        ("error during collection\nSyntaxError: bad", COLLECTION_ERROR),
    ],
)
def test_environment_style_failures_are_classified_and_stop(tail, expected):
    state = {
        "test_results": _result(ok=False, failures=[("tests/t.py::x", "")], tail=tail),
        "baseline_failures": [], "changed_files": [],
    }
    out = failure_analysis(state, _ctx())
    assert out["failure_analysis"]["failure_type"] == expected
    assert out["failure_analysis"]["next_action"] == "stop"


def test_a_timeout_is_classified_as_a_timeout():
    state = {"test_results": _result(ok=False, timed_out=True), "baseline_failures": []}
    assert failure_analysis(state, _ctx())["failure_analysis"]["failure_type"] == TIMEOUT


def test_a_repeated_pre_existing_failure_is_still_ignored():
    """A pre-existing failure fails on EVERY test run, so its signature always
    repeats. Letting the repetition check win meant a benign, correctly
    diagnosed failure killed the run on the second pass."""
    failures = [("tests/test_seat.py::test_default", "assert False is True")]
    sig = signature([{"nodeid": n, "message": m} for n, m in failures])
    state = {
        "test_results": _result(ok=False, failures=failures),
        "baseline_failures": ["tests/test_seat.py::test_default"],
        "changed_files": ["pkg/a.py"],
        "failure_history": [sig],          # already seen this run
    }
    out = failure_analysis(state, _ctx())
    assert out["failure_analysis"]["failure_type"] == PRE_EXISTING
    assert out["failure_analysis"]["next_action"] == "ignore"
    assert out["stop_requested"] is False


def test_stop_requested_is_set_only_when_the_run_ends():
    state = {
        "test_results": _result(ok=False, failures=[("tests/t.py::x", "")],
                                tail="ModuleNotFoundError: No module named 'x'"),
        "baseline_failures": [], "changed_files": [],
    }
    assert failure_analysis(state, _ctx())["stop_requested"] is True


def test_a_repeated_failure_signature_stops_the_loop():
    failures = [("pkg/a.py::test_thing", "AssertionError")]
    sig = signature([{"nodeid": n, "message": m} for n, m in failures])
    state = {
        "test_results": _result(ok=False, failures=failures),
        "baseline_failures": [], "changed_files": ["pkg/a.py"],
        "failure_history": [sig],       # seen once already
    }
    out = failure_analysis(state, _ctx())
    assert out["failure_analysis"]["repeated"] is True
    assert out["failure_analysis"]["next_action"] == "stop"


# ==========================================================================
# verify
# ==========================================================================
def _verifiable(**over):
    base = {
        "scope": "pkg", "files": ["pkg/a.py"],
        "review_findings": [_finding()], "bugs": [], "dismissed": [_finding()],
        "fixes": [], "changed_files": [], "test_results": None, "baseline_failures": [],
    }
    base.update(over)
    return base


def test_verification_requires_that_something_was_actually_reviewed():
    out = verify(_verifiable(files=[], review_findings=[], dismissed=[]), _ctx())
    assert out["verification"]["goal_achieved"] is False
    assert any(c["name"] == "scope_reviewed" and not c["ok"]
               for c in out["verification"]["criteria"])


def test_findings_already_fixed_still_count_as_accounted_for():
    """bug_analysis drops fixed bugs so it does not re-propose them, so they sit
    in neither `bugs` nor `dismissed`. They are still accounted for."""
    finding = _finding(file="pkg/a.py", line=3, code="F401")
    state = _verifiable(
        review_findings=[finding],
        bugs=[], dismissed=[],
        fixes=[{"file": "pkg/a.py", "line": 3, "code": "F401",
                "applied": True, "outcome": "applied", "reason": "applied ruff_fix"}],
        changed_files=["pkg/a.py"],
        test_results={**_result(ok=True), "stage": "regression"},
    )
    out = verify(state, _ctx())
    criterion = next(c for c in out["verification"]["criteria"] if c["name"] == "findings_triaged")
    assert criterion["ok"] is True


def test_a_finding_that_vanished_without_explanation_fails_verification():
    state = _verifiable(review_findings=[_finding(line=99)], bugs=[], dismissed=[], fixes=[])
    out = verify(state, _ctx())
    criterion = next(c for c in out["verification"]["criteria"] if c["name"] == "findings_triaged")
    assert criterion["ok"] is False


def test_green_tests_alone_do_not_verify_the_goal():
    """pytest exit 0 with an empty review is not an achievement."""
    state = _verifiable(files=[], review_findings=[], dismissed=[],
                        test_results=_result(ok=True))
    assert verify(state, _ctx())["verification"]["goal_achieved"] is False


def test_a_new_test_failure_fails_verification():
    state = _verifiable(
        changed_files=["pkg/a.py"],
        test_results={**_result(ok=False, failures=[("tests/t.py::x", "")]), "stage": "regression"},
        baseline_failures=[],
    )
    out = verify(state, _ctx())
    assert out["verification"]["goal_achieved"] is False
    assert any(c["name"] == "no_new_test_failures" and not c["ok"]
               for c in out["verification"]["criteria"])


def test_pre_existing_failures_do_not_fail_verification_but_are_a_caveat():
    state = _verifiable(
        changed_files=["pkg/a.py"],
        test_results={**_result(ok=False, failures=[("tests/t.py::x", "")]), "stage": "regression"},
        baseline_failures=["tests/t.py::x"],
    )
    out = verify(state, _ctx())
    assert out["verification"]["goal_achieved"] is True
    assert any("pre-existing" in c for c in out["verification"]["caveats"])


def test_changing_a_file_outside_the_scope_fails_verification():
    state = _verifiable(changed_files=["other/secret.py"], test_results=_result(ok=True))
    out = verify(state, _ctx())
    assert any(c["name"] == "no_collateral_damage" and not c["ok"]
               for c in out["verification"]["criteria"])


def test_unfixable_bugs_are_reported_as_a_caveat_not_a_failure():
    state = _verifiable(
        bugs=[_finding(auto_fixable=False, not_fixable_reason="touches payroll")],
        dismissed=[],
    )
    out = verify(state, _ctx())
    assert out["verification"]["goal_achieved"] is True
    assert any("need a human" in c for c in out["verification"]["caveats"])


# ==========================================================================
# report
# ==========================================================================
def test_report_marks_an_achieved_goal_done():
    out = report({"verification": {"goal_achieved": True}}, _ctx())
    assert (out["status"], out["stop_reason"]) == ("done", "goal_achieved")


def test_report_records_a_repeated_failure_stop():
    out = report({"failure_analysis": {"repeated": True, "next_action": "stop"}}, _ctx())
    assert out["stop_reason"] == "repeated_failure"


def test_report_records_an_unfixable_stop():
    out = report({"failure_analysis": {"repeated": False, "next_action": "stop"}}, _ctx())
    assert out["stop_reason"] == "failure_not_auto_fixable"


def test_a_closure_called_in_place_is_dismissed_not_flagged():
    """B023 only matters when the closure outlives its iteration."""
    source = (
        "for item in items:\n"
        "    def build():\n"
        "        return item\n"
        "    record = build()\n"
    )
    ctx = _ctx(FakeRepo(contents={"pkg/a.py": source}))
    out = bug_analysis({"review_findings": [_finding(code="B023", line=3)]}, ctx)

    assert out["bugs"] == []
    assert "late binding never occurs" in out["dismissed"][0]["dismiss_reason"]


def test_a_closure_that_escapes_the_loop_stays_a_bug():
    source = (
        "handlers = []\n"
        "for item in items:\n"
        "    def build():\n"
        "        return item\n"
        "    handlers.append(build)\n"
    )
    ctx = _ctx(FakeRepo(contents={"pkg/a.py": source}))
    out = bug_analysis({"review_findings": [_finding(code="B023", line=4)]}, ctx)

    assert len(out["bugs"]) == 1
    assert out["dismissed"] == []


_BEST_EFFORT_SRC = (
    "import logging\n"
    "logger = logging.getLogger(__name__)\n"
    "def f(h):\n"
    "    try:\n"
    "        h.close()\n"
    "    except OSError:\n"
    "        pass\n"
)
_UNPROVEN_SRC = (
    "import logging\n"
    "logger = logging.getLogger(__name__)\n"
    "def g():\n"
    "    try:\n"
    "        compute_salary()\n"
    "    except ValueError:\n"
    "        pass\n"
)
_NO_LOGGER_SRC = "def f(h):\n    try:\n        h.close()\n    except OSError:\n        pass\n"


def test_provable_best_effort_handler_is_auto_fixable():
    ctx = _ctx(FakeRepo(contents={"pkg/a.py": _BEST_EFFORT_SRC}))
    out = bug_analysis({"review_findings": [_finding(code=ast_checks.EXCEPT_PASS, line=6)]}, ctx)
    assert out["bugs"][0]["auto_fixable"] is True
    assert out["bugs"][0]["fix_strategy"] == "except_pass_logging"


def test_except_pass_fix_only_adds_logging_and_keeps_control_flow(monkeypatch):
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    repo = FakeRepo(contents={"pkg/a.py": _BEST_EFFORT_SRC})
    bug = _bug(line=6, code=ast_checks.EXCEPT_PASS, fix_strategy="except_pass_logging")
    out = fix({"bugs": [bug], "fix_budget": 3}, _ctx(repo, apply_fixes=True))

    written = repo.written["pkg/a.py"]
    assert out["fixes"][0]["applied"] is True
    assert "logger.debug(" in written and "exc_info=True" in written
    assert "        pass\n" not in written
    # The handler still swallows the exception: no raise, no re-raise introduced.
    assert "raise" not in written
    assert written.count("except OSError:") == 1


_PARSE_FALLBACK_SRC = (
    "import logging\n"
    "logger = logging.getLogger(__name__)\n"
    "def f(s):\n"
    "    try:\n"
    "        dt = datetime.fromisoformat(s)\n"
    "    except ValueError:\n"
    "        pass\n"
)
_SESSION_TEARDOWN_SRC = (
    "import logging\n"
    "logger = logging.getLogger(__name__)\n"
    "def f(c):\n"
    "    try:\n"
    "        c.logout()\n"
    "    except Exception:\n"
    "        pass\n"
)


@pytest.mark.parametrize(
    "source,label",
    [(_PARSE_FALLBACK_SRC, "parse fallback"), (_SESSION_TEARDOWN_SRC, "session teardown")],
)
def test_provable_best_effort_shapes_are_auto_fixable(source, label):
    ctx = _ctx(FakeRepo(contents={"pkg/a.py": source}))
    out = bug_analysis({"review_findings": [_finding(code=ast_checks.EXCEPT_PASS, line=6)]}, ctx)
    assert out["bugs"][0]["auto_fixable"] is True, label


def test_unproven_handler_gets_an_honest_warning_not_a_debug_lie(monkeypatch):
    """Calling a swallowed worker-thread crash "non-critical" would be false."""
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    source = (
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def f(fut):\n"
        "    try:\n"
        "        late = fut.result()\n"
        "    except Exception:\n"
        "        pass\n"
    )
    repo = FakeRepo(contents={"pkg/a.py": source})
    bug = _bug(line=6, code=ast_checks.EXCEPT_PASS, fix_strategy="except_pass_logging")
    out = fix({"bugs": [bug], "fix_budget": 3}, _ctx(repo, apply_fixes=True))

    written = repo.written["pkg/a.py"]
    assert out["fixes"][0]["applied"] is True
    assert 'logger.warning("fut.result failed", exc_info=True)' in written
    assert "non-critical" not in written
    assert "raise" not in written and "return" not in written


def test_a_module_without_a_logger_gets_one_injected(monkeypatch):
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    source = (
        '"""Doc."""\n'
        "import os\n"
        "def f(p):\n"
        "    try:\n"
        "        os.unlink(p)\n"
        "    except OSError:\n"
        "        pass\n"
    )
    repo = FakeRepo(contents={"pkg/a.py": source})
    bug = _bug(line=6, code=ast_checks.EXCEPT_PASS, fix_strategy="except_pass_logging")
    out = fix({"bugs": [bug], "fix_budget": 3}, _ctx(repo, apply_fixes=True))

    written = repo.written["pkg/a.py"]
    assert out["fixes"][0]["applied"] is True
    assert "import logging" in written
    assert "logger = logging.getLogger(__name__)" in written
    assert written.index("import logging") > written.index('"""Doc."""')
    assert "        pass\n" not in written


def test_float_parse_is_a_proven_fallback_but_round_is_not():
    """float() as validation means "not a number"; round() failing means something
    else, so it must not inherit the parse-fallback exemption."""
    base = ("import logging\n"
            "logger = logging.getLogger(__name__)\n"
            "def f(x):\n"
            "    try:\n"
            "        return {call}\n"
            "    except ValueError:\n"
            "        pass\n")
    ctx_float = _ctx(FakeRepo(contents={"pkg/a.py": base.format(call="float(x)")}))
    ctx_round = _ctx(FakeRepo(contents={"pkg/a.py": base.format(call="round(x, 2)")}))
    finding = _finding(code=ast_checks.EXCEPT_PASS, line=6)

    assert ast_checks.except_pass_intent(
        ctx_float.repo.read("pkg/a.py"), 6)["provable"] is True
    assert ast_checks.except_pass_intent(
        ctx_round.repo.read("pkg/a.py"), 6)["provable"] is False
    # Both are still fixable -- round just gets the honest warning, not debug.
    assert bug_analysis({"review_findings": [finding]}, ctx_round)["bugs"][0]["auto_fixable"]


def test_a_missing_logger_does_not_mask_a_best_effort_proof():
    """Regression: the no-logger check used to return before the proof ran, so
    `sftp.close()` in a logger-less module was labelled an unproven fault."""
    source = ("import os\n"
              "def f(c):\n"
              "    try:\n"
              "        c.close()\n"
              "    except Exception:\n"
              "        pass\n")
    intent = ast_checks.except_pass_intent(source, 5)
    assert intent["provable"] is True
    assert intent["logger_name"] is None, "still reports that a logger must be injected"


_RAISE_BOUND = ("def f():\n    try:\n        g()\n"
                "    except ValueError as exc:\n        raise RuntimeError('bad')\n")
_RAISE_UNBOUND = ("def f():\n    try:\n        g()\n"
                  "    except ValueError:\n        raise RuntimeError('bad')\n")


def test_raise_from_chains_to_the_bound_exception(monkeypatch):
    """Sets __cause__ only: same exception, same control flow, better traceback."""
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    repo = FakeRepo(contents={"pkg/a.py": _RAISE_BOUND})
    bug = _bug(line=4, code="B904", fix_strategy="raise_from")
    out = fix({"bugs": [bug], "fix_budget": 3}, _ctx(repo, apply_fixes=True))

    written = repo.written["pkg/a.py"]
    assert out["fixes"][0]["applied"] is True
    assert "raise RuntimeError('bad') from exc" in written
    assert "from None" not in written, "suppressing the cause is never inferred"
    assert written.count("raise") == 1


def test_raise_from_declines_when_the_handler_binds_no_name(monkeypatch):
    """Without `as e` there is nothing to chain, so the site stays manual."""
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    repo = FakeRepo(contents={"pkg/a.py": _RAISE_UNBOUND})
    bug = _bug(line=4, code="B904", fix_strategy="raise_from")
    out = fix({"bugs": [bug], "fix_budget": 3}, _ctx(repo, apply_fixes=True))

    assert repo.written == {}
    assert out["fixes"][0]["outcome"] == "declined"
    assert "does not bind" in out["fixes"][0]["reason"]
