"""Graph-level tests: whole runs, asserted on the path taken and the outcome.

The repository and the test runner are injected fakes, so no file is written and
no pytest subprocess is spawned. Everything between `preflight` and `report` is
the real implementation, including all routing and both loops.
"""
import pytest

from graph_engine.config import EngineConfig
from graph_engine.graph import build_graph, run_engine
from graph_engine.state import (
    STOP_GOAL_ACHIEVED,
    STOP_REPEATED_FAILURE,
    STOP_UNSAFE,
)

CLEAN = "value = 1\n"
ONE_BUG = "value = 1\nif value == None:\n    pass\n"


class FakeRepo:
    def __init__(self, contents):
        self._contents = dict(contents)
        self.writes = []

    def list_python_files(self, scope=None):
        if scope == "tests":
            # `test_widget` matches a change to `widget.py`; `test_other` does not.
            return ["tests/test_widget.py", "tests/test_other.py"]
        return [p for p in self._contents if not p.startswith("tests/")]

    def list_source_files(self, scope=None, suffixes=None):
        return self.list_python_files(scope)

    def read(self, rel):
        return self._contents[rel]

    def write(self, rel, content, reason):
        self._contents[rel] = content
        self.writes.append((rel, reason))


class FakeTests:
    __test__ = False

    def __init__(self, *results):
        self._queue = list(results)
        self.calls = []

    def run(self, targets, *, stage):
        self.calls.append(stage)
        result = self._queue.pop(0) if self._queue else _tests(ok=True)
        return {**result, "stage": stage}


def _tests(*, ok=True, failures=(), tail=""):
    return {
        "command": "pytest", "exit_code": 0 if ok else 1, "timed_out": False,
        "ok": ok, "passed": 100, "failed": len(failures), "skipped": 0,
        "duration_s": 1.0, "tail": tail,
        "failures": [{"nodeid": n, "message": m} for n, m in failures],
    }


@pytest.fixture(autouse=True)
def no_ruff(monkeypatch):
    """Findings come only from the AST checks, so cases are exactly controlled."""
    monkeypatch.setattr("graph_engine.nodes.review.linters.run_ruff", lambda *a, **k: [])


@pytest.fixture(autouse=True)
def clean_git(monkeypatch):
    monkeypatch.setattr("graph_engine.nodes.fix.git_tools.dirty_files", lambda root: {})
    monkeypatch.setattr("graph_engine.nodes.preflight.git_tools.dirty_files", lambda root: {})
    monkeypatch.setattr("graph_engine.nodes.preflight.git_tools.current_branch", lambda root: "test")


def _run(contents, *, tests=None, scope="graph_engine", **config_kwargs):
    config = EngineConfig(goal="fix the bugs", scope=scope, **config_kwargs)
    repo = FakeRepo(contents)
    result = run_engine(config, repo=repo, tests=tests or FakeTests())
    return result, repo


def _nodes(result):
    return [t["node"] for t in result["trace"]]


# ==========================================================================
# happy paths
# ==========================================================================
def test_clean_scope_skips_fixing_entirely():
    result, repo = _run({"graph_engine/mod.py": CLEAN})

    assert result["status"] == "done"
    assert _nodes(result) == ["preflight", "review", "bug_analysis", "verify", "report"]
    assert repo.writes == []
    assert result["verification"]["goal_achieved"] is True


def test_bug_found_and_fixed_then_tests_run_and_verify_passes():
    # `widget.py` gives target selection a keyword to match, so the run exercises
    # the targeted-then-regression path rather than falling back to the suite.
    tests = FakeTests(_tests(ok=True), _tests(ok=True), _tests(ok=True))
    result, repo = _run({"graph_engine/widget.py": ONE_BUG}, tests=tests, apply_fixes=True)

    assert result["status"] == "done"
    assert result["stop_reason"] == STOP_GOAL_ACHIEVED
    assert repo.writes and "is None" in repo._contents["graph_engine/widget.py"]
    visited = _nodes(result)
    assert visited[:5] == ["preflight", "review", "bug_analysis", "fix", "test"]
    assert "verify" in visited and visited[-1] == "report"
    assert tests.calls == ["baseline", "targeted", "regression"]


def test_full_suite_fallback_is_reported_as_regression_not_targeted():
    """A fallback run covers everything; labelling it 'targeted' would make the
    verifier claim regression coverage was missing when it was not."""
    tests = FakeTests(_tests(ok=True), _tests(ok=True))
    result, _ = _run({"graph_engine/mod.py": ONE_BUG}, tests=tests, apply_fixes=True)

    assert tests.calls == ["baseline", "regression"]
    assert result["test_results"]["stage"] == "regression"
    assert result["status"] == "done"


def test_dry_run_reaches_verify_without_testing_or_writing():
    result, repo = _run({"graph_engine/mod.py": ONE_BUG}, apply_fixes=False)

    assert repo.writes == []
    assert "test" not in _nodes(result), "nothing changed, so there is nothing to test"
    assert result["fixes"][0]["applied"] is False
    assert result["status"] == "done"


# ==========================================================================
# failure paths and the repair loop
# ==========================================================================
def test_a_new_failure_routes_to_failure_analysis_then_back_to_fix():
    tests = FakeTests(
        _tests(ok=True),                                              # baseline
        _tests(ok=False, failures=[("graph_engine/mod.py::test_a", "AssertionError")]),  # targeted
        _tests(ok=True),                                              # after refix
        _tests(ok=True),                                              # regression
    )
    result, _ = _run({"graph_engine/mod.py": ONE_BUG}, tests=tests, apply_fixes=True)

    visited = _nodes(result)
    assert "failure_analysis" in visited
    assert visited.count("fix") >= 2, "the loop must return to fix"
    assert result["failure_analysis"]["failure_type"] == "implementation_bug"


def test_the_same_failure_twice_stops_the_loop():
    """Repetition, not budget exhaustion, is what ends a deterministic failure."""
    same = _tests(ok=False, failures=[("graph_engine/mod.py::test_a", "AssertionError")])
    tests = FakeTests(_tests(ok=True), same, same, same, same, same, same)
    result, _ = _run({"graph_engine/mod.py": ONE_BUG}, tests=tests, apply_fixes=True)

    assert result["status"] == "stopped"
    assert result["stop_reason"] == STOP_REPEATED_FAILURE
    assert len(result["failure_history"]) <= 3


def test_a_pre_existing_failure_is_ignored_and_the_run_still_succeeds():
    failing = [("tests/test_seat.py::test_default", "assert False is True")]
    tests = FakeTests(
        _tests(ok=False, failures=failing),   # baseline already failing
        _tests(ok=False, failures=failing),   # targeted: same failure
    )
    result, _ = _run({"graph_engine/mod.py": ONE_BUG}, tests=tests, apply_fixes=True)

    assert result["failure_analysis"]["failure_type"] == "pre_existing"
    assert result["failure_analysis"]["next_action"] == "ignore"
    assert result["status"] == "done"
    assert any("pre-existing" in c for c in result["verification"]["caveats"])


def test_a_pre_existing_failure_surviving_two_test_runs_does_not_stop_the_run():
    """Regression: a pre-existing failure repeats by definition, and used to end
    the run with stop_reason=repeated_failure on the second pass."""
    failing = [("tests/test_seat.py::test_default", "assert False is True")]
    tests = FakeTests(
        _tests(ok=False, failures=failing),   # baseline
        _tests(ok=False, failures=failing),   # first pass
        _tests(ok=False, failures=failing),   # second pass, identical
    )
    many = "\n".join(f"if v{i} == None:\n    pass" for i in range(8))
    result, _ = _run({"graph_engine/mod.py": many}, tests=tests,
                     apply_fixes=True, fix_budget=8)

    assert result["stop_reason"] != STOP_REPEATED_FAILURE
    assert result["failure_analysis"]["failure_type"] == "pre_existing"
    assert result["failure_analysis"]["next_action"] == "ignore"


def test_a_stopping_run_still_verifies_against_final_state():
    """The stop path routes through verify, so the reported criteria are not
    stale figures from an earlier pass."""
    tests = FakeTests(
        _tests(ok=True),
        _tests(ok=False, failures=[("tests/t.py::x", "")],
               tail="ModuleNotFoundError: No module named 'onnxruntime'"),
    )
    result, _ = _run({"graph_engine/mod.py": ONE_BUG}, tests=tests, apply_fixes=True)

    visited = _nodes(result)
    assert visited.index("verify") > visited.index("failure_analysis")
    assert visited[-1] == "report"
    # The criteria were computed after the fix, so they see the changed file.
    changed_criterion = next(
        c for c in result["verification"]["criteria"] if c["name"] == "no_collateral_damage"
    )
    assert str(len(result["changed_files"])) in changed_criterion["evidence"]


def test_an_environment_failure_stops_instead_of_looping():
    tests = FakeTests(
        _tests(ok=True),
        _tests(ok=False, failures=[("tests/t.py::x", "")],
               tail="ModuleNotFoundError: No module named 'onnxruntime'"),
    )
    result, _ = _run({"graph_engine/mod.py": ONE_BUG}, tests=tests, apply_fixes=True)

    assert result["status"] == "stopped"
    assert result["failure_analysis"]["failure_type"] == "dependency"
    assert _nodes(result).count("fix") == 1, "an environment failure must not be re-fixed"


# ==========================================================================
# bounds and safety
# ==========================================================================
def test_a_missing_scope_stops_before_any_code_is_read():
    result, repo = _run({"graph_engine/mod.py": ONE_BUG}, scope="no/such/directory")

    assert result["status"] == "stopped"
    assert result["stop_reason"] == STOP_UNSAFE
    assert _nodes(result) == ["preflight", "report"]
    assert repo.writes == []


def test_the_fix_budget_bounds_how_many_files_change():
    many = "\n".join(f"if v{i} == None:\n    pass" for i in range(10))
    result, repo = _run({"graph_engine/mod.py": many}, apply_fixes=True, fix_budget=2)

    applied = [f for f in result["fixes"] if f["applied"]]
    assert len(applied) == 2
    assert any("budget" in f["reason"] for f in result["fixes"] if not f["applied"])


def test_iterations_are_bounded_even_with_many_bugs():
    many = "\n".join(f"if v{i} == None:\n    pass" for i in range(40))
    result, _ = _run({"graph_engine/mod.py": many}, apply_fixes=True, fix_budget=99, max_iterations=3)

    assert result["iteration"] <= 3
    assert result["status"] in ("done", "stopped")


def test_a_node_exception_is_reported_not_raised(monkeypatch):
    def boom(state, ctx):
        raise RuntimeError("review exploded")

    monkeypatch.setattr("graph_engine.graph.review", boom)
    # The compiled graph is a module-level singleton; rebuild it for this test.
    monkeypatch.setattr("graph_engine.graph._compiled", None)

    result, _ = _run({"graph_engine/mod.py": CLEAN})
    assert result["status"] in ("failed", "stopped")
    assert result["error"]["kind"] == "node_exception"
    assert _nodes(result)[-1] == "report"


# ==========================================================================
# observability + topology
# ==========================================================================
def test_every_run_produces_a_trace_and_metrics():
    result, _ = _run({"graph_engine/mod.py": ONE_BUG}, apply_fixes=True)

    assert result["run_id"]
    assert result["metrics"]["node_count"] == len(result["trace"])
    for entry in result["trace"]:
        assert {"run_id", "seq", "node", "status", "duration_ms"} <= set(entry)
    # Each node records which skill governed it.
    skills = {e["meta"].get("skill") for e in result["trace"] if e["meta"]}
    assert {"code_review", "bug_analysis", "bug_fixing"} <= skills


def test_trace_carries_decisions_not_source_code():
    result, _ = _run({"graph_engine/mod.py": ONE_BUG}, apply_fixes=True)
    blob = str(result["trace"])
    assert "if value == None" not in blob, "source must never enter the trace"


def test_graph_compiles_and_every_edge_resolves():
    build_graph().compile()


def test_mermaid_shows_both_loops():
    diagram = build_graph().mermaid()
    assert "failure_analysis -->|fix| fix" in diagram
    assert "verify -->|retry| bug_analysis" in diagram
    assert "test -->|fail| failure_analysis" in diagram
