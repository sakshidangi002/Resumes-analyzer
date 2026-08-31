"""`test` node (module named `testing` so pytest does not collect it) — run the real suite. No LLM has a vote on whether tests passed.

Two stages inside one node execution, because they are sequential by nature:

1. **targeted** — test files whose names relate to the changed modules. Fast, and
   if these fail there is no point paying for the full suite.
2. **regression** — the whole suite, run only once targeted is green, to catch
   damage outside the changed area.

Target selection is a filename-keyword heuristic, not a real import graph. It is
honest about that: when it finds nothing it falls back to the full suite rather
than reporting a green run it never really covered.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from graph_engine.context import EngineContext
from graph_engine.skills import skill_for_node

logger = logging.getLogger(__name__)

#: Path components too generic to identify a module.
_STOPWORDS = frozenset({
    "app", "backend", "nodes", "tools", "services", "api", "routes", "models",
    "schemas", "core", "db", "__init__", "main", "config", "utils", "graph_engine",
    "attendance management", "tests", "test", "graph", "state", "base",
})

TESTS_ROOT = "tests"


def _keywords(changed_files: list[str], scope: str) -> set[str]:
    sources = list(changed_files) or [scope]
    words: set[str] = set()
    for rel in sources:
        for part in rel.replace("\\", "/").split("/"):
            stem = part[:-3] if part.endswith(".py") else part
            stem = stem.strip().lower()
            if stem and stem not in _STOPWORDS and len(stem) > 3:
                words.add(stem)
    return words


def select_targets(all_test_files: list[str], changed_files: list[str], scope: str) -> list[str]:
    """Test files related to the change, or `["tests"]` when nothing matches."""
    words = _keywords(changed_files, scope)
    if not words:
        return [TESTS_ROOT]

    matched = sorted(
        path for path in all_test_files
        if any(word in path.replace("\\", "/").lower() for word in words)
    )
    return matched or [TESTS_ROOT]


def run_tests(state: Mapping[str, Any], ctx: EngineContext) -> Mapping[str, Any]:
    skill_name, _ = skill_for_node("test")
    changed = list(state.get("changed_files") or [])

    try:
        all_tests = [p for p in ctx.repo.list_python_files(TESTS_ROOT)
                     if p.rsplit("/", 1)[-1].startswith("test_")]
    except Exception:  # noqa: BLE001 - a missing tests/ dir must not crash the run
        all_tests = []

    targets = select_targets(all_tests, changed, state.get("scope") or "")
    targets = list(dict.fromkeys([*targets, *ctx.config.extra_test_targets]))

    # When selection falls back to the whole suite, this *is* the regression
    # pass. Labelling it "targeted" would make the verifier report that no
    # regression coverage ran, when in fact everything ran.
    first_stage = "regression" if targets == [TESTS_ROOT] else "targeted"

    ctx.counters["test_runs"] += 1
    result = ctx.tests.run(targets, stage=first_stage)

    # Only pay for the full suite once the targeted subset is clean.
    if result["ok"] and not ctx.config.skip_regression and first_stage == "targeted":
        ctx.counters["test_runs"] += 1
        regression = ctx.tests.run([TESTS_ROOT], stage="regression")
        regression["targeted_before"] = {
            "command": result["command"], "passed": result["passed"],
        }
        result = regression

    logger.info(
        "graph_engine.test stage=%s targets=%d ok=%s failed=%d",
        result["stage"], len(targets), result["ok"], result["failed"],
    )
    return {
        "test_results": result,
        "_trace": {
            "skill": skill_name,
            "stage": result["stage"],
            "targets": targets[:8],
            "target_count": len(targets),
            "exit_code": result["exit_code"],
            "passed": result["passed"],
            "failed": result["failed"],
            "duration_s": result["duration_s"],
            "timed_out": result["timed_out"],
        },
    }
