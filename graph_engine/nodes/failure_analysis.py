"""`failure_analysis` — classify a test failure and decide the next action.

Runs only when tests failed. The classification is deterministic pattern matching
over the *actual* pytest output, because the categories that matter here are
distinguishable by text: an `ImportError` is an environment problem, a node id
that also failed in the baseline is pre-existing, an assertion in a file the run
just edited is an implementation bug.

The rule that keeps the loop finite: a failure whose signature has already been
seen returns `stop`. Three attempts at the same error means the diagnosis is
wrong, and trying a fourth fix is not going to discover that.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping

from graph_engine.context import EngineContext
from graph_engine.skills import skill_for_node

logger = logging.getLogger(__name__)

# failure_type values
PRE_EXISTING = "pre_existing"
IMPLEMENTATION_BUG = "implementation_bug"
TEST_BUG = "test_bug"
ENVIRONMENT = "environment"
DEPENDENCY = "dependency"
CONFIGURATION = "configuration"
COLLECTION_ERROR = "collection_error"
TIMEOUT = "timeout"
UNKNOWN = "unknown"

# next_action values
ACTION_FIX = "fix"
ACTION_IGNORE = "ignore"
ACTION_STOP = "stop"

_ENVIRONMENT_MARKERS = ("ImportError", "DLL load failed", "No module named", "libGL",
                        "onnxruntime", "cv2", "Tesseract", "CUDA")
_DEPENDENCY_MARKERS = ("ModuleNotFoundError", "VersionConflict", "incompatible")
_CONFIGURATION_MARKERS = ("SECRET_KEY", "ValidationError", "missing environment",
                          "POSTGRES", "settings")
_COLLECTION_MARKERS = ("error during collection", "SyntaxError", "IndentationError",
                       "conftest")


def signature(failures: list[dict]) -> str:
    """Stable id for "this exact set of failures"."""
    joined = "|".join(sorted(f["nodeid"] for f in failures))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def _classify(result: dict, baseline: set[str], changed: set[str]) -> tuple[str, str, list[str]]:
    """(failure_type, root_cause, affected_files)"""
    if result.get("timed_out"):
        return (TIMEOUT,
                f"pytest exceeded the {result.get('duration_s')}s limit without completing",
                [])

    failures = result.get("failures") or []
    blob = f"{result.get('tail', '')}\n" + "\n".join(f.get("message", "") for f in failures)

    if failures and all(f["nodeid"] in baseline for f in failures):
        return (PRE_EXISTING,
                "every failing test also failed in the baseline run, before any change",
                sorted({f["nodeid"].split("::")[0] for f in failures}))

    if any(marker in blob for marker in _COLLECTION_MARKERS):
        return (COLLECTION_ERROR, "a module failed to import or parse at collection time",
                sorted({f["nodeid"].split("::")[0] for f in failures}))
    if any(marker in blob for marker in _DEPENDENCY_MARKERS):
        return (DEPENDENCY, "a required package is missing or version-incompatible", [])
    if any(marker in blob for marker in _ENVIRONMENT_MARKERS):
        return (ENVIRONMENT, "a native dependency or model asset is unavailable here", [])
    if any(marker in blob for marker in _CONFIGURATION_MARKERS):
        return (CONFIGURATION, "configuration or environment variables are not as the tests expect",
                [])

    new_failures = [f for f in failures if f["nodeid"] not in baseline]
    touched = sorted({
        f["nodeid"].split("::")[0] for f in new_failures
        if f["nodeid"].split("::")[0] in changed
    })
    if touched:
        return (IMPLEMENTATION_BUG, "a test that passed in the baseline now fails in a file "
                                    "this run changed", touched)
    if new_failures and changed:
        return (IMPLEMENTATION_BUG,
                "new failures appeared after this run's changes, in tests covering them",
                sorted(changed))
    if new_failures:
        return (TEST_BUG,
                "tests fail but this run changed nothing that they cover — the expectation "
                "itself is likely stale",
                sorted({f["nodeid"].split("::")[0] for f in new_failures}))
    return (UNKNOWN, "failure could not be attributed from the test output", [])


def failure_analysis(state: Mapping[str, Any], ctx: EngineContext) -> Mapping[str, Any]:
    skill_name, _ = skill_for_node("failure_analysis")
    result = state.get("test_results") or {}
    failures = result.get("failures") or []
    baseline = set(state.get("baseline_failures") or [])
    changed = set(state.get("changed_files") or [])
    history = list(state.get("failure_history") or [])

    sig = signature(failures)
    repeated = sig in history
    failure_type, root_cause, affected = _classify(result, baseline, changed)

    if repeated:
        next_action, decision_reason = ACTION_STOP, "identical failure already seen this run"
    elif failure_type == PRE_EXISTING:
        next_action, decision_reason = ACTION_IGNORE, "not caused by this run"
    elif failure_type in (IMPLEMENTATION_BUG, COLLECTION_ERROR) and (set(affected) & changed):
        # The overlap with `changed` is the point: a collection error in a file
        # this run never touched is someone else's problem, and handing it to the
        # fixer would mean editing code we have no mandate over.
        next_action, decision_reason = ACTION_FIX, "cause is inside a file this run changed"
    else:
        next_action, decision_reason = ACTION_STOP, f"{failure_type} needs a human"

    analysis = {
        "failure_type": failure_type,
        "root_cause": root_cause,
        "affected_files": affected,
        "next_action": next_action,
        "signature": sig,
        "repeated": repeated,
        "decision_reason": decision_reason,
        "failing_node_ids": [f["nodeid"] for f in failures][:20],
    }
    logger.info(
        "graph_engine.failure_analysis type=%s action=%s repeated=%s sig=%s",
        failure_type, next_action, repeated, sig,
    )
    return {
        "failure_analysis": analysis,
        "failure_history": history + [sig],
        "_trace": {
            "skill": skill_name,
            "failure_type": failure_type,
            "next_action": next_action,
            "repeated": repeated,
            "signature": sig,
            "failing_count": len(failures),
        },
    }
