"""`preflight` — safety checks and the test baseline, before anything is read.

Two jobs, both of which must happen before a single byte is written:

1. **Git safety.** Record the branch and the set of uncommitted files. This tree
   routinely carries dozens of dirty files, so knowing which ones are dirty is
   what lets the fix node refuse to layer an autonomous edit on top of
   in-progress work.
2. **Baseline.** Capture which tests already fail. Without this, a pre-existing
   failure gets blamed on the run's changes — and worse, gets "fixed".

The baseline suite is only run when fixes may actually be applied; a dry run has
nothing to attribute and should not pay 30 seconds for the privilege.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from graph_engine.context import EngineContext
from graph_engine.state import STOP_UNSAFE
from graph_engine.tools import git_tools

logger = logging.getLogger(__name__)


def preflight(state: Mapping[str, Any], ctx: EngineContext) -> Mapping[str, Any]:
    config = ctx.config

    # Every declared scope must exist. A typo in one root of a whole-application
    # run would otherwise silently review less than the operator asked for.
    missing = [
        name for name, path in zip(config.scopes, config.scope_paths())
        if not path.exists()
    ]
    if missing:
        return {
            "status": "stopped",
            "stop_reason": STOP_UNSAFE,
            "errors": [{"node": "preflight",
                        "message": f"scope(s) do not exist: {missing}"}],
            "_trace": {"scope_exists": False, "missing_scopes": missing},
        }

    branch = git_tools.current_branch(config.repo_root)
    dirty = git_tools.dirty_files(config.repo_root)

    baseline_failures: list[str] = []
    baseline_ran = False
    if config.apply_fixes:
        ctx.counters["test_runs"] += 1
        result = ctx.tests.run(["tests"], stage="baseline")
        baseline_failures = [f["nodeid"] for f in result["failures"]]
        baseline_ran = True
        logger.info(
            "graph_engine.baseline failures=%d passed=%d",
            len(baseline_failures), result["passed"],
        )

    return {
        "baseline_failures": baseline_failures,
        "_trace": {
            "branch": branch,
            "dirty_file_count": len(dirty),
            "scope": config.scope,
            "apply_fixes": config.apply_fixes,
            "baseline_ran": baseline_ran,
            "baseline_failures": baseline_failures,
        },
    }
