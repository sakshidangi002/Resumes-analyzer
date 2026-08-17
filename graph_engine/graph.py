"""Graph assembly and `run_engine` — the single public entry point.

    preflight -+- stopped -----------------------------------> report -> END
               |
               +- ok -> review -> bug_analysis -+- none ----------> verify
                                                |                     |
                                                +- bugs -> fix -+- verify
                                                        ^        |
                                                        |        +- test -+- pass -> verify
                                                        |                 |
                                                        |                 +- fail -> failure_analysis
                                                        |                                |
                                                        +--------- fix -----------------+- verify
                                                        |                                |
                                                        |                                +- stop -> report
                                                        |
                                                 verify +- retry (unaddressed bugs)
                                                        +- done -> report -> END

Two cycles, both bounded by `iteration` / `max_iterations` and `fix_budget`:

* **fix -> test -> failure_analysis -> fix** — the repair loop.
* **verify -> bug_analysis -> fix -> ... -> verify** — the drain loop, because the
  fix node deliberately handles a bounded number of bugs per pass.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from graph_engine import edges
from graph_engine._runtime import END, CompiledGraph, StateGraph
from graph_engine.config import EngineConfig
from graph_engine.context import EngineContext
from graph_engine.nodes.bug_analysis import bug_analysis
from graph_engine.nodes.failure_analysis import failure_analysis
from graph_engine.nodes.fix import fix
from graph_engine.nodes.preflight import preflight
from graph_engine.nodes.report import report
from graph_engine.nodes.review import review
from graph_engine.nodes.testing import run_tests
from graph_engine.nodes.verify import verify
from graph_engine.state import new_state, summarize
from graph_engine.tools.repository import Repository
from graph_engine.tools.test_runner import TestRunner

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Construct without compiling, so tests and docs can inspect the topology."""
    g = StateGraph()

    g.add_node("preflight", preflight)
    g.add_node("review", review)
    g.add_node("bug_analysis", bug_analysis)
    g.add_node("fix", fix)
    g.add_node("test", run_tests)
    g.add_node("failure_analysis", failure_analysis)
    g.add_node("verify", verify)
    g.add_node("report", report)

    g.set_entry("preflight")
    # Any node that raises lands in `report`, which records the failure rather
    # than letting a traceback escape a long-running autonomous run.
    g.set_error_node("report")

    g.add_conditional_edges(
        "preflight", edges.route_after_preflight,
        {"ok": "review", "stopped": "report"},
    )
    g.add_edge("review", "bug_analysis")
    g.add_conditional_edges(
        "bug_analysis", edges.route_after_triage,
        {"bugs": "fix", "none": "verify"},
    )
    g.add_conditional_edges(
        "fix", edges.route_after_fix,
        {"test": "test", "verify": "verify"},
    )
    g.add_conditional_edges(
        "test", edges.route_after_test,
        {"pass": "verify", "fail": "failure_analysis"},
    )
    g.add_conditional_edges(
        "failure_analysis", edges.route_after_failure_analysis,
        {"fix": "fix", "verify": "verify", "stop": "report"},
    )
    g.add_conditional_edges(
        "verify", edges.route_after_verify,
        {"done": "report", "retry": "bug_analysis"},
    )
    g.add_edge("report", END)
    return g


_compiled: Optional[CompiledGraph] = None


def get_compiled_graph() -> CompiledGraph:
    global _compiled
    if _compiled is None:
        # Generous step cap: the two cycles are bounded by state, and this is
        # only the engine refusing to spin if a predicate is ever wrong.
        _compiled = build_graph().compile(max_steps=60)
    return _compiled


def run_engine(
    config: EngineConfig,
    *,
    repo: Any = None,
    tests: Any = None,
    llm: Any = None,
) -> dict:
    """Execute one engineering run. Returns final state plus trace and metrics.

    `repo` and `tests` are injectable so the engine's own tests can drive the
    whole graph without touching the filesystem or spawning pytest.
    """
    run_id = uuid.uuid4().hex[:12]
    ctx = EngineContext(
        run_id=run_id,
        config=config,
        repo=repo if repo is not None else Repository(config=config, run_id=run_id),
        tests=tests if tests is not None else TestRunner(
            repo_root=config.repo_root, timeout=config.test_timeout_seconds
        ),
        llm=llm,
        deadline=time.monotonic() + config.timeout_seconds,
    )

    state = new_state(
        goal=config.goal,
        scope=config.scope,
        fix_budget=config.fix_budget,
        max_iterations=config.max_iterations,
        apply_fixes=config.apply_fixes,
    )

    logger.info(
        "graph_engine.start run_id=%s goal=%r scope=%r apply_fixes=%s",
        run_id, config.goal[:120], config.scope, config.apply_fixes,
    )
    started = time.monotonic()
    final = get_compiled_graph().invoke(state, ctx)
    elapsed_ms = (time.monotonic() - started) * 1000

    logger.info(
        "graph_engine.finish run_id=%s duration_ms=%.0f nodes=%d state=%s",
        run_id, elapsed_ms, len(ctx.traces), summarize(final),
    )
    for trace in ctx.traces:
        logger.info("graph_engine.node %s", trace.as_dict())

    final["run_id"] = run_id
    final["trace"] = [t.as_dict() for t in ctx.traces]
    final["metrics"] = {
        "duration_ms": round(elapsed_ms, 1),
        "node_count": len(ctx.traces),
        **ctx.counters,
    }
    return final
