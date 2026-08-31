"""Per-run context: everything a node needs that must not live in graph state.

Satisfies the runtime's context protocol (`run_id`, `traces`, `counters`,
`expired()`, `time_left()`) so the borrowed `StateGraph` can drive it, while
carrying the engine's own handles — config, the sandboxed repository, the test
runner and an optional LLM.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from graph_engine._runtime import NodeTrace
from graph_engine.config import EngineConfig


@dataclass
class EngineContext:
    run_id: str
    config: EngineConfig
    #: graph_engine.tools.repository.Repository — the ONLY way nodes touch disk.
    repo: Any = None
    #: graph_engine.tools.test_runner.TestRunner
    tests: Any = None
    #: Optional callable(messages, max_new_tokens) -> str. None = deterministic.
    llm: Optional[Any] = None

    deadline: Optional[float] = None
    traces: list[NodeTrace] = field(default_factory=list)
    counters: dict = field(
        default_factory=lambda: {
            "llm_calls": 0,
            "files_read": 0,
            "files_written": 0,
            "test_runs": 0,
            "fix_attempts": 0,
        }
    )

    def time_left(self) -> Optional[float]:
        return None if self.deadline is None else self.deadline - time.monotonic()

    def expired(self) -> bool:
        left = self.time_left()
        return left is not None and left <= 0
