"""Adapter that borrows the graph runtime already proven in this repo.

`Attendance Management/backend/app/assistant/runtime.py` implements exactly the
primitives this engine needs — nodes, static and conditional edges, parallel
fan-out, per-node tracing, cycle bounding and a wall-clock deadline — and it has
its own test suite. Shipping a second engine would mean two implementations of
the same semantics drifting apart.

The coupling is deliberately confined to this one module: the runtime is generic
(no HRM imports, no database), so if it ever needs to be vendored, only this file
changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_HRMS_BACKEND = REPO_ROOT / "Attendance Management" / "backend"

for _path in (str(REPO_ROOT), str(_HRMS_BACKEND)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.assistant.runtime import (  # noqa: E402  (path setup must precede import)
    END,
    CompiledGraph,
    GraphError,
    NodeTrace,
    StateGraph,
)

__all__ = ["END", "CompiledGraph", "GraphError", "NodeTrace", "StateGraph", "REPO_ROOT"]
