"""Terminal-intent nodes: nothing to fetch, just set an outcome.

`fallback_node` is the graph's honest "no" — and it does something useful with
it. The HRMS already has an employee↔HR ticket system (`POST /api/queries`), so
an unanswerable question suggests filing one instead of dead-ending. That reuses
a shipped feature rather than inventing an escalation path.
"""
from __future__ import annotations

from typing import Any, Mapping

from app.assistant import failures
from app.assistant.runtime import RunContext
from app.assistant.state import INTENT_LEAVE, INTENT_PAYROLL

#: Recognised-but-not-built capabilities. Naming them beats a generic shrug and
#: tells the user (and the logs) exactly what phase 2 should add.
_KNOWN_UNBUILT = {
    INTENT_LEAVE: "I can't answer leave questions yet — attendance only for now.",
    INTENT_PAYROLL: "I can't answer payroll or payslip questions yet — attendance only for now.",
}

_UNKNOWN_TEXT = (
    "I can currently answer attendance questions — for example "
    "\"how many days was I absent last month?\". "
    "For anything else, you can raise a query with HR."
)


def fallback_node(state: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
    intent = state.get("intent")
    message = _KNOWN_UNBUILT.get(intent, _UNKNOWN_TEXT)
    return {
        "status": "unsupported",
        "error": {
            "kind": failures.UNSUPPORTED,
            "message": message,
            "node": "fallback_node",
        },
        "_trace": {"intent": intent, "recognised": intent in _KNOWN_UNBUILT},
    }
