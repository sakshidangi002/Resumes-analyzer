"""`bug_analysis` — triage. Which findings are real, why, and which are safe to fix.

The rules here are the engine's judgement, and they are deliberately conservative:
an autonomous fixer that is occasionally wrong is worse than one that hands more
work to a human. Every dismissal carries a reason, so "we looked and decided not
to" is visible in the report rather than looking like an oversight.

The sharpest rule is the SQLAlchemy one. `col == None` inside `filter()` is
correct and required — `is None` there evaluates to a plain `bool` and silently
drops the filter, which would turn a soft-delete query into one that returns
deleted rows. A naive fixer "correcting" that would introduce a data-exposure
bug while reporting a successful cleanup.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from graph_engine.context import EngineContext
from graph_engine.skills import skill_for_node
from graph_engine.tools import ast_checks

logger = logging.getLogger(__name__)

#: Path fragments where an automated edit is never acceptable, regardless of how
#: mechanical the change looks. Mirrors `skills/bug_fixing.md`.
SENSITIVE_PATH_PARTS = (
    "payroll", "salary", "security.py", "auth", "deps.py", "alembic",
    "attendance_service", "leave_service", "encrypted_types", "pii.py",
)

#: code -> mechanical fix strategy. Absence means "no known safe transformation".
FIX_STRATEGIES = {
    "F401": "ruff_fix",             # unused import — ruff supplies a verified edit
    "F841": "ruff_fix",             # unused local
    ast_checks.CMP_NONE: "none_comparison",
}

_PRIORITY = {"high": 1, "medium": 2, "low": 3}


def _is_sensitive(rel_path: str) -> str | None:
    lowered = rel_path.lower()
    for part in SENSITIVE_PATH_PARTS:
        if part in lowered:
            return part
    return None


def _triage(finding: dict) -> dict:
    """Decide on one finding. Returns it enriched, with `dismissed` set."""
    code = finding["code"]
    rel_path = finding["file"]
    result = dict(finding)

    # --- false positives -------------------------------------------------
    if code == ast_checks.CMP_NONE and finding.get("in_query_context"):
        return {
            **result,
            "dismissed": True,
            "dismiss_reason": "SQLAlchemy filter expression — `is None` would drop the "
                              "filter and change the result set",
            "root_cause": None,
            "auto_fixable": False,
            "fix_strategy": None,
        }

    if code == "F401" and rel_path.endswith("__init__.py"):
        return {
            **result,
            "dismissed": True,
            "dismiss_reason": "re-export from a package __init__; removing it breaks importers",
            "root_cause": None,
            "auto_fixable": False,
            "fix_strategy": None,
        }

    # --- real findings ---------------------------------------------------
    strategy = FIX_STRATEGIES.get(code)
    auto_fixable = strategy is not None
    reason_not_fixable = None

    if auto_fixable:
        sensitive = _is_sensitive(rel_path)
        if sensitive:
            auto_fixable, strategy = False, None
            reason_not_fixable = f"touches sensitive area ({sensitive}) — needs human review"
    else:
        reason_not_fixable = "no known mechanical fix; requires understanding the intent"

    root_cause = {
        "F401": "import left behind after the code that used it was removed or moved",
        "F841": "assignment kept after its consumer was removed",
        "F821": "name is referenced but never bound in any reachable scope",
        "E999": "file does not parse, so it cannot be imported at all",
        "E722": "bare `except:` also catches KeyboardInterrupt and SystemExit",
        "B006": "mutable default is created once at definition and shared across calls",
        ast_checks.CMP_NONE: "identity comparison written as equality",
        ast_checks.EXCEPT_PASS: "handler discards the exception, so the failure is unobservable",
    }.get(code, "not determined from static analysis alone")

    return {
        **result,
        "dismissed": False,
        "dismiss_reason": None,
        "root_cause": root_cause,
        "auto_fixable": auto_fixable,
        "fix_strategy": strategy,
        "not_fixable_reason": reason_not_fixable,
    }


def bug_analysis(state: Mapping[str, Any], ctx: EngineContext) -> Mapping[str, Any]:
    skill_name, _ = skill_for_node("bug_analysis")
    findings = list(state.get("review_findings") or [])

    # Bugs already fixed in an earlier iteration must not be re-triaged, or the
    # loop would keep proposing the same fix.
    already_fixed = {
        (f["file"], f["line"], f.get("code"))
        for f in (state.get("fixes") or [])
        if f.get("applied")
    }

    bugs: list[dict] = []
    dismissed: list[dict] = []
    for finding in findings:
        key = (finding["file"], finding["line"], finding.get("code"))
        if key in already_fixed:
            continue
        triaged = _triage(finding)
        (dismissed if triaged["dismissed"] else bugs).append(triaged)

    bugs.sort(key=lambda b: (_PRIORITY.get(b["severity"], 9), not b["auto_fixable"], b["file"]))
    for index, bug in enumerate(bugs, 1):
        bug["priority"] = index

    actionable = [b for b in bugs if b["auto_fixable"]]
    logger.info(
        "graph_engine.triage findings=%d bugs=%d dismissed=%d actionable=%d",
        len(findings), len(bugs), len(dismissed), len(actionable),
    )
    return {
        "bugs": bugs,
        "dismissed": dismissed,
        "_trace": {
            "skill": skill_name,
            "findings_in": len(findings),
            "bugs": len(bugs),
            "dismissed": len(dismissed),
            "actionable": len(actionable),
            "dismiss_reasons": sorted({d["dismiss_reason"] for d in dismissed if d.get("dismiss_reason")}),
        },
    }
