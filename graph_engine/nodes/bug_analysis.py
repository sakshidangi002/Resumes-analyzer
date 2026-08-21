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
import re
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
    "F841": "ruff_fix",             # unused local (ruff's fix is unsafe; often declines)
    "B033": "ruff_fix",             # duplicate item in a set literal — sets dedupe anyway
    "F541": "ruff_fix",             # f-string with no placeholder -> plain string
    "B007": "ruff_fix",             # unused loop variable -> renamed to `_`
    "B905": "ruff_fix",             # zip() -> zip(..., strict=False), preserving behaviour
    ast_checks.CMP_NONE: "none_comparison",
    ast_checks.EXCEPT_PASS: "except_pass_logging",
    "B904": "raise_from",          # `from e` only; `from None` stays manual
}

#: Deliberately NOT here, and why:
#:   B023  rebinding a loop variable in a closure is a real code change.
#:   GE002 is always fixable now: proven best-effort handlers get a debug log,
#:         everything else an honest `logger.warning("<operation> failed")`.
#:         Neither touches control flow. Sensitive paths are still refused.

#: Codes whose fix is pure hygiene — no runtime behaviour can change.
IMPORT_HYGIENE_CODES = frozenset({"F401", "F541", "B033"})

_PRIORITY = {"high": 1, "medium": 2, "low": 3}

#: Callables that are *supposed* to appear in an argument default. FastAPI's
#: entire dependency-injection and parameter-declaration mechanism is built on
#: them, so ruff's B008 ("do not perform function call in argument defaults")
#: is a false positive for every route in this application — 436 of them on a
#: whole-application run, which is more than two thirds of all findings.
#: Rewriting them would break every endpoint.
_INTENTIONAL_DEFAULT_CALLS = frozenset({
    "Depends", "Query", "Path", "Body", "Header", "Cookie",
    "Form", "File", "Security",
    # This application's own dependency factories, used exactly like Depends():
    # `current_user: User = Depends(require_roles(["Admin"]))`.
    "require_roles", "require_media_access", "get_current_user", "get_db",
})

_B008_CALLABLE = re.compile(r"function call `([A-Za-z_][A-Za-z0-9_.]*)`")


def _is_sensitive(rel_path: str) -> str | None:
    lowered = rel_path.lower()
    for part in SENSITIVE_PATH_PARTS:
        if part in lowered:
            return part
    return None


def _triage(finding: dict, ctx: EngineContext | None = None) -> dict:
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

    if code == "B023" and ctx is not None:
        try:
            source = ctx.repo.read(rel_path)
            in_place = ast_checks.closure_is_called_in_place(source, finding["line"])
        except Exception:  # noqa: BLE001 - an unreadable file just means no verdict
            in_place = None
        if in_place is True:
            return {
                **result,
                "dismissed": True,
                "dismiss_reason": "the closure is only ever called inside its own loop "
                                  "iteration, so the loop variable holds the intended "
                                  "value — late binding never occurs",
                "root_cause": None,
                "auto_fixable": False,
                "fix_strategy": None,
            }

    if code == "B008":
        match = _B008_CALLABLE.search(finding.get("problem", ""))
        callable_name = (match.group(1).rsplit(".", 1)[-1] if match else "")
        if callable_name in _INTENTIONAL_DEFAULT_CALLS:
            return {
                **result,
                "dismissed": True,
                "dismiss_reason": f"`{callable_name}()` in an argument default is the "
                                  f"FastAPI declaration idiom, not a defect — moving it "
                                  f"would break the endpoint",
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
        if sensitive and code in IMPORT_HYGIENE_CODES:
            # Removing an unused import or an empty f-string prefix cannot alter
            # behaviour, so the sensitive-area guard does not apply to them.
            sensitive = None
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
        triaged = _triage(finding, ctx)
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
