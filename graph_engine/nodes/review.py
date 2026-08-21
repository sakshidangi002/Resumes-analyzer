"""`review` — inspect the scope and report findings. Writes nothing.

Two independent analysers feed it:

* **ruff**, restricted to correctness families (`E9`, `F`, `B`). Style rules are
  excluded deliberately — see `config.REVIEW_RULE_SELECT`.
* **AST checks** for patterns ruff cannot judge in this codebase, which attach
  the context facts triage needs (`in_query_context`, …).

The node reports facts and assigns severity. It does **not** decide what is real
— that is `bug_analysis`. Keeping the two apart is what makes it possible to see
that a finding was considered and rejected, rather than never noticed.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from graph_engine.config import REVIEW_RULE_SELECT
from graph_engine.context import EngineContext
from graph_engine.skills import skill_for_node
from graph_engine.tools import ast_checks, linters, typescript

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

#: Human-readable guidance per code, so a finding says what to do about it.
_ACTIONS = {
    "F401": "Remove the unused import, or re-export it explicitly via __all__.",
    "F811": "Remove or rename the duplicate definition — the earlier one is dead.",
    "F821": "Define the name, fix the typo, or add the missing import.",
    "F841": "Remove the unused local, or use it.",
    "E722": "Catch a specific exception type instead of a bare `except:`.",
    "E999": "Fix the syntax error — this file cannot be imported at all.",
    "B006": "Use `None` as the default and build the mutable value inside the function.",
    "B008": "Move the function call out of the default argument.",
    "B904": "Use `raise ... from err` inside `except` to preserve the cause.",
    ast_checks.CMP_NONE: "Use `is None` / `is not None` — unless this is a SQLAlchemy filter.",
    ast_checks.EXCEPT_PASS: "Log the exception, or narrow the handler and comment why it is ignored.",
}


def _dedupe(findings: list[dict]) -> list[dict]:
    """One record per (file, line, code). Both analysers can see the same defect."""
    seen: set[tuple] = set()
    unique: list[dict] = []
    for finding in findings:
        key = (finding["file"], finding["line"], finding["code"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


def review(state: Mapping[str, Any], ctx: EngineContext) -> Mapping[str, Any]:
    skill_name, _skill_text = skill_for_node("review")
    files = ctx.repo.list_source_files()
    python_files = [f for f in files if f.endswith('.py')]
    ts_files = [f for f in files if f.endswith(('.ts', '.tsx'))]

    if not files:
        return {
            "files": [],
            "review_findings": [],
            "_trace": {"skill": skill_name, "files": 0, "findings": 0, "empty_scope": True},
        }

    findings: list[dict] = list(
        linters.run_ruff(ctx.config.repo_root, python_files, REVIEW_RULE_SELECT)
    )
    ruff_count = len(findings)
    ast_count = 0
    tsc_findings = typescript.run_tsc(ctx.config.repo_root) if ts_files else []
    findings.extend(tsc_findings)

    for rel_path in python_files:
        try:
            source = ctx.repo.read(rel_path)
        except OSError as exc:
            findings.append({
                "file": rel_path, "line": 0, "code": "GE000", "severity": "medium",
                "problem": f"Could not read file: {type(exc).__name__}",
                "source": "engine", "ruff_fixable": False,
            })
            continue
        ctx.counters["files_read"] += 1
        ast_findings = ast_checks.analyze_file(rel_path, source)
        ast_count += len(ast_findings)
        findings.extend(ast_findings)

    # Counted before dedupe. Deriving the AST count as `len(findings) - ruff_count`
    # went negative as soon as dedupe removed more overlaps than the AST checks
    # contributed, which it does on a whole-application run.
    before_dedupe = len(findings)
    findings = _dedupe(findings)
    duplicates = before_dedupe - len(findings)
    for finding in findings:
        finding.setdefault("reason", finding.get("problem", ""))
        finding["recommended_action"] = _ACTIONS.get(
            finding["code"], "Inspect manually — no mechanical remedy is known."
        )

    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f["severity"], 9), f["file"], f["line"]))

    by_severity: dict[str, int] = {}
    for finding in findings:
        by_severity[finding["severity"]] = by_severity.get(finding["severity"], 0) + 1

    logger.info(
        "graph_engine.review files=%d (py=%d ts=%d) findings=%d ruff=%d ast=%d tsc=%d deduped=%d",
        len(files), len(python_files), len(ts_files), len(findings),
        ruff_count, ast_count, len(tsc_findings), duplicates,
    )
    return {
        "files": files,
        "review_findings": findings,
        "_trace": {
            "skill": skill_name,
            "files": len(files),
            "python_files": len(python_files),
            "ts_files": len(ts_files),
            "tsc_findings": len(tsc_findings),
            "findings": len(findings),
            "by_severity": by_severity,
            "ruff_findings": ruff_count,
            "ast_findings": ast_count,
            "deduped": duplicates,
        },
    }
