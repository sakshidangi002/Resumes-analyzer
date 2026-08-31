"""Ruff as a review input.

Rule selection is passed explicitly rather than inherited from `pyproject.toml`,
because the ambient config yields mostly style findings (`UP`, `RUF`) and a
fixer aimed at those would rewrite large amounts of code that has nothing wrong
with it. `REVIEW_RULE_SELECT` narrows this to correctness: syntax errors,
pyflakes, and bugbear.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from graph_engine.tools.test_runner import _interpreter

logger = logging.getLogger(__name__)

_TIMEOUT = 120

#: Ruff codes that are genuine defects rather than style opinions. Used by the
#: review node to assign severity before triage sees them.
_HIGH = frozenset({"F821", "F811", "F502", "F506", "F601", "F602", "E999", "B002", "B006", "B012"})
_MEDIUM = frozenset({"F401", "F841", "B007", "B008", "B011", "B904", "E722"})


def severity_for(code: str) -> str:
    if code in _HIGH or code.startswith("E9"):
        return "high"
    if code in _MEDIUM:
        return "medium"
    return "low"


def run_ruff(repo_root: Path, files: list[str], select: tuple[str, ...]) -> list[dict]:
    """Return normalised findings. An unavailable ruff yields [], never an error.

    The engine must still work on a machine without ruff — the AST checks and the
    test suite carry the run on their own, with fewer findings.
    """
    if not files:
        return []

    argv = [
        _interpreter(repo_root), "-m", "ruff", "check",
        "--select", ",".join(select),
        "--no-cache",
        "--output-format", "json",
        *files,
    ]
    try:
        proc = subprocess.run(
            argv, cwd=str(repo_root), capture_output=True, text=True,
            timeout=_TIMEOUT, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("graph_engine.ruff_unavailable exc=%s", type(exc).__name__)
        return []

    # Exit 1 means "findings exist" and is the normal case; only a crash (2+)
    # with no parsable payload is a real problem.
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        logger.warning("graph_engine.ruff_unparsable exit=%s", proc.returncode)
        return []

    findings: list[dict] = []
    root = repo_root.resolve()
    for item in payload:
        try:
            rel = Path(item["filename"]).resolve().relative_to(root).as_posix()
        except (ValueError, KeyError):
            continue
        code = item.get("code") or "UNKNOWN"
        findings.append({
            "file": rel,
            "line": int((item.get("location") or {}).get("row") or 0),
            "code": code,
            "severity": severity_for(code),
            "problem": item.get("message", ""),
            "source": "ruff",
            # Ruff supplies a machine-applicable edit for some rules; the fix
            # node uses this to decide whether a safe mechanical fix exists.
            "ruff_fixable": bool(item.get("fix")),
        })
    return findings
