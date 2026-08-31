"""TypeScript review via the project's own compiler.

The frontend has no eslint and no test runner, so `tsc --noEmit` is the only
static analysis available — and it is a strong one: a type error is a genuine
correctness defect, not a style opinion.

The compiler is the one already pinned in the frontend's `node_modules`, so this
reports exactly what `npm run build` would, never a different version's opinion.
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_TIMEOUT = 300

#: `src/pages/Leave.tsx(42,7): error TS2345: Argument of type ...`
_TSC_LINE = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+),(?P<col>\d+)\):\s+"
    r"(?P<severity>error|warning)\s+(?P<code>TS\d+):\s+(?P<message>.*)$",
    re.MULTILINE,
)


def frontend_root(repo_root: Path) -> Path | None:
    """The frontend project directory, or None when there isn't one."""
    candidate = repo_root / "Attendance Management" / "frontend"
    return candidate if (candidate / "tsconfig.json").exists() else None


def _tsc_binary(project: Path) -> Path | None:
    for name in ("tsc.cmd", "tsc"):
        candidate = project / "node_modules" / ".bin" / name
        if candidate.exists():
            return candidate
    return None


def run_tsc(repo_root: Path) -> list[dict]:
    """Type-check the frontend and return normalised findings.

    An absent frontend, absent `node_modules` or a crashed compiler yields [] —
    the engine must still run on a machine that has never done `npm install`.
    """
    project = frontend_root(repo_root)
    if project is None:
        return []
    binary = _tsc_binary(project)
    if binary is None:
        logger.warning("graph_engine.tsc_unavailable — run npm install in the frontend")
        return []

    try:
        proc = subprocess.run(
            [str(binary), "--noEmit", "-p", "tsconfig.json"],
            cwd=str(project), capture_output=True, text=True,
            timeout=_TIMEOUT, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("graph_engine.tsc_failed exc=%s", type(exc).__name__)
        return []

    findings: list[dict] = []
    for match in _TSC_LINE.finditer(f"{proc.stdout}\n{proc.stderr}"):
        rel = (project / match.group("file")).resolve()
        try:
            rel_path = rel.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            rel_path = match.group("file")
        findings.append({
            "file": rel_path,
            "line": int(match.group("line")),
            "code": match.group("code"),
            # A type error stops the build; nothing about it is cosmetic.
            "severity": "high" if match.group("severity") == "error" else "medium",
            "problem": match.group("message").strip(),
            "source": "tsc",
            "ruff_fixable": False,
        })
    logger.info("graph_engine.tsc files_checked=project findings=%d", len(findings))
    return findings
