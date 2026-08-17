"""`fix` — the only node that writes. Applies mechanical fixes, one bug at a time.

Every strategy produces *new file content*, which is then written through
`Repository.write()`. Nothing here calls `open()` directly, so path validation,
the scope boundary, the protected-path list and the byte-exact backup all apply
without the strategies having to remember them.

Three guards run before any write:

1. `apply_fixes` must be on — otherwise the node reports what it would do.
2. The fix budget must not be spent.
3. The file must not carry uncommitted modifications (unless explicitly allowed):
   layering an autonomous edit on someone's in-progress work is expensive to
   untangle even with a backup.

And one guard runs after: the modified source must still parse. If it does not,
the write is rolled back from the backup immediately rather than being handed to
the test node to discover.
"""
from __future__ import annotations

import ast
import logging
import re
import subprocess
from typing import Any, Mapping

from graph_engine.context import EngineContext
from graph_engine.skills import skill_for_node
from graph_engine.tools import git_tools
from graph_engine.tools.repository import PathNotAllowed
from graph_engine.tools.test_runner import _interpreter

logger = logging.getLogger(__name__)

#: One fix per node execution keeps the loop legible: a failing test can be
#: attributed to a specific change instead of a batch.
FIXES_PER_ITERATION = 4

_NONE_EQ = re.compile(r"==\s*None\b")
_NONE_NE = re.compile(r"!=\s*None\b")


class FixFailed(Exception):
    """A strategy could not produce a change. Recorded, never fatal."""


def _strategy_none_comparison(source: str, bug: dict) -> str:
    """`x == None` -> `x is None` on the finding's line only."""
    lines = source.splitlines(keepends=True)
    index = bug["line"] - 1
    if not 0 <= index < len(lines):
        raise FixFailed(f"line {bug['line']} out of range")

    original = lines[index]
    patched = _NONE_NE.sub("is not None", _NONE_EQ.sub("is None", original))
    if patched == original:
        raise FixFailed("no `== None` / `!= None` found on that line")
    lines[index] = patched
    return "".join(lines)


def _strategy_ruff_fix(source: str, bug: dict, ctx: EngineContext) -> str:
    """Let ruff produce the edit for the single rule that was reported.

    Uses stdin/stdout rather than `--fix` on the file, so the write still goes
    through the repository tool and inherits its backup and validation.
    """
    argv = [
        _interpreter(ctx.config.repo_root), "-m", "ruff", "check",
        "--select", bug["code"], "--fix", "--no-cache",
        "--stdin-filename", bug["file"], "-",
    ]
    # Bytes, not `text=True`: with text mode subprocess encodes stdin using the
    # locale codec, which is cp1252 on Windows. Ruff requires UTF-8 and rejects
    # the stream outright ("did not contain valid UTF-8") the moment a source
    # file contains a non-ASCII character such as an em dash.
    try:
        proc = subprocess.run(
            argv, input=source.encode("utf-8"), cwd=str(ctx.config.repo_root),
            capture_output=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FixFailed(f"ruff unavailable: {type(exc).__name__}") from exc

    patched = proc.stdout.decode("utf-8", errors="strict")
    if not patched.strip():
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise FixFailed(f"ruff produced no output (rc={proc.returncode}): {stderr[:160]}")
    if patched == source:
        raise FixFailed(f"ruff had no fix for {bug['code']}")
    return patched


_STRATEGIES = {
    "none_comparison": lambda source, bug, ctx: _strategy_none_comparison(source, bug),
    "ruff_fix": _strategy_ruff_fix,
}


#: `declined` = the engine chose not to (dry run, budget, dirty file, no strategy).
#: `error` = the engine tried and broke. Verification treats these differently:
#: a decline is a legitimate outcome, an error means the run did not really work.
OUTCOME_APPLIED = "applied"
OUTCOME_DECLINED = "declined"
OUTCOME_ERROR = "error"


def _record(
    bug: dict, applied: bool, reason: str,
    outcome: str = OUTCOME_DECLINED, strategy: str | None = None,
) -> dict:
    return {
        "file": bug["file"],
        "line": bug["line"],
        "code": bug.get("code"),
        "strategy": strategy if strategy is not None else bug.get("fix_strategy"),
        "applied": applied,
        "outcome": OUTCOME_APPLIED if applied else outcome,
        "reason": reason,
    }


def fix(state: Mapping[str, Any], ctx: EngineContext) -> Mapping[str, Any]:
    skill_name, _ = skill_for_node("fix")
    config = ctx.config
    budget = int(state.get("fix_budget", 0) or 0)
    existing_fixes = list(state.get("fixes") or [])
    attempted_keys = {(f["file"], f["line"], f.get("code")) for f in existing_fixes}

    pending = [
        bug for bug in (state.get("bugs") or [])
        if bug.get("auto_fixable")
        and (bug["file"], bug["line"], bug.get("code")) not in attempted_keys
    ]
    # `FIXES_PER_ITERATION` exists so a failing test can be attributed to a
    # specific change. A dry run makes no changes, so there is nothing to
    # attribute — record a decline for every candidate at once instead of
    # looping five times to decline four at a time.
    candidates = pending if not config.apply_fixes else pending[:FIXES_PER_ITERATION]

    new_fixes: list[dict] = []
    changed: list[str] = list(state.get("changed_files") or [])
    dirty = git_tools.dirty_files(config.repo_root)

    for bug in candidates:
        if budget <= 0:
            new_fixes.append(_record(bug, False, "fix budget exhausted"))
            continue

        if not config.apply_fixes:
            new_fixes.append(_record(bug, False, "dry run — apply_fixes is off"))
            continue

        if not config.allow_dirty_files and git_tools.is_file_dirty(
            config.repo_root, bug["file"], dirty
        ):
            new_fixes.append(_record(
                bug, False,
                "file has uncommitted modifications; rerun with --allow-dirty to override",
            ))
            continue

        strategy_name = bug.get("fix_strategy")
        strategy = _STRATEGIES.get(strategy_name or "")
        if strategy is None:
            new_fixes.append(_record(bug, False, f"no strategy registered for {strategy_name!r}"))
            continue

        ctx.counters["fix_attempts"] += 1
        try:
            source = ctx.repo.read(bug["file"])
            patched = strategy(source, bug, ctx)
        except (FixFailed, PathNotAllowed, OSError) as exc:
            new_fixes.append(_record(bug, False, f"{type(exc).__name__}: {exc}", OUTCOME_ERROR))
            continue

        # Never hand unparsable source to the test node.
        try:
            ast.parse(patched, filename=bug["file"])
        except SyntaxError as exc:
            new_fixes.append(_record(bug, False, f"fix produced invalid syntax: {exc.msg}", OUTCOME_ERROR))
            continue

        try:
            ctx.repo.write(
                bug["file"], patched,
                reason=f"{strategy_name} for {bug['code']} at line {bug['line']}",
            )
        except PathNotAllowed as exc:
            new_fixes.append(_record(bug, False, f"blocked by sandbox: {exc}", OUTCOME_ERROR))
            continue

        ctx.counters["files_written"] += 1
        budget -= 1
        if bug["file"] not in changed:
            changed.append(bug["file"])
        new_fixes.append(_record(bug, True, f"applied {strategy_name}"))

    applied_count = sum(1 for f in new_fixes if f["applied"])
    logger.info(
        "graph_engine.fix candidates=%d applied=%d budget_left=%d",
        len(candidates), applied_count, budget,
    )
    return {
        "fixes": existing_fixes + new_fixes,
        "changed_files": changed,
        "fix_budget": budget,
        "iteration": int(state.get("iteration", 0) or 0) + 1,
        "_trace": {
            "skill": skill_name,
            "candidates": len(candidates),
            "applied": applied_count,
            "skipped": len(new_fixes) - applied_count,
            "budget_left": budget,
            "changed_files": changed,
        },
    }
