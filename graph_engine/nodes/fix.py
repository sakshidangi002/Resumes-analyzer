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
from graph_engine.tools import ast_checks, git_tools
from graph_engine.tools.repository import PathNotAllowed
from graph_engine.tools.test_runner import _interpreter

logger = logging.getLogger(__name__)

#: Default is on EngineConfig.fixes_per_pass; see there for the trade-off.

_NONE_EQ = re.compile(r"==\s*None\b")
_NONE_NE = re.compile(r"!=\s*None\b")


class FixFailed(Exception):
    """A strategy broke while trying. Recorded as an engine error, never fatal."""


class FixUnavailable(Exception):
    """No mechanical fix exists for this finding.

    Distinct from `FixFailed`: nothing went wrong, the tooling simply has
    nothing to offer. Recorded as a decline so it does not fail verification.
    """


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
        # Usually because a sibling finding in this same file already triggered
        # the whole-file fix — see `_siblings_resolved_by`. Not an error.
        raise FixUnavailable(f"ruff has no remaining {bug['code']} fix for this file")
    return patched



def _strategy_except_pass_logging(source: str, bug: dict, ctx: EngineContext) -> str:
    """Make a swallowed exception observable, without touching control flow.

    Three cases, in order of specificity:

    1. the handler is provably best-effort  -> `logger.debug("ignored, non-critical")`
    2. the module has no logger             -> inject one, then re-decide
    3. anything else                        -> `logger.warning("<operation> failed")`

    The warning branch exists because calling a swallowed camera-worker crash
    "non-critical" would be a lie. The message names the actual operation taken
    from the `try` body, so the log says what really failed.

    The exception is still swallowed either way: no raise, no return, no change
    of exception type. Only a log line is added.
    """
    lines = source.splitlines(keepends=True)

    # 2. Inject a module logger if there is none. Approved as mechanical: it adds
    #    no control flow and no business behaviour.
    plan = ast_checks.module_logger_plan(source)
    if plan["needed"]:
        newline = "\r\n" if source.endswith("\r\n") or (
            lines and lines[0].endswith("\r\n")) else "\n"
        block = []
        if plan["needs_import"]:
            block.append("import logging" + newline)
        block.append(newline)
        block.append("logger = logging.getLogger(__name__)" + newline)
        lines[plan["insert_at"]:plan["insert_at"]] = block
        source = "".join(lines)
        bug = {**bug, "line": bug["line"] + len(block)}

    intent = ast_checks.except_pass_intent(source, bug["line"])
    if intent["logger_name"] is None:
        raise FixFailed("logger injection did not take effect")

    lines = source.splitlines(keepends=True)
    if intent["provable"]:
        pass_line = intent["lineno"]
        indent = intent["indent"]
        message = '"ignored, non-critical"'
        level = "debug"
    else:
        pass_line = indent = None
        for i, line in enumerate(lines, 1):
            if line.strip() == "pass" and i >= bug["line"]:
                pass_line = i
                indent = line[: len(line) - len(line.lstrip())]
                break
        if pass_line is None:
            raise FixUnavailable("the handler body is no longer a bare `pass`")
        operation = ast_checks.swallowed_operation(source, bug["line"])
        message = '"%s failed"' % operation
        level = "warning"

    index = pass_line - 1
    if lines[index].strip() != "pass":
        raise FixUnavailable("the handler body is no longer a bare `pass`")
    newline = "\r\n" if lines[index].endswith("\r\n") else "\n"
    lines[index] = (
        indent + intent["logger_name"] + "." + level
        + "(" + message + ", exc_info=True)" + newline
    )
    return "".join(lines)



def _strategy_raise_from(source: str, bug: dict, ctx: EngineContext) -> str:
    """Append ` from <exc>` to a raise inside an except block.

    Sets __cause__ and nothing else: same exception, same control flow, same
    handler catches it. Only the traceback improves. `from None` -- which hides
    the cause -- is never inferred; those sites stay manual.
    """
    plan = ast_checks.raise_from_plan(source, bug["line"])
    if not plan["provable"]:
        raise FixUnavailable(plan["reason"])

    lines = source.splitlines(keepends=True)
    index = plan["insert_line"] - 1
    if not 0 <= index < len(lines):
        raise FixUnavailable("raise line out of range")

    line = lines[index]
    col = plan["insert_col"]
    if col is None or col > len(line):
        raise FixUnavailable("could not locate the end of the raised expression")
    lines[index] = line[:col] + " from " + plan["name"] + line[col:]
    return "".join(lines)


_STRATEGIES = {
    "none_comparison": lambda source, bug, ctx: _strategy_none_comparison(source, bug),
    "ruff_fix": _strategy_ruff_fix,
    "except_pass_logging": _strategy_except_pass_logging,
    "raise_from": _strategy_raise_from,
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


def _siblings_resolved_by(bug: dict, pending: list[dict], handled: set) -> list[dict]:
    """Findings the same whole-file fix already removed."""
    resolved: list[dict] = []
    for other in pending:
        key = (other["file"], other["line"], other.get("code"))
        if key in handled:
            continue
        if other["file"] == bug["file"] and other.get("code") == bug.get("code"):
            handled.add(key)
            resolved.append(_record(
                other, True,
                f"resolved by the whole-file {bug['code']} fix at line {bug['line']}",
                strategy="ruff_fix",
            ))
    return resolved


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
    # `fixes_per_pass` exists so a failing test can be attributed to a specific
    # change. A dry run makes no changes, so there is nothing to attribute —
    # record a decline for every candidate at once instead of looping to decline
    # a handful at a time.
    candidates = pending if not config.apply_fixes else pending[:config.fixes_per_pass]

    new_fixes: list[dict] = []
    changed: list[str] = list(state.get("changed_files") or [])
    dirty = git_tools.dirty_files(config.repo_root)
    #: Keys recorded this pass, so a sibling resolved by a whole-file fix is not
    #: then processed again in its own right.
    handled: set = set()

    for bug in candidates:
        if (bug["file"], bug["line"], bug.get("code")) in handled:
            continue
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
        except FixUnavailable as exc:
            new_fixes.append(_record(bug, False, str(exc), OUTCOME_DECLINED))
            continue
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
        handled.add((bug["file"], bug["line"], bug.get("code")))

        # `ruff --fix` rewrites the WHOLE file, so it removes every instance of
        # the rule at once. Without this, the second F401 in the same file would
        # be attempted against an already-clean file, find nothing, and be
        # reported as a failure — when in fact it was just fixed.
        if strategy_name == "ruff_fix":
            new_fixes.extend(_siblings_resolved_by(bug, pending, handled))

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
