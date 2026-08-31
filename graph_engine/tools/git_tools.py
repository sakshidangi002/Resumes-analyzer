"""Read-only git access.

Every function here inspects. Nothing commits, pushes, merges, stashes, resets
or checks out. That is a hard rule, not a current limitation: this working tree
carries dozens of uncommitted files, so a "helpful" `git checkout` would delete
work the engine never wrote. Rollback lives in `repository.restore_all()`, which
restores from byte-exact backups instead.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

#: Only these git subcommands may ever run. Anything else raises.
_ALLOWED_SUBCOMMANDS = frozenset({"status", "diff", "rev-parse"})

_TIMEOUT = 30


class GitCommandNotAllowed(Exception):
    """A mutating git subcommand was requested."""


def _git(repo_root: Path, *args: str) -> tuple[int, str]:
    if not args or args[0] not in _ALLOWED_SUBCOMMANDS:
        raise GitCommandNotAllowed(
            f"git {args[0] if args else ''!r} is not read-only; "
            f"allowed: {sorted(_ALLOWED_SUBCOMMANDS)}"
        )
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("graph_engine.git_failed args=%s exc=%s", args, type(exc).__name__)
        return 1, ""
    return proc.returncode, proc.stdout


def current_branch(repo_root: Path) -> str:
    code, out = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    return out.strip() if code == 0 else "unknown"


def dirty_files(repo_root: Path) -> dict[str, str]:
    """Repo-relative path -> porcelain status code for every uncommitted change."""
    code, out = _git(repo_root, "status", "--porcelain")
    if code != 0:
        return {}

    result: dict[str, str] = {}
    for line in out.splitlines():
        if len(line) < 4:
            continue
        status, path = line[:2].strip(), line[3:].strip()
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        # Renames are reported as "old -> new"; the new path is what matters.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        result[path] = status
    return result


def is_file_dirty(repo_root: Path, rel_path: str, dirty: dict[str, str] | None = None) -> bool:
    """True when the file has uncommitted *modifications* to tracked content.

    Untracked files (`??`) are deliberately not "dirty": there is no committed
    baseline to lose, and the repository tool backs them up before writing
    anyway.
    """
    dirty = dirty_files(repo_root) if dirty is None else dirty
    status = dirty.get(rel_path.replace("\\", "/"))
    return bool(status) and status != "??"


def diff_stat(repo_root: Path, paths: list[str]) -> str:
    """`git diff --stat` limited to the given paths. Empty string when clean."""
    if not paths:
        return ""
    code, out = _git(repo_root, "diff", "--stat", "--", *paths)
    return out.strip() if code == 0 else ""


def diff_for_file(repo_root: Path, rel_path: str) -> str:
    code, out = _git(repo_root, "diff", "--", rel_path)
    return out if code == 0 else ""
