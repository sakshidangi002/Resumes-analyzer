"""Sandboxed repository access: the only way a node touches the filesystem.

Every path is validated before use:

* resolved and required to sit inside `repo_root` (blocks `../` traversal and
  absolute paths pointing anywhere else on the machine);
* rejected if any component is in `EXCLUDED_DIRS` (`.venv`, `node_modules`,
  `third_party`, …) — those are not this repository's code;
* for **writes**, additionally required to sit inside the run's declared scope
  and not to match `PROTECTED_PATH_PARTS` (migrations, `.env`, `conftest.py`).

Writes take a byte-exact backup first. Rollback restores from that backup and
never touches git — this working tree carries dozens of uncommitted files, and
`git checkout`/`stash`/`reset` would destroy work the engine did not create.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from graph_engine.config import EXCLUDED_DIRS, PROTECTED_PATH_PARTS, EngineConfig

logger = logging.getLogger(__name__)


class PathNotAllowed(Exception):
    """A node asked for a path outside the sandbox."""


@dataclass
class WriteRecord:
    rel_path: str
    backup_path: str
    reason: str


@dataclass
class Repository:
    config: EngineConfig
    run_id: str
    writes: list[WriteRecord] = field(default_factory=list)
    reads: set[str] = field(default_factory=set)

    # -- validation ---------------------------------------------------------
    def _resolve(self, rel_path: str) -> Path:
        root = self.config.repo_root.resolve()
        candidate = (root / rel_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PathNotAllowed(f"{rel_path!r} resolves outside the repository") from exc

        rel_parts = candidate.relative_to(root).parts
        blocked = EXCLUDED_DIRS.intersection(rel_parts)
        if blocked:
            raise PathNotAllowed(f"{rel_path!r} is inside an excluded directory: {sorted(blocked)}")
        return candidate

    def _resolve_for_write(self, rel_path: str) -> Path:
        path = self._resolve(rel_path)

        # A whole-application run declares several scopes; the write must fall
        # inside at least one of them.
        in_scope = any(
            path == scope or (scope.is_dir() and scope in path.parents)
            for scope in self.config.scope_paths()
        )
        if not in_scope:
            raise PathNotAllowed(
                f"{rel_path!r} is outside the run scope(s) {self.config.scope!r}"
            )

        parts = path.relative_to(self.config.repo_root.resolve()).parts
        protected = PROTECTED_PATH_PARTS.intersection(parts)
        if protected:
            raise PathNotAllowed(f"{rel_path!r} is protected from automated edits: {sorted(protected)}")
        if path.suffix != ".py":
            raise PathNotAllowed(f"{rel_path!r} is not a Python source file")
        return path

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.config.repo_root.resolve()).as_posix()

    # -- read ---------------------------------------------------------------
    def list_python_files(self, scope: str | None = None) -> list[str]:
        """Python files in the given scope, or across every configured scope.

        De-duplicated: whole-application runs can declare overlapping roots, and
        reviewing the same file twice would double every finding in it.
        """
        scopes = [scope] if scope is not None else list(self.config.scopes)

        found: list[str] = []
        seen: set[str] = set()
        for entry in scopes:
            target = self._resolve(entry)
            if not target.exists():
                continue
            candidates = [target] if target.is_file() else sorted(target.rglob("*.py"))
            for path in candidates:
                if path.suffix != ".py":
                    continue
                rel_parts = path.resolve().relative_to(self.config.repo_root.resolve()).parts
                if EXCLUDED_DIRS.intersection(rel_parts):
                    continue
                rel = self.rel(path)
                if rel not in seen:
                    seen.add(rel)
                    found.append(rel)
        return sorted(found)

    def read(self, rel_path: str) -> str:
        """Read source the way Python's own import machinery does.

        `utf-8-sig`, not `utf-8`: several files in this repo carry a UTF-8 BOM
        (`app/core/config.py`, `routes/employees.py`). Python strips it on
        import, but plain `utf-8` leaves U+FEFF at the start of the string, and
        `ast.parse` then reports a syntax error for a file that imports fine —
        a false "this file cannot be imported at all" on healthy code.
        """
        path = self._resolve(rel_path)
        self.reads.add(rel_path)
        return path.read_text(encoding="utf-8-sig")

    def read_lines(self, rel_path: str) -> list[str]:
        return self.read(rel_path).splitlines()

    def exists(self, rel_path: str) -> bool:
        try:
            return self._resolve(rel_path).exists()
        except PathNotAllowed:
            return False

    # -- write --------------------------------------------------------------
    def write(self, rel_path: str, content: str, reason: str) -> WriteRecord:
        """Back up, then overwrite. Raises `PathNotAllowed` if out of bounds."""
        path = self._resolve_for_write(rel_path)

        backup_root = self.config.backup_dir(self.run_id)
        backup_path = backup_root / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():  # first write wins: the original, not an intermediate
            shutil.copy2(path, backup_path)

        # newline="" keeps the file's existing line endings; this repo has a mix
        # of CRLF and LF and rewriting them would swamp the diff with noise.
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)

        record = WriteRecord(rel_path=rel_path, backup_path=str(backup_path), reason=reason)
        self.writes.append(record)
        logger.info("graph_engine.write run_id=%s file=%s reason=%s", self.run_id, rel_path, reason)
        return record

    def changed_files(self) -> list[str]:
        seen: list[str] = []
        for record in self.writes:
            if record.rel_path not in seen:
                seen.append(record.rel_path)
        return seen

    def restore_all(self) -> list[str]:
        """Undo every write from this run, byte for byte. Never calls git."""
        restored: list[str] = []
        for record in reversed(self.writes):
            backup = Path(record.backup_path)
            if not backup.exists():
                continue
            shutil.copy2(backup, self._resolve(record.rel_path))
            restored.append(record.rel_path)
        logger.warning(
            "graph_engine.restore run_id=%s files=%s", self.run_id, sorted(set(restored))
        )
        return restored
