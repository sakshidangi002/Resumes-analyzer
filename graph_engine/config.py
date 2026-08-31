"""Engine configuration — every budget, boundary and safety switch in one place.

Defaults are chosen to be safe rather than convenient: fixes are applied only
when explicitly enabled, the scope must be named, and the engine refuses to
touch anything outside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from graph_engine._runtime import REPO_ROOT

#: Hard ceiling on the fix -> test -> analyse loop.
MAX_ITERATIONS = 5
#: Distinct fix attempts allowed across the whole run.
DEFAULT_FIX_BUDGET = 3
#: Whole-run wall clock. A pytest run over this repo takes ~30s, so a run that
#: needs several iterations still fits comfortably.
DEFAULT_TIMEOUT_SECONDS = 900.0
#: Single pytest invocation timeout.
DEFAULT_TEST_TIMEOUT_SECONDS = 300.0

#: Only these ruff rule families are treated as review input. Deliberately
#: narrow: correctness, not style. Style findings would drown the signal and
#: tempt the fixer into touching code that has nothing wrong with it.
#:   E9  syntax / IO errors      F   pyflakes (undefined name, unused import)
#:   B   bugbear (mutable default arg, bare except, ...)
REVIEW_RULE_SELECT = ("E9", "F", "B")

#: Directories never inspected or modified, regardless of scope.
EXCLUDED_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    "third_party", "dist", "build", "chromadb", "uploads", "logs", "backups",
    "frontend_build", "site-packages", ".ruff_cache", "HCNetSDK", "models",
})

#: Paths the fixer must never write to even when they fall inside the scope.
#: Editing these autonomously is either destructive or a production risk.
PROTECTED_PATH_PARTS = frozenset({
    "alembic",           # migrations: ordering and revision ids are load-bearing
    ".env", ".env.example",
    "conftest.py",       # breaking this breaks every test's ability to report
})


#: Every code root in this repository. `--all` expands to these, skipping the
#: vendored, generated and deployment trees that `EXCLUDED_DIRS` already blocks.
ALL_SCOPES: tuple[str, ...] = (
    "Attendance Management/backend/app",   # HRM backend  (~121 files)
    "backend",                             # Resume Analyzer (~14 files)
    "Attendance Management/frontend/src",  # React/TypeScript UI (~59 files)
    "graph_engine",                        # this engine  (~24 files)
    "services",
    "scripts",
    "tests",
)


@dataclass
class EngineConfig:
    """One run's configuration.

    Pass `scope=` for a single directory or `scopes=` for several. A whole-app
    run is `scopes=ALL_SCOPES`. Both forms end up in `self.scopes`; `self.scope`
    is only a display label from then on.
    """

    goal: str
    #: Convenience single scope. Mutually redundant with `scopes`.
    scope: str = ""
    #: Repo-relative directories or files the run may read and change.
    scopes: tuple[str, ...] = ()
    repo_root: Path = field(default_factory=lambda: REPO_ROOT)

    #: When False the fix node reports what it *would* change and changes nothing.
    apply_fixes: bool = False
    #: Allow editing files that have uncommitted modifications. Off by default:
    #: this working tree carries 55 dirty files, and a half-applied autonomous
    #: edit on top of someone's in-progress work is expensive to untangle even
    #: with backups.
    allow_dirty_files: bool = False

    max_iterations: int = MAX_ITERATIONS
    fix_budget: int = DEFAULT_FIX_BUDGET
    #: Fixes attempted per pass before running the tests. Small values make a
    #: failing test attributable to a specific change; large values trade that
    #: attribution for far fewer test cycles, which is the right call for a
    #: whole-application sweep of mechanical fixes.
    fixes_per_pass: int = 4
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    test_timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS

    #: Extra pytest node-ids / paths to run beyond the auto-selected ones.
    extra_test_targets: tuple[str, ...] = ()
    #: Skip the full-suite regression pass (targeted tests only). Faster, weaker.
    skip_regression: bool = False

    def __post_init__(self) -> None:
        if not self.scopes:
            if not self.scope:
                raise ValueError("EngineConfig needs either `scope` or `scopes`")
            self.scopes = (self.scope,)
        # From here on `scope` is a label for reports and state, never a path.
        self.scope = self.scope or ", ".join(self.scopes)

    def scope_paths(self) -> list[Path]:
        return [(self.repo_root / s).resolve() for s in self.scopes]

    def scope_path(self) -> Path:
        """The first scope. Kept for single-scope callers and tests."""
        return self.scope_paths()[0]

    def backup_dir(self, run_id: str) -> Path:
        return self.repo_root / ".graph_engine_runs" / run_id / "backup"
