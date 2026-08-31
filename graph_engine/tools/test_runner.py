"""Real pytest execution in a subprocess.

The whole point of this tool is that **no LLM decides whether tests passed**.
The exit code comes from pytest, the failure list is parsed from its output, and
the duration is measured by the clock.

Command safety: the executable is the repo's own interpreter, the subcommand is
always `-m pytest`, and every target is validated to be a repo-relative path (or
a `path::node` id) before it reaches the argv. Targets beginning with `-` are
rejected outright, so a crafted finding cannot smuggle in `--exec` style flags.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_FAILED_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s+-\s+(.*))?$", re.MULTILINE)
_COUNT_RE = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "errors": re.compile(r"(\d+) error"),
    "skipped": re.compile(r"(\d+) skipped"),
}


class UnsafeTestTarget(Exception):
    """A test target looked like a flag or escaped the repository."""


def _interpreter(repo_root: Path) -> str:
    """The project's virtualenv interpreter, which is where pytest lives."""
    for candidate in (
        repo_root / ".venv" / "Scripts" / "python.exe",   # Windows
        repo_root / ".venv" / "bin" / "python",           # POSIX
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


@dataclass
class TestRunner:
    #: pytest would otherwise try to collect this as a test class because of the
    #: `Test` prefix, and warn that it cannot instantiate it.
    __test__ = False

    repo_root: Path
    timeout: float = 300.0

    def _validate(self, target: str) -> str:
        if not target or target.startswith("-"):
            raise UnsafeTestTarget(f"{target!r} looks like a command-line flag")
        path_part = target.split("::", 1)[0]
        resolved = (self.repo_root / path_part).resolve()
        try:
            resolved.relative_to(self.repo_root.resolve())
        except ValueError as exc:
            raise UnsafeTestTarget(f"{target!r} points outside the repository") from exc
        if not resolved.exists():
            raise UnsafeTestTarget(f"{target!r} does not exist")
        return target

    def run(self, targets: list[str], *, stage: str) -> dict:
        """Execute pytest and return a structured result. Never raises on failure."""
        safe_targets = [self._validate(t) for t in targets]
        argv = [_interpreter(self.repo_root), "-m", "pytest", "-q", "--no-header",
                "-p", "no:cacheprovider", *safe_targets]

        started = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            stdout, stderr, code, timed_out = proc.stdout, proc.stderr, proc.returncode, False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            code, timed_out = -1, True
            logger.error("graph_engine.tests_timeout stage=%s after=%.0fs", stage, self.timeout)

        duration = time.monotonic() - started
        result = self._parse(stdout, stderr, code, duration, stage, argv, timed_out)
        logger.info(
            "graph_engine.tests stage=%s exit=%s passed=%s failed=%s duration=%.1fs",
            stage, code, result["passed"], result["failed"], duration,
        )
        return result

    @staticmethod
    def _parse(
        stdout: str, stderr: str, code: int, duration: float,
        stage: str, argv: list[str], timed_out: bool,
    ) -> dict:
        combined = f"{stdout}\n{stderr}"
        failures = [
            {"nodeid": nodeid, "message": (message or "").strip()[:400]}
            for nodeid, message in _FAILED_RE.findall(combined)
        ]
        counts = {}
        for name, pattern in _COUNT_RE.items():
            match = pattern.search(combined)
            counts[name] = int(match.group(1)) if match else 0

        return {
            "stage": stage,
            # Command is recorded without the absolute interpreter path, which is
            # machine-specific noise in a trace.
            "command": " ".join(["pytest", *argv[3:]]),
            "exit_code": code,
            "timed_out": timed_out,
            "ok": code == 0 and not timed_out,
            "passed": counts["passed"],
            "failed": counts["failed"] + counts["errors"],
            "skipped": counts["skipped"],
            "duration_s": round(duration, 2),
            "failures": failures,
            # Tail only: full pytest output on this repo is thousands of lines,
            # and the failure list above already carries the signal.
            "tail": combined.strip()[-2000:],
        }
