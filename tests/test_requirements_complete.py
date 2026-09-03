"""Every third-party import must be declared in a requirements file.

Cloning this repo onto a second machine failed at the first undeclared import.
That failure is expensive out of proportion to its cause: the error names a
MODULE ("No module named psutil"), not a PACKAGE, and it surfaces one at a time
— install, re-run, hit the next one. Eight were missing when this was written.

This test is the same check the audit script runs, so it fails in CI the moment
someone adds an import without declaring it, rather than on a colleague's
machine a week later.
"""
import ast
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".venv", "venv", "env", ".env", "site-packages", "node_modules",
        "__pycache__", ".git", ".graph_engine_runs", "deploy", "third_party",
        "build", "dist", "release", ".pytest_cache", ".mypy_cache"}
REQ_FILES = [
    ROOT / "requirements.txt",
    ROOT / "Attendance Management" / "backend" / "requirements.txt",
    ROOT / "requirements-test.txt",
    ROOT / "scripts" / "requirements-sync.txt",
]
# Top-level modules that are this repo's own code, imported without a package
# prefix because of how the entry points set sys.path.
OURS = {"app", "backend", "frontend", "scripts", "tests", "net", "conftest",
        "graph_engine", "api", "services", "email_service", "extraction_v3",
        "resume_parser_v2", "email_resume_pipeline", "indeed_resume_pipeline",
        "indeed_resume_downloader"}


def _declared() -> set[str]:
    out = set()
    for rf in REQ_FILES:
        if not rf.exists():
            continue
        for line in rf.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if not line or line.startswith("-"):
                continue
            pkg = line.split("==")[0].split(">=")[0].split("[")[0].split("<")[0]
            if pkg.strip():
                out.add(pkg.strip().lower().replace("_", "-"))
    return out


def _module_to_dist() -> dict[str, str]:
    m: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        for mod in (dist.read_text("top_level.txt") or "").split():
            m.setdefault(mod, name)
        for f in dist.files or []:
            parts = f.parts
            if len(parts) == 1 and parts[0].endswith(".py"):
                m.setdefault(parts[0][:-3], name)
            elif len(parts) > 1 and not parts[0].endswith((".dist-info", ".data")):
                m.setdefault(parts[0], name)
    return m


def _imports() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    found.setdefault(a.name.split(".")[0], rel)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                found.setdefault(node.module.split(".")[0], rel)
    return found


def test_every_third_party_import_is_declared():
    declared = _declared()
    mod_to_dist = _module_to_dist()
    stdlib = set(sys.stdlib_module_names)

    missing = []
    for mod, where in sorted(_imports().items()):
        if mod in stdlib or mod in OURS or mod.startswith("_"):
            continue
        dist = mod_to_dist.get(mod)
        if dist is None:
            # Not installed here, so this environment cannot judge it. The audit
            # script lists these for a human; a test must not guess.
            continue
        if dist.lower().replace("_", "-") not in declared:
            missing.append(f"{mod} (pip install {dist}) — used in {where}")

    assert not missing, (
        "these are imported but declared in no requirements file, so a fresh "
        "clone breaks:\n  " + "\n  ".join(missing)
    )


def test_the_audit_script_agrees_and_exits_clean():
    """The script is what a developer runs by hand; keep it honest too."""
    script = ROOT / "scripts" / "audit_requirements.py"
    if not script.exists():
        pytest.skip("audit script not present")
    r = subprocess.run([sys.executable, str(script)], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, f"scripts/audit_requirements.py reports gaps:\n{r.stdout}"
