"""Which third-party packages does this code import but never declare?

Run:  python scripts/audit_requirements.py     (exit 1 if anything is missing)

Which third-party packages does this code import but never declare?

Cloning the repo onto a second machine fails at the first missing import, and
the error names a MODULE ("No module named cv2") not a PACKAGE ("opencv-python"),
so the fix is guesswork. This walks every .py file, resolves each top-level
import to its distribution name via the installed metadata, and diffs that
against the requirements files.
"""
import ast
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(r"c:\sakshi folder\application\Resume analyzer")
# There is a SECOND virtualenv at Attendance Management/backend/venv, so
# skipping only ".venv" walked 40k library files and buried the real answer.
SKIP = {".venv", "venv", "env", ".env", "site-packages", "node_modules",
        "__pycache__", ".git", ".graph_engine_runs", "deploy", "third_party",
        "build", "dist", "release", ".pytest_cache", ".mypy_cache"}

# ONE file. The dependencies used to live in four, which is how eight imports
# ended up declared in none of them and thirteen packages ended up declared
# twice with different pins.
REQ_FILES = [ROOT / "requirements.txt"]

# ── every top-level module imported anywhere in our own code ───────────────
imported: dict[str, set[str]] = {}
files = 0
for path in ROOT.rglob("*.py"):
    if any(part in SKIP for part in path.parts):
        continue
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        continue
    files += 1
    rel = str(path.relative_to(ROOT))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.setdefault(a.name.split(".")[0], set()).add(rel)
        elif isinstance(node, ast.ImportFrom):
            if node.level:          # relative import — our own code
                continue
            if node.module:
                imported.setdefault(node.module.split(".")[0], set()).add(rel)

# ── module -> distribution, from what is actually installed ────────────────
mod_to_dist: dict[str, set[str]] = {}
for dist in metadata.distributions():
    name = dist.metadata["Name"]
    if not name:
        continue
    for mod in (dist.read_text("top_level.txt") or "").split():
        mod_to_dist.setdefault(mod, set()).add(name)
    # Wheels without top_level.txt: fall back to the recorded file paths.
    for f in dist.files or []:
        parts = f.parts
        if parts and parts[0].endswith(".py") and len(parts) == 1:
            mod_to_dist.setdefault(parts[0][:-3], set()).add(name)
        elif len(parts) > 1 and not parts[0].endswith((".dist-info", ".data")):
            mod_to_dist.setdefault(parts[0], set()).add(name)

STDLIB = set(sys.stdlib_module_names)
OURS = {"app", "backend", "frontend", "scripts", "tests", "net", "conftest"}

declared: dict[str, Path] = {}
for rf in REQ_FILES:
    if not rf.exists():
        continue
    for line in rf.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        pkg = line.split("==")[0].split(">=")[0].split("[")[0].split("<")[0].strip()
        if pkg:
            declared[pkg.lower().replace("_", "-")] = rf

print(f"scanned {files} python files, {len(imported)} distinct top-level imports")
print(f"declared across {len([f for f in REQ_FILES if f.exists()])} requirements files: "
      f"{len(declared)} packages\n")

missing: list[tuple[str, str, str, int]] = []
unresolved: list[tuple[str, str]] = []
for mod, users in sorted(imported.items()):
    if mod in STDLIB or mod in OURS or mod.startswith("_"):
        continue
    dists = mod_to_dist.get(mod)
    if not dists:
        # Imported but not installed here either — a genuinely unknown module.
        unresolved.append((mod, sorted(users)[0]))
        continue
    dist = sorted(dists)[0]
    if dist.lower().replace("_", "-") not in declared:
        try:
            ver = metadata.version(dist)
        except Exception:
            ver = "?"
        missing.append((mod, dist, ver, len(users)))

if missing:
    print("MISSING from every requirements file — a fresh clone breaks on these")
    print(f"{'import':<22}{'pip package':<26}{'installed':<14}{'files using it':>15}")
    print("-" * 78)
    for mod, dist, ver, n in sorted(missing, key=lambda x: -x[3]):
        print(f"{mod:<22}{dist:<26}{ver:<14}{n:>15}")
else:
    print("nothing missing")

if unresolved:
    print("\nImported but not installed here either (check these by hand):")
    for mod, where in unresolved:
        print(f"  {mod:<22} first seen in {where}")

# Exit non-zero so this is usable as a gate, not just a report.
sys.exit(1 if missing else 0)
