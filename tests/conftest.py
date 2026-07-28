"""Test bootstrap.

Puts the HRMS backend on sys.path and supplies the settings the app refuses to
boot without, so these tests run on a bare CI machine with no .env and no
database. Nothing here connects to PostgreSQL — every test in this directory is
a pure unit test by design, because the bugs they cover were all pure-logic
bugs that a running database would not have exposed.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HRMS_BACKEND = REPO_ROOT / "Attendance Management" / "backend"

for path in (REPO_ROOT, HRMS_BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Settings has no default for secret_key (deliberately — a missing key must fail
# fast rather than silently sign tokens with a placeholder), so supply one.
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-unit-tests-only-not-real-0123456789")
os.environ.setdefault("POSTGRES_PASSWORD", "unused-in-unit-tests")
