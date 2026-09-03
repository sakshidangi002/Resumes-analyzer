"""Test bootstrap.

Puts the HRMS backend on sys.path and supplies the settings the app refuses to
boot without, so these tests run on a bare CI machine with no .env and no
database. Nothing here connects to PostgreSQL — every test in this directory is
a pure unit test by design, because the bugs they cover were all pure-logic
bugs that a running database would not have exposed.
"""
import os
import sys
import tempfile
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

# Keep the test suite OUT of the production log.
#
# app/main.py writes to $HRMS_LOG_DIR/hrms.log, defaulting to the repo's own
# logs/ directory — the same file the live cameras write to. Running pytest
# therefore injected fabricated attendance traffic into production diagnostics:
# writes for `emp=7 camera=53` (a camera that does not exist in this
# deployment) alongside genuine-looking ERRORs — "ATTN-WRITE LOST ... attendance
# NOT recorded", "queue overflow ... the database is not keeping up",
# "RuntimeError: this camera explodes". Every one of those is a fixture.
#
# That is not cosmetic. Diagnosing why a real employee was not marked means
# grepping this file, and it was salted with alarming failures that never
# happened, at timestamps that interleave with real camera activity.
_TEST_LOG_DIR = Path(tempfile.gettempdir()) / "hrms-test-logs"
_TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HRMS_LOG_DIR"] = str(_TEST_LOG_DIR)
