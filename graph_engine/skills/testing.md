# Skill: Testing

Instruction set for the `test` node.

## Non-negotiable

**Test results come from running tests.** Never infer, assume or reason that
tests pass. The exit code comes from pytest; the failure list is parsed from its
output. An LLM has no vote here.

## This repository

- pytest, `testpaths = ["tests"]` in `pyproject.toml`.
- `tests/conftest.py` puts the repo root and `Attendance Management/backend` on
  `sys.path` and supplies `SECRET_KEY` / `POSTGRES_PASSWORD`.
- Every test is a pure unit test. **No test touches PostgreSQL, the network, a
  camera or a model.** A test that needs any of those is a design error.
- Run with the project virtualenv (`.venv`), not the system interpreter —
  pytest and the dependencies live there.

## Test selection, in order

1. **Targeted** — the test files that cover the changed modules. Fast; run first.
2. **Regression** — the full suite. Run once the targeted tests are green, to
   catch damage outside the changed area.

Skipping the regression pass makes a run faster and its conclusion weaker. If it
is skipped, say so in the report; do not present a targeted pass as a clean run.

## Baseline

Capture failures **before** any change. This repository currently has a
pre-existing failure (`test_seat_anchoring_is_enabled_by_default`, caused by an
uncommitted `config.py` edit). A pre-existing failure must never be attributed to
the run's changes, and must never be "fixed" as though it were.

## Reporting

Capture and report: command, exit code, pass/fail/skip counts, per-failure node
id and message, duration, and whether the run timed out.

A timeout is not a pass and not a failure — it is a timeout. Report it as such.

## Rules

- Never pass unvalidated strings into the pytest argv. A target beginning with
  `-` is a flag, not a test.
- Never run a test target outside the repository.
- Never delete or disable a failing test to make a run green.
