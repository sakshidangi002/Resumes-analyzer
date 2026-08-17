# Skill: Bug Fixing

Instruction set for the `fix` node. This is the only node that writes.

## Root cause first

Fix the cause, not the symptom. Never make a test pass by weakening the
assertion, adding a `try/except` around the failure, or special-casing the input
the test happens to use. If the only way to make it pass is to change what the
test checks, the finding was misdiagnosed — send it back to triage.

## Smallest safe change

- One finding, one change.
- Touch only the lines the root cause requires.
- Preserve the surrounding architecture, naming and idiom. Code should read as
  though the original author wrote it.
- No opportunistic refactoring, no reformatting, no import reordering, no
  renaming while you are in the file.
- No new dependencies.

## Never modify

- `alembic/` migrations — revision ids and ordering are load-bearing.
- `.env`, `.env.example`, or anything holding credentials.
- `tests/conftest.py` — breaking it removes every test's ability to report.
- Files outside the declared scope, for any reason.
- Files with uncommitted modifications, unless explicitly allowed: a partial
  autonomous edit layered on someone's in-progress work is expensive to untangle.

## Stop and hand back to a human

Do not attempt a fix that would touch:

- authentication or authorisation logic;
- payroll or salary arithmetic;
- attendance status rules;
- database schema or migrations;
- concurrency, threading or locking;
- any public function signature with callers outside the scope.

Record it as `applied: false` with the reason. An unfixed, reported bug is a
good outcome. A silently wrong fix to payroll is not.

## Tests

If the fix changes behaviour, the change must include a test that fails before
and passes after. If no test can express it, say so rather than claiming
coverage.

## Record keeping

Every write records: file, line, strategy, and why. A file changed without a
recorded reason is indistinguishable from an accident.
