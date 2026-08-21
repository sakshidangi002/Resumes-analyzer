---
description: Do the task through graph_engine (review → triage → fix → test → verify), not by hand
argument-hint: <goal> [--scope PATH ...] [--all] [--apply] [--budget N]
allowed-tools: Bash(.venv/Scripts/python.exe -m graph_engine:*), Bash(git diff:*), Bash(git status:*), Read, Grep, Glob
---

Run the engineering workflow graph for this goal. **The graph does the work — you
orchestrate and report it.**

## Arguments

`$ARGUMENTS`

Parse them as:

- everything before the first `--` flag is the **goal** (quote it when passing on)
- `--scope PATH` — repeatable. Defaults to `--all` when no scope is given.
- `--all` — whole application (203 files: HRM backend, Resume Analyzer,
  graph_engine, services, scripts, tests)
- `--apply` — pass `--apply-fixes` to the engine. **Without it the run is
  read-only**, which is the correct default.
- `--budget N` — pass as `--fix-budget N` (engine default 3; use 40–80 for a
  whole-application sweep)

## What to run

From the repo root, using the project virtualenv:

```
.venv/Scripts/python.exe -m graph_engine --goal "<goal>" <scopes> [--apply-fixes] [--fix-budget N] [--fixes-per-pass 20] [--timeout 1800]
```

Add `--fixes-per-pass 20` whenever `--apply` is used with `--all`, or the run
needs one test cycle per four fixes and takes far longer than it should.

## Rules — these are the point of the command

1. **Do not hand-edit application code to satisfy the goal.** If the engine
   declines a fix, report that it declined and why. Do not "help" by making the
   change yourself; the whole value of this command is that every change went
   through the sandbox, the backup, and the test gate.
2. **Do not claim a result the engine did not produce.** Quote its numbers:
   files reviewed, findings, bugs, dismissed, fixes applied, test counts,
   verification criteria.
3. **Report the caveats.** `bug(s) need a human`, `never attempted because the
   fix budget was spent`, and any pre-existing failure are part of the result,
   not noise to trim.
4. **If the engine errors** (`no_engine_errors` fails, or a node raises), that
   is a defect in `graph_engine/`, not in the target code. Diagnose and fix the
   engine — that part *is* hand work, because the engine has only mechanical fix
   strategies and cannot repair its own logic.
5. **Never pass `--allow-dirty` unless the user asks for it.** Files with
   uncommitted modifications are refused on purpose.

## After the run

- If files changed: run `git diff --stat` on the changed paths and show the
  user, then confirm the app still imports:
  `.venv/Scripts/python.exe -c "import sys,os; sys.path.insert(0,'Attendance Management/backend'); os.environ.setdefault('SECRET_KEY','x'*40); os.environ.setdefault('POSTGRES_PASSWORD','x'); from app.api.routes import api_router; print(len(api_router.routes),'routes OK')"`
- Tell the user where the backups are: `.graph_engine_runs/<run_id>/backup/`
- Give them the one-line revert for the changed paths.
- Do **not** commit unless asked.

## What the engine cannot do

It has two fix strategies (`ruff_fix`, `none_comparison`). It cannot write new
code, add tests, refactor, or repair logic errors, and it never auto-fixes
payroll, auth, `deps.py`, `attendance_service`, `leave_service`, `alembic/`,
`.env` or `conftest.py`.

If the goal needs any of that, say so plainly before running: the engine will
review and report, but the fixing is not something it can do. Do not silently
substitute your own edits for the graph's.
