# Graph Engineering — executable engineering workflow

An orchestration layer that runs a software-engineering workflow **against this
repository**. It is a developer tool, not part of the served HRM application: it
imports no HRM business logic, and the application does not import it.

```
review -> bug analysis -> fix -> test -> verify
                           ^              |
                           |              v
                    failure analysis <- (tests failed)
```

## Using it from Claude Code

`/graph <goal> [--scope PATH ...] [--all] [--apply] [--budget N]`

Defined in [.claude/commands/graph.md](../.claude/commands/graph.md). The command
exists to enforce a rule: **the graph does the work, Claude orchestrates and
reports it.** It forbids hand-editing application code to satisfy the goal, and
requires the reported numbers to be the engine's own.

```
/graph find correctness defects --all
/graph clean up unused imports in the HRM routes --scope "Attendance Management/backend/app/api/routes" --apply
/graph fix everything mechanically safe across the app --all --apply --budget 80
```

Read-only unless `--apply` is given.

## Quick start (plain CLI)

```bash
# Whole application, read-only. 203 files in ~2s.
python -m graph_engine --goal "Find correctness defects" --all

# One module.
python -m graph_engine --goal "Find correctness defects" --scope "backend"

# Several modules: repeat --scope.
python -m graph_engine --goal "Audit the HRM services and routes" \
  --scope "Attendance Management/backend/app/services" \
  --scope "Attendance Management/backend/app/api"

# Apply mechanical fixes, run the tests, verify the goal.
python -m graph_engine \
  --goal "Remove dead code from the assistant package without breaking its tests" \
  --scope "Attendance Management/backend/app/assistant" \
  --apply-fixes
```

`--all` expands to `ALL_SCOPES` in [config.py](config.py): the HRM backend, the
Resume Analyzer backend, this engine, `services`, `scripts` and `tests`.
Overlapping scopes are de-duplicated, so a file is never reviewed twice.

**Dry runs do not loop.** With `--apply-fixes` off there is nothing left to
attempt after one pass, so the drain loop is skipped and every actionable bug
gets its decline recorded in a single pass — 6 nodes instead of 18.

Exit codes: `0` goal achieved · `1` ran but did not achieve it · `2` stopped on a
precondition or safe stopping condition.

Useful flags: `--fix-budget N` · `--max-iterations N` · `--allow-dirty`
`--skip-regression` · `--json` · `-v`

## Why not LangGraph

The HRM backend and the Resume Analyzer share one virtualenv, and
`requirements.txt` pins `langchain-core==0.1.52` for the NuExtract extraction
chain. Current LangGraph needs `>=0.2.43`, so installing it would break resume
extraction.

Instead this reuses the runtime already proven in
`Attendance Management/backend/app/assistant/runtime.py` — nodes, conditional
edges, parallel fan-out, tracing, cycle bounding, deadline — which has its own
test suite. The coupling is confined to `_runtime.py`, so vendoring it later
means changing one file.

## Layout

| Path | Role |
|---|---|
| `graph.py` | Assembly + `run_engine()`, the only public entry point |
| `state.py` | `EngineState` TypedDict + stop reasons. **No file contents ever** |
| `edges.py` | Routing predicates — pure functions of state |
| `config.py` | Budgets, rule selection, excluded and protected paths |
| `context.py` | `EngineContext`: config, repo, test runner, optional LLM, counters |
| `nodes/` | `preflight`, `review`, `bug_analysis`, `fix`, `testing`, `failure_analysis`, `verify`, `report` |
| `tools/` | `repository` (sandboxed IO), `linters` (ruff), `ast_checks`, `test_runner`, `git_tools` |
| `skills/` | Agent instruction markdown, one per node |
| `_runtime.py` | Adapter to the borrowed `StateGraph` |

Tests live in the repo's `tests/` (`test_engine_*.py`), following
`testpaths = ["tests"]` rather than the nested layout.

## Nodes

| Node | Does | Writes? |
|---|---|---|
| `preflight` | Records branch + dirty files; captures the **test baseline** so pre-existing failures are never blamed on the run | no |
| `review` | ruff (`E9,F,B` only) + AST checks; reports facts with severity | no |
| `bug_analysis` | Triage: real vs false positive, root cause, priority, `auto_fixable` | no |
| `fix` | Applies mechanical strategies through the sandbox | **yes** |
| `testing` | Real pytest subprocess: targeted, then full regression | no |
| `failure_analysis` | Classifies the actual failure and picks the next action | no |
| `verify` | Evidence-backed acceptance criteria — not "pytest exit 0" | no |
| `report` | Single terminal node; sets status + `stop_reason` | no |

## Deterministic vs LLM

**Deterministic (everything that matters):** routing, ruff + AST review, triage
rules, fix strategies, test execution and parsing, failure classification,
verification criteria, iteration/budget accounting, git inspection, file IO.

**LLM-optional:** `ctx.llm` may be supplied for review reasoning, root-cause
analysis and verification judgement. It defaults to `None`, and every run above
completed with `llm_calls: 0`. The only chat model in this repo is a local
TinyLlama, which is not capable of code review; plugging in a capable model is a
config change, not a redesign.

## Loops and stopping conditions

Two cycles, both bounded:

* **repair** — `fix -> test -> failure_analysis -> fix`
* **drain** — `verify -> bug_analysis -> fix -> ... -> verify`, because `fix`
  deliberately handles at most `FIXES_PER_ITERATION` bugs per pass

Stops: `max_iterations` (5) · `fix_budget` (3) · **repeated failure signature** ·
whole-run timeout (900s) · single-test timeout (300s) · runtime `max_steps` (60).

The signature check is the important one — a budget alone still lets a
deterministic failure consume every attempt.

## Safety

**Filesystem.** Every path is resolved and required to be inside the repo
(blocking `../`), rejected if in an excluded tree (`.venv`, `node_modules`,
`third_party`, …), and for writes additionally required to be inside the run
scope, to be `.py`, and not to match a protected path (`alembic/`, `.env`,
`conftest.py`).

**Git.** Read-only, enforced by an allowlist — `status`, `diff`, `rev-parse`.
`checkout`, `reset`, `stash`, `commit`, `push` raise. This matters concretely:
the working tree carries dozens of uncommitted files, so rollback uses byte-exact
**file backups** under `.graph_engine_runs/<run_id>/backup/`, never git.

**Writes.** Off unless `--apply-fixes`. Files with uncommitted modifications are
refused unless `--allow-dirty`. Every write is backed up first, and a patch that
does not parse is rejected before the test node ever sees it.

**Sensitive areas** are never auto-fixed regardless of how mechanical the change
looks: payroll, salary, `security.py`, auth, `deps.py`, `alembic`,
`attendance_service`, `leave_service`, encryption, PII.

**Shell.** No general shell access. Two allowlisted subprocesses only: the
project interpreter running `-m pytest` with validated targets, and `-m ruff`.

**Secrets.** No credential ever enters a prompt; `.env` is unreadable through the
sandbox; state and traces carry paths, counts and decisions — never source.

## Skills

`skills/*.md` are **agent instruction text**, not the graph. Each becomes the
system prompt when a node delegates judgement to an LLM, and documents the
contract the deterministic path implements. Each node records which skill
governed it in its trace. No routing decision is ever read from markdown.

## Human approval boundary

Today the boundary is expressed by refusal: `auto_fixable: false` for sensitive
areas, protected paths, and anything without a registered mechanical strategy.
Those become a reported bug plus a caveat rather than an edit. Adding an approval
gate later means one conditional edge before `fix` — the classification it would
consult already exists.

## Known limitations

1. **Test selection is a filename-keyword heuristic**, not an import graph. It
   falls back to the full suite rather than silently under-testing.
2. **Two fix strategies only** (`ruff_fix`, `none_comparison`). Everything else
   is reported for a human.
3. **No LLM wired up**, so review depth is whatever ruff + the AST checks find.
4. **No worktree isolation.** Runs edit the working tree directly, protected by
   backups and the dirty-file refusal.
5. `verify` checks that every finding was *accounted for*, not that the codebase
   is now defect-free — an honest limit of static analysis.
