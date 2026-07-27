# Code Review & Refactoring Playbook

Use this file when reviewing or improving an existing FastAPI backend. Read it before generating a "findings report" so categorization is consistent.

## Review workflow

1. **Read three files first**, in this order:
   - `app/main.py` (or wherever the FastAPI app is) — middleware, mounting, lifespan
   - One representative router
   - One representative model + its CRUD/service
   This gives you the project's actual layering, not the README's claims.

2. **Run the checklists below** and collect findings.

3. **Categorize each finding** as:
   - **🔴 Critical** — security hole, data corruption risk, broken in prod (fix before next deploy).
   - **🟡 Important** — performance, scalability, correctness drift (fix this sprint).
   - **🟢 Nice-to-have** — readability, naming, mild duplication (backlog).

4. **Report in this format:**

   ```markdown
   ## Findings — <project area>

   ### 🔴 Critical
   1. **<short title>** — `path/to/file.py:42`
      - **What**: <one-line description>
      - **Why it matters**: <impact>
      - **Fix**: <minimal change>

   ### 🟡 Important
   ...

   ### 🟢 Nice-to-have
   ...

   ## Suggested order of attack
   1. <highest leverage critical>
   2. ...
   ```

5. **Offer to apply fixes incrementally.** One category at a time. Never silently rewrite a whole file.

## Critical-level checklist

### Security
- [ ] No secrets in source (`SECRET_KEY`, DB URL, SMTP password).
- [ ] All protected endpoints have `Depends(get_current_user)` or stronger.
- [ ] No `allow_origins=["*"]` with `allow_credentials=True`.
- [ ] No raw SQL with f-string / `+` interpolation of user input.
- [ ] Password storage uses bcrypt/argon2; NEVER SHA-256 / MD5 / plain.
- [ ] No PII / password hash leaked in `response_model`.
- [ ] File uploads validate MIME + extension + size + magic bytes; UUID-prefixed names.
- [ ] No `eval` / `exec` / `pickle.loads` on untrusted data.
- [ ] JWT validates `type` (access vs refresh) on every protected endpoint.
- [ ] Login error message identical for "no user" and "wrong password" (no enumeration).

### Data integrity
- [ ] `session.commit()` happens at service boundary, not scattered in CRUD.
- [ ] `get_session` rolls back on exception.
- [ ] No long-running transactions wrapping external HTTP calls.
- [ ] Foreign keys declared on models; cascade behavior explicit.
- [ ] Unique constraints on natural keys.
- [ ] Migrations reviewed; no silent column drops in autogen output.

### Correctness
- [ ] Async/sync not mixed (no sync `Session` inside `async def` handlers).
- [ ] Routes that take `{id}` actually look up the resource (no implicit trust on path).
- [ ] Resource-owner checks (employee accessing only their own payslip).

## Important-level checklist

### Architecture
- [ ] Business logic lives in services, not routers.
- [ ] DB queries live in CRUD or services, not routers.
- [ ] One responsibility per file (no 800-line god routers).
- [ ] Schemas separated into `Create` / `Update` / `Read`.
- [ ] Custom exceptions used, not raw `HTTPException` everywhere.
- [ ] Settings via `pydantic-settings`, not `os.environ` scattered.

### Performance
- [ ] No N+1 — every relationship accessed in a loop has `selectinload` / `joinedload`.
- [ ] Indexes on FK columns and frequently-filtered columns.
- [ ] Pagination on every list endpoint that can return >50 rows.
- [ ] `COUNT(*)` on large tables is avoided or cached.
- [ ] `pool_pre_ping=True` and `pool_recycle` set.
- [ ] Heavy compute (PDF parsing, ML inference) is not blocking the event loop — use `run_in_threadpool` or a worker.

### Reliability
- [ ] `lifespan` is used (not deprecated `@app.on_event`).
- [ ] `/healthz` or `/health` endpoint exists and checks DB.
- [ ] Structured logging configured; no `print` calls.
- [ ] Exception handlers return JSON, not HTML.
- [ ] CORS list is explicit per environment.

### Testing
- [ ] `conftest.py` provides isolated test DB + per-test rollback.
- [ ] Auth tests cover 401/403 for every protected endpoint family.
- [ ] Services have unit tests for business rules.
- [ ] CI runs lint + tests + migrations.

## Nice-to-have checklist

- [ ] Consistent naming: `snake_case` everywhere, `Pascalcase` for classes, `UPPER_SNAKE` for constants.
- [ ] Type hints on every public function signature.
- [ ] No commented-out code (delete it; git remembers).
- [ ] Imports grouped: stdlib / third-party / first-party (ruff handles this).
- [ ] Routers use `tags=["..."]` for cleaner OpenAPI docs.
- [ ] `response_model_exclude_none=True` where partial responses are common.
- [ ] API surface versioned under `/api/v1`.
- [ ] OpenAPI docs disabled or auth-gated in production.

## Common refactors (and how to phase them)

### "God router" → layered

**Symptom:** one router file with 600+ lines doing DB queries, business logic, validation, and email sending.

**Refactor steps:**

1. Identify the 3-5 distinct operations. Each becomes a service function.
2. Move DB queries into a per-model CRUD file.
3. Move business rules into a service file.
4. Router endpoints now: parse → call service → return.
5. **Run tests after each step**, not after the whole refactor.

### Sync codebase → async (when it's worth it)

**Don't do this if:** you have < 50 endpoints and CPU is the bottleneck. Async pays off for I/O-bound workloads with high concurrency.

**Refactor steps:**

1. Add async engine + `get_async_session` alongside the existing sync session.
2. Pick ONE router (smallest), convert it. Switch its dependency to `get_async_session`.
3. Convert its service + CRUD. Run tests.
4. Repeat per router. Don't try to convert all at once.
5. When all converted, remove sync session.

### `passlib` → native `bcrypt`

If you see version pin pain (`bcrypt==4.0.1` only) or `__about__` errors:

1. Add `bcrypt>=4.1` to requirements; remove `passlib`.
2. Replace `pwd_context.hash(...)` → `bcrypt.hashpw(...)`.
3. Replace `pwd_context.verify(...)` → `bcrypt.checkpw(...)`.
4. Existing hashes are compatible — bcrypt hashes look the same.

### Tests that hit dev DB → isolated test DB

1. Create `test_app` Postgres DB (or in CI service).
2. Add `TEST_DATABASE_URL` env var.
3. Replace direct session use in tests with the per-test rollback fixture.
4. Delete any `session.execute("TRUNCATE ...")` cleanup hacks.

### Settings via `os.environ` → `pydantic-settings`

```python
# Before
import os
DB_URL = os.environ["DATABASE_URL"]
SECRET = os.getenv("SECRET", "changeme")  # BAD: default secret
```

```python
# After
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str   # no default — fails fast at startup if missing
```

The Pydantic version fails at startup if `SECRET_KEY` is missing, instead of silently using a weak default.

## Red flags to call out by name

When you see these, name them explicitly in the review — they have well-known names:

- **N+1 query problem** — fix with eager loading.
- **God object / god router** — split by responsibility.
- **Anemic domain model** — service has all logic, models are bags of fields. Often fine for backends, but call it out if logic could live closer to the data.
- **Stringly-typed status** — `status: str` instead of `Enum`.
- **Magic numbers** — `if days > 24:` → `MAX_LEAVE_DAYS_PER_YEAR = 24`.
- **Catching everything** — `except Exception:` without re-raise or log.
- **Long parameter list** — > 5 positional params → take a dataclass / Pydantic model.
- **Premature abstraction** — generic factory wrapping 2 concrete things. Inline until you have 3+ uses.
- **Inappropriate intimacy** — router reaches into another module's CRUD bypassing the service.

## Output style for review reports

- One sentence per finding in the bullet — link to the file, then the fix.
- Don't show 100 lines of "after" code. Show the diff or the key 5 lines.
- Group by category, then by file (not the other way around).
- End with a "Suggested order of attack" — your top 3 picks, with effort estimate.
- If a finding is on a hot path or in a critical area, label it. The user reads top-down.

## What to do AFTER the review

- Offer to **apply Critical fixes immediately**.
- Offer to **draft a refactor plan** for Important findings (split into PRs).
- Offer to **add tests** for any untested critical path you noticed.
- Update the project's `AGENTS.md` / `.cursor/rules/` if you found a recurring anti-pattern worth a project rule.
