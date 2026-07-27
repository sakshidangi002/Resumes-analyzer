---
name: backend
description: >-
  FastAPI + SQLAlchemy + PostgreSQL + Alembic + JWT (bcrypt) backend work for
  Softwiz HRMS and the Resume Analyzer. Activate whenever the user edits Python
  code under `Attendance Management/backend/app/` or repo-root `backend/`, adds
  or changes an API endpoint or Pydantic schema, modifies a SQLAlchemy model or
  Alembic migration, debugs JWT 401/403, configures role-based access (Admin /
  HR / Manager / Employee), touches the unified `app/main.py` mount of `/api`,
  `/resume-api`, or `/resume/`, writes uvicorn / startup code, or mentions
  FastAPI, dependency injection, CORS, bcrypt, postgres, sqlalchemy session,
  alembic revision, RBAC, JWT decode, or the `/api/auth/login` route.
---

# Activation signals

Apply this skill when ANY of these are true:

- Editing files under `Attendance Management/backend/app/` (HRMS backend).
- Editing files under repo-root `backend/` that import FastAPI / SQLAlchemy / Pydantic.
- File extension is `.py` AND the imports include `fastapi`, `sqlalchemy`, `pydantic`, `alembic`, `jose`, or `bcrypt`.
- User mentions: endpoint, route, handler, model, schema, migration, JWT, auth, role, RBAC, Admin/HR/Manager/Employee, `/api`, `/resume-api`, uvicorn, PYTHONPATH, alembic, bcrypt, postgres, sqlalchemy session, dependency, depends, CORS.

Skip this skill when working purely on the React UI — use `frontend` instead. For AI / ChromaDB / embedding code in `backend/main.py` or `backend/extraction_v3.py`, use `langchain` instead.

# Read these before changing anything

1. `Attendance Management/backend/app/main.py` — unified mount points and middleware order. NEVER restructure `/api`, `/resume-api`, `/resume/` mounts.
2. The existing route in the same router file you intend to edit — match its dependency-injection pattern, response_model, and status_code.
3. Any related SQLAlchemy model in `app/models/` — reuse, do not duplicate.

# Hard rules

- ONE FastAPI process, ONE port in production. Do NOT introduce additional ASGI apps, alternate ports, or a separate Resume server.
- NEVER bypass JWT: every protected endpoint MUST depend on the existing auth dependency. Resume Analyzer endpoints (`/resume-api/*`) are Admin + HR only — preserve that.
- NEVER use `passlib`. HRMS uses **native `bcrypt>=4.0.1,<5`**. Mixing the two breaks ChromaDB's transitive deps.
- NEVER write a destructive Alembic migration (column drop / rename / type narrowing) without an explicit user confirmation step.
- ALWAYS reuse existing service functions, models, and Pydantic schemas. Grep before writing new ones.
- Public endpoints MUST declare a Pydantic `response_model` — no raw dicts to clients.
- DB queries live in service functions, NOT in route handlers.

# Workflow: adding a new endpoint

1. Find the matching router file (`app/api/<area>.py` for HRMS, `backend/api.py` for Resume).
2. Read 2 nearby routes to mirror style.
3. Add the route with: `response_model=...`, `status_code=...`, `Depends(get_current_user)`, role check if needed.
4. Add the Pydantic request + response schemas in the schemas module.
5. Move DB work into a new or existing service function under `app/services/`.
6. Verify with: uvicorn auto-reload + a real JWT against the endpoint (200 path), a missing-token call (401), and a wrong-role call (403).

# Workflow: schema / model change

1. Edit the SQLAlchemy model in `app/models/`.
2. From `Attendance Management/backend/`: `alembic revision --autogenerate -m "describe change"`.
3. REVIEW the generated migration — autogenerate sometimes drops indexes / unique constraints. Patch before applying.
4. `alembic upgrade head` on a fresh local DB to confirm idempotency.
5. Update the matching Pydantic schema and any service code that touches the column.

# Workflow: debugging JWT / role failures

1. Decode the token at jwt.io and confirm the `sub`, `roles`, and `exp` claims.
2. Confirm the route's `Depends()` chain includes the role check (look at how the working route nearby does it).
3. Confirm the frontend Axios client sends `Authorization: Bearer <token>` — see `src/api/client.ts` request interceptor.
4. Check `SECRET_KEY` is the same on the server that issued vs. the server that's verifying.

# Stop and ask the user when

- A migration would drop or rename a column, or change a primary key.
- A change would alter the `/api`, `/resume-api`, or `/resume/` URL surface.
- You'd need a second port, second process, or second uvicorn app.
- Anything would invalidate existing JWTs (`SECRET_KEY` / algorithm change).
- Adding a new heavyweight dependency (celery, redis client, etc.).

# Verification (run before declaring done)

- Hit the endpoint with an Admin JWT (expect 200) AND with the lowest-privileged role that should access it.
- Hit the endpoint with no token (expect 401) AND with a forbidden role (expect 403).
- Tail uvicorn log — no new ERROR / WARNING lines.
- If models changed: `alembic upgrade head` ran clean on a fresh local DB.
- If routing changed: `GET /api/openapi.json` includes the new path.

# Related skills

- Build, .env, Windows port issues → `deployment` skill.
- AI / ChromaDB / embeddings / Indeed pipeline in `backend/main.py` or `backend/extraction_v3.py` → `langchain` skill.
- Final QA checklist → `testing` skill.
