---
name: deployment
description: >-
  Deployment, environment, build, Windows server, and runtime troubleshooting
  for the unified Softwiz HRMS + AI Resume Analyzer FastAPI app. Activate
  whenever the user edits `.env`, `requirements*.txt`, or `pyproject.toml`,
  runs `uvicorn`, `python run_app.py`, `start.bat`, or `run-server.bat`, sets
  PYTHONPATH, runs `npm run build`, builds the deploy zip, edits files under
  `deploy/` or `scripts/`, encounters a Windows reserved-port bind error,
  `UnicodeDecodeError` on `.env` load, bcrypt-vs-passlib conflict, stale JS in
  production, or mentions deploy, deployment, build, package, zip, env, port,
  Windows, server, production, venv, virtualenv, PYTHONPATH, requirements,
  bcrypt, passlib, BOM, IPv4 excluded, or reserved port.
---

# Activation signals

Apply this skill when ANY of these are true:

- Editing `.env`, `.env.example`, `requirements*.txt`, `pyproject.toml`.
- Editing `run_app.py`, `start.bat`, `run-server.bat`, anything in `scripts/`, anything in `deploy/`.
- Running `npm run build`, `pip install`, `python -m venv`, `uvicorn`, `alembic upgrade`.
- User mentions: deploy, deployment, build, package, zip, env, port, Windows, server, production, virtualenv, PYTHONPATH, requirements, bcrypt, passlib, BOM, IPv4 excluded, reserved port.

# Architecture invariants (NEVER violate)

- ONE Windows server.
- ONE FastAPI process.
- ONE port.
- ONE shared domain.
- Everything runs through `Attendance Management/backend/app/main.py`.
- Resume API is mounted at `/resume-api` and `/resume/` on the SAME process.

If a change would create a second process, second port, or second deployable for the Resume Analyzer — STOP and ask the user.

# Single-port rule

Production must use ONE port only. Do NOT use:

- `8000` (commonly occupied)
- `8001` (often Hyper-V reserved on Windows)
- `8501` (Streamlit default — confusing)

Allowed examples: `5050`, `5555`, `9000`.

Check the Windows reserved range before choosing a port:

```powershell
netsh interface ipv4 show excludedportrange protocol=tcp
```

If the port lands in the excluded range, the FastAPI process binds but never receives traffic. Switch to an allowed port and retest.

# Environment files (TWO required)

| File | Path | Contains |
|---|---|---|
| HRMS backend env | `Attendance Management/backend/.env` | `POSTGRES_*`, `SECRET_KEY`, `SMTP_*` |
| Repo-root env | `<repo root>/.env` | `DATABASE_URL`, `CHAT_MODEL`, `IMAP_*`, `INDEED_*` |

Hard rules:

- NEVER commit either `.env`.
- Encoding MUST be UTF-8 **WITHOUT BOM**. A BOM causes `UnicodeDecodeError` at startup.
- `SECRET_KEY` MUST be identical across local + server. Changing it invalidates all existing JWTs.
- Missing root `.env` causes the Resume API to silently fall back to `localhost:5432` and fail on the server.

# Frontend build rules

`npm run build` is for PRODUCTION + DEPLOY only. For daily development see the `frontend` skill (`npm run dev` HMR at `localhost:5173`).

After ANY production frontend change:

```powershell
cd "Attendance Management/frontend"
npm run build
```

Build output: `Attendance Management/backend/frontend_build/`.

Without rebuild: production serves the OLD JS bundle and users see stale URLs / ports / bugs.

# Dependency pins (do not bump without reading)

```text
bcrypt>=4.0.1,<5
```

- HRMS uses **native bcrypt**, NOT `passlib`.
- Older bcrypt versions conflict with ChromaDB's transitive deps.
- Mixing bcrypt and passlib silently breaks password verification on some accounts.

# Local development startup

```powershell
python run_app.py
```

Equivalent manual command:

```powershell
cd "Attendance Management/backend"
$env:PYTHONPATH = "$PWD;..\.."
python -m uvicorn app.main:app --host 0.0.0.0 --port 5001
```

PYTHONPATH MUST include BOTH the HRMS backend folder and the repo root, so Resume API imports resolve.

# Production package build

```powershell
.\scripts\package-for-server.ps1 -Zip
```

The script must NOT bundle `.env`, `chromadb/` data, or `backend/uploads/`. Sanity-check: resulting zip < 50 MB.

# Server setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-all.txt
```

Copy BOTH `.env` files into place BEFORE first start.

Run:

```powershell
.\run-server.bat 5050
```

Port arg must match the allowed list.

# Common deployment pitfalls (always check this list)

- [ ] Forgot `npm run build` → old JS bundle in production
- [ ] Wrong PYTHONPATH → Resume API imports fail
- [ ] Missing root `.env` → Resume API hits `localhost:5432` on server
- [ ] UTF-8 BOM in `.env` → `UnicodeDecodeError` at startup
- [ ] Different `SECRET_KEY` → all existing JWTs invalidated, users logged out
- [ ] Blocked Windows port → process starts but accepts no traffic
- [ ] `.env` shipped inside deploy zip → secret leak
- [ ] Second FastAPI process / alternate port added → breaks unified architecture
- [ ] `bcrypt<4` installed → ChromaDB conflict
- [ ] `passlib` added back → password verification regressions

# Stop and ask the user when

- A change would alter the single-port single-process architecture.
- You'd need a new system dependency (Redis, Nginx in front, etc.).
- A bcrypt version change is necessary.
- `.env` keys must be added or renamed on a deployed server.

# Verification (run after every deploy)

- FastAPI starts and reports the chosen port in the uvicorn banner.
- `GET /health` → 200 (if implemented).
- `GET /` serves the HRMS landing / login.
- `GET /resume/` serves the Resume Analyzer HTML.
- `POST /api/auth/login` returns a valid JWT.
- `GET /resume-api/...` with Admin JWT returns data; with Employee JWT returns 403.
- PostgreSQL connects (no SQLAlchemy retries in uvicorn log).
- ChromaDB loads (no `OperationalError` on first `/resume-api` call).
- Frontend bundle hash in `frontend_build/assets/` matches the just-built one.

# Related skills

- Backend code changes → `backend` skill.
- UI code + `npm run dev` workflow → `frontend` skill.
- AI pipeline changes that affect deploy → `langchain` skill.
- Full smoke-test checklist → `testing` skill.
