---
name: testing
description: >-
  Smoke testing, verification, and QA checklist runner for Softwiz HRMS + AI
  Resume Analyzer. Activate when the user asks to verify, smoke-test,
  sanity-check, QA, validate, or confirm a change works — across JWT auth,
  role-based access (Admin / HR / Manager / Employee), React production build
  correctness, frontend / backend API integration, PostgreSQL connectivity,
  Alembic migration health, ChromaDB queries, resume parsing, embedding
  generation, candidate ranking, AI chat answers, IMAP / Indeed pipeline,
  `/resume/` and `/resume-api/` routing, and post-deploy server health. Use
  after any non-trivial backend, frontend, AI, or deployment change.
---

# Activation signals

Apply this skill when the user says: test, verify, check, smoke, sanity, QA, validate, confirm, ensure — OR immediately after a backend / frontend / AI / deployment change finishes.

This skill describes WHAT to verify. WHERE / HOW (which terminal, which URL) lives in the relevant peer skill.

# Always-on checklist (after ANY change)

- [ ] Browser console has no red errors on the affected page.
- [ ] Uvicorn log has no new ERROR or unexpected WARNING lines.
- [ ] You tested with the LOWEST role that should be allowed.
- [ ] You also tested ONE role below to confirm it gets blocked (RBAC works in both directions).

# Backend changes — verify

- API responds with the expected JSON shape (use the Pydantic schema as the contract).
- 200 path with a real JWT.
- 401 path with NO `Authorization` header.
- 403 path with the wrong role.
- DB state matches the change — manually check one row.
- Alembic migration: `alembic upgrade head` on a fresh local DB; no SQL errors.

# Frontend changes — verify

- The page renders at `localhost:5173/<route>` (dev HMR) AND at `localhost:5001/<route>` (production bundle).
- Forms validate; submit succeeds and shows the success state.
- Tables: `<th>` and `<td>` alignment match; numeric columns right-aligned with `tabular-nums`; action columns centered.
- Responsive: shrink to ~600 px width; `hide-md` / `hide-sm` columns collapse correctly.
- Vite proxy works in dev: Network tab shows `/api/...` returning 200.
- After production change: `cd "Attendance Management/frontend" ; npm run build` ran clean; browser at `localhost:5001` shows the change.

# AI / Resume Analyzer — verify

- Upload one PDF AND one DOCX; both parse without errors.
- Extracted skills include known primary skills from the source resume.
- Embedding generated (ChromaDB count + 1).
- Semantic search returns the just-uploaded candidate when queried by their unique skill.
- AI Chat about that candidate cites their actual experience.
- Indeed pipeline (if changed): a fresh search yields downloadable resumes that then ingest.
- Admin sees Resume Analyzer; Employee receives 403 on `/resume-api/*`.

# Deployment / runtime — verify

- `.env` loaded (no `UnicodeDecodeError`, no missing-variable warning at startup).
- PYTHONPATH includes BOTH the HRMS backend and the repo root.
- Unified routing: `/`, `/api/auth/login`, `/resume/`, `/resume-api/...` all respond.
- Server is listening on the configured port — `netstat -an | findstr :<port>` shows `LISTENING`.
- Production frontend bundle hash matches the latest `npm run build`.
- ChromaDB persistent directory is writable by the user running the service.

# When verification fails

- DO NOT silently retry. Capture the exact error (uvicorn log line OR browser console line OR Network tab response) and report it with the most likely cause from the relevant peer skill's pitfall list.

# Related skills

- Reviewing backend code → `backend` skill.
- Reviewing frontend code → `frontend` skill.
- Reviewing AI / vector code → `langchain` skill.
- Reviewing server / build → `deployment` skill.
