# Resume Analyzer – Project Structure Review

## 1. Current Folder Structure (Summary)

```
Resume analyzer/
├── .venv/                          # Virtual environment (keep, add to .gitignore)
├── .env                            # Secrets (keep, never commit)
├── backend/
│   ├── __init__.py
│   ├── api.py                      # Main FastAPI app (entry: uvicorn backend.api:app)
│   ├── main.py                     # LLM extraction, chat, embeddings, rank
│   ├── a.py                        # OLD duplicate API – not used
│   ├── chromadb/                   # Chroma vector DB files (runtime)
│   ├── uploads/                    # Stored PDF/DOCX (runtime)
│   ├── resume.db                   # Leftover SQLite (project uses PostgreSQL)
│   ├── temp_*.pdf                  # Stale temp files
│   └── __pycache__/
├── chromadb/                       # Duplicate chroma at root? (check usage)
├── frontend/
│   └── app.py                      # Streamlit UI
├── check_database.py               # One-off DB query script
├── db_query.py                     # One-off DB query script
├── delete_db.py                    # Chroma SQLite delete (niche)
├── final_clean.py                  # Chroma cleanup script
├── force_clean.py                  # Chroma cleanup script (overlaps final_clean)
├── test_extraction.py              # Manual upload test
├── upload_test.py                  # Minimal upload test (duplicate purpose)
├── README.md
└── requirements.txt
```

---

## 2. Files That Can Be Safely Removed

| File | Reason |
|------|--------|
| **backend/a.py** | Duplicate/legacy FastAPI app. Entry point is `backend.api:app`. Same logic as api.py but older (different DB URL, no DOCX, etc.). Safe to delete. |
| **backend/resume.db** | Project uses PostgreSQL. This SQLite file is leftover. Safe to delete (ensure PostgreSQL is your source of truth). |
| **backend/temp_*.pdf** | Stale temp files from uploads. Add `temp_*.pdf` to .gitignore and delete existing. |
| **upload_test.py** | Minimal duplicate of test_extraction.py. Keep one test script (e.g. test_extraction.py) or move to tests/. |
| **final_clean.py** **and** **force_clean.py** | Both remove chromadb artifacts. Keep one (e.g. force_clean.py), delete the other, or merge into a single `scripts/clean_chroma.py`. |
| **db_query.py** | One-off ad-hoc query (e.g. by email). Either remove or move to `scripts/` as optional utility. |
| **check_database.py** | One-off “recent resumes” query. Same: remove or move to `scripts/`. |

**Optional (consolidate later):**

- **test_extraction.py** – Keep for manual API testing, or move to `tests/test_upload.py` and run with pytest.
- **delete_db.py** – Very specific (Chroma SQLite). Can move to `scripts/delete_chroma_db.py` if you still need it.

---

## 3. Cleaned and Improved Folder Structure

Target layout (minimal change, good practices):

```
Resume analyzer/
├── .env
├── .gitignore                      # Add: .venv, .env, backend/uploads/, backend/chromadb/, temp_*.pdf, *.db
├── README.md
├── requirements.txt
│
├── backend/
│   ├── __init__.py
│   ├── api.py                      # FastAPI app + routes (keep as single app for simplicity)
│   ├── main.py                     # LLM + extraction + embeddings (or split later into services/)
│   ├── config.py                   # [NEW] Settings from env (DATABASE_URL, UPLOAD_DIR, etc.)
│   ├── models.py                   # [OPTIONAL] Move ResumeDB, Base here from api.py
│   ├── schemas.py                  # [OPTIONAL] Move Pydantic schemas from api.py
│   ├── uploads/                    # Runtime uploads (gitignore)
│   └── chromadb/                   # Runtime vector store (gitignore)
│
├── frontend/
│   └── app.py
│
├── scripts/                        # [NEW] One-off and maintenance scripts
│   ├── check_db.py                 # Renamed from check_database.py
│   ├── clean_chroma.py             # Single script (merge final_clean + force_clean)
│   └── delete_chroma_db.py         # Optional, from delete_db.py
│
└── tests/                          # [OPTIONAL] For pytest later
    └── test_upload.py              # From test_extraction.py
```

**Remove from repo (or move as above):**

- `backend/a.py`
- `backend/resume.db`
- `backend/temp_*.pdf`
- `upload_test.py`
- `final_clean.py` or `force_clean.py` (keep one, merge into scripts/clean_chroma.py)
- `db_query.py` (or move to scripts/)
- `check_database.py` (move to scripts/check_db.py)
- `delete_db.py` (move to scripts/ if needed)
- Root-level `chromadb/` if it’s unused (Chroma should live under backend/)

---

## 4. Better Placement (Routers, Services, Models, Utils)

**Keep it simple for now:**

- **FastAPI app**: Keep a single `backend/api.py` with all routes. No need for multiple routers until the file grows much larger (~800+ lines).
- **Models**: Optionally move `ResumeDB` and `Base` to `backend/models.py` and import in api.py. Reduces api.py size and separates DB from HTTP.
- **Schemas**: Optionally move Pydantic models to `backend/schemas.py`. Same benefit.
- **Services**: Optional next step: create `backend/services/llm.py` (extract_resume, chatbot_answer, rank_candidates) and `backend/services/vector_store.py` (Chroma add/query). Then api.py only does HTTP and calls services. Do this when main.py becomes hard to maintain.
- **Utils**: Helpers like `_sanitize_filename`, `_safe_get`, `_apply_skill_filter` can stay in api.py or move to `backend/utils.py` when you add more helpers.
- **Config**: Add `backend/config.py` with a single `Settings` (e.g. pydantic-settings) for `DATABASE_URL`, `UPLOAD_DIR`, Chroma path. Use it in api.py and main.py instead of hardcoding.

**Recommended order:**

1. Remove duplicate/unused files and add .gitignore.
2. Add scripts/ and move/merge cleanup and DB-check scripts.
3. Add config.py and use it in api.py.
4. Optionally split models.py and schemas.py.
5. Optionally split services/ when main.py grows.

---

## 5. Practices for FastAPI + Streamlit

- **Backend**: Single FastAPI app in `backend/api.py`; run with `uvicorn backend.api:app --reload`. Use Depends(get_db) for DB; keep Chroma/embedding init in app state or a single module.
- **Frontend**: Single Streamlit app in `frontend/app.py`; run with `streamlit run frontend/app.py`. All server calls go to `API_URL` (e.g. http://127.0.0.1:8000).
- **Secrets**: Use `.env` and load in backend (e.g. config.py). Do not commit .env.
- **Ignore**: `.venv`, `.env`, `backend/uploads/`, `backend/chromadb/`, `**/__pycache__/`, `*.pyc`, `temp_*.pdf`, `*.db`.
- **Single source of truth**: PostgreSQL for resumes; Chroma for vectors. No SQLite (remove resume.db).

---

## 6. Quick Checklist

- [ ] Delete `backend/a.py`
- [ ] Delete `backend/resume.db` (after confirming PostgreSQL is used)
- [ ] Delete `backend/temp_*.pdf` and add `temp_*.pdf` to .gitignore
- [ ] Add/update `.gitignore` (uploads, chromadb, .venv, .env, temp, *.db)
- [ ] Remove or merge `final_clean.py` / `force_clean.py` → `scripts/clean_chroma.py`
- [ ] Move `check_database.py` → `scripts/check_db.py` (or remove)
- [ ] Move `delete_db.py` → `scripts/delete_chroma_db.py` (or remove)
- [ ] Remove `upload_test.py` or move to `tests/test_upload.py`
- [ ] Remove `db_query.py` or move to `scripts/`
- [ ] Create `backend/config.py` and use for DATABASE_URL and paths
- [ ] If root `chromadb/` exists and is unused, remove it

This keeps the project simple, removes duplicates, and sets you up for a cleaner structure without a big refactor.
