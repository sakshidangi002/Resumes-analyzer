---
name: langchain
description: >-
  AI / ML / NLP work for the Softwiz Resume Analyzer: LangChain pipelines,
  ChromaDB vector store, sentence-transformers embeddings, spaCy entity
  extraction, PyMuPDF + pdfplumber PDF parsing, candidate ranking, AI chat, and
  the Indeed resume pipeline. Activate when editing `backend/api.py`,
  `backend/main.py`, `backend/extraction_v3.py`, `backend/indeed_resume_pipeline.py`,
  `backend/indeed_service.py`, anything in `chromadb/`, or any file that imports
  `langchain`, `chromadb`, `sentence_transformers`, `spacy`, `fitz`,
  `pymupdf`, or `pdfplumber`. Use also when the user mentions resume parsing,
  skill extraction, embedding, vector search, semantic search, RAG, candidate
  ranking, AI chat answer, IMAP resume fetch, or `/resume/` and `/resume-api/`.
---

# Activation signals

Apply this skill when ANY of these are true:

- Editing files under repo-root `backend/` whose imports include `langchain`, `chromadb`, `sentence_transformers`, `spacy`, `fitz`, or `pdfplumber`.
- Editing anything inside `chromadb/`.
- User mentions: resume parse, skill extract, embedding, vector, similarity, semantic search, RAG, LangChain chain, ChromaDB collection, ranking, AI chat, IMAP fetch, Indeed pipeline.

For pure FastAPI routing without AI work, use `backend` instead.

# Read these before changing anything

1. `backend/main.py` — single source of truth for embedding pipeline + ChromaDB persistence.
2. `backend/extraction_v3.py` — current resume extraction (skills, experience, education).
3. `backend/api.py` — `/resume-api` routes surfacing the AI features.

# Hard rules

- ONE embedding model loaded ONCE at startup. Do NOT reload the model per request.
- ONE ChromaDB persistent client. Do NOT instantiate new clients inside route handlers.
- ChromaDB collection schema is FIXED. Changing embedding dimension or distance metric requires a documented re-index of `chromadb/`.
- All AI endpoints sit behind the same JWT + Admin/HR check as the rest of `/resume-api`. Preserve this.
- Uploaded files MUST be validated for MIME and max size BEFORE the PDF parser runs.
- Long-running calls (LLM, embedding generation) MUST be `async` or wrapped in a threadpool. Do not block the event loop.
- spaCy `nlp` object is loaded once and reused. Do not re-load.

# Performance discipline

- Cache extracted skill vocab where possible.
- Batch embedding calls when N > 5. Do not loop one-at-a-time.
- ChromaDB `where` filters BEFORE vector search shrink the candidate set — use them.
- For Indeed pipeline: respect existing throttle config; do not parallelize beyond it.

# Workflow: adding a new ranking criterion

1. Extract the new field in `extraction_v3.py` so every candidate has it persisted.
2. Extend the scoring function in `main.py`. Do NOT add a second scoring module.
3. Update the `/resume-api` response schema in `api.py`.
4. Backfill existing candidates with a one-shot script before re-ranking.

# Workflow: adjusting embeddings

1. If the embedding model changes, ALL existing candidates must be re-embedded.
2. Write a one-shot script under `backend/scripts/` (do not silently re-embed at startup).
3. Confirm disk space — `chromadb/` will grow.
4. Snapshot the existing `chromadb/` before re-embedding (rename folder).

# Workflow: Indeed pipeline changes

- Storage state lives in `backend/indeed_storage_state.json` — sensitive, never commit.
- Pipeline orchestration: `backend/indeed_resume_pipeline.py` → `backend/indeed_service.py`.
- Test changes against a small fixed search query first; do not run a wide crawl.
- Re-create the storage state with `backend/create_indeed_storage_state.py` after Indeed UI / auth changes.

# Stop and ask the user when

- A change would re-index the entire ChromaDB collection.
- A change would download new model weights at runtime.
- You'd need a new heavyweight dependency (transformers from-source, torch GPU, etc.).
- You'd alter the JWT / role guard on a `/resume-api` route.

# Verification (run before declaring done)

- Upload one PDF and one DOCX end-to-end from the UI; watch uvicorn log for warnings.
- Query the AI chat about a known candidate; confirm the answer cites THEIR actual experience (not hallucinated).
- ChromaDB collection count went up by exactly the number of new uploads.
- Admin sees results; an Employee JWT returns 403 on `/resume-api/*`.
- For Indeed pipeline: a fresh run downloads at least one resume and ingests successfully.

# Related skills

- Request routing, JWT, role guards → `backend` skill.
- Environment variables (`CHAT_MODEL`, `IMAP_*`, `INDEED_*`) in root `.env` → `deployment` skill.
- AI QA checklist → `testing` skill.
