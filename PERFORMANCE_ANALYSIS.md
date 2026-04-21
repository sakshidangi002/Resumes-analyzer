# Resume Analyzer – Backend Performance Analysis

## Exact causes of slowness

### 1. **Lazy model loading (first-request spike)**
- **Chat model (Qwen)**: Loaded on **first** `/chat` or `/resume/{id}/chat` request via `_get_chat_pipe()` in `main.py`. First chat can take **~20+ seconds**.
- **NuExtract**: Loaded on **first** resume upload via `_get_extract_model()`. First upload is much slower than subsequent ones.
- **Fix**: Preload models at startup (configurable) so first user request doesn’t pay the cost.

### 2. **Blocking the event loop**
- **Sync endpoints**: `/chat`, `/resume/{id}/chat`, `list_resumes`, `reextract_*`, `compare_resumes`, etc. are defined with `def` (sync). They block the single uvicorn worker for the whole duration of LLM inference and DB work.
- **upload_resume**: Declared `async def` but calls blocking code (`extract_resume()`, `embedding_manager.embedder.encode()`, DB operations) **without** `run_in_executor`, so it still blocks the event loop.
- **Effect**: One slow request (e.g. chat or upload) blocks all other requests until it finishes.
- **Fix**: Run CPU/IO-heavy work in a thread pool (`run_in_executor`) so the event loop stays responsive.

### 3. **N+1 database queries in `/chat`**
- For each vector search result, the code does:
  `db.query(ResumeDB).filter(ResumeDB.vector_id == vid).first()` — **one query per result** (up to 25).
- **Fix**: Single batch query: `db.query(ResumeDB).filter(ResumeDB.vector_id.in_(vector_ids)).all()` and then map by `vector_id` in memory.

### 4. **Network / IP access**
- **run_app.py** starts uvicorn with `--host 127.0.0.1`, so the API is **not reachable from another machine** by IP. If you changed to `0.0.0.0` elsewhere, the backend is reachable but:
- **BASE_URL** defaults to `http://127.0.0.1:8001`. When the app is opened from another system (e.g. `http://192.168.x.x:8501`), resume links and file URLs still point to `127.0.0.1`, so they break or point to the client’s localhost. This makes the app feel broken/slow when “accessed via IP”.
- **Fix**: Use `--host 0.0.0.0` when serving over the network and set `BASE_URL` to the actual host (e.g. `http://192.168.x.x:8001` or your domain).

### 5. **Single worker**
- Uvicorn runs with a single worker by default. Any blocking sync endpoint holds that worker for the whole request.
- **Fix**: Either run blocking work in threads (as above) or, with care, use multiple workers (note: each worker loads its own models and ChromaDB; use only if memory allows).

### 6. **list_resumes and skill filtering**
- `list_resumes` uses `q.all()` with no pagination — all rows are loaded into memory. Skills filter is applied **in Python** after the full fetch. For large tables this is slow and memory-heavy.
- **Fix**: Add pagination (limit/offset or cursor) and, where possible, push skill filtering into the DB (e.g. `ilike` on a normalized skills column or full-text search).

### 7. **_db_search_candidates**
- Fetches `limit * 2` rows and filters by skill terms **in Python**. No DB index on `skills` / `experience_summary` for text search.
- **Fix**: Consider PostgreSQL full-text search or GIN index on a normalized skills field to reduce rows and move filtering into the DB.

### 8. **Embedding / ChromaDB**
- Embedding model is loaded once (at import via `ResumeEmbedding()` and warmed in lifespan). No per-request model reload.
- Each `/chat` request encodes the question with `embedding_manager.embedder.encode(question)` — no caching for repeated identical questions. Cost is small compared to LLM, but caching can help for repeated queries.

---

## Code changes applied

1. **Preload chat model at startup** (optional): set `PRELOAD_CHAT_MODEL=1` in `.env` so the first chat request isn’t ~20s.
2. **Batch vector_id lookup** in `/chat`: one `ResumeDB.vector_id.in_(vector_ids)` query instead of N single-row queries.
3. **Thread pool for blocking work**: `app.state.executor` (4 workers); `/upload` runs `extract_resume` and embedding encode in executor; `/chat` and `/resume/{id}/chat` run question encode and `chatbot_answer` in executor so the event loop isn’t blocked.
4. **run_app.py**: backend host is read from `HOST` (default `127.0.0.1`). Use `HOST=0.0.0.0` when serving from another machine.

---

## Server configuration for IP access

- **Backend reachable by IP**: Start with `HOST=0.0.0.0` (e.g. `set HOST=0.0.0.0` then `python run_app.py`, or in `.env` add `HOST=0.0.0.0`).
- **Resume and file links**: Set `BASE_URL` to the URL clients use to reach the API (e.g. `http://192.168.1.100:8001`). Otherwise links in the app point to `127.0.0.1` and break when opened from another PC.
- **Streamlit**: To open the UI from another machine, run Streamlit with `--server.address=0.0.0.0` (e.g. `streamlit run frontend/app.py --server.address=0.0.0.0`) and use `http://<this-pc-ip>:8501`.

---

## Architectural recommendations

- **Background worker for uploads**: Move PDF parsing + NuExtract + embedding to a Celery/Redis (or RQ) worker so the API returns quickly and processing runs in the background; frontend can poll or use SSE for status.
- **Pagination**: Add `limit`/`offset` (or cursor) to `GET /resumes` and use them in the UI to avoid loading thousands of rows.
- **Caching**: Cache LLM responses for identical (question, context_hash) in Redis or in-memory with TTL to reduce repeated inference.
- **Read replicas**: If DB becomes the bottleneck, use a read replica for list/search and keep writes on the primary.
- **Model serving**: For scale, serve the chat (and optionally extraction) model via a separate inference service (e.g. TGI, vLLM) with a small timeout and retries, so the API stays non-blocking and scalable.
