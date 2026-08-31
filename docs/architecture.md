# Architecture notes

The product is deployed as a unified process, but the resume analyzer is kept
as a separate Python package boundary:

```text
HTTP request
    -> backend/api.py          FastAPI adapters, validation, response mapping
    -> backend/services/       dependency-light domain operations
    -> backend/main.py         document extraction and AI/model integrations
    -> PostgreSQL / Chroma     persistence and vector search
```

`backend/config.py` is the only place that defines environment-backed runtime
settings. New code should receive `Settings` or a service dependency instead of
reading `os.environ` directly.

The first service seam is `backend/services/resume_utils.py`. It contains pure
normalization, legacy text-list conversion, and embedding-cleanup operations;
these are deliberately independent of FastAPI, SQLAlchemy, and model loading.
This makes duplicate detection and text preparation testable in milliseconds.

When adding an endpoint, keep request/response concerns in `api.py` and move
database writes, extraction workflows, and external integrations into a service
module. Avoid importing model-heavy modules from unit tests.

## Validation layers

- Unit tests: pure domain behavior in `tests/`.
- Lint: Ruff on the new configuration, service, and test seams.
- Integration tests: add database-backed API tests when a disposable PostgreSQL
  service is available in deployment CI.
