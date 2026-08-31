# Resume Analyzer

An AI-powered resume analyzer that extracts structured information from PDF/DOCX resumes, stores candidates in PostgreSQL, and provides HR search and chat features.

## Run locally

1. Create a virtual environment and install dependencies: `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and set `DATABASE_URL` and `SECRET_KEY`.
3. Start the unified server with `python run_app.py --no-browser`.

The API is available at `/resume-api`; the HRMS UI mounts the resume UI at `/resume/`.

## Project structure

- `backend/api.py` contains HTTP adapters and persistence integration.
- `backend/main.py` contains extraction and AI integrations.
- `backend/services/resume_utils.py` contains dependency-free domain helpers.
- `backend/config.py` is the single source for environment-backed settings.
- `tests/` contains fast unit tests that do not load AI models or require PostgreSQL.

## Validation

Run `pytest -q` for unit tests and `ruff check backend/config.py backend/services tests` for the CI lint target. GitHub Actions runs both checks for pushes and pull requests.

See [docs/architecture.md](docs/architecture.md) for module boundaries and
guidance for extending the API safely.
