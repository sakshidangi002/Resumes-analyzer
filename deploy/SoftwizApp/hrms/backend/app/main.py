"""
Attendance & HRMS – FastAPI application.
PostgreSQL database; SMTP email; role-based access (Admin, HR, Manager, Employee).

Also acts as the unified entry point that exposes Resume Analyzer (UI + API)
under the same origin, so the whole product is reachable on a single URL:

    /              -> HRMS React SPA
    /api/...       -> HRMS API
    /resume/       -> Resume Analyzer UI (static)
    /resume-api/   -> Resume Analyzer API
    /portal.html   -> Landing portal
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.types import Scope
from sqlalchemy.orm import Session
from app.api.routes import api_router
from app.core.config import get_settings
from app.core.security import decode_access_token
from app.db.session import SessionLocal
from app.models import User


# --- Resume Analyzer integration -------------------------------------------
# The Resume Analyzer lives at the repo root (../../.. from this file):
#   <repo>/backend/api.py        -> Resume API FastAPI app
#   <repo>/frontend/             -> Resume Analyzer static UI
#   <repo>/portal.html           -> Unified landing page
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

resume_api_app = None
try:
    from backend.api import app as resume_api_app  # type: ignore
except Exception as _exc:  # noqa: BLE001 — Resume API is optional at runtime
    print(f"[unified] Resume Analyzer API not available: {_exc}")


# MIME types that should declare UTF-8. Without this, browsers default to
# Latin-1 / Windows-1252 and multi-byte characters (₹, ·, em-dash, etc.) in
# the JS/CSS bundle render as mojibake ("â,¹", "Â·").
_UTF8_MIME_PREFIXES = (
    "application/javascript",
    "text/javascript",
    "application/json",
    "text/css",
    "image/svg+xml",
)


class Utf8StaticFiles(StaticFiles):
    """StaticFiles that forces charset=utf-8 on text-like assets."""

    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        try:
            ctype = response.headers.get("content-type", "") or ""
            base = ctype.split(";", 1)[0].strip().lower()
            if base and "charset=" not in ctype.lower() and any(
                base.startswith(p) for p in _UTF8_MIME_PREFIXES
            ):
                response.headers["content-type"] = f"{base}; charset=utf-8"
        except Exception:
            pass
        return response

@asynccontextmanager
async def _unified_lifespan(parent_app: FastAPI):
    """Bridge the Resume API's lifespan so mounted sub-app startup hooks run."""
    if resume_api_app is not None and getattr(resume_api_app.router, "lifespan_context", None):
        async with resume_api_app.router.lifespan_context(resume_api_app):
            yield
    else:
        yield


app = FastAPI(
    title=get_settings().app_name,
    description="Attendance, Leave, Payroll, HR Letters, Notifications. Indian HR practices; 30-day month salary.",
    version="1.0.0",
    lifespan=_unified_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api")

# Mount Resume Analyzer API at /resume-api so the existing Resume UI works
# without code changes other than its base URL.
if resume_api_app is not None:
    app.mount("/resume-api", resume_api_app, name="resume_api")


# ---------------------------------------------------------------------------
# Resume Analyzer access control
# ---------------------------------------------------------------------------
# The Resume Analyzer (UI + API) must only be reachable by users whose JWT
# carries the Admin or HR role. Anyone else hitting /resume/* or /resume-api/*
# directly gets redirected to login (UI) or a 403 (API).

RESUME_ALLOWED_ROLES = {"Admin", "HR"}


def _extract_token(request: Request) -> str:
    """Pull JWT from Authorization header or `?token=` query string (used by
    the Resume UI when opened via the HRMS sidebar)."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return request.query_params.get("token", "").strip()


def _user_roles_from_token(token: str) -> set[str]:
    """Decode a JWT and return the user's role names (empty set if invalid)."""
    if not token:
        return set()
    payload = decode_access_token(token)
    if not payload:
        return set()
    # Prefer roles embedded in the token (fast path, no DB hit).
    roles = payload.get("roles")
    if isinstance(roles, list) and roles:
        return {str(r) for r in roles}
    # Fallback: look up the user in the DB.
    user_id = payload.get("sub")
    if not user_id:
        return set()
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user or not user.is_active:
            return set()
        return {r.name for r in user.roles}
    except Exception:
        return set()
    finally:
        db.close()


@app.middleware("http")
async def resume_access_guard(request: Request, call_next):
    """Block non-Admin/HR users from /resume/* and /resume-api/*."""
    path = request.url.path
    is_ui  = path == "/resume" or path.startswith("/resume/")
    is_api = path == "/resume-api" or path.startswith("/resume-api/")

    if is_ui or is_api:
        roles = _user_roles_from_token(_extract_token(request))
        if not (roles & RESUME_ALLOWED_ROLES):
            if is_api:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Resume Analyzer is restricted to Admin / HR users."},
                )
            # Static UI: send them back to the HRMS login.
            return RedirectResponse(url="/", status_code=302)

    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the Resume Analyzer static frontend at /resume/  (index.html, app.py, styles.css, assets/)
RESUME_UI_DIR = PROJECT_ROOT / "frontend"
if RESUME_UI_DIR.is_dir() and (RESUME_UI_DIR / "index.html").exists():
    app.mount("/resume", Utf8StaticFiles(directory=str(RESUME_UI_DIR), html=True), name="resume_ui")


# Serve the landing portal at /portal.html (and /portal -> /portal.html)
PORTAL_HTML = PROJECT_ROOT / "portal.html"
if PORTAL_HTML.exists():
    @app.get("/portal.html", include_in_schema=False)
    def _portal_page():
        return FileResponse(str(PORTAL_HTML))

    @app.get("/portal", include_in_schema=False)
    def _portal_redirect():
        return RedirectResponse(url="/portal.html")


# Serve React frontend build (production mode)
def _get_frontend_dist() -> Path:
    """
    Resolve the built React app folder (Vite `dist/`) in this order:
    1) FRONTEND_BUILD_PATH (absolute/relative) — for deployment flexibility
    2) `backend/frontend_build/` (integrated copy, preferred)
    3) PyInstaller: `frontend_build/` or legacy `frontend/dist` next to the EXE
    4) Legacy dev layout: `../frontend/dist` from repo root
    """
    override = (os.getenv("FRONTEND_BUILD_PATH") or "").strip()
    if override:
        return Path(override).expanduser().resolve()

    backend_dir = Path(__file__).resolve().parent.parent
    integrated = backend_dir / "frontend_build"
    if integrated.exists():
        return integrated

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        for candidate in (exe_dir / "frontend_build", exe_dir / "frontend" / "dist"):
            if candidate.is_dir() and (candidate / "index.html").exists():
                return candidate
        return exe_dir / "frontend_build"

    legacy = backend_dir.parent / "frontend" / "dist"
    if legacy.exists():
        return legacy
    return integrated


FRONTEND_DIST = _get_frontend_dist()

if FRONTEND_DIST.exists():
    # Vite build uses /assets for hashed JS/CSS. Incomplete copies sometimes omit this folder.
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", Utf8StaticFiles(directory=str(assets_dir)), name="assets")

    # Serve favicon and other root-level static files
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        f = FRONTEND_DIST / "favicon.ico"
        if f.exists():
            return FileResponse(str(f))
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    # SPA fallback — serve index.html for ALL non-API routes
    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        index = FRONTEND_DIST / "index.html"
        return FileResponse(str(index))
