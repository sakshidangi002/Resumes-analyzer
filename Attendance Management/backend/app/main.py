"""
Attendance & HRMS – FastAPI application.
PostgreSQL database; SMTP email; role-based access (Admin, HR, Manager, Employee).
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.types import Scope
from app.api.routes import api_router
from app.core.config import get_settings


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

app = FastAPI(
    title=get_settings().app_name,
    description="Attendance, Leave, Payroll, HR Letters, Notifications. Indian HR practices; 30-day month salary.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}


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
