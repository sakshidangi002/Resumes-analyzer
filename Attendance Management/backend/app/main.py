"""
Attendance & HRMS – FastAPI application.
PostgreSQL database; SMTP email; role-based access (Admin, HR, Manager, Employee).

Also acts as the unified entry point that exposes Resume Analyzer (UI + API)
under the same origin, so the whole product is reachable on a single URL:

    /              -> HRMS React SPA (auto-redirects to /login if unauthenticated)
    /api/...       -> HRMS API
    /resume/       -> Resume Analyzer UI (static)
    /resume-api/   -> Resume Analyzer API
"""
import logging
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

logger = logging.getLogger(__name__)


# --- Resume Analyzer integration -------------------------------------------
# The Resume Analyzer lives at the repo root (../../.. from this file):
#   <repo>/backend/api.py        -> Resume API FastAPI app
#   <repo>/frontend/             -> Resume Analyzer static UI
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

def _warn_on_weak_secret_key() -> None:
    """Loudly flag a JWT secret key that is too short or matches a known placeholder.

    The application refuses to boot if SECRET_KEY is missing (Pydantic raises);
    this guard catches the next-worst case: a configured but weak key.
    """
    s = get_settings().secret_key or ""
    KNOWN_WEAK = {"abc2025", "change-me-in-production", "secret", "changeme"}
    if s in KNOWN_WEAK or len(s) < 32:
        logger.warning(
            "SECURITY: SECRET_KEY is weak (length=%d). "
            "Generate a strong key with `openssl rand -hex 32` and set it in .env "
            "BEFORE deploying to production. Existing JWTs will be invalidated on rotation.",
            len(s),
        )


def _dsr_reminder_tick():
    """Run once per minute: if IST hour:minute matches the HR-configured
    reminder time on an enabled weekday, fire the reminder job.

    The reminder job is itself idempotent per IST day, so even if the tick
    runs a second time (misfire / coalesce) only one notification is created
    per user per day.
    """
    try:
        from datetime import datetime, timedelta
        from app.db.session import SessionLocal
        from app.services.reminder_settings import read_schedule
        from app.services.dsr_reminder import send_dsr_reminders
    except Exception:
        logger.exception("DSR reminder tick: failed to import dependencies")
        return

    _WEEKDAY_BY_INDEX = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

    db = SessionLocal()
    try:
        enabled, target_h, target_m, weekday_set = read_schedule(db)
    except Exception:
        logger.exception("DSR reminder tick: failed to read schedule")
        db.close()
        return

    try:
        if not enabled:
            return
        ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        if ist_now.hour != target_h or ist_now.minute != target_m:
            return
        weekday_token = _WEEKDAY_BY_INDEX[ist_now.weekday()]
        if weekday_token not in weekday_set:
            return
        logger.info(
            "DSR reminder tick FIRING at IST %02d:%02d (%s)",
            target_h,
            target_m,
            weekday_token,
        )
        send_dsr_reminders(db)
    except Exception:
        logger.exception("DSR reminder tick: send failed")
    finally:
        try:
            db.close()
        except Exception:
            pass


def _start_background_scheduler():
    """Start APScheduler. We tick every minute and decide inside the tick
    whether to fire the reminder — that way HR can change the time via
    /api/dsr/reminder-settings and the new time takes effect at the next
    minute, no restart needed.

    Returns the scheduler instance (or ``None`` if APScheduler isn't
    installed) so the lifespan can shut it down on exit.
    """
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
    except Exception:
        logger.exception(
            "APScheduler not available — DSR reminder will NOT run. "
            "Install with: pip install apscheduler tzdata"
        )
        return None

    try:
        sched = AsyncIOScheduler()
        sched.add_job(
            _dsr_reminder_tick,
            trigger=IntervalTrigger(seconds=60),
            id="dsr_reminder_tick",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=120,
        )
        sched.start()
        logger.info(
            "Background scheduler started: DSR reminder tick is active "
            "(time configurable via /api/dsr/reminder-settings, IST)."
        )
        return sched
    except Exception:
        logger.exception("Failed to start background scheduler for DSR reminder")
        return None


def _sync_blocked_employee_users() -> None:
    """Retroactively deactivate user accounts whose employee record is already
    marked Resigned or Terminated. Run once on startup so the new access rule
    takes effect without HR having to re-save every old record.
    """
    try:
        from app.models.employee import Employee, EmploymentStatus

        db = SessionLocal()
        try:
            blocked_emp_ids = [
                row.id for row in db.query(Employee.id).filter(
                    Employee.employment_status.in_([
                        EmploymentStatus.RESIGNED.value,
                        EmploymentStatus.TERMINATED.value,
                    ])
                ).all()
            ]
            if not blocked_emp_ids:
                return
            updated = (
                db.query(User)
                .filter(User.employee_id.in_(blocked_emp_ids), User.is_active.is_(True))
                .update({"is_active": False}, synchronize_session=False)
            )
            if updated:
                db.commit()
                logger.info(
                    "Deactivated %d user account(s) linked to resigned/terminated employees.",
                    updated,
                )
        finally:
            db.close()
    except Exception:
        logger.exception("Failed to sync resigned/terminated employee access")


@asynccontextmanager
async def _unified_lifespan(parent_app: FastAPI):
    """Bridge the Resume API's lifespan so mounted sub-app startup hooks run,
    and run our background scheduler (5 PM IST DSR reminder) alongside it."""
    _warn_on_weak_secret_key()
    _sync_blocked_employee_users()

    # Capture the running asyncio loop so that synchronous request handlers
    # (e.g. /api/dsr POST) can fan-out real-time WebSocket events via
    # ``connection_manager.publish_sync``.
    try:
        import asyncio as _asyncio

        from app.ws.manager import connection_manager as _wsmgr

        _wsmgr.set_loop(_asyncio.get_running_loop())
    except Exception:
        logger.exception("Failed to capture event loop for WebSocket manager")

    scheduler = _start_background_scheduler()
    try:
        if resume_api_app is not None and getattr(resume_api_app.router, "lifespan_context", None):
            async with resume_api_app.router.lifespan_context(resume_api_app):
                yield
        else:
            yield
    finally:
        if scheduler is not None:
            try:
                if scheduler.running:
                    scheduler.shutdown(wait=False)
            except Exception:
                logger.exception("Error shutting down background scheduler")


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

# Real-time notification WebSocket. Mounted at the application root (NOT under
# /api/) so the URL is the conventional ws[s]://host/ws/notifications and
# isn't swallowed by the SPA fallback below.
try:
    from app.ws.notifications import router as ws_router

    app.include_router(ws_router)
except Exception:
    logger.exception("Failed to mount WebSocket router /ws/notifications")

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
    """Block non-Admin/HR users from /resume/* and /resume-api/*.

    NOTE: Static branding assets under `/resume/assets/*` (logo, css, fonts,
    images) bypass the role check. Browser `<img>` / `<link>` requests can't
    send an `Authorization: Bearer ...` header, so gating them would cause
    the sidebar logo etc. to render as broken-image icons. These files
    contain no sensitive data — the real API and parsed-candidate data
    remain gated via `/resume-api/*`.
    """
    path = request.url.path
    is_ui  = path == "/resume" or path.startswith("/resume/")
    is_api = path == "/resume-api" or path.startswith("/resume-api/")

    # Public static branding bundle (no auth required).
    if is_ui and path.startswith("/resume/assets/"):
        return await call_next(request)

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


# Legacy /portal.html and /portal routes have been removed — the application
# now lands directly on the HRMS login page (the React SPA handles auth).
@app.get("/portal.html", include_in_schema=False)
@app.get("/portal", include_in_schema=False)
def _portal_redirect_to_root():
    return RedirectResponse(url="/", status_code=302)


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

    # Service worker for Web Push notifications. MUST be served from the root
    # of the origin so its scope is "/" (otherwise the browser scopes it to
    # /assets/* which means push events from /dsr etc. would be ignored).
    # The SPA fallback below would otherwise intercept /sw.js and return
    # index.html — we explicitly route around it.
    @app.get("/sw.js", include_in_schema=False)
    def service_worker():
        f = FRONTEND_DIST / "sw.js"
        if f.exists():
            return FileResponse(
                str(f),
                media_type="application/javascript",
                headers={
                    # Prevent stale SW from sticking around forever after a deploy.
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    # Allow root scope even if the file is later moved.
                    "Service-Worker-Allowed": "/",
                },
            )
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    # SPA fallback — serve index.html for ALL non-API routes
    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        index = FRONTEND_DIST / "index.html"
        return FileResponse(str(index))
