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
import threading
import time
from contextlib import asynccontextmanager
from datetime import timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.types import Scope
from sqlalchemy.orm import Session
from app.api.routes import api_router
from app.core.config import get_settings
from app.core.net import client_ip
from app.core.security import decode_access_token
from app.db.session import SessionLocal
from app.models import User

# Timestamped logs (HH:MM:SS.mmm) so recognition-pipeline stages (STEP-1 frame
# received … STEP-11 complete, detect/match ms) can be measured to the
# millisecond. force=True replaces uvicorn's default handler so app loggers get
# the timestamp prefix.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

logger = logging.getLogger(__name__)
_SENSITIVE_RATE_STATE: dict[tuple[str, str], list[float]] = {}
_SENSITIVE_RATE_LOCK = threading.Lock()


_GENERIC_SERVER_ERROR = "Internal server error. Please try again later."


# --- Resume Analyzer integration -------------------------------------------
# The Resume Analyzer lives at the repo root (../../.. from this file):
#   <repo>/backend/api.py        -> Resume API FastAPI app
#   <repo>/frontend/             -> Resume Analyzer static UI
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Tell the Resume API it is running MOUNTED inside HRMS, not standalone. When
# mounted, this HRMS app's `resume_access_guard` (below) is the single
# authentication for every /resume-api/* request, so the Resume API skips its
# own duplicate token check — which also stops it running two synchronous DB
# queries on the event loop for every resume request. Set before the import so
# the flag is in place no matter when the middleware reads it.
os.environ["RESUME_API_MOUNTED"] = "1"

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
    settings = get_settings()
    s = settings.secret_key or ""
    KNOWN_WEAK = {"abc2025", "change-me-in-production", "secret", "changeme"}
    if s in KNOWN_WEAK or len(s) < 32:
        logger.warning(
            "SECURITY: SECRET_KEY is weak (length=%d). "
            "Generate a strong key with `openssl rand -hex 32` and set it in .env "
            "BEFORE deploying to production. Existing JWTs will be invalidated on rotation.",
            len(s),
        )
        if not settings.embedding_encryption_key:
            logger.warning(
                "BEFORE ROTATING SECRET_KEY: set EMBEDDING_ENCRYPTION_KEY to the "
                "CURRENT SECRET_KEY value first. Stored face embeddings are "
                "encrypted with SECRET_KEY while EMBEDDING_ENCRYPTION_KEY is unset, "
                "so rotating without pinning it makes every enrolled face "
                "permanently unreadable and requires re-enrolling all employees."
            )


# In-memory guard: the IST date on which the reminder job last ran to
# completion in THIS process. Combined with the per-day DB de-dupe inside
# ``send_dsr_reminders`` this guarantees exactly one reminder per user per day
# even across server restarts.
_last_dsr_reminder_date = None


def _dsr_reminder_tick():
    """Run once per minute and fire the DSR reminder as soon as the IST clock
    is at OR PAST the HR-configured time on an enabled weekday.

    Using "at or after" (rather than an exact-minute match) makes the reminder
    resilient to the server not being alive at the precise target minute: if
    the process is started/restarted any time between the target time and IST
    midnight, that day's reminder still goes out. An in-memory date guard plus
    the per-day DB de-dupe inside ``send_dsr_reminders`` prevent duplicates.
    """
    global _last_dsr_reminder_date
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
        ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        today_ist = ist_now.date()

        # Already completed a run today in this process — nothing to do.
        if _last_dsr_reminder_date == today_ist:
            return

        weekday_token = _WEEKDAY_BY_INDEX[ist_now.weekday()]
        if weekday_token not in weekday_set:
            return

        # Only fire once we're at or past the configured time of day.
        if (ist_now.hour, ist_now.minute) < (target_h, target_m):
            return

        logger.info(
            "DSR reminder tick FIRING at IST %02d:%02d (target %02d:%02d, %s)",
            ist_now.hour,
            ist_now.minute,
            target_h,
            target_m,
            weekday_token,
        )
        send_dsr_reminders(db)
        _last_dsr_reminder_date = today_ist
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


def _initialize_database_defaults() -> None:
    """Create one-time default rows during startup, never from GET handlers."""
    db = SessionLocal()
    try:
        from app.services.reminder_settings import get_or_create_config
        from app.api.routes.leave import _ensure_default_leave_types
        from app.services.letter_templates_defaults import ensure_default_letter_templates

        get_or_create_config(db)
        _ensure_default_leave_types(db)
        ensure_default_letter_templates(db)
    except Exception:
        db.rollback()
        logger.exception("Failed to initialize database defaults")
    finally:
        db.close()


@asynccontextmanager
async def _unified_lifespan(parent_app: FastAPI):
    """Bridge the Resume API's lifespan so mounted sub-app startup hooks run,
    and run our background scheduler (5 PM IST DSR reminder) alongside it.
    Also starts persistent CCTV camera workers on boot.
    """
    _warn_on_weak_secret_key()
    _sync_blocked_employee_users()
    _initialize_database_defaults()

    # Capture the running asyncio loop so that synchronous request handlers
    # (e.g. /api/dsr POST) can fan-out real-time WebSocket events via
    # ``connection_manager.publish_sync``.
    try:
        import asyncio as _asyncio

        from app.ws.manager import connection_manager as _wsmgr

        _wsmgr.set_loop(_asyncio.get_running_loop())
    except Exception:
        logger.exception("Failed to capture event loop for WebSocket manager")

    # --- Pre-load employee face embeddings ---------------------------------
    # Build the recognition cache ONCE at startup so the first CCTV frames
    # don't pay the full DB load on a recognition thread (which caused slow /
    # missed first detections right after boot).
    try:
        from app.services.embedding_cache import warm_embedding_cache
        _n = warm_embedding_cache()
        logger.info("Embedding cache warmed: %d enrolled employee(s)", _n)
    except Exception:
        logger.exception("Failed to warm embedding cache – first recognitions may be slow")

    # --- Start CCTV camera workers -----------------------------------------
    #
    # CCTV_WORKERS_ENABLED=0 starts this process WITHOUT the camera pipeline:
    # no capture threads, no YOLO/ArcFace inference, no DVR connection. The API,
    # payroll, reports and the Resume Analyzer all still work.
    #
    # This exists so camera work can be kept off a process that serves requests
    # — the inference threads compete with request handling for the same CPU,
    # and on this box a single monitor-camera cycle is seconds of saturated CPU.
    # Run one instance with cameras ON and additional API-only instances with it
    # OFF; attendance is exchanged through the database, so the API instances
    # see recognition results normally.
    #
    # LIMITATION — this is not full process separation. Live preview and the
    # MJPEG streams serve frames out of THIS process's memory via
    # camera_manager.get_latest_jpeg(), so an API-only instance has no frames to
    # hand out and those endpoints return 503 there. Point the camera UI at the
    # instance that owns the cameras, or put a frame transport (Redis / shared
    # memory) behind camera_manager before load-balancing them freely.
    if os.getenv("CCTV_WORKERS_ENABLED", "1").strip().lower() not in ("0", "false", "no"):
        try:
            from app.services.camera_service import camera_manager
            camera_manager.start_all_from_db()
            logger.info("CCTV camera manager started")
        except Exception:
            logger.exception("Failed to start CCTV camera manager – cameras will not run")
    else:
        logger.info(
            "CCTV camera manager DISABLED for this process (CCTV_WORKERS_ENABLED=0). "
            "Live preview/stream endpoints will not serve frames here."
        )

    # --- Auto-connect DVR and start its streams (opt-in via .env) ----------
    try:
        from app.core.config import get_settings
        _s = get_settings()
        if os.getenv("CCTV_WORKERS_ENABLED", "1").strip().lower() in ("0", "false", "no"):
            _s = None  # camera pipeline is off for this process; skip the DVR too
        if _s and _s.dvr_autostart and _s.dvr_ip and _s.dvr_username:
            from app.services.dvr_manager import get_dvr_manager
            dvr = get_dvr_manager()
            ok, msg, _dev = dvr.connect(
                _s.dvr_ip, _s.dvr_port, _s.dvr_username, _s.dvr_password
            )
            if ok:
                started = dvr.start_all_streams()
                logger.info("DVR auto-start: connected, %d stream(s) started", started)
            else:
                logger.error("DVR auto-start: connect failed: %s", msg)
    except Exception:
        logger.exception("DVR auto-start failed")

    scheduler = _start_background_scheduler()
    try:
        if resume_api_app is not None and getattr(resume_api_app.router, "lifespan_context", None):
            async with resume_api_app.router.lifespan_context(resume_api_app):
                yield
        else:
            yield
    finally:
        # --- Shutdown camera workers ---------------------------------------
        try:
            from app.services.camera_service import camera_manager
            camera_manager.stop_all()
            logger.info("CCTV camera manager stopped")
        except Exception:
            logger.exception("Error stopping CCTV camera manager")

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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Request validation failed: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=422, content={"detail": "Invalid request data."})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code >= 500:
        logger.error("HTTP %s at %s %s: %s", exc.status_code, request.method, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": _GENERIC_SERVER_ERROR},
            headers=exc.headers,
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception at %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": _GENERIC_SERVER_ERROR})


app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()] or ["http://127.0.0.1:5001", "http://localhost:5001", "http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' ws: wss:; font-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if request.url.scheme == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


_SENSITIVE_RATE_WINDOW_SECONDS = 60
_SENSITIVE_RATE_MAX = int(os.getenv("SENSITIVE_RATE_LIMIT_MAX", "30"))


def _prune_sensitive_rate_state(now: float) -> None:
    """Drop every key whose window has fully expired.

    Without this the dict only ever pruned the key currently being hit, so any
    (address, path) pair seen once stayed resident forever — a slow leak that
    grows with the number of distinct clients the server has ever handled.
    Caller must hold _SENSITIVE_RATE_LOCK.
    """
    cutoff = now - _SENSITIVE_RATE_WINDOW_SECONDS
    for key in [k for k, stamps in _SENSITIVE_RATE_STATE.items() if not stamps or stamps[-1] <= cutoff]:
        _SENSITIVE_RATE_STATE.pop(key, None)


@app.middleware("http")
async def sensitive_rate_limit(request: Request, call_next):
    """Small process-local backstop for expensive/authenticated write paths."""
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        path = request.url.path
        watched = path.startswith("/api/auth/") or path.startswith("/resume-api/upload") or path.endswith("/export-sheets")
        if watched:
            # Same address resolution as the login limiter (app/core/net.py), so
            # the two cannot disagree about who the caller is behind a proxy.
            key = (client_ip(request), path)
            now = time.monotonic()
            # Locked: read-modify-write of the per-key list races across
            # concurrent requests, which silently undercounts.
            with _SENSITIVE_RATE_LOCK:
                _prune_sensitive_rate_state(now)
                attempts = [
                    stamp for stamp in _SENSITIVE_RATE_STATE.get(key, [])
                    if stamp > now - _SENSITIVE_RATE_WINDOW_SECONDS
                ]
                if len(attempts) >= _SENSITIVE_RATE_MAX:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Too many requests. Try again later."},
                        headers={"Retry-After": str(_SENSITIVE_RATE_WINDOW_SECONDS)},
                    )
                attempts.append(now)
                _SENSITIVE_RATE_STATE[key] = attempts
    return await call_next(request)
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
    """Pull JWT from the Authorization header or HttpOnly session cookie."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return request.cookies.get("access_token", "").strip()


def _token_requires_password_change(token: str) -> bool:
    """True when the bearer is still on a temporary password.

    HRMS blocks these users via `get_current_user`, but the Resume Analyzer sits
    behind this middleware instead -- without this check an Admin/HR user on a
    temp password is locked out of HRMS yet can still read every candidate
    record through /resume-api/*.

    Read from the `pwd_change` JWT claim so the hot path stays DB-free.
    /auth/change-password issues a replacement token with the claim cleared.
    Tokens minted before this claim existed simply lack it and are treated as
    "no change pending", which matches their pre-existing behaviour.
    """
    if not token:
        return False
    payload = decode_access_token(token)
    return bool(payload and payload.get("pwd_change"))


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

    # Optional: let the API docs pages (Swagger UI + the OpenAPI schema it
    # loads) be viewed without a login, so an operator can browse the endpoint
    # list in a browser. This exposes only the SHAPE of the API (endpoint names
    # and parameters) -- NOT any candidate data, and NOT the ability to call a
    # protected endpoint (those still require an Admin/HR token). OFF by default;
    # turn on per-deployment with RESUME_DOCS_PUBLIC=1 in the environment.
    if is_api and os.getenv("RESUME_DOCS_PUBLIC", "").strip().lower() in ("1", "true", "yes"):
        if path.rstrip("/") in ("/resume-api/docs", "/resume-api/redoc",
                                "/resume-api/openapi.json"):
            return await call_next(request)

    if is_ui or is_api:
        token = _extract_token(request)
        roles = _user_roles_from_token(token)
        if not (roles & RESUME_ALLOWED_ROLES):
            if is_api:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Resume Analyzer is restricted to Admin / HR users."},
                )
            # Static UI. Distinguish the two failure modes — sending both to "/"
            # silently dropped the user on the HRMS dashboard with no clue why,
            # which is indistinguishable from "the link is broken".
            if not token:
                # No session cookie reached us at all. Usually genuinely signed
                # out, but also what you see if the browser is on a different
                # host than the one the cookie was issued for (localhost vs
                # 127.0.0.1 are separate cookie jars).
                logger.warning(
                    "Resume UI blocked: no session token on %s (cookie not sent?)",
                    path,
                )
                return RedirectResponse(url="/login", status_code=302)
            # Token present and valid, but this user is not Admin/HR.
            logger.warning(
                "Resume UI blocked: roles %s lack Admin/HR for %s", sorted(roles) or "[]", path
            )
            return RedirectResponse(url="/?resume_denied=1", status_code=302)
        # Correct role, but still on a temporary password -> same treatment as
        # HRMS gives them. Checked AFTER roles so we never leak "this token is
        # valid but needs a password change" to an unrelated caller.
        if _token_requires_password_change(token):
            if is_api:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Password change required before accessing Resume Analyzer."},
                    headers={"X-Password-Change-Required": "true"},
                )
            return RedirectResponse(url="/change-password", status_code=302)

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

    # SPA fallback — serve index.html for ALL non-API routes.
    #
    # index.html must NEVER be cached by the browser: it is the only file that
    # references the hashed JS/CSS bundles (e.g. /assets/index-<hash>.js). If a
    # browser serves a stale index.html it keeps loading the OLD bundle even
    # after a fresh deploy, so users see outdated UI until they hard-refresh.
    # The hashed assets themselves are immutable and safe to cache forever.
    _NO_STORE_HEADERS = {"Cache-Control": "no-cache, no-store, must-revalidate"}

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        index = FRONTEND_DIST / "index.html"
        return FileResponse(str(index), headers=_NO_STORE_HEADERS)
