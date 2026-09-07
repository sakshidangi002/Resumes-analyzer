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
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import asynccontextmanager
from datetime import timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.types import Scope
from sqlalchemy.orm import Session
from app.api.routes import api_router
from app.core.config import get_settings
from app.core.net import client_ip
from app.core.security import decode_access_token
from app.db.session import SessionLocal
from app.models import User

# ---------------------------------------------------------------------------
# Logging  (C4 + rotation)
# ---------------------------------------------------------------------------
_CRED_RE = re.compile(r"(?P<scheme>\w+://)(?P<user>[^:/@\s]+):(?P<pw>[^@/\s]+)@")


class RedactCredentialsFilter(logging.Filter):
    """Scrub `scheme://user:password@host` from every log record.

    Defence in depth. Call sites redact explicitly (camera_service._redact_url),
    but a single missed f-string used to be enough to write the DVR password to
    disk on every reconnect — and with the reconnect loop running, that meant
    thousands of times into a 26 MB file.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if "://" in message and "@" in message:
            record.msg = _CRED_RE.sub(r"\g<scheme>\g<user>:***@", message)
            record.args = ()
        return True


def _configure_logging() -> None:
    """Rotating file + console logging.

    Previously logging.basicConfig with no handler configuration: no rotation,
    no size cap, no retention. Combined with the recognition pipeline's
    per-frame INFO trace (STEP-1..STEP-11, ~100 lines/second across the
    cameras) that produced an unbounded log file.

    The STEP trace is a debugging tool, not an operational log, so it is
    silenced by default and re-enabled with RECOGNITION_LOG_LEVEL=DEBUG.
    """
    log_dir = Path(os.getenv("HRMS_LOG_DIR", Path(__file__).resolve().parents[2] / "logs"))
    formatter = logging.Formatter(
        # Date included: the old "%H:%M:%S" format made a multi-day log
        # impossible to correlate — every day looked like the same 24 hours.
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s [%(threadName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redact = RedactCredentialsFilter()

    handlers: list[logging.Handler] = []
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "hrms.log",
            maxBytes=int(os.getenv("HRMS_LOG_MAX_BYTES", str(50 * 1024 * 1024))),
            backupCount=int(os.getenv("HRMS_LOG_BACKUPS", "5")),
            encoding="utf-8",
        )
        handlers.append(file_handler)
    except Exception:  # pragma: no cover - read-only FS, permissions, etc.
        # Never let logging setup stop the app from booting; console still works.
        logger.debug("ignored, non-critical", exc_info=True)

    handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(redact)

    root = logging.getLogger()
    root.handlers[:] = handlers          # replaces uvicorn's default handler
    root.setLevel(logging.INFO)

    logging.getLogger("app.services.recognition").setLevel(
        os.getenv("RECOGNITION_LOG_LEVEL", "WARNING").upper()
    )


_configure_logging()

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
            logger.warning("get failed", exc_info=True)
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
            logger.debug("ignored, non-critical", exc_info=True)


async def _attendance_closeout_tick() -> None:
    """Close attendance days the OUT camera never closed.

    Runs at 02:30 IST — after the business-day boundary has passed for the day
    being closed, so a genuine night shift is never truncated mid-shift.

    Set ATTENDANCE_CLOSEOUT_DRY_RUN=1 to log what WOULD be closed without
    writing anything. Recommended for the first week in production.
    """
    import asyncio

    dry_run = os.getenv("ATTENDANCE_CLOSEOUT_DRY_RUN", "").lower() in {"1", "true", "yes"}
    try:
        from app.services.attendance_closeout import run_closeout

        # Blocking DB work — keep it off the event loop.
        await asyncio.to_thread(run_closeout, None, not dry_run)
    except Exception:
        logger.exception("Attendance closeout tick failed")

    try:
        from app.services.attendance_snapshot import prune_snapshots

        # Face snapshots are biometric data on a retention clock; without this
        # the directory grows without bound.
        await asyncio.to_thread(prune_snapshots)
    except Exception:
        logger.exception("Attendance snapshot pruning failed")

    try:
        from app.services.unknown_faces import purge_older_than

        # Unknown-face rows carry an embedding and a face crop — the same class
        # of biometric data as the snapshots above, and they accumulate with
        # every unrecognised passer-by. Retention is not optional.
        await asyncio.to_thread(purge_older_than)
    except Exception:
        logger.exception("Unknown-face purge failed")


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
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
    except Exception:
        logger.exception(
            "APScheduler not available — DSR reminder and attendance closeout "
            "will NOT run. Install with: pip install apscheduler tzdata"
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
        sched.add_job(
            _attendance_closeout_tick,
            trigger=CronTrigger(hour=2, minute=30, timezone="Asia/Kolkata"),
            id="attendance_closeout",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            # An hour of grace so a restart around 02:30 does not skip the
            # night's run entirely — an unclosed day is never picked up again.
            misfire_grace_time=3600,
        )
        sched.start()
        logger.info(
            "Background scheduler started: DSR reminder tick (time configurable "
            "via /api/dsr/reminder-settings, IST) and attendance closeout "
            "(02:30 IST)."
        )
        return sched
    except Exception:
        logger.exception("Failed to start background scheduler")
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
    # ── Bound concurrent sync request handlers ─────────────────────────────
    #
    # Starlette runs every `def` (non-async) route handler in anyio's thread
    # pool, which defaults to 40 threads. Nearly all of this app's handlers are
    # sync and each takes a pooled DB connection, so 40 concurrent requests can
    # claim the ENTIRE pool (pool_size 20 + max_overflow 20) and leave nothing
    # for the work that does not arrive over HTTP: four camera workers, the
    # attendance writer pool, the WebSocket manager and the APScheduler jobs.
    #
    # That is how this database reached "FATAL: sorry, too many clients
    # already" and dropped CCTV transit events.
    #
    # Capped below the connection pool so background work always has headroom.
    # Requests beyond the cap queue for a thread instead of failing on a
    # connection checkout, which is a far better failure mode: a slightly
    # slower response rather than a 500 and a lost attendance event.
    #
    # NOT a substitute for the real fix -- see app/api/deps.get_db_session,
    # which was opening two connections per authenticated request. This is the
    # backstop that keeps a traffic spike from starving the cameras.
    try:
        import anyio.to_thread

        _req_threads = int(os.getenv("REQUEST_THREAD_LIMIT", "24"))
        anyio.to_thread.current_default_thread_limiter().total_tokens = _req_threads
        logger.info(
            "Sync request handlers capped at %d concurrent threads "
            "(DB pool is %s+%s, remainder reserved for camera/scheduler work)",
            _req_threads, os.getenv("DB_POOL_SIZE", "20"),
            os.getenv("DB_MAX_OVERFLOW", "20"),
        )
    except Exception:
        logger.warning("Could not cap the request thread pool", exc_info=True)

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
    if _cctv_workers_enabled():
        try:
            from app.services.camera_service import camera_manager
            camera_manager.start_all_from_db()
            logger.info("CCTV camera manager started")
        except Exception:
            logger.exception("Failed to start CCTV camera manager – cameras will not run")

        # --- Automatic chair inventory ------------------------------------
        # Started only alongside the camera workers: it reads frames out of
        # their memory, so on an API-only instance there would be nothing for it
        # to look at. A failure here must never stop the cameras -- without it
        # the chair map simply stays as configured.
        try:
            from app.cctv_v2.pipeline import chair_sweeper
            chair_sweeper.start()
        except Exception:
            logger.exception(
                "Failed to start the chair sweeper – chair counts will stay as configured"
            )
    else:
        logger.info(
            "CCTV camera manager DISABLED for this process (CCTV_WORKERS_ENABLED=0). "
            "Live preview/stream endpoints will not serve frames here."
        )

    # --- Auto-connect DVR and start its streams (opt-in via .env) ----------
    try:
        from app.core.config import get_settings
        _s = get_settings()
        if not _cctv_workers_enabled():
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
        # Stopped BEFORE the cameras: it reads their frames, and its final act
        # is to persist the inventory, which should happen while the process is
        # still healthy.
        try:
            from app.cctv_v2.pipeline import chair_sweeper
            chair_sweeper.stop()
        except Exception:
            logger.exception("Error stopping the chair sweeper")

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

    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: blob: https://fastapi.tiangolo.com; "
        "connect-src 'self' ws: wss: https://cdn.jsdelivr.net; "
        "font-src 'self' data:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    )

    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault(
        "Referrer-Policy",
        "strict-origin-when-cross-origin"
    )

    if request.url.scheme == "https":
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains"
        )

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


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
# This endpoint used to be `return {"status": "ok"}` — unconditional. It never
# touched the database or looked at a camera, so it reported a healthy service
# while every camera was dead and Postgres was unreachable. That is worse than
# having no endpoint at all: it manufactures confidence, and a supervisor
# configured against it would never restart anything.
#
# LIVENESS vs HEALTH — the split matters, and conflating them is the classic
# way to build a restart loop:
#
#   /health/live  "is this process able to serve a request?"  → supervisor
#                 Restart target. Trivially 200.
#   /health       "is this process doing its job?"            → monitoring
#                 503 when degraded. NOT a restart target: restarting does not
#                 reconnect an unplugged camera or revive a dead database, it
#                 just adds an outage to a fault that is already visible.
#
# HEALTH_STRICT=0 restores the old unconditional behaviour without a code
# change, as the rollback path.

def _cctv_workers_enabled() -> bool:
    """Whether THIS process runs the camera pipeline.

    The health check must agree with the lifespan about this, or an API-only
    instance reports "no cameras" as healthy while the camera host's failure
    goes unnoticed. Single source of truth for a value that was parsed
    identically in three places.
    """
    return os.getenv("CCTV_WORKERS_ENABLED", "1").strip().lower() not in ("0", "false", "no")


_HEALTH_STRICT = os.getenv("HEALTH_STRICT", "1").strip().lower() not in ("0", "false", "no")

# Comfortably above the stream watchdog's own _STALE_TIMEOUT (15s): by the time
# a frame is this old, the forced reconnect has already been tried and failed.
_HEALTH_MAX_FRAME_AGE_SEC = float(os.getenv("HEALTH_MAX_FRAME_AGE_SEC", "60"))

# Deliberately much looser than frame age. Inference is SUPPOSED to pause on a
# static scene — the motion gate coasts monitor cameras for 30s and only forces
# a doorway pass every 8s when nothing moves, on top of a p99 cycle around 25s.
# An empty corridor at night is idle, not broken. 180s clears every legitimate
# combination of those while still catching the failure that prompted this:
# camera 60 produced a 3,360-second gap (host sleep) that nothing detected.
_HEALTH_MAX_INFERENCE_AGE_SEC = float(os.getenv("HEALTH_MAX_INFERENCE_AGE_SEC", "180"))

# Bounded so a hung database cannot hang the probe. A health check that blocks
# looks identical to a dead process from the supervisor's side.
_HEALTH_DB_TIMEOUT_SEC = float(os.getenv("HEALTH_DB_TIMEOUT_SEC", "3.0"))

# Fail before the queue is full, not when it overflows — at 100% writes are
# already being dropped (see camera_service._submit_attendance).
_HEALTH_QUEUE_WARN_RATIO = float(os.getenv("HEALTH_QUEUE_WARN_RATIO", "0.8"))


def _probe_database() -> dict:
    """SELECT 1 on a short-lived session. Blocking — call it via _check_database.

    `SessionLocal()` is INSIDE the try: checking a connection out of an
    exhausted pool raises before any query runs, and "too many clients" is a
    failure this deployment has actually hit. A probe that propagates instead
    of reporting is a probe that cannot describe the outage it exists for.
    """
    from sqlalchemy import text

    db = None
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        # Type name only. The message from a connection failure carries the DSN,
        # and this body is returned to whatever can reach the endpoint.
        return {"ok": False, "error": type(exc).__name__}
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                logger.debug("ignored, non-critical", exc_info=True)


# One dedicated thread for the database probe, and never more than one probe in
# flight.
#
# The obvious implementation — `asyncio.wait_for(asyncio.to_thread(probe))` —
# returns on time but does NOT cancel the thread: a blocking socket read is not
# interruptible. The work keeps running in asyncio's DEFAULT executor, which is
# shared with everything else. A supervisor polling /health every 10s against a
# database that is hung (not refusing — hung) therefore parks a new thread every
# poll, in the pool the rest of the app depends on. The health check becomes an
# outage amplifier.
#
# So: a private single-thread executor, and if the previous probe has not come
# back the answer is already known — the database is not responding — and no
# second probe is started. Bounded at exactly one thread no matter how long the
# fault lasts.
_HEALTH_DB_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="health-db")
_health_db_inflight: "Future | None" = None
_health_db_lock = threading.Lock()


def _check_database() -> dict:
    """Bounded database probe. Never blocks longer than the timeout."""
    global _health_db_inflight

    with _health_db_lock:
        pending = _health_db_inflight
        if pending is not None and not pending.done():
            # A previous probe is still stuck on the socket. That IS the answer.
            return {"ok": False, "error": "database not responding (probe still in flight)"}
        try:
            future = _HEALTH_DB_EXECUTOR.submit(_probe_database)
        except Exception as exc:  # noqa: BLE001 — executor shut down at exit
            return {"ok": False, "error": type(exc).__name__}
        _health_db_inflight = future

    try:
        return future.result(timeout=_HEALTH_DB_TIMEOUT_SEC)
    except FuturesTimeoutError:
        # Leave the future in place: the next call sees it unfinished and
        # returns immediately rather than queueing behind it.
        return {"ok": False, "error": f"timeout after {_HEALTH_DB_TIMEOUT_SEC:.0f}s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}


def _probe_cameras() -> dict:
    """Camera liveness. Never raises — a probe that throws reports nothing."""
    if not _cctv_workers_enabled():
        # Not a fault. This instance was deliberately started without cameras.
        return {"ok": True, "enabled": False, "total": 0, "unhealthy": [], "cameras": []}

    try:
        from app.services.camera_service import camera_manager

        snapshot = camera_manager.health_snapshot()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Health: camera snapshot failed")
        return {"ok": False, "enabled": True, "error": type(exc).__name__,
                "total": 0, "unhealthy": [], "cameras": []}

    cameras = snapshot.get("cameras", [])
    unhealthy: list[dict] = []
    for cam in cameras:
        reasons = []
        frame_age = cam.get("frame_age_sec")
        infer_age = cam.get("inference_age_sec")

        if cam.get("status") in ("error", "unreadable"):
            reasons.append(f"status={cam.get('status')}")
        # None means the camera has never produced a frame. On a worker that is
        # registered and supposedly running, that is a fault, not "no data yet".
        if frame_age is None:
            reasons.append("no frame ever received")
        elif frame_age > _HEALTH_MAX_FRAME_AGE_SEC:
            reasons.append(f"frame {frame_age:.0f}s old (>{_HEALTH_MAX_FRAME_AGE_SEC:.0f}s)")
        if infer_age is not None and infer_age > _HEALTH_MAX_INFERENCE_AGE_SEC:
            reasons.append(
                f"no analysis for {infer_age:.0f}s (>{_HEALTH_MAX_INFERENCE_AGE_SEC:.0f}s)"
            )
        # False means the thread died. None means this worker type never had one
        # (HCNetSDK), which is not a fault.
        if cam.get("stream_thread_alive") is False:
            reasons.append("capture thread dead")
        if cam.get("recognition_thread_alive") is False:
            reasons.append("recognition thread dead")

        if reasons:
            unhealthy.append({
                "camera_id": cam.get("camera_id"),
                "name": cam.get("name"),
                "reasons": reasons,
            })

    return {
        "ok": not unhealthy,
        "enabled": True,
        "total": len(cameras),
        "unhealthy": unhealthy,
        "cameras": cameras,
        "ffmpeg_ok": snapshot.get("ffmpeg_ok"),
    }


def _probe_attendance_writer() -> dict:
    """Queue depth for the attendance write pool. Full queue = dropped writes."""
    if not _cctv_workers_enabled():
        return {"ok": True, "enabled": False}
    try:
        from app.services.camera_service import camera_manager

        snapshot = camera_manager.health_snapshot()
        depth = snapshot.get("attendance_queue_depth")
        maximum = snapshot.get("attendance_queue_max") or 0
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "enabled": True, "error": type(exc).__name__}

    if depth is None or not maximum:
        return {"ok": True, "enabled": True, "queue_depth": depth, "queue_max": maximum}
    return {
        "ok": depth < maximum * _HEALTH_QUEUE_WARN_RATIO,
        "enabled": True,
        "queue_depth": depth,
        "queue_max": maximum,
    }


@app.get("/health/live")
def health_live():
    """Liveness only: the process is up and can serve a request.

    The supervisor's restart target. Deliberately checks NOTHING else — a
    restart cannot fix an unplugged camera or a dead database, so making those
    conditions restart the process converts a visible fault into a restart loop
    on top of it.
    """
    return {"status": "ok"}


@app.get("/health")
async def health():
    """Health: is this process actually doing its job?

    200 when every subsystem is good, 503 when any is not, so a monitor can
    alert on it directly. The body names what failed and why.
    """
    if not _HEALTH_STRICT:
        # Rollback path — the pre-T-01 behaviour, without a code change.
        return {"status": "ok"}

    import asyncio

    # _check_database is bounded and cheap, but it still WAITS — keep that off
    # the event loop so a slow database cannot stall unrelated requests.
    database = await asyncio.to_thread(_check_database)

    checks = {
        "database": database,
        "cameras": _probe_cameras(),
        "attendance_writer": _probe_attendance_writer(),
    }
    healthy = all(check.get("ok") for check in checks.values())

    body = {"status": "ok" if healthy else "degraded", "checks": checks}
    if healthy:
        return body
    # 503, not 500: the service is up and answering, it just is not fit to
    # serve. A 500 would read as "the health check itself is broken".
    return JSONResponse(status_code=503, content=body)


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
def _is_frontend_build(path: Path) -> bool:
    """True when `path` looks like a usable Vite build (has an index.html)."""
    try:
        return path.is_dir() and (path / "index.html").is_file()
    except OSError:
        return False


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

    # A candidate only counts when it actually holds an index.html. A bare
    # directory (an interrupted build, or a `frontend_build/` left behind by a
    # cleanup) would otherwise shadow a perfectly good `frontend/dist` and make
    # every non-API route 500 on a missing file.
    if _is_frontend_build(integrated):
        return integrated

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        for candidate in (exe_dir / "frontend_build", exe_dir / "frontend" / "dist"):
            if _is_frontend_build(candidate):
                return candidate
        return exe_dir / "frontend_build"

    legacy = backend_dir.parent / "frontend" / "dist"
    if _is_frontend_build(legacy):
        return legacy
    return integrated


FRONTEND_DIST = _get_frontend_dist()

if _is_frontend_build(FRONTEND_DIST):
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

else:
    # No built UI on disk. Without this branch every non-API route (including
    # "/") falls through to FastAPI's default 404 — the API and /docs answer
    # fine, so the app looks half-broken with no hint why. Say what is missing
    # and how to fix it instead.
    logger.warning(
        "React frontend build not found at %s - serving build instructions at '/'. "
        "Run build-frontend.bat (or `npm install && npm run build` in frontend/) "
        "to build the UI, or set FRONTEND_BUILD_PATH to an existing build.",
        FRONTEND_DIST,
    )

    _MISSING_UI_HTML = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>UI not built</title></head>
<body style="font-family:system-ui,sans-serif;max-width:44rem;margin:3rem auto;padding:0 1rem;line-height:1.6">
<h1>Frontend build not found</h1>
<p>The API is running, but the React UI has not been built, so there is nothing to serve at this address.</p>
<p>Expected a Vite build (a folder containing <code>index.html</code>) at:</p>
<pre style="background:#f4f4f5;padding:.75rem;border-radius:.375rem;overflow-x:auto">{FRONTEND_DIST}</pre>
<h2>Fix</h2>
<pre style="background:#f4f4f5;padding:.75rem;border-radius:.375rem;overflow-x:auto">cd "Attendance Management"
build-frontend.bat</pre>
<p>Then restart the server. The API docs remain available at <a href="/docs">/docs</a>.</p>
</body></html>"""

    @app.get("/{full_path:path}", include_in_schema=False)
    def missing_frontend(full_path: str):
        return HTMLResponse(
            _MISSING_UI_HTML,
            status_code=503,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
