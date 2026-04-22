import asyncio
import json
import logging
import os
import re
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, time
from collections import Counter
from io import BytesIO, StringIO
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text, Boolean, create_engine, text, or_, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from starlette.responses import FileResponse, JSONResponse, Response

# allow running from workspace root OR from inside the backend/ folder
try:
    from backend.main import (
        analyze_fit,
        calculate_experience_years,
        chatbot_answer,
        extract_resume,
        extract_skills_from_text,
        estimate_experience_years_from_text,
        extract_text_from_docx,
        extract_text_from_pdf,
        format_experience_duration,
        normalize_resume_text,
        validate_and_repair_extraction,
        get_embedding_model,
        preload_chat_model,
        preload_extract_model,
        rank_candidates,
    )
except ImportError:  # pragma: no cover
    from main import (  # type: ignore
        analyze_fit,
        calculate_experience_years,
        chatbot_answer,
        extract_resume,
        extract_skills_from_text,
        estimate_experience_years_from_text,
        extract_text_from_docx,
        extract_text_from_pdf,
        format_experience_duration,
        normalize_resume_text,
        validate_and_repair_extraction,
        get_embedding_model,
        preload_chat_model,
        preload_extract_model,
        rank_candidates,
    )

import pandas as pd
from chromadb import PersistentClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")


def _log_json(event: str, payload: dict) -> None:
    """Structured log helper (no PII redaction beyond caller responsibility)."""
    try:
        logger.info("%s %s", event, json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.info("%s %s", event, str(payload))


def _norm_email(email: Optional[str]) -> str:
    return (email or "").strip().lower()


def _norm_phone(phone: Optional[str]) -> str:
    # digits only
    p = re.sub(r"\D+", "", (phone or ""))
    # drop common leading country codes for comparison (best-effort)
    if len(p) > 10 and p.startswith("1"):
        p = p[1:]
    if len(p) > 10 and p.startswith("91"):
        p = p[2:]
    return p


def _norm_name(name: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _sanitize_embedding_text(text: str) -> str:
    # collapse whitespace and drop extreme symbol junk
    t = (text or "").replace("\r", "\n")
    lines = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        # drop footnote-only lines / symbol-heavy noise
        sym = sum(1 for ch in s if not (ch.isalnum() or ch.isspace()))
        if len(s) > 12 and (sym / max(1, len(s))) > 0.45:
            continue
        lines.append(s)
    return re.sub(r"\s+", " ", "\n".join(lines)).strip()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:root@localhost:5432/Resume_analyzer",
)
# Used for building resume download links — override in .env if behind a proxy
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8501")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
EXCEL_DIR = os.path.join(BASE_DIR, "data")
EXCEL_FILE = os.path.join(EXCEL_DIR, "resumes_data.xlsx")

# Excel column order for append
EXCEL_COLUMNS = ["name", "email", "phone", "skills", "experience", "resume_link", "created_at"]

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,  # reconnect on stale connections
    connect_args={"connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "5"))},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class ResumeDB(Base):
    __tablename__ = "resumes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    phone = Column(String)
    location = Column(Text)
    skills = Column(Text)
    experience_years = Column(Float)
    experience_summary = Column(Text)
    total_experience_years = Column(Float, nullable=True)
    experience_level = Column(String(50), nullable=True)
    internship_present = Column(Boolean, nullable=True, default=False)
    experience_notes = Column(Text, nullable=True)
    education = Column(Text)
    projects = Column(Text)
    resume_link = Column(Text)
    source_file = Column(String)
    vector_id = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    companies_worked_at = Column(Text, nullable=True)
    role = Column(String, nullable=True)
    important_keywords = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    one_liner = Column(String(500), nullable=True)
    experience_line = Column(Text, nullable=True)
    experience_tags = Column(Text, nullable=True)
    is_shortlisted = Column(Boolean, default=False, nullable=False)
    tags = Column(Text, nullable=True)
    deleted_at = Column(DateTime, nullable=True)
    source = Column(String(50), nullable=True, default="manual_upload")
    key_skills = Column(Text, nullable=True)
    primary_skills = Column(Text, nullable=True)
    other_skills = Column(Text, nullable=True)


class CandidateNoteDB(Base):
    __tablename__ = "candidate_notes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resume_id = Column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
    )
    note = Column(Text, nullable=False)
    status = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100), nullable=True)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ResumeSchema(BaseModel):
    name: str = ""
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    skills: List[str] = []
    experience_years: float = 0.0
    total_experience_years: float = 0.0
    experience_level: str = ""
    internship_present: bool = False
    experience_notes: str = ""
    experience_summary: str = ""
    education: List[str] = []
    projects: List[str] = []
    resume_link: Optional[str] = None
    source_file: Optional[str] = None
    summary: str = ""
    companies_worked_at: List[str] = []
    role: Optional[str] = None
    important_keywords: List[str] = []
    experience_line: Optional[str] = None
    experience_tags: List[str] = []
    key_skills: List[str] = []
    primary_skills: List[str] = []
    other_skills: List[str] = []


class AnalyzeFitRequest(BaseModel):
    job_description: str
    resume_id: str


class BulkTagRequest(BaseModel):
    resume_ids: List[str]
    tag: str


class BulkDeleteRequest(BaseModel):
    resume_ids: List[str]


class NoteCreate(BaseModel):
    note: str
    status: Optional[str] = None


# ---------------------------------------------------------------------------
# Vector / embedding manager
# ---------------------------------------------------------------------------

class ResumeEmbedding:
    def __init__(self):
        os.makedirs("./chromadb", exist_ok=True)
        self.chromaclient = PersistentClient(path="./chromadb")
        try:
            self.collection = self.chromaclient.get_collection("resumes")
            logger.info("Found existing 'resumes' ChromaDB collection.")
        except Exception:
            self.collection = self.chromaclient.create_collection(
                name="resumes", 
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Created new 'resumes' ChromaDB collection.")
        self.embedder = get_embedding_model()
        logger.info("Embedding manager ready.")

        
embedding_manager = ResumeEmbedding()


# ---------------------------------------------------------------------------
# FastAPI app with lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

def _run_migrations():
    """
    Safely add any columns that exist in the ORM model but may be missing
    from an older PostgreSQL table (create_all never alters existing tables).
    Uses ADD COLUMN IF NOT EXISTS so it is safe to run on every startup.
    """
    migrations = [
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS is_shortlisted BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS summary TEXT",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS companies_worked_at TEXT",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS role VARCHAR",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS important_keywords TEXT",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS one_liner VARCHAR(500)",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS experience_line TEXT",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS experience_tags TEXT",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS tags TEXT",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS location TEXT",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS key_skills TEXT",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS primary_skills TEXT",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS other_skills TEXT",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'manual_upload'",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS total_experience_years FLOAT",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS experience_level VARCHAR(50)",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS internship_present BOOLEAN DEFAULT FALSE",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS experience_notes TEXT",
        # candidate_notes table
        "ALTER TABLE candidate_notes ADD COLUMN IF NOT EXISTS created_by VARCHAR(100)",
    ]
    try:
        # IMPORTANT: In PostgreSQL, any failed statement can abort the whole transaction.
        # Run each statement in its own transaction so a single failure doesn't poison the connection.
        with engine.connect() as conn:
            for sql in migrations:
                try:
                    conn.execute(text(sql))
                    conn.commit()
                    logger.debug("Migration OK: %s", sql[:60])
                except Exception as col_err:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    logger.debug("Migration skipped (%s): %s", col_err, sql[:60])
        logger.info("Database migrations complete.")
    except Exception as exc:
        logger.warning("Migration step failed (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Run column migrations (adds missing columns to existing tables)
    # NOTE: if PostgreSQL is down/unreachable, don't block API startup forever.
    try:
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(loop.run_in_executor(None, _run_migrations), timeout=15)
    except asyncio.TimeoutError:
        logger.warning("Migration step timed out; continuing startup (DB may be down).")
    except Exception as exc:
        logger.warning("Migration step failed (non-fatal): %s", exc)

    # 2. Thread pool for CPU-bound work (LLM, embedding) so the event loop stays responsive
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="resume_worker")
    app.state.executor = executor

    # 3. Pre-warm embedding model (fast, ~1 sec)
    try:
        logger.info("Loading embedding model…")
        model = get_embedding_model()
        test_vec = model.encode(["test"])
        logger.info("Embedding model OK. Shape: %s", test_vec.shape)
    except Exception as exc:
        logger.error("Embedding model FAILED: %s", exc)

    # 4. Optional: preload chat model in background so first chat is fast without blocking startup
    if os.getenv("PRELOAD_CHAT_MODEL", "").strip().lower() in ("1", "true", "yes"):
        def _preload_chat():
            try:
                logger.info("Preloading chat model in background…")
                preload_chat_model()
                logger.info("Chat model preloaded.")
            except Exception as exc:
                logger.warning("Chat model preload failed (will load on first chat): %s", exc)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, _preload_chat)
        # Do not await — let startup complete immediately; model loads in background

    # 5. Optional: preload extraction model so first upload is fast (no cold start)
    if os.getenv("PRELOAD_EXTRACT_MODEL", "").strip().lower() in ("1", "true", "yes"):
        def _preload_extract():
            try:
                logger.info("Preloading extraction model in background…")
                preload_extract_model()
                logger.info("Extraction model preloaded.")
            except Exception as exc:
                logger.warning("Extraction model preload failed (will load on first upload): %s", exc)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, _preload_extract)

    yield

    executor.shutdown(wait=False)


app = FastAPI(
    title="Resume Analyzer",
    version="2.1.0",
    description="FastAPI backend for Resume Analyzer – upload, search, rank, chat.",
    lifespan=lifespan,
)

Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Middleware / global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    tb = traceback.format_exc()
    logger.error("Unhandled exception: %s\n%s", exc, tb)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _sanitize_filename(filename: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", os.path.basename(filename))


def _safe_get(obj, attr: str, default=None):
    return getattr(obj, attr, default) or default


# Strip any email address from name so "Name\nemail" or "Name email@x.com" becomes "Name" only
_EMAIL_IN_NAME_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_IN_NAME_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){6,14}\d")


def _sanitize_name(name: Optional[str]) -> str:
    """Return name with any embedded email removed; name-only, no email in string."""
    if not name or not isinstance(name, str):
        return (name or "").strip()
    s = _EMAIL_IN_NAME_RE.sub("", name)
    s = _PHONE_IN_NAME_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_skill_query_token(tok: str) -> str:
    tok = (tok or "").strip().lower()
    tok = re.sub(r"\\s+", " ", tok)
    return tok


def _skill_query_key(tok: str) -> str:
    # Compact key to match variations like ".net", "dot net", "DOTNET"
    return re.sub(r"[^a-z0-9]+", "", _norm_skill_query_token(tok))


def _expand_skill_query(skills: str) -> list[list[str]]:
    """
    Expand user-entered skill filter terms into alias groups.
    Each group is OR'd; groups are AND'd.
    Example: ".net" -> [".net", "dotnet", "asp.net", "asp.net mvc", ...]
    """
    raw = (skills or "").strip()
    if not raw:
        return []

    # Prefer comma-separated tokens; fall back to whole string as one token
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if not parts:
        parts = [raw]

    alias_map: dict[str, list[str]] = {
        # .NET family
        "net": [".net", "dotnet", "asp.net", "asp.net mvc", "asp.net core", ".net core", ".net framework"],
        "dotnet": [".net", "dotnet", "asp.net", "asp.net mvc", "asp.net core", ".net core", ".net framework"],
        "aspnet": ["asp.net", "asp.net mvc", "asp.net core", ".net", "dotnet"],
        "aspnetmvc": ["asp.net mvc", "asp.net", ".net", "dotnet"],
        "mvc": ["asp.net mvc", "mvc", ".net", "dotnet"],
    }

    groups: list[list[str]] = []
    for p in parts:
        norm = _norm_skill_query_token(p)
        key = _skill_query_key(p)
        aliases = alias_map.get(key, [])
        group = [norm] + [_norm_skill_query_token(a) for a in aliases]
        # de-dupe while preserving order
        seen: set[str] = set()
        uniq = []
        for g in group:
            if g and g not in seen:
                uniq.append(g)
                seen.add(g)
        groups.append(uniq)

    return groups


def _looks_like_role_label(s: str) -> bool:
    """
    Detect role/designation labels mistakenly stored in primary_skills.
    We keep this conservative to avoid hiding real skills.
    """
    if not s:
        return False
    low = re.sub(r"\s+", " ", str(s).strip().lower())
    return low in {
        "full stack development",
        "frontend development",
        "cloud-native .net backend engineering",
        "backend development",
        "fullstack development",
        "full stack",
    }


def _derive_primary_skills_fallback(r: ResumeDB) -> list[str]:
    """
    Return exactly 3 skill names derived from key_skills/primary_skills/skills.
    Used only when stored primary_skills looks like a role label.
    """
    def _split(csv: str) -> list[str]:
        return [x.strip() for x in (csv or "").split(",") if x.strip()]

    key = _split(_safe_get(r, "key_skills", "") or "")
    prim = _split(_safe_get(r, "primary_skills", "") or "")
    skills = _split(r.skills or "")

    merged: list[str] = []
    seen: set[str] = set()
    for src in (key, prim, skills):
        for x in src:
            if not x:
                continue
            low = x.strip().lower()
            if low in {"and", "or", "with", "using", "based", "of"}:
                continue
            if low in {"chandigarh", "india"}:
                continue
            if low in seen:
                continue
            seen.add(low)
            merged.append(x.strip())
            if len(merged) >= 3:
                return merged
    return merged[:3]


def _resume_to_dict(r: ResumeDB) -> dict:
    primary_list = [
        s.strip()
        for s in (_safe_get(r, "primary_skills") or "").split(",")
        if s.strip()
    ]
    # If DB has a role label in primary_skills, derive 3 real skills at read-time.
    if len(primary_list) == 1 and _looks_like_role_label(primary_list[0]):
        primary_list = _derive_primary_skills_fallback(r)

    return {
        "id": str(r.id),
        "name": _sanitize_name(r.name) or "",
        "email": r.email or "",
        "phone": r.phone or "",
        "location": _safe_get(r, "location", "") or "",
        "skills": [s.strip() for s in (r.skills or "").split(",") if s.strip()],
        "experience_years": r.experience_years,
        "total_experience_years": float(getattr(r, "total_experience_years", 0.0) or 0.0),
        "experience_level": _safe_get(r, "experience_level", "") or "",
        "internship_present": bool(getattr(r, "internship_present", False) or False),
        "experience_notes": _safe_get(r, "experience_notes", "") or "",
        "experience_summary": r.experience_summary or "",
        "education": [s.strip() for s in (r.education or "").split(",") if s.strip()],
        "projects": [s.strip() for s in (r.projects or "").split(",") if s.strip()],
        "resume_link": r.resume_link,
        "summary": _safe_get(r, "summary", "") or "",
        "companies_worked_at": [
            s.strip()
            for s in (_safe_get(r, "companies_worked_at") or "").split(",")
            if s.strip()
        ],
        "role": _safe_get(r, "role", "") or "",
        "important_keywords": [
            s.strip()
            for s in (_safe_get(r, "important_keywords") or "").split(",")
            if s.strip()
        ],
        "is_shortlisted": getattr(r, "is_shortlisted", False) or False,
        "tags": [s.strip() for s in (_safe_get(r, "tags") or "").split(",") if s.strip()],
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "source": _safe_get(r, "source", "") or "",
        "experience_line": _safe_get(r, "experience_line", "") or "",
        "experience_tags": [
            s.strip()
            for s in (_safe_get(r, "experience_tags") or "").split(",")
            if s.strip()
        ],
        "key_skills": [
            s.strip()
            for s in (_safe_get(r, "key_skills") or "").split(",")
            if s.strip()
        ],
        "primary_skills": primary_list,
        "other_skills": [
            s.strip()
            for s in (_safe_get(r, "other_skills") or "").split(",")
            if s.strip()
        ],
    }


def append_resume_to_excel(
    name: str,
    email: Optional[str],
    phone: Optional[str],
    skills: str,
    experience_years: Optional[float],
    resume_link: Optional[str],
    created_at: Optional[datetime],
    excel_path: Optional[str] = None,
) -> bool:
    """
    Append one candidate row to resumes_data.xlsx.
    Creates file with headers if missing. Skips append if email already exists.
    Returns True if row was appended, False if duplicate or error.
    """
    path = excel_path or EXCEL_FILE
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception as e:
        logger.warning("Could not create Excel dir: %s", e)
        return False

    def _safe_str(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v).strip()

    def _safe_float(v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    row = {
        "name": _safe_str(name),
        "email": _safe_str(email),
        "phone": _safe_str(phone),
        "skills": _safe_str(skills),
        "experience": _safe_float(experience_years),
        "resume_link": _safe_str(resume_link),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") and created_at else _safe_str(created_at),
    }

    try:
        if os.path.exists(path):
            df = pd.read_excel(path, engine="openpyxl")
            if not df.empty and "email" in df.columns:
                existing_emails = df["email"].astype(str).str.strip().str.lower()
                if _safe_str(email).lower() in set(existing_emails):
                    logger.info("Excel: skip append (duplicate email): %s", _safe_str(email))
                    return False
            # Ensure columns exist
            for c in EXCEL_COLUMNS:
                if c not in df.columns:
                    df[c] = ""
            df = df[[c for c in EXCEL_COLUMNS if c in df.columns]]
        else:
            df = pd.DataFrame(columns=EXCEL_COLUMNS)

        new_row = pd.DataFrame([{c: row.get(c, "") for c in EXCEL_COLUMNS}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(path, index=False, engine="openpyxl")
        logger.info("Appended to Excel: %s", _safe_str(name))
        return True
    except Exception as exc:
        logger.warning("Excel append failed (non-fatal): %s", exc)
        return False


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.get("/api-version", tags=["Health"])
def api_version():
    routes = sorted(
        f"{list(r.methods)} {r.path}"
        for r in app.routes
        if hasattr(r, "path") and r.path and not r.path.startswith("/openapi")
    )
    return {"version": "2.1.0", "routes": routes}


@app.get("/stats", tags=["Health"])
def stats(db: Session = Depends(get_db)):
    total = db.query(ResumeDB).filter(ResumeDB.deleted_at == None).count()
    return {"total_resumes": total}



# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def _encode_embedding(embedding_text: str):
    """Blocking: encode text and return (vector_id, embedding list). Run in thread pool."""
    vid = str(uuid.uuid4())
    vec = embedding_manager.embedder.encode(embedding_text)
    if len(vec.shape) == 2:
        vec = vec[0]
    return vid, vec.tolist()


def _chroma_add(vector_id: str, embedding: list, embedding_text: str, file_name: str, name: str) -> None:
    """Blocking: add vector to ChromaDB."""
    embedding_manager.collection.add(
        embeddings=[embedding],
        documents=[embedding_text],
        metadatas=[{"vector_id": vector_id, "file": file_name, "name": name}],
        ids=[vector_id],
    )


def _extract_text_from_bytes(file_bytes: bytes, ext: str, base_dir: str) -> str:
    """Extract resume text from file bytes (PDF/DOCX). Runs in thread pool."""
    temp_path = os.path.join(base_dir, f"temp_{uuid.uuid4()}{ext}")
    try:
        with open(temp_path, "wb") as fh:
            fh.write(file_bytes)
        if ext == ".docx":
            return extract_text_from_docx(temp_path)
        return extract_text_from_pdf(temp_path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _slice_skills_section(text: str) -> str:
    """
    Best-effort: keep content under a Skills heading, stop at next common heading.
    Used only as a signal booster for skills extraction/cleanup.
    """
    t = (text or "").replace("\r\n", "\n")
    lower = t.lower()
    start_markers = ["\nskills\n", "\ntechnical skills\n", "\nkey skills\n", "\nskill set\n"]
    start = -1
    for m in start_markers:
        start = lower.find(m)
        if start != -1:
            start = start + len(m)
            break
    if start == -1:
        return ""

    tail = t[start:]
    tail_l = tail.lower()
    end_markers = [
        "\nexperience\n",
        "\nwork experience\n",
        "\nprofessional experience\n",
        "\neducation\n",
        "\nprojects\n",
        "\ncertifications\n",
        "\nsummary\n",
        "\nprofile\n",
    ]
    end = None
    for em in end_markers:
        epos = tail_l.find(em)
        if epos != -1:
            end = epos
            break
    return tail[:end].strip() if end is not None else tail.strip()


def _clean_skill_list(items, *, max_items: int) -> list[str]:
    """
    Remove junk phrases often misclassified as skills.
    Keeps short, skill-like tokens (e.g. 'JavaScript', 'Node.js', 'MongoDB').
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in (items or []):
        s = str(raw or "").strip()
        s = re.sub(
            r"^(programming languages|frontend development|backend development|version control tools|"
            r"database management|web technologies|project management tool|core concepts|technical skills|"
            r"technical proficiencies|skills|languages|databases|frameworks|testing|tools)\s*[:\-]?\s*",
            "",
            s,
            flags=re.IGNORECASE,
        )
        if not s:
            continue

        # Drop overly long / sentence-like strings
        if len(s) > 28:
            continue
        # Drop items with too many words (usually sentences)
        word_count = len(s.split())
        if word_count > 4:
            continue
        if word_count > 2 and not _looks_like_primary_skill(s):
            continue
        # Drop obvious non-skill filler
        low = s.lower().strip(" .,:;|/\\-+_()[]{}")
        if low in {"programmer", "developer", "software", "engineer", "fresher", "student"}:
            continue
        if low in {"asp", "net", "concepts"}:
            continue
        if low in {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}:
            continue
        if "management team collaboration" in low:
            continue
        
        # Aggressive verb/noun filtering for hallucinated strings
        bad_words = ["ready", "good", "passion", "team", "adaptability", "time management", "communication", "work", "thumbnail", "canva", "seo", "management", "designer"]
        if any(w in low for w in bad_words):
            continue

        key = low.replace(" ", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _looks_like_primary_skill(label: str) -> bool:
    s = str(label or "").strip()
    if not s:
        return False
    low = re.sub(r"\s+", " ", s.lower()).strip()
    if not low:
        return False
    if any(ch in s for ch in (".", "#", "+", "/")):
        return True
    primary_terms = {
        "python",
        "java",
        "javascript",
        "typescript",
        "react",
        "node.js",
        "next.js",
        "angular",
        "vue",
        "asp.net",
        ".net",
        "c#",
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "jira",
        "figma",
        "framer",
        "webflow",
        "photoshop",
        "maya",
        "blender",
        "windows",
        "linux",
        "visual studio",
        "entity framework",
        "core java",
        "advanced java",
        "react native",
        "postman",
        "ssms",
    }
    return low in primary_terms


def _derive_clean_skills_from_text(resume_text: str) -> tuple[list[str], list[str]]:
    """
    Prefer extracting skills from the Skills section; fallback to full text.
    """
    skills_text = _slice_skills_section(resume_text or "")
    skills = extract_skills_from_text(skills_text if skills_text.strip() else (resume_text or ""))
    # skills is a list, not a dict
    if not isinstance(skills, list):
        skills = []

    primary = [s for s in skills if _looks_like_primary_skill(s)]
    if not primary:
        primary = []
    else:
        primary = primary[:10]
    other = [s for s in skills if s not in primary] if skills else []

    primary_clean = _clean_skill_list(primary, max_items=10)
    other_clean = _clean_skill_list(other, max_items=40)
    # remove duplicates across lists
    prim_norm = {s.lower().replace(" ", "") for s in primary_clean}
    other_clean = [s for s in other_clean if s.lower().replace(" ", "") not in prim_norm]
    return primary_clean, other_clean


@app.post("/upload", tags=["Resumes"])
async def upload_resume(
    request: Request,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    executor = getattr(request.app.state, "executor", None)
    loop = asyncio.get_event_loop()
    results: List[Optional[dict]] = [None] * len(files)
    try:
        from backend.main import _enrichment_fallback
    except ImportError:
        from main import _enrichment_fallback  # type: ignore[import]

    # Phase 1: read files and extract text; collect valid items for parallel LLM extraction
    valid_items: List[tuple] = []  # (index, filename, file_bytes, ext, resume_text)
    for i, file in enumerate(files):
        try:
            logger.info("Upload started: %s", file.filename)
            file_bytes = await file.read()
            ext = os.path.splitext(file.filename or "")[-1].lower()
            if ext not in (".pdf", ".docx"):
                results[i] = {"status": "error", "file": file.filename, "message": "Only PDF and DOCX are supported."}
                continue
            if executor:
                resume_text = await loop.run_in_executor(
                    executor, _extract_text_from_bytes, file_bytes, ext, BASE_DIR
                )
            else:
                resume_text = _extract_text_from_bytes(file_bytes, ext, BASE_DIR)
            logger.info("Text extracted from %s: %d chars", file.filename, len(resume_text))
            if not (resume_text or resume_text.strip()):
                results[i] = {"status": "error", "file": file.filename, "message": "Text extraction produced no content."}
                continue
            valid_items.append((i, file.filename, file_bytes, ext, resume_text))
        except Exception as exc:
            import traceback
            logger.error("Upload failed for %s: %s\n%s", file.filename, exc, traceback.format_exc())
            results[i] = {"status": "error", "file": file.filename, "message": str(exc)}

    if not valid_items:
        return [r for r in results if r is not None]

    # Phase 2: run LLM extraction in parallel for all valid files (biggest speedup)
    if executor:
        extracted_list = await asyncio.gather(*[
            loop.run_in_executor(executor, extract_resume, item[4])
            for item in valid_items
        ])
    else:
        extracted_list = [extract_resume(item[4]) for item in valid_items]

    # Phase 3: enrich, validate, duplicate check, persist file, embedding, DB (per file, in order)
    for (idx, filename, file_bytes, ext, resume_text), extracted in zip(valid_items, extracted_list):
        try:
            enrichment = _enrichment_fallback(extracted)
            if not extracted.get("summary"):
                extracted["summary"] = enrichment.get("summary", "")
            extracted["one_liner"] = enrichment.get("one_liner", "")
            extracted["experience_line"] = enrichment.get("experience_line", "")
            extracted["experience_tags"] = enrichment.get("experience_tags", [])

            extraction_warnings = extracted.pop("extraction_warnings", None)
            if extraction_warnings:
                logger.info("Extraction warnings for %s: %s", filename, extraction_warnings)

            # Final plausibility gate (generalizes across messy PDFs):
            # validate/repair extracted fields against normalized text BEFORE persisting.
            try:
                norm_text, _meta = normalize_resume_text(resume_text or "")
                repaired, added = validate_and_repair_extraction(extracted, norm_text)
                if added:
                    logger.info("Extraction repairs for %s: %s", filename, added)
                extracted.update(repaired)
            except Exception as _gate_exc:
                logger.warning("Extraction validation gate failed for %s: %s", filename, _gate_exc)

            # Skills repair: prefer deterministic section-aware extraction over
            # the broader LLM-derived list when we have any usable signal.
            try:
                primary_clean, other_clean = _derive_clean_skills_from_text(norm_text or resume_text or "")
                deterministic_skills = primary_clean + other_clean
                if deterministic_skills:
                    extracted["primary_skills"] = primary_clean
                    extracted["other_skills"] = other_clean
                    extracted["skills"] = deterministic_skills
                    extracted["key_skills"] = primary_clean[:15] if primary_clean else deterministic_skills[:15]
            except Exception as _skills_exc:
                logger.debug("Skills split skipped for %s: %s", filename, _skills_exc)

            cleaned = ResumeSchema(**extracted)

            if cleaned.experience_years == 0.0:
                cleaned.experience_years = calculate_experience_years(cleaned.experience_summary or "")
                if cleaned.experience_years == 0.0:
                    approx = estimate_experience_years_from_text(resume_text)
                    if approx > 0:
                        cleaned.experience_years = approx
                        if not cleaned.experience_summary:
                            cleaned.experience_summary = f"{approx} years experience (from date ranges)"

            # Duplicate detection (normalized): email > phone > (name + corroborator)
            existing = None
            email_n = _norm_email(cleaned.email)
            phone_n = _norm_phone(cleaned.phone)
            name_n = _norm_name(cleaned.name)

            if email_n:
                existing = (
                    db.query(ResumeDB)
                    .filter(ResumeDB.email != None)
                    .filter(func.lower(ResumeDB.email) == email_n)
                    .first()
                )
            if not existing and phone_n:
                # Best-effort DB narrowing using last digits, then normalize in Python.
                last7 = phone_n[-7:] if len(phone_n) >= 7 else phone_n
                cand = (
                    db.query(ResumeDB)
                    .filter(ResumeDB.phone != None)
                    .filter(ResumeDB.phone.ilike(f"%{last7}%"))
                    .limit(200)
                    .all()
                )
                for r0 in cand:
                    if _norm_phone(r0.phone) == phone_n:
                        existing = r0
                        break
            if not existing and name_n:
                # Name alone is not enough; require corroborator (same location or overlapping skill token)
                cand = (
                    db.query(ResumeDB)
                    .filter(ResumeDB.name != None)
                    .filter(func.lower(ResumeDB.name) == name_n)
                    .order_by(ResumeDB.created_at.desc())
                    .limit(200)
                    .all()
                )
                for r0 in cand:
                    loc0 = (getattr(r0, "location", "") or "").strip().lower()
                    loc1 = (cleaned.location or "").strip().lower()
                    if loc0 and loc1 and loc0 == loc1:
                        existing = r0
                        break
                    blob0 = " ".join([(r0.skills or ""), (getattr(r0, "primary_skills", "") or ""), (getattr(r0, "key_skills", "") or "")]).lower()
                    blob1 = " ".join([( ", ".join(cleaned.skills or []) ), (", ".join(cleaned.primary_skills or []) ), (", ".join(cleaned.key_skills or []) )]).lower()
                    if any(t and (t in blob0 and t in blob1) for t in [".net", "python", "react", "java", "sql", "aws", "azure", "docker", "kubernetes"]):
                        existing = r0
                        break

            if existing:
                _log_json("duplicate_detected", {"file": filename, "reason": "email" if email_n and _norm_email(existing.email)==email_n else ("phone" if phone_n and _norm_phone(existing.phone)==phone_n else "name+corroborator"), "existing_id": str(existing.id)})
                results[idx] = {
                    "status": "duplicate",
                    "file": filename,
                    "message": f"Candidate already exists: {existing.name}",
                    "existing_id": str(existing.id),
                }
                continue

            os.makedirs(UPLOAD_DIR, exist_ok=True)
            safe_name = _sanitize_filename(filename or "resume")
            new_name = f"{uuid.uuid4()}_{safe_name}"
            with open(os.path.join(UPLOAD_DIR, new_name), "wb") as fh:
                fh.write(file_bytes)
            resume_link = f"{BASE_URL}/files/{new_name}"

            embedding_text = (
                f"Name: {cleaned.name}\n"
                f"Email: {cleaned.email}\n"
                f"Phone: {cleaned.phone}\n"
                f"Location: {cleaned.location or ''}\n"
                f"Skills: {', '.join(cleaned.skills)}\n"
                f"Experience: {cleaned.experience_years} years\n"
                f"Summary: {cleaned.experience_summary}\n"
                f"Education: {', '.join(cleaned.education)}\n"
                f"Role: {cleaned.role or ''}\n"
                f"Companies: {', '.join(cleaned.companies_worked_at)}\n"
                f"Keywords: {', '.join(cleaned.important_keywords)}"
            ).strip()
            embedding_text = _sanitize_embedding_text(embedding_text)

            if executor:
                vector_id, embedding = await loop.run_in_executor(executor, _encode_embedding, embedding_text)
            else:
                vector_id, embedding = _encode_embedding(embedding_text)

            def _make_record(include_new_cols: bool = True) -> ResumeDB:
                base = dict(
                    name=_sanitize_name(cleaned.name) or cleaned.name or "",
                    email=cleaned.email,
                    phone=cleaned.phone,
                    location=cleaned.location or "",
                    skills=", ".join(cleaned.skills),
                    experience_years=cleaned.experience_years,
                    experience_summary=cleaned.experience_summary,
                    total_experience_years=getattr(cleaned, "total_experience_years", None),
                    experience_level=getattr(cleaned, "experience_level", None),
                    internship_present=getattr(cleaned, "internship_present", False),
                    experience_notes=getattr(cleaned, "experience_notes", None),
                    education=", ".join(cleaned.education),
                    projects=", ".join(cleaned.projects),
                    resume_link=resume_link,
                    source_file=new_name,
                    vector_id=vector_id,
                    source="manual_upload",
                )
                if include_new_cols:
                    base.update(
                        summary=cleaned.summary or None,
                        companies_worked_at=", ".join(cleaned.companies_worked_at) if cleaned.companies_worked_at else None,
                        role=cleaned.role,
                        important_keywords=", ".join(cleaned.important_keywords) if cleaned.important_keywords else None,
                        experience_line=getattr(cleaned, "experience_line", None),
                        experience_tags=", ".join(getattr(cleaned, "experience_tags", []) or []) or None,
                        key_skills=", ".join(getattr(cleaned, "key_skills", []) or []) or None,
                        primary_skills=", ".join(getattr(cleaned, "primary_skills", []) or []) or None,
                        other_skills=", ".join(getattr(cleaned, "other_skills", []) or []) or None,
                        is_shortlisted=False,
                        tags=None,
                        deleted_at=None,
                    )
                return ResumeDB(**base)

            db_record = _make_record(include_new_cols=True)
            db.add(db_record)
            try:
                db.commit()
                db.refresh(db_record)
            except (OperationalError, ProgrammingError) as db_err:
                msg = str(getattr(db_err, "orig", db_err))
                if "column" in msg.lower() and ("does not exist" in msg or "undefined" in msg):
                    db.rollback()
                    db_record = _make_record(include_new_cols=False)
                    db.add(db_record)
                    db.commit()
                    db.refresh(db_record)
                else:
                    raise

            # Multi-storage consistency: Chroma + Excel are best-effort with retries and logs
            chroma_ok = False
            for attempt in range(3):
                try:
                    if executor:
                        await loop.run_in_executor(
                            executor, _chroma_add, vector_id, embedding, embedding_text, new_name, cleaned.name
                        )
                    else:
                        _chroma_add(vector_id, embedding, embedding_text, new_name, cleaned.name)
                    chroma_ok = True
                    break
                except Exception as ch_exc:
                    _log_json(
                        "chroma_add_failed",
                        {
                            "attempt": attempt + 1,
                            "vector_id": vector_id,
                            "resume_id": str(getattr(db_record, "id", "")),
                            "error": str(ch_exc),
                        },
                    )
                    await asyncio.sleep(0.2 * (2 ** attempt))

            excel_ok = False
            for attempt in range(2):
                try:
                    ok = append_resume_to_excel(
                        name=db_record.name or "",
                        email=db_record.email,
                        phone=db_record.phone,
                        skills=db_record.skills or "",
                        experience_years=db_record.experience_years,
                        resume_link=db_record.resume_link,
                        created_at=db_record.created_at,
                    )
                    excel_ok = bool(ok)
                    break
                except Exception as xl_exc:
                    _log_json(
                        "excel_append_failed",
                        {
                            "attempt": attempt + 1,
                            "resume_id": str(getattr(db_record, "id", "")),
                            "error": str(xl_exc),
                        },
                    )
                    await asyncio.sleep(0.1 * (2 ** attempt))

            _log_json(
                "upload_sinks",
                {
                    "resume_id": str(getattr(db_record, "id", "")),
                    "db_ok": True,
                    "chroma_ok": chroma_ok,
                    "excel_ok": excel_ok,
                    "vector_id": vector_id,
                },
            )

            logger.info("Upload complete: %s → %s", filename, cleaned.name)
            results[idx] = {"status": "success", "candidate_name": cleaned.name, "resume_link": resume_link}

        except Exception as exc:
            import traceback
            logger.error("Upload failed for %s: %s\n%s", filename, exc, traceback.format_exc())
            db.rollback()
            results[idx] = {"status": "error", "file": filename, "message": str(exc)}

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Skills-only extraction (no DB write)
# ---------------------------------------------------------------------------

@app.post("/extract/skills", tags=["Resumes"])
async def extract_skills_only(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Extract ONLY skills from a resume file.
    - Does not write anything to the database.
    - Keeps existing analyzer/extraction logic untouched by using the existing helpers.
    """
    try:
        file_bytes = await file.read()
        ext = os.path.splitext(file.filename or "")[-1].lower()
        if ext not in (".pdf", ".docx"):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX are supported.")

        executor = getattr(request.app.state, "executor", None)
        loop = asyncio.get_event_loop()
        if executor:
            resume_text = await loop.run_in_executor(executor, _extract_text_from_bytes, file_bytes, ext, BASE_DIR)
        else:
            resume_text = _extract_text_from_bytes(file_bytes, ext, BASE_DIR)

        if not (resume_text or "").strip():
            raise HTTPException(status_code=400, detail="Text extraction produced no content.")

        primary, other = _derive_clean_skills_from_text(resume_text or "")

        return {
            "ok": True,
            "file": file.filename,
            "primary_skills": primary,
            "other_skills": other,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Skills-only extraction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Resume list & detail
# ---------------------------------------------------------------------------

@app.get("/resumes", tags=["Resumes"])
def list_resumes(
    db: Session = Depends(get_db),
    shortlisted: Optional[bool] = None,
    min_experience: Optional[float] = None,
    max_experience: Optional[float] = None,
    skills: Optional[str] = None,
    added_after: Optional[str] = None,
    added_before: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    company: Optional[str] = None,
    education: Optional[str] = None,
):
    """List resumes (excludes soft-deleted). Supports filters, date range, and sorting."""
    q = db.query(ResumeDB).filter(ResumeDB.deleted_at == None)

    if shortlisted is not None:
        q = q.filter(ResumeDB.is_shortlisted == shortlisted)
    if min_experience is not None:
        q = q.filter(ResumeDB.experience_years >= min_experience)
    if max_experience is not None:
        q = q.filter(ResumeDB.experience_years <= max_experience)
    if added_after:
        try:
            dt = datetime.fromisoformat(added_after.replace("Z", "+00:00"))
            q = q.filter(ResumeDB.created_at >= dt)
        except Exception:
            pass
    if added_before:
        try:
            dt = datetime.fromisoformat(added_before.replace("Z", "+00:00"))
            end_of_day = datetime.combine(dt.date(), time(23, 59, 59, 999999))
            q = q.filter(ResumeDB.created_at <= end_of_day)
        except Exception:
            pass
    if company:
        q = q.filter(ResumeDB.companies_worked_at.ilike(f"%{company}%"))
    if education:
        q = q.filter(ResumeDB.education.ilike(f"%{education}%"))

    # Order by: name, experience, created_at, shortlisted; asc or desc
    order_col = ResumeDB.created_at
    if sort_by == "name":
        order_col = ResumeDB.name
    elif sort_by == "experience":
        order_col = ResumeDB.experience_years
    elif sort_by == "created_at":
        order_col = ResumeDB.created_at
    elif sort_by == "shortlisted":
        order_col = ResumeDB.is_shortlisted
    if sort_order == "asc":
        q = q.order_by(order_col.asc().nulls_last())
    else:
        q = q.order_by(order_col.desc().nulls_last())

    rows = q.all()
    result = [_resume_to_dict(r) for r in rows]

    # Skills filter: case-insensitive, partial match, supports common aliases/variations.
    # Checks: skills, primary_skills, other_skills, key_skills (all stored as CSV/text).
    if skills:
        groups = _expand_skill_query(skills)
        if groups:
            def _skill_blob(r: dict) -> str:
                parts = [
                    " ".join(r.get("skills") or []),
                    " ".join(r.get("primary_skills") or []),
                    " ".join(r.get("other_skills") or []),
                    " ".join(r.get("key_skills") or []),
                ]
                return " ".join(parts).lower()

            def _matches(r: dict) -> bool:
                blob = _skill_blob(r)
                # AND across user tokens; OR across aliases within a token.
                # For plain word tokens (e.g. "java") use word-boundary regex
                # so that "java" does not match "javascript". For tokens that
                # contain punctuation (".net", "c#") fall back to substring.
                for group in groups:
                    found_any = False
                    for a in group:
                        if not a:
                            continue
                        if re.fullmatch(r"[a-z0-9]+", a):
                            # pure alphanumeric → word boundary
                            if re.search(rf"\b{re.escape(a)}\b", blob):
                                found_any = True
                                break
                        else:
                            if a in blob:
                                found_any = True
                                break
                    if not found_any:
                        return False
                return True

            result = [r for r in result if _matches(r)]

    return result


@app.post("/resumes/{resume_id}/reextract-skills", tags=["Resumes"])
def reextract_skills(resume_id: str, db: Session = Depends(get_db)):
    """
    Re-extract skills from the stored source file (PDF/DOCX) and update DB.
    This is useful for older uploads where skills were missed.
    """
    resume = (
        db.query(ResumeDB)
        .filter(ResumeDB.id == resume_id, ResumeDB.deleted_at == None)
        .first()
    )
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    if not resume.source_file:
        raise HTTPException(status_code=400, detail="No source_file stored for this resume")

    file_path = os.path.join(UPLOAD_DIR, resume.source_file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Stored resume file not found on server")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        resume_text = extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        resume_text = extract_text_from_docx(file_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Use full extraction so noise filters apply (prevents education/form text in skills)
    extracted = extract_resume(resume_text)
    skills = extracted.get("skills") or []
    primary_skills = extracted.get("primary_skills") or []
    other_skills = extracted.get("other_skills") or []
    key_skills = extracted.get("key_skills") or []

    if not skills:
        return {"updated": False, "skills": [], "message": "No skills found in resume text."}

    resume.skills = ", ".join(skills)
    resume.key_skills = ", ".join(key_skills) if key_skills else None
    resume.primary_skills = ", ".join(primary_skills) if primary_skills else None
    resume.other_skills = ", ".join(other_skills) if other_skills else None
    db.add(resume)
    db.commit()
    db.refresh(resume)
    return {
        "updated": True,
        "resume": _resume_to_dict(resume),
        "skills": skills,
        "primary_skills": primary_skills,
        "other_skills": other_skills,
    }


@app.post("/resumes/{resume_id}/reextract-experience", tags=["Resumes"])
def reextract_experience(resume_id: str, db: Session = Depends(get_db)):
    """
    Recalculate experience_years from the stored source file by scanning date ranges.
    Useful when a resume didn't explicitly say 'X years' and older parsing missed it.
    """
    resume = (
        db.query(ResumeDB)
        .filter(ResumeDB.id == resume_id, ResumeDB.deleted_at == None)
        .first()
    )
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    if not resume.source_file:
        raise HTTPException(status_code=400, detail="No source_file stored for this resume")

    file_path = os.path.join(UPLOAD_DIR, resume.source_file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Stored resume file not found on server")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        resume_text = extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        resume_text = extract_text_from_docx(file_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Use the conservative calculator from full extraction (excludes internships)
    extracted = extract_resume(resume_text)
    yrs = float(extracted.get("experience_years") or 0.0)
    if not yrs or yrs <= 0:
        return {"updated": False, "experience_years": 0, "message": "Experience cannot be reliably calculated."}

    resume.experience_years = float(yrs)
    resume.total_experience_years = float(extracted.get("total_experience_years") or yrs)
    resume.experience_level = extracted.get("experience_level") or None
    resume.internship_present = bool(extracted.get("internship_present") or False)
    resume.experience_notes = extracted.get("experience_notes") or None
    if not (resume.experience_summary or "").strip():
        resume.experience_summary = extracted.get("experience_summary") or f"{float(yrs)} years experience (from date ranges)"
    db.add(resume)
    db.commit()
    db.refresh(resume)
    return {"updated": True, "resume": _resume_to_dict(resume)}


@app.post("/resumes/{resume_id}/reextract-key-skills", tags=["Resumes"])
def reextract_key_skills(resume_id: str, db: Session = Depends(get_db)):
    """
    Re-extract primary key skills from the stored source file (full extraction).
    Updates key_skills in DB. Use for existing resumes that have no key_skills.
    """
    resume = (
        db.query(ResumeDB)
        .filter(ResumeDB.id == resume_id, ResumeDB.deleted_at == None)
        .first()
    )
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    if not resume.source_file:
        raise HTTPException(status_code=400, detail="No source_file stored for this resume")

    file_path = os.path.join(UPLOAD_DIR, resume.source_file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Stored resume file not found on server")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        resume_text = extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        resume_text = extract_text_from_docx(file_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    extracted = extract_resume(resume_text)
    key_skills = extracted.get("key_skills") or []
    if not key_skills and extracted.get("skills"):
        key_skills = extracted["skills"][:15]

    # Ensure primary/other skills are also refreshed
    primary_skills = extracted.get("primary_skills") or []
    other_skills = extracted.get("other_skills") or []

    # Also refresh name/location when full extraction is run (helps fix older bad headers)
    new_name = (extracted.get("name") or "").strip()
    if new_name and new_name.lower() not in {"unknown", "unknown candidate", "candidate"}:
        resume.name = new_name[:120]
    new_location = (extracted.get("location") or "").strip()
    if new_location and new_location.lower() not in {"unknown", "n/a", "na"}:
        resume.location = new_location[:120]

    # Refresh experience fields from latest extraction logic
    resume.experience_years = float(extracted.get("experience_years") or 0.0)
    resume.total_experience_years = float(extracted.get("total_experience_years") or resume.experience_years or 0.0)
    resume.experience_level = extracted.get("experience_level") or None
    resume.internship_present = bool(extracted.get("internship_present") or False)
    resume.experience_notes = extracted.get("experience_notes") or None

    resume.key_skills = (
        ", ".join(str(s).strip() for s in key_skills if s) if key_skills else None
    )
    resume.primary_skills = (
        ", ".join(str(s).strip() for s in primary_skills if s)
        if primary_skills
        else None
    )
    resume.other_skills = (
        ", ".join(str(s).strip() for s in other_skills if s)
        if other_skills
        else None
    )
    db.add(resume)
    db.commit()
    db.refresh(resume)
    return {
        "updated": True,
        "key_skills": key_skills,
        "primary_skills": primary_skills,
        "other_skills": other_skills,
        "resume": _resume_to_dict(resume),
    }


@app.get("/resumes/compare", tags=["Resumes"])
def compare_resumes(
    ids: str,
    job_description: Optional[str] = None,
    db: Session = Depends(get_db),
):
    id_list = [x.strip() for x in ids.split(",") if x.strip()][:5]
    if not id_list:
        raise HTTPException(status_code=400, detail="ids required")
    rows = db.query(ResumeDB).filter(
        ResumeDB.id.in_(id_list), ResumeDB.deleted_at == None
    ).all()
    result = [_resume_to_dict(r) for r in rows]
    if job_description and result:
        for r in result:
            # Provide richer evidence for fit analysis without changing API shape.
            skills = ", ".join((r.get("skills") or [])[:30])
            key_skills = ", ".join((r.get("key_skills") or [])[:15])
            primary = ", ".join((r.get("primary_skills") or [])[:10])
            tags = ", ".join((r.get("experience_tags") or [])[:15])
            companies = ", ".join((r.get("companies_worked_at") or [])[:8])
            education = ", ".join((r.get("education") or [])[:6])
            context = "\n".join(
                [
                    f"Name: {r.get('name','')}",
                    f"Role: {r.get('role','')}",
                    f"Experience: {r.get('experience_years')} years",
                    f"Primary skills: {primary}",
                    f"Key skills: {key_skills}",
                    f"Skills: {skills}",
                    f"Experience tags: {tags}",
                    f"Companies: {companies}",
                    f"Education: {education}",
                    f"Summary: {r.get('summary','')}",
                    f"Experience summary: {r.get('experience_summary','')}",
                ]
            ).strip()
            fit = analyze_fit(job_description, context)
            r["fit_score"] = fit.get("score_1_10")
            r["fit_summary"] = fit.get("fit_summary", "")
    return result


@app.get("/resumes/by-field", tags=["Resumes"])
def list_resumes_by_field(
    field: str,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    """
    Fast field/domain filter for Compare UI.
    Uses ResumeDB.primary_skills (CSV text) with case-insensitive partial matching.
    """
    field_norm = (field or "").strip().lower()
    if not field_norm:
        raise HTTPException(status_code=400, detail="field required")

    # Map UI field → canonical token we can expand
    field_to_token = {
        ".net": ".net",
        "dotnet": ".net",
        "python": "python",
        "react": "react",
        "seo": "seo",
        "devops": "devops",
        "java": "java",
    }
    token = field_to_token.get(field_norm, field_norm)

    # Reuse existing alias expansion used by skills filter
    groups = _expand_skill_query(token)
    aliases = groups[0] if groups else [token]

    conds = [ResumeDB.primary_skills.ilike(f"%{a}%") for a in aliases if a]
    q = db.query(ResumeDB).filter(ResumeDB.deleted_at == None)
    if conds:
        q = q.filter(or_(*conds))

    rows = q.order_by(ResumeDB.created_at.desc()).limit(min(max(limit, 1), 500)).all()

    # Minimal payload for dropdown performance
    return [
        {
            "id": str(r.id),
            "name": r.name or "",
            "role": _safe_get(r, "role", "") or "",
            "experience_years": r.experience_years,
            "primary_skills": [s.strip() for s in (_safe_get(r, "primary_skills") or "").split(",") if s.strip()],
            "resume_link": r.resume_link,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@app.get("/resumes/{resume_id}/json", tags=["Resumes"])
def get_resume_json(resume_id: str, db: Session = Depends(get_db)):
    """Return full resume data as JSON. Used by the frontend 'Skills' dialog."""
    r = db.query(ResumeDB).filter(
        ResumeDB.id == resume_id, ResumeDB.deleted_at == None
    ).first()
    if not r:
        raise HTTPException(status_code=404, detail="Resume not found")
    return _resume_to_dict(r)


@app.get("/resume/{resume_id}", tags=["Resumes"])
async def get_resume(resume_id: str, db: Session = Depends(get_db)):
    resume = db.query(ResumeDB).filter(ResumeDB.id == resume_id).first()
    if resume and resume.source_file:
        file_path = os.path.join(UPLOAD_DIR, resume.source_file)
        if os.path.exists(file_path):
            return FileResponse(
                file_path,
                filename=resume.source_file,
                media_type="application/pdf",
            )
    raise HTTPException(status_code=404, detail="Resume file not found")


@app.get("/files/{filename}", tags=["Files"])
def serve_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, media_type="application/pdf")


# ---------------------------------------------------------------------------
# Shortlist
# ---------------------------------------------------------------------------

@app.post("/resumes/{resume_id}/shortlist", tags=["Resumes"])
def shortlist_resume(resume_id: str, db: Session = Depends(get_db)):
    r = db.query(ResumeDB).filter(
        ResumeDB.id == resume_id, ResumeDB.deleted_at == None
    ).first()
    if not r:
        raise HTTPException(status_code=404, detail="Resume not found")
    r.is_shortlisted = True
    db.commit()
    return {"status": "shortlisted", "resume_id": resume_id}


@app.delete("/resumes/{resume_id}/shortlist", tags=["Resumes"])
def unshortlist_resume(resume_id: str, db: Session = Depends(get_db)):
    r = db.query(ResumeDB).filter(ResumeDB.id == resume_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Resume not found")
    r.is_shortlisted = False
    db.commit()
    return {"status": "removed from shortlist", "resume_id": resume_id}


# ---------------------------------------------------------------------------
# Bulk actions
# ---------------------------------------------------------------------------


@app.delete("/resumes/bulk", tags=["Resumes"])
def bulk_delete(body: BulkDeleteRequest, db: Session = Depends(get_db)):
    for rid in body.resume_ids:
        r = db.query(ResumeDB).filter(ResumeDB.id == rid).first()
        if not r:
            continue
        # Remove related notes
        db.query(CandidateNoteDB).filter(CandidateNoteDB.resume_id == rid).delete(
            synchronize_session=False
        )
        # Remove vector from ChromaDB if present
        if getattr(r, "vector_id", None):
            try:
                embedding_manager.collection.delete(ids=[r.vector_id])
            except Exception:
                pass
        # Hard delete candidate row so it is fully removed from the database
        db.delete(r)
    db.commit()
    return {"status": "ok", "deleted_count": len(body.resume_ids)}


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

@app.get("/resumes/{resume_id}/notes", tags=["Resumes"])
def get_notes(resume_id: str, db: Session = Depends(get_db)):
    r = db.query(ResumeDB).filter(
        ResumeDB.id == resume_id, ResumeDB.deleted_at == None
    ).first()
    if not r:
        raise HTTPException(status_code=404, detail="Resume not found")
    notes = (
        db.query(CandidateNoteDB)
        .filter(CandidateNoteDB.resume_id == resume_id)
        .order_by(CandidateNoteDB.created_at.desc())
        .all()
    )
    return [
        {
            "id": str(n.id),
            "note": n.note,
            "status": n.status,
            "created_at": n.created_at.isoformat() if n.created_at else None,
        }
        for n in notes
    ]


@app.post("/resumes/{resume_id}/notes", tags=["Resumes"])
def add_note(resume_id: str, body: NoteCreate, db: Session = Depends(get_db)):
    r = db.query(ResumeDB).filter(
        ResumeDB.id == resume_id, ResumeDB.deleted_at == None
    ).first()
    if not r:
        raise HTTPException(status_code=404, detail="Resume not found")
    n = CandidateNoteDB(resume_id=r.id, note=body.note, status=body.status)
    db.add(n)
    db.commit()
    return {"id": str(n.id), "note": body.note, "status": body.status}


@app.delete("/resumes/{resume_id}/notes/{note_id}", tags=["Resumes"])
def delete_note(resume_id: str, note_id: str, db: Session = Depends(get_db)):
    n = (
        db.query(CandidateNoteDB)
        .filter(
            CandidateNoteDB.id == note_id,
            CandidateNoteDB.resume_id == resume_id,
        )
        .first()
    )
    if not n:
        raise HTTPException(status_code=404, detail="Note not found")
    db.delete(n)
    db.commit()
    return {"deleted": True, "note_id": note_id}



# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

# Skill/keyword groups for candidate search (query → DB filter)
_SEARCH_SKILL_GROUPS = {
    "dotnet": [".net", "dotnet", "asp.net", "c#", "c sharp", "asp.net core", "entity framework"],
    "python": ["python", "django", "flask", "fastapi", "pandas", "numpy"],
    "react": ["react", "reactjs", "react.js", "next.js", "nextjs"],
    "java": ["java", "spring", "spring boot", "jvm"],
    "frontend": ["frontend", "javascript", "typescript", "react", "angular", "vue", "html", "css"],
    "devops": ["devops", "docker", "kubernetes", "aws", "azure", "ci/cd"],
}

def _extract_search_intent(question: str) -> tuple[list[str], float]:
    """
    Detect if the question is a candidate search and extract skill keywords and min experience.
    Returns (list of search terms for DB ilike, min_experience in years or 0).
    """
    q = (question or "").lower().strip()
    search_terms: list[str] = []
    min_exp = 0.0

    # Min experience: e.g. "3+ years", "1 year", "5 years experience"
    exp_match = re.search(r"(\d+)\s*\+\s*years?", q)
    if exp_match:
        min_exp = float(exp_match.group(1))
    else:
        exp_match = re.search(r"(?:with|having)?\s*(\d+)\s*years?\s*(?:experience|exp)?", q)
        if exp_match:
            min_exp = float(exp_match.group(1))

    for key, tokens in _SEARCH_SKILL_GROUPS.items():
        matched = [t for t in tokens if t in q]
        if matched:
            # Keep the query narrow. If the user asks for TypeScript, do not
            # expand into the whole frontend bucket (React/Angular/Vue/HTML/CSS).
            if key == "frontend":
                if "typescript" in matched:
                    search_terms.extend(["typescript", "ts"])
                elif "javascript" in matched:
                    search_terms.extend(["javascript", "js"])
                elif "react" in matched:
                    search_terms.extend(["react", "reactjs", "react.js", "next.js", "nextjs"])
                elif "angular" in matched:
                    search_terms.extend(["angular"])
                elif "vue" in matched:
                    search_terms.extend(["vue"])
                elif "html" in matched:
                    search_terms.extend(["html"])
                elif "css" in matched:
                    search_terms.extend(["css"])
                elif "frontend" in matched:
                    search_terms.extend(["frontend"])
            else:
                search_terms.extend(matched)
    # Also single-word tech mentions
    for word in ["developer", "developers", "candidates", "show", "find", "list"]:
        q = q.replace(word, " ")
    words = re.findall(r"[a-z0-9.#+]+", q)
    for w in words:
        if len(w) >= 2 and w not in search_terms and w not in {"or", "and", "with", "year", "years", "experience"}:
            if any(tech in w or w in tech for tech in [".net", "python", "react", "java", "node", "sql", "vue", "angular"]):
                search_terms.append(w)
    search_terms = list(dict.fromkeys(search_terms))[:15]  # dedupe, cap
    return search_terms, min_exp


def _db_search_candidates(
    db: Session,
    skill_terms: list[str],
    min_experience: float,
    limit: int = 50,
) -> list:
    """Return resume ORM objects matching any of the skill terms and optional min experience."""
    if not skill_terms and min_experience <= 0:
        return []
    q = db.query(ResumeDB).filter(ResumeDB.deleted_at == None)
    if min_experience > 0:
        q = q.filter(ResumeDB.experience_years >= min_experience)
    rows = q.order_by(ResumeDB.created_at.desc()).limit(limit * 2).all()  # fetch extra then filter
    if not skill_terms:
        return rows[:limit]
    combined = []
    for r in rows:
        # For skill lookups, match only the explicit skill columns.
        # Projects / summary text can mention technologies in narrative form
        # and should not make a candidate appear as a skill match.
        blob = " ".join([
            (r.skills or ""),
            (_safe_get(r, "primary_skills") or ""),
            (_safe_get(r, "key_skills") or ""),
        ]).lower()
        if any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", blob) for term in skill_terms):
            combined.append(r)
    return combined[:limit]


def _apply_skill_filter(question: str, resumes: list) -> list:
    """Hard-filter resumes for technology queries (e.g. .NET, Python, React)."""
    q = (question or "").lower()
    must_tokens: list[str] = []
    for _, tokens in _SEARCH_SKILL_GROUPS.items():
        if any(t in q for t in tokens):
            must_tokens.extend(tokens)
    if any(t in q for t in ["frontend", "developer", "developers"]):
        must_tokens.extend(["frontend", "javascript", "react", "angular", "vue", "html", "css"])
    must_tokens = list(dict.fromkeys(must_tokens))[:20]

    if not must_tokens:
        return resumes

    filtered = []
    for r in resumes:
        blob = " ".join([
            str(r.skills or ""),
            str(_safe_get(r, "primary_skills") or ""),
            str(_safe_get(r, "key_skills") or ""),
        ]).lower()
        if any(re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", blob) for tok in must_tokens):
            filtered.append(r)
    return filtered if filtered else resumes


def _resume_row_for_chat(r: ResumeDB) -> dict:
    """One row for global chat best_matches table (same shape as resume table for consistent UI)."""
    primary = [
        s.strip() for s in (_safe_get(r, "primary_skills") or "").split(",") if s.strip()
    ]
    return {
        "id": str(r.id),
        "name": _sanitize_name(r.name) or "",
        "email": r.email or "",
        "phone": r.phone or "",
        "primary_skills": ", ".join(primary) if primary else (r.skills or "")[:200],
        "experience_years": r.experience_years,
        "experience": r.experience_years,
        "companies_worked_at": ", ".join([
            s.strip() for s in (_safe_get(r, "companies_worked_at") or "").split(",") if s.strip()
        ]),
        "resume_link": r.resume_link,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "is_shortlisted": getattr(r, "is_shortlisted", False) or False,
        "skills": r.skills,
    }


def _encode_question(question: str):
    """Blocking: encode question for vector search. Run in thread pool."""
    q_vec = embedding_manager.embedder.encode(question)
    return q_vec[0] if len(q_vec.shape) == 2 else q_vec


def _safe_chroma_query(query_embeddings, n_results: int) -> dict:
    """
    Best-effort Chroma query wrapper.
    If the collection has an internal metadata/segment problem, return an empty
    result instead of failing the whole chat request.
    """
    try:
        collection = getattr(embedding_manager, "collection", None)
        if collection is None:
            return {}
        return collection.query(query_embeddings=query_embeddings, n_results=n_results)
    except Exception as exc:
        logger.warning("Chroma query skipped after failure: %s", exc)
        return {}


def _safe_chroma_count() -> int:
    try:
        collection = getattr(embedding_manager, "collection", None)
        if collection is None:
            return 0
        return int(collection.count())
    except Exception as exc:
        logger.warning("Chroma count skipped after failure: %s", exc)
        return 0


@app.post("/chat", tags=["Chat"])
async def chat(request: Request, payload: dict, db: Session = Depends(get_db)):
    question = (payload.get("question") or "").strip()
    if not question:
        return {"answer": "Please ask a question."}

    # Detect candidate-search intent and run DB search for multiple matches
    skill_terms, min_exp = _extract_search_intent(question)
    is_search_like = bool(skill_terms or min_exp > 0) or any(
        x in question.lower() for x in ["show", "find", "list", "candidates", "developers", "who has"]
    )

    executor = getattr(request.app.state, "executor", None)
    loop = asyncio.get_event_loop()

    # Keep chat independent from Chroma/embedding persistence. Search-like
    # questions are answered from stored resume rows only.
    seen_ids = set()
    merged: list = []
    if is_search_like and (skill_terms or min_exp > 0):
        db_resumes = _db_search_candidates(db, skill_terms, min_exp, limit=50)
        for r in db_resumes:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                merged.append(r)
    resumes = _apply_skill_filter(question, merged) if merged else merged
    if not resumes and is_search_like and skill_terms:
        resumes = _db_search_candidates(db, skill_terms, min_exp, limit=50)
    if not resumes and is_search_like:
        resumes = db.query(ResumeDB).filter(ResumeDB.deleted_at == None).order_by(ResumeDB.created_at.desc()).limit(50).all()

    if not resumes:
        return {"answer": "No matching candidates found.", "best_matches": []}

    if is_search_like and resumes:
        terms = [t.strip().lower() for t in (skill_terms or []) if t and t.strip()][:25]

        def _rank_score(r: ResumeDB) -> float:
            primary = (getattr(r, "primary_skills", "") or "").lower()
            key = (getattr(r, "key_skills", "") or "").lower()
            skills_blob = (r.skills or "").lower()
            projects = (r.projects or "").lower()
            exp_sum = (r.experience_summary or "").lower()
            score = 0.0
            for t in terms:
                if not t:
                    continue
                if t in primary or t in key:
                    score += 10.0
                elif t in skills_blob:
                    score += 6.0
                elif t in projects or t in exp_sum:
                    score += 2.0
            try:
                y = float(r.experience_years or 0.0)
            except Exception:
                y = 0.0
            if min_exp and y >= float(min_exp):
                score += 8.0
            # Penalty for tool-dump strings (lots of commas / extremely long skills blobs)
            if len(skills_blob) > 900 or skills_blob.count(",") > 60:
                score -= 4.0
            return score

        resumes = sorted(resumes, key=_rank_score, reverse=True)

    # Return table-ready rows for all matched candidates (up to 50)
    best_matches = [_resume_row_for_chat(r) for r in resumes[:50]]

    # For search-like questions, deterministic summary; otherwise run LLM in thread pool
    if is_search_like:
        top_n = min(10, len(best_matches))
        lines = [f"Found {len(best_matches)} matching candidate(s). Showing top {top_n}:"]
        for i, m in enumerate(best_matches[:top_n], 1):
            nm = (m.get("name") or "").strip() or (m.get("email") or "").strip() or f"Candidate {i}"
            exp = m.get("experience_years")
            try:
                exp_s = f"{float(exp):g} yr" if exp is not None else "—"
            except Exception:
                exp_s = "—"
            skills = (m.get("primary_skills") or "").strip() or "—"
            lines.append(f"{i}. {nm} • Exp: {exp_s} • Skills: {skills}")
        answer = "\n".join(lines)
    else:
        context = ""
        for i, r in enumerate(resumes[:20], 1):
            context += (
                f"\nCandidate {i}\n"
                f"Name: {_sanitize_name(r.name)}\nEmail: {r.email}\nPhone: {r.phone}\n"
                f"Skills: {r.skills}\nExperience: {r.experience_years} years\n"
                f"Summary: {r.experience_summary}\nEducation: {r.education}\n"
                f"Projects: {r.projects}\nResume Link: {r.resume_link}\n---\n"
            )
        if executor:
            answer = await loop.run_in_executor(executor, chatbot_answer, question, context)
        else:
            answer = chatbot_answer(question, context)

    return {
        "answer": answer,
        "best_matches": best_matches,
    }


@app.post("/resume/{resume_id}/chat", tags=["Chat"])
async def chat_resume(request: Request, resume_id: str, payload: dict, db: Session = Depends(get_db)):
    """Ask a question scoped to a single resume."""
    resume = db.query(ResumeDB).filter(ResumeDB.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    question = (payload.get("question") or "").strip()
    if not question:
        return {"answer": "Please provide a question."}
    context = (
        f"Name: {resume.name}\nEmail: {resume.email}\nPhone: {resume.phone}\n"
        f"Skills: {resume.skills}\nExperience: {resume.experience_years} years\n"
        f"Summary: {resume.experience_summary}\nEducation: {resume.education}\n"
        f"Projects: {resume.projects}\n"
        f"Role: {_safe_get(resume, 'role', '') or ''}\n"
        f"Companies: {_safe_get(resume, 'companies_worked_at', '') or ''}\n"
        f"Keywords: {_safe_get(resume, 'important_keywords', '') or ''}"
    )
    executor = getattr(request.app.state, "executor", None)
    if executor:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(executor, chatbot_answer, question, context)
    else:
        answer = chatbot_answer(question, context)
    return {"answer": answer, "resume_id": resume_id, "candidate_name": resume.name}



# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

@app.post("/resumes/check-duplicates", tags=["Resumes"])
def check_duplicates(payload: dict, db: Session = Depends(get_db)):
    resume_id = payload.get("resume_id")
    if resume_id:
        r = db.query(ResumeDB).filter(
            ResumeDB.id == resume_id, ResumeDB.deleted_at == None
        ).first()
        if not r:
            raise HTTPException(status_code=404, detail="Resume not found")
        candidates = [r]
    else:
        candidates = db.query(ResumeDB).filter(ResumeDB.deleted_at == None).all()

    all_resumes = db.query(ResumeDB).filter(ResumeDB.deleted_at == None).all()
    results = []
    for c in candidates:
        dupes = []
        c_email = _norm_email(c.email)
        c_phone = _norm_phone(c.phone)
        c_name = _norm_name(c.name)
        c_loc = (getattr(c, "location", "") or "").strip().lower()
        c_blob = " ".join(
            [
                (c.skills or ""),
                (getattr(c, "primary_skills", "") or ""),
                (getattr(c, "key_skills", "") or ""),
            ]
        ).lower()
        for other in all_resumes:
            if str(other.id) == str(c.id):
                continue
            o_email = _norm_email(other.email)
            o_phone = _norm_phone(other.phone)
            o_name = _norm_name(other.name)
            o_loc = (getattr(other, "location", "") or "").strip().lower()
            o_blob = " ".join(
                [
                    (other.skills or ""),
                    (getattr(other, "primary_skills", "") or ""),
                    (getattr(other, "key_skills", "") or ""),
                ]
            ).lower()

            reason = ""
            score = 0
            if c_email and o_email and c_email == o_email:
                reason = "email"
                score = 100
            elif c_phone and o_phone and c_phone == o_phone:
                reason = "phone"
                score = 100
            elif c_name and o_name and c_name == o_name:
                # Require corroborator
                corroborator = False
                if c_loc and o_loc and c_loc == o_loc:
                    corroborator = True
                if any(t and (t in c_blob and t in o_blob) for t in [".net", "python", "react", "java", "sql", "aws", "azure", "docker", "kubernetes"]):
                    corroborator = True
                if corroborator:
                    reason = "name+corroborator"
                    score = 60

            if score:
                dupes.append({"resume_id": str(other.id), "name": other.name, "score": min(100, score)})
        if dupes:
            results.append({"resume_id": str(c.id), "name": c.name, "possible_duplicates": dupes})
    return {"results": results}





# ---------------------------------------------------------------------------
# Bulk ZIP upload
# ---------------------------------------------------------------------------

@app.post("/upload/bulk", tags=["Resumes"])
async def upload_bulk_zip(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files accepted")

    buf = BytesIO(await file.read())
    results = []
    try:
        with zipfile.ZipFile(buf, "r") as zf:
            for name in zf.namelist():
                if name.startswith("__") or "/." in name:
                    continue
                ext = os.path.splitext(name)[-1].lower()
                if ext not in (".pdf", ".docx"):
                    results.append({"file": name, "status": "skipped", "message": "Not PDF/DOCX"})
                    continue
                try:
                    data = zf.read(name)
                    uf = UploadFile(filename=os.path.basename(name), file=BytesIO(data))
                    # Reuse single-file upload logic, passing through the same Request + DB session
                    upload_result = await upload_resume(request=request, files=[uf], db=db)
                    first = upload_result[0] if upload_result else {}
                    results.append({
                        "file": name,
                        "status": first.get("status", "unknown"),
                        "message": first.get("message", ""),
                    })
                except Exception as exc:
                    results.append({"file": name, "status": "error", "message": str(exc)})
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    return results
