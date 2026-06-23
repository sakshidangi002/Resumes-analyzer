from __future__ import annotations

import io
from pathlib import Path
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models import Base

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(get_settings().database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db() -> None:
    ensure_directories()
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def embedding_to_blob(embedding: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(embedding, dtype=np.float32))
    return buffer.getvalue()

def blob_to_embedding(blob: bytes) -> np.ndarray:
    buffer = io.BytesIO(blob)
    buffer.seek(0)
    return np.load(buffer, allow_pickle=False)

import contextlib

@contextlib.contextmanager
def db_cursor():
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        
        class CursorWrapper:
            def __init__(self, c):
                self._c = c
            def execute(self, sql, params=()):
                if engine.dialect.name == "postgresql" and "?" in sql:
                    sql = sql.replace("?", "%s")
                self._c.execute(sql, params)
                return self
            def fetchall(self):
                rows = self._c.fetchall()
                if rows and not isinstance(rows[0], dict) and hasattr(self._c, "description"):
                    cols = [d[0] for d in self._c.description]
                    return [dict(zip(cols, row)) for row in rows]
                return rows
            def fetchone(self):
                row = self._c.fetchone()
                if row and not isinstance(row, dict) and hasattr(self._c, "description"):
                    cols = [d[0] for d in self._c.description]
                    return dict(zip(cols, row))
                return row
            
        yield conn, CursorWrapper(cur)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def get_app_meta(key: str) -> str | None:
    with db_cursor() as (_, cur):
        try:
            row = cur.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None
        except Exception:
            return None

def set_app_meta(key: str, value: str) -> None:
    with db_cursor() as (_, cur):
        cur.execute(
            """
            INSERT INTO app_meta (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value
            """,
            (key, value)
        )
