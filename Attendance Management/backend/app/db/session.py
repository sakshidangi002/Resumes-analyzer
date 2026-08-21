"""
PostgreSQL database session. Uses URL from config (postgresql+psycopg2).
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import get_settings

# Import all models so Base.metadata knows them
from app.models import *  # noqa: F401, F403


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


# Pool sizing matters here because the database is remote (measured ~29 ms per
# round trip), so a connection is expensive to establish and worth keeping.
# The default pool of 5+10 was too small once several cameras and the
# scheduler share the app with live requests -- callers then block waiting for
# a free connection on top of the network latency.
#
# pool_pre_ping issues an extra SELECT 1 on every checkout, which costs a full
# round trip on a link this slow. It stays ON by default because dropping a
# stale WAN connection mid-request is worse than the latency, but it can be
# turned off with DB_PRE_PING=0 once the DB is moved onto the same network.
engine = create_engine(
    get_settings().database_url,
    pool_pre_ping=(os.getenv("DB_PRE_PING", "1").strip().lower() not in ("0", "false", "no")),
    pool_size=_int_env("DB_POOL_SIZE", 20),
    max_overflow=_int_env("DB_MAX_OVERFLOW", 20),
    pool_recycle=_int_env("DB_POOL_RECYCLE", 1800),
    pool_timeout=_int_env("DB_POOL_TIMEOUT", 30),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
