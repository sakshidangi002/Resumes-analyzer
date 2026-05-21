"""
PostgreSQL database session. Uses URL from config (postgresql+psycopg2).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import get_settings
from app.db.base_class import Base

# Import all models so Base.metadata knows them
from app.models import *  # noqa: F401, F403

engine = create_engine(
    get_settings().database_url,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
