"""
Create all tables in the database from SQLAlchemy models.
Run from backend dir: python -m scripts.create_tables
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import engine
from app.db.base_class import Base
from app.models import *  # noqa: F401, F403

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")
