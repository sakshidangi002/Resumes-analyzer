"""
create_cameras_table.py
=======================
Safe one-time migration: creates the 'cameras' table in PostgreSQL
if it does not already exist.

Run with:
    cd "Attendance Management/backend"
    .\venv\Scripts\python.exe scripts\create_cameras_table.py
"""
import os
import sys

# Allow running from the backend directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import inspect, text
from app.db.session import engine
from app.models.camera import CameraConfig  # noqa: ensure model is registered
from app.db.base_class import Base

print("=" * 60)
print("Camera Table Migration")
print("=" * 60)

inspector = inspect(engine)
existing_tables = inspector.get_table_names()

if "cameras" in existing_tables:
    print("[OK] Table 'cameras' already exists – no action needed.")
    # Still check/add any new columns
    existing_cols = {c["name"] for c in inspector.get_columns("cameras")}
    desired_cols = {
        "camera_purpose": "VARCHAR(10) NOT NULL DEFAULT 'IN'",
        "location": "VARCHAR(200)",
        "source_type": "VARCHAR(20) NOT NULL DEFAULT 'rtsp'",
        "interval_sec": "FLOAT NOT NULL DEFAULT 2.0",
        "enabled": "BOOLEAN NOT NULL DEFAULT FALSE",
        "threshold": "FLOAT NOT NULL DEFAULT 0.45",
        "updated_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
    }
    with engine.begin() as conn:
        for col, definition in desired_cols.items():
            if col not in existing_cols:
                print(f"  Adding missing column: {col}")
                conn.execute(text(f"ALTER TABLE cameras ADD COLUMN IF NOT EXISTS {col} {definition}"))
                print(f"  [OK] Column '{col}' added.")
            else:
                print(f"  [OK] Column '{col}' already exists.")
else:
    print("Creating table 'cameras'...")
    Base.metadata.create_all(engine, tables=[CameraConfig.__table__])
    print("[OK] Table 'cameras' created successfully.")

print()
print("Migration complete.")
print("=" * 60)
