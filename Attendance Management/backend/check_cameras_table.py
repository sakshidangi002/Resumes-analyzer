"""Check existing cameras table structure"""
from app.db.session import SessionLocal
from sqlalchemy import inspect, text

with SessionLocal() as db:
    inspector = inspect(db.bind)
    columns = inspector.get_columns('cameras')
    print("Existing columns in cameras table:")
    for col in columns:
        print(f"  {col['name']}: {col['type']} (nullable={col['nullable']})")
