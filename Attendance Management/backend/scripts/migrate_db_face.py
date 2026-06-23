import sys
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import engine

def migrate():
    print("Connecting to database...")
    with engine.connect() as conn:
        print("Checking/Adding photo_path column...")
        try:
            conn.execute(text("ALTER TABLE employees ADD COLUMN IF NOT EXISTS photo_path VARCHAR(500)"))
            print("photo_path column added/verified.")
        except Exception as e:
            print(f"photo_path column error: {e}")
        
        print("Checking/Adding embedding column...")
        try:
            conn.execute(text("ALTER TABLE employees ADD COLUMN IF NOT EXISTS embedding BYTEA"))
            print("embedding column added/verified.")
        except Exception as e:
            print(f"embedding column error: {e}")
            
        print("Checking/Adding sample_count column...")
        try:
            conn.execute(text("ALTER TABLE employees ADD COLUMN IF NOT EXISTS sample_count INTEGER DEFAULT 0"))
            print("sample_count column added/verified.")
        except Exception as e:
            print(f"sample_count column error: {e}")
            
        conn.commit()
    print("Database schema migration complete.")

if __name__ == "__main__":
    migrate()
