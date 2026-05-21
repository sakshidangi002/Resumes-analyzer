"""
One-time migration: add new columns to resumes and create candidate_notes table.
Run from project root: python -m backend.migrate_db
Or from backend: python migrate_db.py
Uses PostgreSQL-compatible ALTER TABLE ... ADD COLUMN IF NOT EXISTS.
"""
import os
import sys

def run():
    try:
        from sqlalchemy import create_engine, text
        from dotenv import load_dotenv
        load_dotenv()
        url = os.getenv("DATABASE_URL", "postgresql://postgres:d5x8JGZ%40CH5td7X@43.205.127.72:5432/Resume_analyzer")
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS one_liner VARCHAR(500);"))
            conn.execute(text("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS is_shortlisted BOOLEAN DEFAULT FALSE;"))
            conn.execute(text("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS tags TEXT;"))
            conn.execute(text("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP;"))
            conn.execute(text("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS source VARCHAR(50);"))
            conn.execute(text("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS experience_line TEXT;"))
            conn.execute(text("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS experience_tags TEXT;"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS candidate_notes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    resume_id UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
                    note TEXT NOT NULL,
                    status VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by VARCHAR(100)
                );
            """))
            conn.commit()
        print("Resumes table columns updated. Candidate_notes table ensured.")
    except Exception as e:
        print("Migration error:", e)
        sys.exit(1)

if __name__ == "__main__":
    run()
