import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# Strictly import from the backend package to avoid root directory conflicts
from backend.main import extract_text_from_pdf, extract_text_from_docx, extract_location_from_text

Base = declarative_base()

class ResumeDB(Base):
    __tablename__ = "resumes"
    id = Column(String(36), primary_key=True)
    name = Column(String(120))
    location = Column(Text)
    source_file = Column(String(255))
    deleted_at = Column(DateTime, nullable=True)

# Force UPLOAD_DIR to point to backend/uploads
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "backend", "uploads")

def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:root@localhost:5432/Attendance_system")
    print(f"Connecting to database at {db_url}...")
    
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    resumes = session.query(ResumeDB).filter(ResumeDB.deleted_at == None).all()
    print(f"Found {len(resumes)} resumes in database.")

    updated_count = 0
    for r in resumes:
        if not r.source_file:
            continue
        file_path = os.path.join(UPLOAD_DIR, r.source_file)
        if not os.path.exists(file_path):
            print(f"Skipping {r.name}: file {r.source_file} not found in {UPLOAD_DIR}.")
            continue

        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                text = extract_text_from_pdf(file_path)
            elif ext in (".docx", ".doc"):
                text = extract_text_from_docx(file_path)
            else:
                continue

            # Drop symbols/heavy footnote lines just like in main.py
            lines = []
            for s in text.splitlines():
                s = s.strip()
                if not s:
                    continue
                sym = sum(1 for ch in s if not (ch.isalnum() or ch.isspace()))
                if len(s) > 12 and (sym / max(1, len(s))) > 0.45:
                    continue
                lines.append(s)
            clean_text = "\n".join(lines)

            # Use the newly updated robust location extraction
            new_loc = extract_location_from_text(clean_text)
            
            if new_loc != r.location:
                print(f"Updating {r.name}: '{r.location}' -> '{new_loc}'")
                r.location = new_loc
                updated_count += 1
        except Exception as e:
            print(f"Error processing {r.name}: {e}")

    session.commit()
    print(f"Migration completed! Updated {updated_count} resumes.")
    session.close()

if __name__ == "__main__":
    main()
