"""
Re-run name extraction for all resumes in the DB using the latest rules
(email reconciliation + rejection of Contact/Bootstrap/etc.).
"""
import os

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.main import (
    extract_resume,
    extract_text_from_docx,
    extract_text_from_pdf,
    reconcile_name_with_email,
)

Base = declarative_base()


class ResumeDB(Base):
    __tablename__ = "resumes"
    id = Column(String(36), primary_key=True)
    name = Column(String(120))
    email = Column(String(120))
    source_file = Column(String(255))
    deleted_at = Column(DateTime, nullable=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "backend", "uploads")


def main() -> None:
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("Set DATABASE_URL in .env")

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    resumes = session.query(ResumeDB).filter(ResumeDB.deleted_at.is_(None)).all()
    print(f"Total candidates: {len(resumes)}")

    updated = 0
    for r in resumes:
        old_name = (r.name or "").strip()
        email = (r.email or "").strip()
        new_name = ""

        if r.source_file:
            file_path = os.path.join(UPLOAD_DIR, r.source_file)
            ext = os.path.splitext(file_path)[1].lower() if os.path.exists(file_path) else ""
            if ext == ".pdf":
                text = extract_text_from_pdf(file_path)
            elif ext in (".docx", ".doc"):
                text = extract_text_from_docx(file_path)
            else:
                text = ""
            if text:
                extracted = extract_resume(text)
                new_name = (extracted.get("name") or "").strip()
                email = email or (extracted.get("email") or "").strip()

        if not new_name and email:
            new_name, _ = reconcile_name_with_email(old_name, email)

        if new_name and new_name != old_name and new_name.lower() not in {"unknown", "unknown candidate"}:
            print(f"  {old_name!r} -> {new_name!r}  ({email})")
            r.name = new_name[:120]
            updated += 1

    session.commit()
    session.close()
    print(f"Done. Updated {updated} name(s).")


if __name__ == "__main__":
    main()
