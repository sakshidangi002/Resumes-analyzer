"""Interview question PDFs, grouped by hiring position.

HR/Admin upload a PDF of interview questions for a hiring position (e.g. ".NET
Developer"). Each row is one uploaded document with its metadata; the PDF itself
lives on disk under backend/data/interview_questions and is referenced by
pdf_path. Titles are unique per position so the same question set is not
uploaded twice.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint
from app.db.base_class import Base

from app.core.datetime_utils import get_ist_now


class InterviewQuestion(Base):
    __tablename__ = "interview_questions"
    __table_args__ = (
        UniqueConstraint("position", "title", name="uq_interview_position_title"),
    )

    id = Column(Integer, primary_key=True, index=True)
    position = Column(String(100), nullable=False, index=True)  # hiring position
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    pdf_path = Column(String(500), nullable=False)   # server-side stored path
    pdf_name = Column(String(255), nullable=False)   # original filename (for download)
    uploaded_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    uploaded_by_name = Column(String(150), nullable=True)
    created_at = Column(DateTime, default=get_ist_now)
    updated_at = Column(DateTime, default=get_ist_now, onupdate=get_ist_now)
