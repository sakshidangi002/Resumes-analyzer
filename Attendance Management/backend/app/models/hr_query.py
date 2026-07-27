"""Employee ↔ HR query system (lightweight, non-realtime).

An employee raises a query (subject + message); HR replies and moves it through
OPEN → PENDING → RESOLVED. Threaded replies keep the full conversation history.
"""
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.db.base_class import Base

from app.core.datetime_utils import get_ist_now


class HRQuery(Base):
    __tablename__ = "hr_queries"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)  # sender
    subject = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    category = Column(String(50), nullable=True)  # Attendance, Document, Salary, Leave, General
    status = Column(String(20), nullable=False, default="OPEN", index=True)  # OPEN, PENDING, RESOLVED
    created_at = Column(DateTime, default=get_ist_now, index=True)
    updated_at = Column(DateTime, default=get_ist_now, onupdate=get_ist_now)

    employee = relationship("Employee", backref="hr_queries")
    replies = relationship(
        "HRQueryReply",
        back_populates="query",
        cascade="all, delete-orphan",
        order_by="HRQueryReply.created_at",
    )


class HRQueryReply(Base):
    __tablename__ = "hr_query_replies"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("hr_queries.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # author
    author_name = Column(String(150), nullable=True)  # snapshot of author display name
    author_role = Column(String(20), nullable=True)   # "HR" or "Employee" (for display)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_ist_now)

    query = relationship("HRQuery", back_populates="replies")
