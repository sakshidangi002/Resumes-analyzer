"""Email send log for notifications and letters."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from app.db.base_class import Base


class EmailLog(Base):
    __tablename__ = "email_logs"

    id = Column(Integer, primary_key=True, index=True)
    to_email = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=True)
    body = Column(Text, nullable=True)
    template_code = Column(String(50), nullable=True)
    status = Column(String(20), default="PENDING")  # SENT, FAILED
    sent_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    related_entity_type = Column(String(50), nullable=True)
    related_entity_id = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
