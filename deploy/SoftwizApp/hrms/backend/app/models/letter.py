"""HR letter templates and generated letter instances."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from app.db.base_class import Base


class LetterTemplate(Base):
    __tablename__ = "letter_templates"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    subject_template = Column(String(255), nullable=True)
    body_template = Column(Text, nullable=False)
    is_editable = Column(Boolean, default=True)

    instances = relationship("LetterInstance", back_populates="template")


class LetterInstance(Base):
    __tablename__ = "letter_instances"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    template_id = Column(Integer, ForeignKey("letter_templates.id"), nullable=False)
    generated_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    generated_at = Column(DateTime, default=datetime.utcnow)
    data_snapshot = Column(Text, nullable=True)  # JSON
    subject = Column(String(255), nullable=True)
    body = Column(Text, nullable=True)
    sent_via_email = Column(Boolean, default=False)
    email_log_id = Column(Integer, ForeignKey("email_logs.id"), nullable=True)

    employee = relationship("Employee", backref="letter_instances")
    template = relationship("LetterTemplate", back_populates="instances")


class LetterReply(Base):
    __tablename__ = "letter_replies"

    id = Column(Integer, primary_key=True, index=True)
    letter_instance_id = Column(Integer, ForeignKey("letter_instances.id"), nullable=False)
    author_employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    author_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    letter = relationship("LetterInstance", backref="replies")
