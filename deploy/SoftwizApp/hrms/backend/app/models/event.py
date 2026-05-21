"""Calendar events: holidays, birthdays, anniversaries, announcements."""
from datetime import date
from sqlalchemy import Column, Integer, String, Date, ForeignKey, Text
from app.db.base_class import Base


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    date = Column(Date, nullable=False)
    event_type = Column(String(30), nullable=False)  # HOLIDAY, BIRTHDAY, ANNIVERSARY, ANNOUNCEMENT
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    description = Column(Text, nullable=True)
