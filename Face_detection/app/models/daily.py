from sqlalchemy import Column, Integer, DateTime, ForeignKey, Float, Date
from sqlalchemy.sql import func
from . import Base

class DailyAttendance(Base):
    __tablename__ = "daily_attendance"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    first_in = Column(DateTime(timezone=True), nullable=True)
    last_out = Column(DateTime(timezone=True), nullable=True)
    total_hours = Column(Float, nullable=True, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
