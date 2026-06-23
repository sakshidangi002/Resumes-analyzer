from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.sql import func
from . import Base

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, nullable=False)
    source_url = Column(String, nullable=False)
    source_type = Column(String, nullable=False, default="rtsp")
    camera_type = Column(String, nullable=False, default="IN")
    enabled = Column(Boolean, nullable=False, default=False)
    threshold = Column(Float, nullable=False, default=0.45)
    interval_sec = Column(Float, nullable=False, default=1.5)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
