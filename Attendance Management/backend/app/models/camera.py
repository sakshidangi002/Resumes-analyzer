"""Camera configuration model for Hikvision DVR / RTSP attendance cameras."""
from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func
from app.db.base_class import Base


class CameraConfig(Base):
    """Persistent configuration + runtime statistics for each camera stream.

    camera_purpose:
        "IN"  → Check-in camera  (forces IN event on recognition)
        "OUT" → Check-out camera (forces OUT event on recognition)
    """
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # --- Identity ---
    name = Column(String(100), nullable=False)
    location = Column(String(200), nullable=True)

    # --- Stream ---
    source_url = Column(String(500), nullable=False)  # Database uses source_url, not stream_url
    source_type = Column(String(20), nullable=False, default="rtsp")  # rtsp | usb | http

    # --- Purpose ---
    camera_type = Column(String(10), nullable=False, default="IN")  # Database uses camera_type, not camera_purpose
    camera_purpose = Column(String(10), nullable=False, default="IN")  # IN | OUT (legacy field)

    # --- Recognition settings ---
    threshold = Column(Float, nullable=False, default=0.45)
    interval_sec = Column(Float, nullable=False, default=2.0)
    frame_skip = Column(Integer, nullable=False, default=0)  # Skip N frames between recognition
    tracking_max_distance = Column(Float, nullable=False, default=100.0)  # Max pixels for face tracking
    tracking_cooldown = Column(Float, nullable=False, default=3.0)  # Seconds between recognitions
    enabled = Column(Boolean, nullable=False, default=False)

    # --- Doorway line-crossing (entry/exit detection) ---
    # When enabled, a virtual line is drawn across the frame; a person crossing
    # it in `entry_direction` fires this camera's event (IN/OUT). Works from an
    # overhead camera where faces can't be recognised, using body tracking.
    crossing_enabled = Column(Boolean, nullable=False, default=False)
    line_orientation = Column(String(10), nullable=False, default="horizontal")  # horizontal | vertical
    line_position = Column(Float, nullable=False, default=0.5)  # 0..1 across the frame
    entry_direction = Column(String(10), nullable=False, default="down")  # down|up (horizontal); right|left (vertical)

    # --- Timestamps ---
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=datetime.utcnow,
        nullable=False,
    )
