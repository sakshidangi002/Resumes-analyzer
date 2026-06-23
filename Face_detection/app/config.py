from __future__ import annotations

from datetime import timedelta, timezone
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    default_threshold: float = 0.45
    min_match_margin: float = 0.05
    repeat_match_delta: float = 0.10
    repeat_match_floor: float = 0.38
    timezone_offset_hours: int = 5
    timezone_offset_minutes: int = 30
    attendance_page_size: int = 100
    webcam_recognition_interval_ms: int = 1500
    database_url: str = "postgresql://postgres:postgres@localhost:5432/face_attendance"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def get_db_timezone() -> timezone:
    settings = get_settings()
    return timezone(
        timedelta(hours=settings.timezone_offset_hours, minutes=settings.timezone_offset_minutes)
    )
