"""Application configuration.

All runtime values are read from the environment in one place.  Keeping this
module dependency-light makes it safe to import from tests and scripts.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:root@localhost:5432/Resume_analyzer"
    base_url: str = "http://127.0.0.1:8001"
    db_connect_timeout: int = 5
    upload_dir: Path = PROJECT_ROOT / "uploads"
    chroma_dir: Path = PROJECT_ROOT / "chromadb"
    excel_file: Path = PROJECT_ROOT / "backend" / "data" / "resumes_data.xlsx"
    secret_key: str = ""
    api_url: str = "http://127.0.0.1:8001"

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / "Attendance Management" / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def resolve_project_paths(self) -> "Settings":
        for field in ("upload_dir", "chroma_dir", "excel_file"):
            path = getattr(self, field)
            if not path.is_absolute():
                setattr(self, field, PROJECT_ROOT / path)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable-by-convention settings object."""
    return Settings()
