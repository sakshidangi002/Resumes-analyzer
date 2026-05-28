"""
Application configuration. Database is PostgreSQL; email uses simple SMTP.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Load from environment. Use .env for local overrides."""

    # App
    app_name: str = "Attendance & HRMS"
    debug: bool = False

    # PostgreSQL (override via .env on each system)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_db: str = "attendance_hrms"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # JWT
    # SECURITY: no default — must come from environment / .env so that a
    # missing config fails fast at startup instead of silently signing tokens
    # with a guessable placeholder. Recommended: `openssl rand -hex 32`.
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # SMTP (simple SMTP for all email)
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_from_email: str = "noreply@company.com"
    smtp_from_name: str = "HRMS"

    # Comma-separated address(es) that receive HR-bound notifications
    # (e.g. "new leave request"). If empty, falls back to all users with the
    # HR role in the database.
    hr_notification_email: str = ""

    # Public URL where the HRMS is reachable (used in outbound emails like the
    # 5 PM DSR reminder). Leave empty in dev; in production set e.g.
    #   APP_BASE_URL=https://hrms.softwiz.local
    app_base_url: str = ""

    # ---- Web Push (VAPID) -------------------------------------------------
    # Generate once with `python scripts/gen_vapid_keys.py` and paste the
    # output into your .env. Browsers verify pushes against the public key
    # registered at subscribe time; the private key signs each push.
    vapid_public_key: str = ""
    vapid_private_key: str = ""
    vapid_claim_email: str = "mailto:noreply@company.com"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
