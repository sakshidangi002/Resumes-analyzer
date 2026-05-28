"""Read / normalize the HR-configurable DSR reminder schedule.

The schedule lives on the singleton ``CompanyConfig`` row:

    dsr_reminder_enabled   : bool
    dsr_reminder_time      : "HH:MM" 24h IST
    dsr_reminder_weekdays  : "mon,tue,wed,thu,fri"

Both the scheduler (minute-ticker in ``app/main.py``) and the
``/api/dsr/reminder-settings`` endpoint read through these helpers so
validation lives in one place.
"""
from __future__ import annotations

import re
from typing import Iterable

from sqlalchemy.orm import Session

from app.models import CompanyConfig


DEFAULT_REMINDER_TIME = "17:00"
DEFAULT_REMINDER_WEEKDAYS = "mon,tue,wed,thu,fri"
DEFAULT_REMINDER_ENABLED = True

_WEEKDAY_TOKENS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_HHMM_RE = re.compile(r"^(?P<h>\d{1,2}):(?P<m>\d{2})$")


def normalize_time(value: str) -> str:
    """Normalize ``"5:00"``, ``"05:0"``, ``"17:00"`` → ``"17:00"``. Raises
    ``ValueError`` for anything that isn't a valid 24h clock time."""
    if value is None:
        raise ValueError("Time is required.")
    s = str(value).strip()
    m = _HHMM_RE.match(s)
    if not m:
        raise ValueError("Time must look like 'HH:MM' (24h), e.g. '17:00'.")
    h, mm = int(m.group("h")), int(m.group("m"))
    if not (0 <= h <= 23 and 0 <= mm <= 59):
        raise ValueError("Time out of range; use 00:00 – 23:59.")
    return f"{h:02d}:{mm:02d}"


def normalize_weekdays(value: str | Iterable[str]) -> str:
    """Normalize a comma-separated weekday string into canonical form.

    Accepts any case + whitespace; raises ``ValueError`` for unknown tokens
    or empty input. Returns e.g. ``"mon,tue,wed,thu,fri"``.
    """
    if value is None:
        raise ValueError("weekdays is required")

    parts: list[str]
    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.split(",")]
    else:
        parts = [str(p).strip().lower() for p in value]

    seen: list[str] = []
    for token in parts:
        if not token:
            continue
        if token not in _WEEKDAY_TOKENS:
            raise ValueError(
                f"Unknown weekday '{token}'. Use mon / tue / wed / thu / fri / sat / sun."
            )
        if token not in seen:
            seen.append(token)
    if not seen:
        raise ValueError("At least one weekday must be selected.")
    # Always serialize in calendar order so reads compare predictably.
    return ",".join(d for d in _WEEKDAY_TOKENS if d in seen)


def get_or_create_config(db: Session) -> CompanyConfig:
    """Return the singleton company config row, creating it if missing."""
    cfg = db.query(CompanyConfig).first()
    if cfg is None:
        cfg = CompanyConfig(
            name="Default Company",
            dsr_reminder_enabled=DEFAULT_REMINDER_ENABLED,
            dsr_reminder_time=DEFAULT_REMINDER_TIME,
            dsr_reminder_weekdays=DEFAULT_REMINDER_WEEKDAYS,
        )
        db.add(cfg)
        db.commit()
        db.refresh(cfg)
        return cfg

    changed = False
    if cfg.dsr_reminder_time is None or not str(cfg.dsr_reminder_time).strip():
        cfg.dsr_reminder_time = DEFAULT_REMINDER_TIME
        changed = True
    if (
        cfg.dsr_reminder_weekdays is None
        or not str(cfg.dsr_reminder_weekdays).strip()
    ):
        cfg.dsr_reminder_weekdays = DEFAULT_REMINDER_WEEKDAYS
        changed = True
    if cfg.dsr_reminder_enabled is None:
        cfg.dsr_reminder_enabled = DEFAULT_REMINDER_ENABLED
        changed = True
    if changed:
        db.commit()
        db.refresh(cfg)
    return cfg


def read_schedule(db: Session) -> tuple[bool, int, int, set[str]]:
    """Return ``(enabled, hour, minute, weekday_set)`` for the scheduler tick."""
    cfg = get_or_create_config(db)
    try:
        hhmm = normalize_time(cfg.dsr_reminder_time or DEFAULT_REMINDER_TIME)
    except ValueError:
        hhmm = DEFAULT_REMINDER_TIME
    h, m = hhmm.split(":")
    try:
        weekdays_csv = normalize_weekdays(
            cfg.dsr_reminder_weekdays or DEFAULT_REMINDER_WEEKDAYS
        )
    except ValueError:
        weekdays_csv = DEFAULT_REMINDER_WEEKDAYS
    return (
        bool(cfg.dsr_reminder_enabled),
        int(h),
        int(m),
        set(d for d in weekdays_csv.split(",") if d),
    )
