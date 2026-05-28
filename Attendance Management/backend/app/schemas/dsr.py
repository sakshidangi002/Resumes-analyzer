"""Pydantic schemas for Daily Status Report (DSR)."""
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DSRStatus = Literal["DRAFT", "SUBMITTED"]


class DSRBase(BaseModel):
    report_date: date
    project_work: str | None = Field(None, max_length=255)
    work_location: str | None = Field("Office", max_length=50)
    total_hours: Decimal | None = Field(None, ge=0, le=24)
    work_done: str = Field(..., min_length=1)
    plan_for_tomorrow: str | None = None

    @field_validator("project_work", "work_location", "plan_for_tomorrow", mode="before")
    @classmethod
    def _strip_or_none(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("work_done", mode="before")
    @classmethod
    def _strip_required(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v


class DSRCreate(DSRBase):
    status: DSRStatus = "DRAFT"


class DSRUpdate(BaseModel):
    project_work: str | None = Field(None, max_length=255)
    work_location: str | None = Field(None, max_length=50)
    total_hours: Decimal | None = Field(None, ge=0, le=24)
    work_done: str | None = Field(None, min_length=1)
    plan_for_tomorrow: str | None = None
    status: DSRStatus | None = None


class DSRResponse(DSRBase):
    id: int
    employee_id: int
    employee_name: str | None = None
    employee_code: str | None = None
    designation: str | None = None
    status: DSRStatus
    submitted_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DSRSummary(BaseModel):
    """Aggregated DSR counts for a given month (used by the right-rail card)."""

    year: int
    month: int
    total: int = 0
    submitted: int = 0
    draft: int = 0
    pending: int = 0  # working days in the month that have no DSR yet
