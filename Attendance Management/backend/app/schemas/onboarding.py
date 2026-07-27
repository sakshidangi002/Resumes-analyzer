from datetime import datetime
from pydantic import BaseModel, Field


class OnboardingTaskCreate(BaseModel):
    employee_id: int
    title: str = Field(..., min_length=1, max_length=255)
    priority: str = "Medium"
    due_date: datetime | None = None
    sort_order: int = 0


class OnboardingTaskUpdate(BaseModel):
    is_completed: bool | None = None
    title: str | None = Field(None, min_length=1, max_length=255)
    priority: str | None = None
    due_date: datetime | None = None
    sort_order: int | None = None


class OnboardingTaskResponse(BaseModel):
    id: int
    employee_id: int
    title: str
    priority: str
    due_date: datetime | None
    is_completed: bool
    completed_at: datetime | None
    sort_order: int
    created_at: datetime

    model_config = {"from_attributes": True}
