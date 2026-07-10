from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import date, datetime


class PolicyVersionResponse(BaseModel):
    id: int
    name: str
    title: Optional[str] = None
    category: Optional[str] = None
    content: Optional[str] = None
    effective_date: date
    version: int
    attachment_name: Optional[str] = None  # present ⇒ a downloadable file exists
    published_by_name: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PolicyGroup(BaseModel):
    """One named policy: its current (latest) version + how many versions exist."""
    name: str
    category: Optional[str] = None
    current: PolicyVersionResponse
    versions_count: int


class PolicyHistory(BaseModel):
    name: str
    versions: List[PolicyVersionResponse]
