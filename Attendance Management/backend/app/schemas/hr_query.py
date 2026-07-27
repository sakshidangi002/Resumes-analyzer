from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime


class HRQueryReplyCreate(BaseModel):
    message: str


class HRQueryReplyResponse(BaseModel):
    id: int
    query_id: int
    user_id: Optional[int] = None
    author_name: Optional[str] = None
    author_role: Optional[str] = None
    message: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HRQueryCreate(BaseModel):
    subject: str
    message: str
    category: Optional[str] = None


class HRQueryStatusUpdate(BaseModel):
    status: str  # OPEN, PENDING, RESOLVED


class HRQueryResponse(BaseModel):
    id: int
    employee_id: int
    employee_name: Optional[str] = None
    subject: str
    message: str
    category: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    replies: List[HRQueryReplyResponse] = []

    model_config = ConfigDict(from_attributes=True)
