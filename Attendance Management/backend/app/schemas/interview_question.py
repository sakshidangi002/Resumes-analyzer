from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class InterviewQuestionResponse(BaseModel):
    id: int
    position: str
    title: str
    description: Optional[str] = None
    pdf_name: str
    uploaded_by_name: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class InterviewQuestionUpdate(BaseModel):
    """Edit an uploaded question set's metadata (the PDF is not replaced here)."""
    position: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
