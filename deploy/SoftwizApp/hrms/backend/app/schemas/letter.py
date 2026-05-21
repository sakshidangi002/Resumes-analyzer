from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime


class LetterTemplateCreate(BaseModel):
    code: str
    name: str
    subject_template: Optional[str] = None
    body_template: str
    is_editable: bool = True


class LetterTemplateResponse(BaseModel):
    id: int
    code: str
    name: str
    subject_template: Optional[str] = None
    body_template: str
    is_editable: bool

    class Config:
        from_attributes = True


class LetterInstanceResponse(BaseModel):
    id: int
    employee_id: int
    template_id: int
    generated_at: datetime
    subject: Optional[str] = None
    sent_via_email: bool
    employee_code: Optional[str] = None
    employee_name: Optional[str] = None
    employee_official_email: Optional[str] = None
    employee_personal_email: Optional[str] = None

    class Config:
        from_attributes = True


class LetterReplyResponse(BaseModel):
    id: int
    letter_instance_id: int
    author_employee_id: Optional[int] = None
    author_user_id: Optional[int] = None
    message: str
    created_at: datetime

    class Config:
        from_attributes = True


class LetterReplyCreate(BaseModel):
    message: str


class LetterPreviewRequest(BaseModel):
    template_code: str
    employee_id: int
    extra_context: Optional[Dict[str, Any]] = None


class LetterPreviewResponse(BaseModel):
    subject: str
    body: str


class LetterGenerateOverrides(BaseModel):
    subject: Optional[str] = None
    body: Optional[str] = None
    from_email: Optional[str] = None
