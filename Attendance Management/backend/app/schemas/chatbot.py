"""Request/response schemas for the HRMS chatbot endpoint."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatbotAskRequest(BaseModel):
    # Capped because a question is a question. Anything longer is either a
    # mistake or an attempt to stuff the prompt.
    question: str = Field(..., min_length=1, max_length=1000)
    #: The skill that answered the previous turn, echoed back by the UI. Lets a
    #: bare refinement ("now for August") re-run that lookup instead of being
    #: read as a brand-new question with no context. Advisory only: it can never
    #: widen what the caller is allowed to see, because the permission check
    #: runs on the resolved skill exactly as it does for a fresh question.
    last_skill: Optional[str] = Field(default=None, max_length=64)
    #: The employee the previous answer was about, so "what about her leave?"
    #: keeps that person. Re-loaded through the caller's own scope server-side,
    #: so echoing back an id they may not see resolves to nothing.
    last_employee_id: Optional[int] = Field(default=None, ge=1)
    #: The period the previous answer used, so "and Neha?" keeps the month.
    last_period: Optional[str] = Field(default=None, max_length=48)


class ChatbotAnswerResponse(BaseModel):
    #: Always present and always safe to show on its own — refusals,
    #: clarifications and errors all arrive as text with an appropriate `kind`.
    answer: str
    #: answer | clarify | denied | not_found | general | capabilities | error
    kind: str
    #: Which skill produced it, or None when the model answered.
    skill: Optional[str] = None
    #: The figures behind the sentence, for the UI to render as chips.
    data: Optional[dict[str, Any]] = None
    #: Tables and services read. Lets HR trace a number back to its origin.
    sources: list[str] = []
    #: Who the answer was about, when it was about a person.
    employee: Optional[dict[str, Any]] = None
    period: Optional[dict[str, Any]] = None
    #: Skill scores for this question. Admin/HR only — a tuning aid.
    debug: Optional[dict[str, Any]] = None


class ChatbotSkill(BaseModel):
    name: str
    topic: str
    summary: str
    examples: list[str] = []
    company_wide: bool = False
    salary: bool = False


class ChatbotCapabilities(BaseModel):
    #: What this caller may ask about, in words: "the whole company" etc.
    scope: str
    can_see_company: bool
    can_see_salary: bool
    skills: list[ChatbotSkill]
