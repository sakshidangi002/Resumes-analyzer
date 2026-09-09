"""HRMS chatbot — the HTTP surface.

Authentication and authorisation are unchanged: this route uses the same
`require_roles` dependency as every other HRMS endpoint, and the authenticated
`User` is the *only* source of identity handed to the chatbot. Nothing in the
request body can influence whose data is read — `Actor` is built from the token,
not from the question.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db_session, require_roles
from app.chatbot import Actor, ask
from app.chatbot import registry
from app.models import User
from app.schemas.chatbot import (
    ChatbotAnswerResponse,
    ChatbotAskRequest,
    ChatbotCapabilities,
    ChatbotSkill,
)

router = APIRouter()

_ALL_ROLES = ["Admin", "HR", "Manager", "Employee"]


@router.post("/ask", response_model=ChatbotAnswerResponse)
def chatbot_ask(
    payload: ChatbotAskRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_roles(_ALL_ROLES)),
):
    """Answer one question.

    Every role may call this; what each role can *see* is decided inside by
    `Actor`, which mirrors the permissions the REST routes already apply. An
    Employee asking a company-wide question gets a refusal, not an error.
    """
    actor = Actor.from_user(current_user)
    answer = ask(
        question=payload.question,
        actor=actor,
        db=db,
        # The skill scores are a tuning aid, not something an employee needs.
        include_debug=actor.is_hr,
        last_skill=payload.last_skill,
        last_employee_id=payload.last_employee_id,
        last_period=payload.last_period,
    )
    return ChatbotAnswerResponse(
        answer=answer.text,
        kind=answer.kind,
        skill=answer.skill,
        data=answer.data,
        sources=answer.sources,
        employee=answer.employee,
        period=answer.period,
        debug=answer.debug,
    )


@router.get("/capabilities", response_model=ChatbotCapabilities)
def chatbot_capabilities(
    current_user: User = Depends(require_roles(_ALL_ROLES)),
):
    """What this caller can ask about.

    Filtered by role, so the UI can offer prompts that will actually work rather
    than advertising company-wide questions to an employee who cannot run them.
    """
    actor = Actor.from_user(current_user)
    skills = [
        ChatbotSkill(
            name=s.name,
            topic=s.topic,
            summary=s.summary,
            examples=list(s.examples),
            company_wide=s.company_wide,
            salary=s.salary,
        )
        for s in registry.all_skills()
        if not (s.company_wide and not actor.may_see_company)
    ]
    return ChatbotCapabilities(
        scope=actor.describe_scope(),
        can_see_company=actor.may_see_company,
        can_see_salary=actor.may_see_salary,
        skills=skills,
    )
