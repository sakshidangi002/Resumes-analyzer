"""HRM assistant chat — the HTTP surface of the graph.

Authentication and authorisation are unchanged: this endpoint uses the same
`require_roles` dependency as every other HRMS route, so a request that could
not reach `/api/attendance` cannot reach the assistant either. The authenticated
`User` is the *only* source of employee identity handed to the graph — nothing
in the request body can influence whose data is read.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db_session, require_roles
from app.assistant.graph import run_assistant
from app.assistant.llm import get_llm
from app.db.session import SessionLocal
from app.models import User
from app.schemas.assistant import AssistantChatRequest, AssistantChatResponse

router = APIRouter()

#: Roles allowed to see the per-node trace in the response.
_TRACE_ROLES = {"Admin", "HR"}


@router.post("/chat", response_model=AssistantChatResponse)
def assistant_chat(
    payload: AssistantChatRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    roles = [r.name for r in current_user.roles]

    result = run_assistant(
        message=payload.message,
        actor_user_id=current_user.id,
        actor_employee_id=current_user.employee_id,
        actor_roles=roles,
        db=db,
        # Parallel branches must not share the request-scoped Session; they mint
        # their own from the factory and close them.
        db_factory=SessionLocal,
        llm=get_llm(),
    )

    return AssistantChatResponse(
        run_id=result.get("run_id", ""),
        status=result.get("status") or "failed",
        answer=result.get("response") or "",
        data=result.get("data"),
        intent=result.get("intent"),
        metrics=result.get("metrics"),
        trace=result.get("trace") if set(roles) & _TRACE_ROLES else None,
    )
