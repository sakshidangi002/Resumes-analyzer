"""Request/response schemas for the assistant chat endpoint."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class AssistantChatRequest(BaseModel):
    # Capped: the graph's analyzer is regex-based and the message is only ever a
    # question, so anything longer is either a mistake or an attempt to abuse it.
    message: str = Field(..., min_length=1, max_length=1000)


class AssistantChatResponse(BaseModel):
    run_id: str
    status: str
    answer: str
    #: Structured payload for the UI (period, buckets, percentage). None for
    #: refusals and clarifications.
    data: Optional[dict[str, Any]] = None
    intent: Optional[str] = None
    metrics: Optional[dict[str, Any]] = None
    #: Per-node execution trace. Populated for Admin/HR only — it reveals the
    #: internal decision path, which is a debugging aid, not employee-facing.
    trace: Optional[list[dict[str, Any]]] = None
