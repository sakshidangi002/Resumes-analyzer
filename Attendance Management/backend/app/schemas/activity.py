from datetime import datetime
from pydantic import BaseModel


class AppNotificationResponse(BaseModel):
    id: int
    title: str
    body: str | None
    kind: str
    link_path: str | None
    read_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}
