"""Per-user Web Push (VAPID) subscriptions.

Each row is one browser/device for one user. A user can have several (laptop
Chrome + phone Chrome + work Edge). When we fire a 5 PM IST reminder, we push
to every active subscription. Dead subscriptions (HTTP 404/410 from the push
service) are cleaned up automatically — see ``app/services/web_push_service.py``.
"""
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class PushSubscription(Base):
    __tablename__ = "push_subscriptions"
    __table_args__ = (
        UniqueConstraint("user_id", "endpoint", name="uq_push_user_endpoint"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # The Push Service URL the browser gave us. Always opaque to us.
    endpoint = Column(Text, nullable=False)

    # ECDH public key (P-256) used to encrypt the payload for this device.
    p256dh = Column(String(255), nullable=False)
    # Authentication secret shared between the browser and the push service.
    auth = Column(String(255), nullable=False)

    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)

    user = relationship("User", backref="push_subscriptions")
