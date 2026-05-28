"""In-process WebSocket connection manager keyed by ``user_id``.

The manager is intentionally tiny: a dict ``user_id -> set[WebSocket]`` plus
two helpers (``publish`` async, ``publish_sync`` for use from request
handlers). It's safe to call from any thread because we always hop back to
the captured asyncio loop via :func:`asyncio.run_coroutine_threadsafe`.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Tracks live WebSockets per user and broadcasts notifications.

    Concurrency-safe within a single asyncio event loop. ``publish_sync``
    bridges sync request handlers to the loop using
    :func:`asyncio.run_coroutine_threadsafe`.
    """

    def __init__(self) -> None:
        self._sockets: dict[int, set[WebSocket]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the running event loop (called once at FastAPI startup)."""
        self._loop = loop

    async def connect(self, user_id: int, websocket: WebSocket) -> None:
        """Register an already-accepted WebSocket for ``user_id``."""
        bucket = self._sockets.setdefault(user_id, set())
        bucket.add(websocket)
        logger.info(
            "WS connect: user_id=%s total_for_user=%d total_users=%d",
            user_id,
            len(bucket),
            len(self._sockets),
        )

    def disconnect(self, user_id: int, websocket: WebSocket) -> None:
        """Remove ``websocket`` from ``user_id``'s bucket (no-op if absent)."""
        bucket = self._sockets.get(user_id)
        if not bucket:
            return
        bucket.discard(websocket)
        if not bucket:
            self._sockets.pop(user_id, None)
        logger.info(
            "WS disconnect: user_id=%s remaining_for_user=%d total_users=%d",
            user_id,
            len(bucket),
            len(self._sockets),
        )

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------
    async def publish(self, user_id: int, payload: dict[str, Any]) -> int:
        """Send ``payload`` to every live socket for ``user_id``.

        Returns the number of sockets the payload was *successfully* sent to.
        Dead sockets (any send-raising) are dropped from the bucket.
        """
        bucket = self._sockets.get(user_id)
        if not bucket:
            return 0
        dead: list[WebSocket] = []
        sent = 0
        for ws in list(bucket):
            try:
                await ws.send_json(payload)
                sent += 1
            except Exception:
                dead.append(ws)
        for ws in dead:
            bucket.discard(ws)
        if not bucket:
            self._sockets.pop(user_id, None)
        return sent

    def publish_sync(self, user_id: int, payload: dict[str, Any]) -> None:
        """Schedule a publish from a synchronous context.

        Never raises. If the loop is not yet captured (e.g. during a test
        without lifespan), the call is silently dropped — the source-of-truth
        ``AppNotification`` row is still in the DB and clients will get it
        on the next poll / refresh.
        """
        if self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.publish(user_id, payload), self._loop
            )
        except Exception:
            logger.exception(
                "WS publish_sync: failed to schedule broadcast for user_id=%s",
                user_id,
            )


# Singleton — imported by routes/services that want to publish.
connection_manager = ConnectionManager()
