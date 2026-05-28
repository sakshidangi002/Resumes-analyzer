"""In-process WebSocket layer for real-time notifications.

This package exposes a single ``ConnectionManager`` keyed by ``user_id`` and a
``publish_sync`` helper that synchronous request handlers (e.g. the FastAPI
route that creates an ``AppNotification``) can call to push a payload to every
live socket belonging to that user.

Design notes:

* One process, one event loop. We capture a reference to the main asyncio
  loop during FastAPI startup so that ``publish_sync`` — which can be called
  from any thread / sync context — can schedule the actual ``send_json`` on
  the right loop via :func:`asyncio.run_coroutine_threadsafe`. This avoids
  the need for Redis or any IPC layer for the current single-worker NSSM
  deployment.

* If you later scale to multiple uvicorn workers, swap the in-memory store
  for a Redis pub/sub channel keyed by ``user_id`` — the public API stays
  the same.

* All sends are best-effort: a closed / stale socket is silently dropped on
  the next publish attempt.
"""

from app.ws.manager import ConnectionManager, connection_manager  # noqa: F401
