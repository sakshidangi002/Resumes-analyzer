"""``/ws/notifications`` — real-time notification feed for the logged-in user.

Auth: browsers cannot send custom headers on the WebSocket handshake, so we
accept the JWT either as a ``?token=`` query parameter (preferred) or as a
``Sec-WebSocket-Protocol`` subprotocol (more secure — token never lands in
proxy access logs as a query string). Both are checked.

Wire protocol (server → client JSON frames):

    {"type": "hello",   "user_id": 7, "ts": "<utc iso>"}
    {"type": "ping",                    "ts": "<utc iso>"}
    {"type": "notification", "data": {...AppNotificationResponse...}}
    {"type": "refresh"}        # generic "go re-poll" nudge

Wire protocol (client → server):

    {"type": "ping"}            # optional keepalive; server echoes "pong"

The route module is intentionally thin: every push goes through the
``connection_manager`` singleton so callers don't need to know about
WebSockets directly.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from app.core.security import decode_access_token
from app.db.session import SessionLocal
from app.models import User
from app.ws.manager import connection_manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_user_id_from_token(token: str) -> int | None:
    """Decode ``token`` and return the active user's id, or ``None``."""
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    sub = payload.get("sub")
    if sub is None:
        return None
    try:
        user_id = int(sub)
    except (TypeError, ValueError):
        return None
    # Verify the user still exists and is active.
    db = SessionLocal()
    try:
        u = db.query(User.id, User.is_active).filter(User.id == user_id).first()
        if not u or not u.is_active:
            return None
        return user_id
    except Exception:
        logger.exception("WS auth: DB lookup failed for sub=%s", sub)
        return None
    finally:
        db.close()


def _token_from_subprotocols(websocket: WebSocket) -> str | None:
    """Some clients pass the JWT via ``Sec-WebSocket-Protocol: bearer.<token>``."""
    protocols = websocket.headers.get("sec-websocket-protocol", "")
    if not protocols:
        return None
    for raw in protocols.split(","):
        item = raw.strip()
        if item.lower().startswith("bearer."):
            return item[len("bearer."):]
    return None


@router.websocket("/ws/notifications")
async def notifications_socket(
    websocket: WebSocket,
    token: str = Query(default=""),
):
    """Single live notification channel for the connected user.

    Client URL examples:

        ws://host/ws/notifications?token=<JWT>
        wss://host/ws/notifications?token=<JWT>
    """
    actual_token = token or _token_from_subprotocols(websocket) or ""
    user_id = _resolve_user_id_from_token(actual_token)
    if user_id is None:
        # 4401 is the conventional "auth failed" close code for WebSocket
        # (the 1xxx range is reserved by RFC 6455; 4xxx is app-level).
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    await connection_manager.connect(user_id, websocket)
    try:
        await websocket.send_json(
            {
                "type": "hello",
                "user_id": user_id,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )
        # Keep the socket open. We don't expect much client→server traffic,
        # but we still read so the connection's close event fires promptly.
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=45.0)
            except asyncio.TimeoutError:
                # Idle keepalive — also helps detect half-open sockets.
                try:
                    await websocket.send_json(
                        {"type": "ping", "ts": datetime.utcnow().isoformat() + "Z"}
                    )
                except Exception:
                    break
                continue
            except WebSocketDisconnect:
                break
            except Exception:
                # Non-JSON / garbage — just keep going.
                continue

            if isinstance(msg, dict) and msg.get("type") == "ping":
                try:
                    await websocket.send_json(
                        {"type": "pong", "ts": datetime.utcnow().isoformat() + "Z"}
                    )
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        connection_manager.disconnect(user_id, websocket)
