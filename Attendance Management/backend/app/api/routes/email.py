"""Company mailbox, read over MCP.

The mail itself lives on Gmail; an external MCP server owns the IMAP/SMTP
credentials and this app never sees them. That is the point of routing through
``app.integrations.mcp_client`` rather than talking to IMAP here.

Admin/HR only — this is the shared company mailbox, not a personal one.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import require_roles
from app.integrations import mcp_client
from app.models import User

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/messages")
async def list_messages(
    folder: str = Query("INBOX"),
    limit: int = Query(20, ge=1, le=100),
    search: str = Query("ALL", description='IMAP criteria, e.g. UNSEEN'),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
) -> list[dict]:
    """Newest messages in the mailbox, as header summaries."""
    try:
        rows = await mcp_client.get_emails(folder=folder, limit=limit, search=search)
    except Exception:
        # The MCP server is a separate process: a missing venv, bad credentials
        # or an IMAP outage all surface here. Log the detail, tell the client
        # something actionable without leaking the mail server's error text.
        logger.exception("MCP list_emails failed (folder=%s search=%s)", folder, search)
        raise HTTPException(status_code=502, detail="Mail server unavailable")
    return rows if isinstance(rows, list) else [rows]


@router.get("/messages/{uid}")
async def read_message(
    uid: str,
    folder: str = Query("INBOX"),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
) -> dict:
    """One message in full, by the uid from ``/messages``."""
    try:
        result = await mcp_client.call_tool("read_email", {"uid": uid, "folder": folder})
    except Exception:
        logger.exception("MCP read_email failed (uid=%s)", uid)
        raise HTTPException(status_code=502, detail="Mail server unavailable")
    if isinstance(result, list):
        result = result[0] if result else {}
    if not isinstance(result, dict):
        raise HTTPException(status_code=404, detail="Message not found")
    return result
