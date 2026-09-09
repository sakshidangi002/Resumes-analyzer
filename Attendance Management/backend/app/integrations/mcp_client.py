from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

MCP_SERVER_DIR = Path(
    os.getenv("MCP_SERVER_DIR", r"c:\sakshi folder\application\mcp_server")
)
MCP_SERVER_PYTHON = Path(
    os.getenv("MCP_SERVER_PYTHON", str(MCP_SERVER_DIR / ".venv" / "Scripts" / "python.exe"))
)


def _mcp():
    """Import the `mcp` package on demand.

    **Imported here rather than at module scope on purpose.** `mcp` is an
    optional dependency: it is not in `requirements.txt`, and the mail server it
    talks to is a separate process that only exists where someone has set it up.
    A top-level import made that optional integration mandatory — the server
    venv had no `mcp`, so `import email -> import mcp_client -> ModuleNotFoundError`
    took down `app.api.routes`, and with it the entire HRMS API. Every route in
    the product went dark because one optional feature's package was absent.

    Now the cost of a missing package is a 502 on the mail endpoints, which is
    exactly what those routes already report when the mail server is unreachable.
    """
    try:
        from mcp import ClientSession, StdioServerParameters  # noqa: PLC0415
        from mcp.client.stdio import stdio_client  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "The 'mcp' package is not installed, so the company mailbox is "
            "unavailable. Install it in this environment to enable it: "
            "pip install mcp"
        ) from exc
    return ClientSession, StdioServerParameters, stdio_client


def available() -> bool:
    """Is the mail integration usable in this environment?"""
    try:
        _mcp()
    except RuntimeError:
        return False
    return True


def _server_params() -> Any:
    _, StdioServerParameters, _stdio = _mcp()
    return StdioServerParameters(
        command=str(MCP_SERVER_PYTHON),
        args=[str(MCP_SERVER_DIR / "server.py")],
        cwd=str(MCP_SERVER_DIR),
    )


async def list_tools() -> list[dict[str, Any]]:
    """Return the tools the email server advertises."""
    ClientSession, _params, stdio_client = _mcp()
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {"name": t.name, "description": (t.description or "").strip()}
                for t in result.tools
            ]


async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call one tool and return its structured result.

    A fresh subprocess per call: correct, and cheap enough at HR volumes. If
    this ever runs per-request on a hot path, hold one session open in the
    FastAPI lifespan instead.
    """
    ClientSession, _params, stdio_client = _mcp()
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments or {})
            if getattr(result, "structuredContent", None):
                return result.structuredContent
            return [_decode_block(c) for c in result.content]


def _decode_block(block: Any) -> Any:
    """Turn one content block into a dict where it holds JSON, else its text.

    The email server returns each message as a JSON string in a text block, so
    without this callers would have to ``json.loads`` every element themselves.
    Non-JSON text (an error string, a saved path) is passed through unchanged.
    """
    text = getattr(block, "text", None)
    if text is None:
        return str(block)
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


async def get_emails(folder: str = "INBOX", limit: int = 10, search: str = "ALL") -> Any:
    """Newest messages in ``folder``. Thin alias over the ``list_emails`` tool."""
    return await call_tool(
        "list_emails", {"folder": folder, "limit": limit, "search": search}
    )


if __name__ == "__main__":
    async def _main() -> None:
        print("Available tools:")
        for tool in await list_tools():
            print(f"  {tool['name']}: {tool['description'].splitlines()[0]}")
        print("\nInbox:")
        print(await get_emails(limit=5))

    asyncio.run(_main())
