"""Optional LLM adapter.

The only chat model in this repo is the Resume Analyzer's local TinyLlama
pipeline (`backend/main.py::_run_chat`). Importing that module pulls in torch,
transformers and spaCy, so the adapter is **lazy and disabled by default**:
`ASSISTANT_LLM_ENABLED=false` means the graph runs a fully deterministic path
and never pays that import cost.

Nothing in the graph *requires* an LLM. It is used for exactly two things —
breaking a router tie, and rephrasing an already-computed answer — and both
degrade cleanly to the deterministic behaviour when this returns None.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cached: Optional[Callable[..., str]] = None
_attempted = False


def _enabled() -> bool:
    return os.getenv("ASSISTANT_LLM_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def get_llm() -> Optional[Callable[..., str]]:
    """Return a `(messages, max_new_tokens) -> str` callable, or None.

    Loaded once per process and cached, including the failure case: a missing
    model must not re-attempt a multi-second import on every chat request.
    """
    global _cached, _attempted

    if not _enabled():
        return None
    if _attempted:
        return _cached

    with _lock:
        if _attempted:
            return _cached
        _attempted = True
        try:
            from backend.main import _run_chat  # noqa: PLC0415 - deliberately lazy

            def _call(messages, max_new_tokens: int = 120) -> str:
                return _run_chat(messages, max_new_tokens=max_new_tokens)

            _cached = _call
            logger.info("assistant.llm_enabled backend=resume_analyzer_tinyllama")
        except Exception:  # noqa: BLE001 - absence is a supported configuration
            logger.warning(
                "assistant.llm_unavailable — running deterministic-only", exc_info=True
            )
            _cached = None
    return _cached
