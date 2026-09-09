"""The chat model — the same one the Resume Analyzer uses.

There is exactly one chat model in this product: the pipeline built in
`backend/main.py::_get_chat_pipe`, `Qwen/Qwen2.5-1.5B-Instruct` by default and
overridable with `CHAT_MODEL`. The chatbot reuses it rather than loading a
second one; the two apps run in a single process (`run_app.py`), and a second
multi-gigabyte model would double resident memory to answer the same questions.

The import is lazy — `backend.main` pulls in torch, transformers and spaCy — so
it happens on the first question that needs a model, not at startup. Set
`PRELOAD_CHAT_MODEL=1` to warm it in the background instead, which is what a
deployed server should do.

**The model is never the source of a figure.** Every number the chatbot states
comes from a skill reading the database. The model does two jobs: phrasing an
answer that has already been computed, and replying to questions no skill
claims. Both degrade to something sensible when it is unavailable.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cached: Optional[Callable[..., str]] = None
_attempted = False


def enabled() -> bool:
    """On unless explicitly switched off."""
    return os.getenv("CHATBOT_LLM_ENABLED", "true").strip().lower() in {"1", "true", "yes"}


def _load() -> None:
    """Import the pipeline and cache the callable. Runs once, possibly off-thread."""
    global _cached, _attempted
    with _lock:
        if _attempted:
            return
        _attempted = True
    try:
        from backend.main import CHAT_MODEL, _run_chat  # noqa: PLC0415 - lazy

        def _call(messages, max_new_tokens: int = 160) -> str:
            return _run_chat(messages, max_new_tokens=max_new_tokens)

        _cached = _call
        logger.info("chatbot.llm_ready model=%s", CHAT_MODEL)
    except Exception:  # noqa: BLE001 - absence is a supported configuration
        logger.warning("chatbot.llm_unavailable — deterministic only", exc_info=True)
        _cached = None


def get_llm(*, block: bool = False) -> Optional[Callable[..., str]]:
    """Return a `(messages, max_new_tokens) -> str` callable, or None if not ready.

    **Non-blocking by default**, and that is the important part. Loading the
    pipeline imports torch and transformers and reads a multi-gigabyte model; on
    CPU that is tens of seconds. Doing it inside a request meant the first
    question nobody had a lookup for simply timed out in the browser, which
    looks like a broken chatbot rather than a cold cache.

    So the first call starts the load on a background thread and returns None —
    the caller falls back to something instant — and questions after it get the
    model. Set `PRELOAD_CHAT_MODEL=1` and the Resume API warms it during startup,
    which is what a deployed server should do.
    """
    if not enabled():
        return None
    if _attempted:
        return _cached
    if block:
        _load()
        return _cached

    # Not loaded yet: start it and answer this turn without it.
    thread = threading.Thread(target=_load, name="chatbot-llm-load", daemon=True)
    thread.start()
    logger.info("chatbot.llm_warming — answering this turn without the model")
    return None


def is_ready() -> bool:
    return _cached is not None


#: Generating ~180 tokens from a 1.5B model on CPU is seconds at best and can be
#: much worse on a loaded box. A single worker runs every generation so two
#: requests cannot fight over the CPU, and each waits only `_TIMEOUT` before the
#: caller gives up and answers deterministically. Without this the browser sees
#: "that took too long" — which reads as a broken chatbot, not a slow model.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chatbot-llm")
_busy = threading.Event()


def _timeout(default: float) -> float:
    try:
        return float(os.getenv("CHATBOT_LLM_TIMEOUT", "") or default)
    except ValueError:
        return default


def _generate(messages, max_new_tokens: int, timeout: float) -> Optional[str]:
    """Run one generation under a deadline, or give up and return None.

    If a generation is already running we do not queue behind it: the second
    caller would wait for both, and a deterministic answer now beats a better
    answer after a minute.
    """
    llm = get_llm()
    if llm is None:
        return None
    if _busy.is_set():
        logger.info("chatbot.llm_busy — answering without the model")
        return None

    _busy.set()

    def _run() -> str:
        try:
            return llm(messages, max_new_tokens=max_new_tokens)
        finally:
            _busy.clear()

    try:
        return _EXECUTOR.submit(_run).result(timeout=timeout)
    except FuturesTimeout:
        # The worker keeps running and clears the flag when it finishes; the
        # result is simply discarded. Nothing is cancelled mid-generation.
        logger.warning("chatbot.llm_timeout after %.0fs", timeout)
        return None
    except Exception:  # noqa: BLE001 - a weak local model failing is expected
        _busy.clear()
        logger.warning("chatbot.llm_failed", exc_info=True)
        return None


def reset_cache() -> None:
    """Forget the cached handle. Tests only — a process should load once."""
    global _cached, _attempted
    with _lock:
        _cached = None
        _attempted = False


# --- the two jobs the model is trusted with --------------------------------

_GENERAL_PROMPT = (
    "You are the HR assistant inside the Softwiz HRMS, used by staff of one "
    "company. Answer in at most three short sentences of plain English. You are "
    "explaining how the HRMS and general HR processes work. You do NOT have "
    "access to any employee record in this answer, so never state anyone's "
    "attendance, leave balance or salary, and never invent a company policy "
    "number. If the question needs real data, tell the user to ask for it "
    "directly. If you do not know, say so and suggest raising a query with HR."
)

_FIGURE = re.compile(r"\d")

_UNVERIFIED = (
    " (I answered that from general knowledge, not from your HRMS records — "
    "please confirm any figures with HR.)"
)


def _trim(text: str) -> str:
    """Cut a small model's reply down to something a chat bubble can hold."""
    text = (text or "").strip()
    if not text:
        return ""
    for marker in ("\nUser:", "\nUSER:", "\nQuestion:", "\nAssistant:", "\nSystem:"):
        index = text.find(marker)
        if index > 0:
            text = text[:index]
    text = text.strip()
    if len(text) > 700:
        text = text[:700]
        stop = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if stop > 200:
            text = text[: stop + 1]
    return text.strip()


def general_answer(question: str) -> tuple[str, bool] | None:
    """Answer a question no skill claimed. Returns (text, flagged) or None.

    `flagged` means the reply stated a number the model had no source for, so
    the caller appends a warning rather than presenting it as an HRMS fact.

    Returns None while the model is still warming, so the caller answers with
    the capability list immediately instead of holding the request open.
    """
    if not question.strip():
        return None
    raw = _generate(
        [
            {"role": "system", "content": _GENERAL_PROMPT},
            {"role": "user", "content": question[:600]},
        ],
        max_new_tokens=140,
        timeout=_timeout(25.0),
    )
    if raw is None:
        return None

    text = _trim(raw)
    if not text:
        return None
    flagged = bool(_FIGURE.search(text))
    return (text + _UNVERIFIED if flagged else text), flagged


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def polish(answer: str, question: str) -> str:
    """Optionally rephrase a computed answer — accepted only if every figure survives.

    A small model will produce fluent, wrong arithmetic given the chance. The
    rewrite is compared digit-for-digit against the template and thrown away on
    any drift, so the worst case is an answer that reads a little flat.

    It is also skipped whenever the answer carries employee names or a list:
    this check verifies numbers, and would not notice a model dropping a person
    or reordering a roster.
    """
    if not answer:
        return answer
    if "\n" in answer or "•" in answer:
        return answer
    candidate = _generate(
        [
            {
                "role": "system",
                "content": (
                    "Rewrite the HR answer as one or two friendly sentences. "
                    "Keep every number and every name exactly as given. "
                    "Add no new facts."
                ),
            },
            {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"},
        ],
        max_new_tokens=120,
        # Phrasing is a nicety. It is never worth making someone wait for it.
        timeout=_timeout(25.0) / 2,
    )
    if candidate is None:
        return answer

    candidate = _trim(candidate)
    if not candidate or _numbers(candidate) != _numbers(answer):
        return answer
    return candidate
