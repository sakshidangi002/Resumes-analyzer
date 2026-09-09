"""The skill catalogue, one module per area of the application.

Importing a module is what registers its skills, so `load_skills()` is the only
thing that has to know they exist. Adding an area of the app to the chatbot is
therefore: write a module, add it to `_MODULES`, done.

The import is deferred into a function rather than done at module import time
because these modules pull in ORM models and services; keeping it explicit means
`app.chatbot.registry` stays importable on its own, which tests rely on.
"""
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_MODULES = (
    "people",
    "attendance",
    "breakdown",
    "leave",
    "payroll",
    "workforce",
    "company_calendar",
    "documents",
)

_loaded = False


def load_skills() -> None:
    """Import every skill module once, populating the registry."""
    global _loaded
    if _loaded:
        return
    for name in _MODULES:
        try:
            importlib.import_module(f"app.chatbot.skills.{name}")
        except Exception:  # noqa: BLE001 - one bad module must not kill the chatbot
            logger.exception("chatbot.skill_module_failed module=%s", name)
    _loaded = True
