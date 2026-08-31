"""Loader for the node skill files.

Honest statement of what these are for: the markdown in `graph_engine/skills/`
is **agent instruction text**. It becomes the system prompt when a node delegates
judgement to an LLM, and it documents the contract the deterministic path
implements. It is not the graph, and no routing decision is read from it — that
would mean a prose file silently controlling execution.

Each node records which skill it applied in its trace, so a run shows which
instruction set was in force.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

#: node name -> skill file stem
NODE_SKILLS = {
    "review": "code_review",
    "bug_analysis": "bug_analysis",
    "fix": "bug_fixing",
    "test": "testing",
    "failure_analysis": "failure_analysis",
    "verify": "verification",
}


@lru_cache(maxsize=None)
def load_skill(name: str) -> str:
    """Return skill text, or '' if the file is absent.

    A missing skill degrades the LLM path to a generic prompt; it never stops a
    run, because the deterministic path does not depend on it.
    """
    path = SKILLS_DIR / f"{name}.md"
    if not path.exists():
        logger.warning("graph_engine.skill_missing name=%s path=%s", name, path)
        return ""
    return path.read_text(encoding="utf-8")


def skill_for_node(node: str) -> tuple[str, str]:
    """(skill_name, skill_text) for a node."""
    skill_name = NODE_SKILLS.get(node, "")
    return skill_name, load_skill(skill_name) if skill_name else ""


def available() -> list[str]:
    if not SKILLS_DIR.exists():
        return []
    return sorted(p.stem for p in SKILLS_DIR.glob("*.md"))
