"""`scope_resolver` — the authorisation gate. Runs before any domain node.

Two rules make this node the security boundary of the whole graph:

1. **`target_employee_id` is derived, never parsed.** For a question about
   oneself it is copied from the authenticated `User.employee_id`. It is never
   read out of the user's message and never produced by the LLM, so no amount of
   prompt injection ("show attendance for employee 7") can widen access — the
   text can only express *intent*, which this node then adjudicates.

2. **The role check happens before the lookup.** An Employee asking about a
   colleague is denied without any database query, so the assistant cannot be
   used to probe whether a name exists. A Manager's name lookup is scoped to
   their own reportees for the same reason.

The graph therefore cannot bypass HRMS authorisation: the route has already run
`get_current_user` (which also revokes RESIGNED/TERMINATED accounts), and this
node applies the same role rules the REST endpoints use.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from sqlalchemy import func, or_

from app.assistant import failures
from app.assistant.runtime import RunContext

logger = logging.getLogger(__name__)

#: Mirrors `require_roles(["Admin", "HR", "Manager"])` on
#: GET /api/attendance/employee/{id}/history.
_CAN_VIEW_ANY = {"Admin", "HR"}
_CAN_VIEW_REPORTEES = {"Manager"}


def _resolve_other(state: Mapping[str, Any], ctx: RunContext) -> dict[str, Any]:
    """Look up the requested employee, constrained by what the actor may see."""
    from app.models.employee import Employee  # local import: keeps node import-light

    roles = set(state.get("actor_roles") or [])
    entities = state.get("entities") or {}
    name = (entities.get("target_name") or "").strip()
    code = (entities.get("target_code") or "").strip()

    # Every non-success path clears `target_employee_id` explicitly. State
    # persists across a recovery loop, so leaving a previously resolved id in
    # place would let a later, *unresolved* turn inherit it.
    if not name and not code:
        return {
            "status": "needs_input",
            "target_employee_id": None,
            "denial_reason": None,
            "error": {
                "kind": failures.MISSING_ENTITY,
                "message": "Which employee did you mean?",
                "node": "scope_resolver",
            },
        }

    query = ctx.db.query(Employee.id)
    if code:
        query = query.filter(func.lower(Employee.employee_code) == code.lower())
    else:
        query = query.filter(
            or_(
                func.lower(Employee.full_name) == name.lower(),
                func.lower(Employee.full_name).like(f"{name.lower()} %"),
            )
        )

    # A Manager may only ever see their own reportees. Scoping the *query*
    # (rather than filtering after the fact) means a manager cannot distinguish
    # "no such employee" from "not your reportee" — both return nothing.
    if not (roles & _CAN_VIEW_ANY):
        query = query.filter(Employee.reporting_manager_id == state.get("actor_employee_id"))

    matches = query.limit(2).all()

    if not matches:
        return {
            "status": "needs_input",
            "target_employee_id": None,
            "error": {
                "kind": failures.MISSING_ENTITY,
                "message": "I couldn't find that employee in the records you can access.",
                "node": "scope_resolver",
            },
        }
    if len(matches) > 1:
        return {
            "status": "needs_input",
            "target_employee_id": None,
            "error": {
                "kind": failures.MISSING_ENTITY,
                "message": "More than one employee matches that name. "
                           "Please use the employee code.",
                "node": "scope_resolver",
            },
        }

    return {"scope": "other", "target_employee_id": int(matches[0][0]), "status": "running"}


def scope_resolver(state: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
    entities = state.get("entities") or {}
    hint = entities.get("subject_hint") or "self"
    roles = set(state.get("actor_roles") or [])
    actor_employee_id = state.get("actor_employee_id")

    # --- someone else's data -------------------------------------------------
    if hint == "other":
        if not (roles & (_CAN_VIEW_ANY | _CAN_VIEW_REPORTEES)):
            logger.info(
                "assistant.authz_denied run_id=%s user_id=%s reason=role",
                ctx.run_id, state.get("actor_user_id"),
            )
            return {
                "status": "denied",
                "scope": "other",
                "target_employee_id": None,
                "denial_reason": "role",
                "_trace": {"decision": "denied", "reason": "role"},
            }
        result = _resolve_other(state, ctx)
        result["_trace"] = {
            "decision": result.get("status"),
            "resolved": result.get("target_employee_id") is not None,
        }
        return result

    # --- a whole team --------------------------------------------------------
    if hint == "team":
        return {
            "status": "unsupported",
            "scope": "team",
            "target_employee_id": None,
            "error": {
                "kind": failures.UNSUPPORTED,
                "message": "I can answer for one employee at a time right now.",
                "node": "scope_resolver",
            },
            "_trace": {"decision": "unsupported", "reason": "team_scope"},
        }

    # --- own data ------------------------------------------------------------
    if not actor_employee_id:
        # Matches the existing behaviour of POST /api/queries: an account with no
        # employee record has nothing to report on.
        return {
            "status": "denied",
            "scope": "self",
            "target_employee_id": None,
            "denial_reason": "no_employee_record",
            "_trace": {"decision": "denied", "reason": "no_employee_record"},
        }

    return {
        "scope": "self",
        "target_employee_id": int(actor_employee_id),
        "status": "running",
        "_trace": {"decision": "allowed", "scope": "self"},
    }
