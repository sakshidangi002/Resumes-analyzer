"""The chatbot's single entry point.

One question in, one answer out. The pipeline is flat on purpose:

    question
      -> resolve who it is about        (resolver, scoped by role)
      -> resolve when it is about       (period)
      -> pick the skill that claims it  (registry, keyword scoring)
      -> check the actor may run it     (access)
      -> run it                         (a read-only lookup)
      -> phrase it                      (llm, figures verified)

Anything that does not reach a skill is answered by the model with no employee
data attached, so the chatbot always replies — but a figure only ever comes from
the database.

There is no workflow graph here. Every step is a function call, in order, and a
new topic is one new skill rather than a new node, edge and validator branch.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.chatbot import llm as llm_module
from app.chatbot import period as period_module
from app.chatbot import registry, resolver
from app.chatbot.access import Actor, missing_permission_message
from app.chatbot.registry import SkillContext
from app.chatbot.skills import load_skills

logger = logging.getLogger(__name__)

# Importing the skill package is what fills the registry.
load_skills()

#: What kind of reply this is. The UI styles refusals and clarifications
#: differently from answers, and the caller can tell them apart without parsing text.
ANSWER = "answer"
CLARIFY = "clarify"
DENIED = "denied"
NOT_FOUND = "not_found"
GENERAL = "general"
CAPABILITIES = "capabilities"
ERROR = "error"

#: Two skills scoring this close are answering different questions equally well,
#: so the chatbot asks instead of guessing.
_TIE_MARGIN = 0.5

#: Third-person references. "What about her leave?" only means anything if the
#: person from the previous turn is still in play — and answering it about the
#: asker instead, as this did before, is a wrong-person answer stated with full
#: confidence.
_PRONOUN = re.compile(
    r"(?<![a-z])(?:he|she|they|him|her|hers|his|their|theirs|them"
    r"|that person|same person|this employee|that employee)(?![a-z])",
    re.I,
)

#: Openings that mark a turn as a refinement of the last one rather than a new
#: question: "and Neha?", "what about yesterday?", "same for last month".
_FOLLOWUP = re.compile(
    r"^(?:and|also|ok|okay|now|then|plus)\b"
    r"|(?:what|how)\s+about\b"
    r"|\bsame\s+(?:for|as|with)\b"
    r"|\binstead\b|\bas\s+well\b|\btoo\?*$",
    re.I,
)

#: A refinement is short. A long sentence that happens to start with "and" is a
#: new question, and inheriting the previous turn's skill would derail it.
_FOLLOWUP_MAX_WORDS = 9

_CAPABILITY_QUESTION = (
    "what can you do", "what can i ask", "help", "capabilities", "what do you know",
    "who are you", "what are you",
)

#: Questions about *how something works*, which no lookup can answer. "What does
#: LOP stand for?" scores for the leave skill on the word "lop" and would come
#: back with a leave balance — a correct number answering the wrong question.
#: These go straight to the model, which is what a definition needs.
#:
#: Written narrowly on purpose: "how many days" and "how much was" must not
#: match, so "how many"/"how much" are excluded while "how do/can I" are not.
_PROCEDURAL = re.compile(
    r"(?<![a-z])(?:"
    r"how\s+(?:do|can|would|should)\s+(?:i|we|you)"
    r"|how\s+to\s"
    r"|what\s+does\s+.{1,40}\s+(?:mean|stand\s+for)"
    r"|what\s+is\s+(?:the\s+)?(?:meaning|process|procedure|policy)"
    r"|what\s+(?:is|are)\s+the\s+(?:steps|rules)"
    r"|explain|difference\s+between"
    r")",
    re.I,
)


@dataclass
class ChatAnswer:
    text: str
    kind: str = ANSWER
    skill: str | None = None
    data: dict[str, Any] | None = None
    sources: list[str] = field(default_factory=list)
    #: Who the answer was about, when it was about a person.
    employee: dict[str, Any] | None = None
    period: dict[str, Any] | None = None
    #: Ranked skill scores. Returned to Admin/HR only — a tuning aid, not
    #: something an employee needs to see.
    debug: dict[str, Any] | None = None


def _employee_payload(employee) -> dict[str, Any] | None:
    if employee is None:
        return None
    return {
        "id": employee.id,
        "name": employee.full_name,
        "employee_code": employee.employee_code,
        "department": employee.department,
        "designation": employee.designation,
    }


def _period_payload(period) -> dict[str, Any]:
    start, end = period.bounds()
    return {
        "label": period.label,
        "month": period.month,
        "year": period.year,
        "day": period.day.isoformat() if period.day else None,
        "assumed": period.assumed,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


def _capabilities_answer(actor: Actor) -> ChatAnswer:
    """Describe what this particular user can ask, not the whole catalogue."""
    topics: dict[str, list[str]] = {}
    for skill in registry.all_skills():
        if skill.company_wide and not actor.may_see_company:
            continue
        if skill.salary and not actor.may_see_salary and skill.company_wide:
            continue
        topics.setdefault(skill.topic, []).extend(skill.examples[:1])

    lines = [f"I can answer about {actor.describe_scope()}. For example:"]
    for topic, examples in sorted(topics.items()):
        for example in examples[:2]:
            lines.append(f"• {example}")
    lines.append(
        "You can also ask general questions about how the HRMS works — "
        "I'll answer those from general knowledge and say so."
    )
    return ChatAnswer(text="\n".join(lines), kind=CAPABILITIES)


def ask(
    *,
    question: str,
    actor: Actor,
    db: Session,
    include_debug: bool = False,
    last_skill: str | None = None,
    last_employee_id: int | None = None,
    last_period: str | None = None,
) -> ChatAnswer:
    """Answer one question. The only function the route calls."""
    started = time.monotonic()
    text = " ".join((question or "").split())
    if not text:
        return ChatAnswer(text="Ask me something about the company or an employee.",
                          kind=CLARIFY)

    low = text.lower()
    if any(phrase in low for phrase in _CAPABILITY_QUESTION) and len(text) < 60:
        return _capabilities_answer(actor)

    # --- who is this about? -------------------------------------------------
    resolution = resolver.resolve(db, actor, text)

    if resolution.ambiguous:
        return ChatAnswer(
            text=resolver.ambiguity_question(resolution),
            kind=CLARIFY,
            data={"candidates": [_employee_payload(e) for e in resolution.ambiguous]},
        )

    if resolution.unmatched_term:
        return ChatAnswer(
            text=(
                f"I couldn't find anyone matching \"{resolution.unmatched_term}\" "
                f"in {actor.describe_scope()}."
            ),
            kind=NOT_FOUND,
        )

    employee = resolution.employee
    employee_named = not resolution.named_nobody and employee is not None
    if employee is None:
        # Nobody named: a question about the asker, or a company-wide one.
        employee = resolver.self_employee(db, actor)

    period = period_module.parse(text)

    # --- carry the conversation forward -------------------------------------
    # A follow-up states only what changed. Everything it leaves out — who, when,
    # which lookup — comes from the previous turn, or the question is unanswerable.
    prior = registry.get(last_skill) if last_skill else None
    pronoun = bool(_PRONOUN.search(text)) and not employee_named
    is_followup = bool(
        prior
        and len(text.split()) <= _FOLLOWUP_MAX_WORDS
        and (pronoun or _FOLLOWUP.search(text) or not period.assumed or employee_named)
    )

    if is_followup and not employee_named and last_employee_id:
        # Re-loaded through the actor's own scope, never trusted from the client:
        # echoing back an id they may not see resolves to nothing.
        carried = resolver.load(db, actor, int(last_employee_id))
        if carried is not None:
            employee, employee_named = carried, True

    if is_followup and period.assumed and last_period:
        remembered = period_module.parse(last_period)
        if not remembered.assumed:
            period = remembered

    # --- which skill claims it? ---------------------------------------------
    # A "how does this work" question has no lookup, whatever it scores on
    # domain keywords. Checked before matching so a definition is not answered
    # with a figure.
    boost = prior.name if (prior and is_followup) else None
    if _PROCEDURAL.search(text):
        match = None
    else:
        match = registry.best(
            text,
            employee_named=employee_named,
            may_see_company=actor.may_see_company,
            boost_skill=boost,
        )

    debug = None
    if include_debug:
        ranked = registry.rank(
            text,
            employee_named=employee_named,
            may_see_company=actor.may_see_company,
            boost_skill=boost,
        )
        debug = {
            "employee_named": employee_named,
            "followup": is_followup,
            "period": period.label,
            "scores": [{"skill": m.skill.name, "score": m.score} for m in ranked[:6]],
            "chosen": match.skill.name if match else None,
        }

    if match is None:
        return _general(text, period, employee, employee_named, debug)

    # A near-tie between two different skills is a genuinely ambiguous question,
    # not a coin toss to resolve silently. "One by one for each employee" scores
    # identically for attendance, leave and payroll, and picking one would give
    # a confident answer to a question that was never asked. A follow-up is
    # exempt: the previous turn already settled which lookup is meant.
    contenders = [] if is_followup else registry.rank(
        text,
        employee_named=employee_named,
        may_see_company=actor.may_see_company,
    )
    tied = [
        m for m in contenders
        if m.score >= match.score - _TIE_MARGIN
        and (actor.may_see_company or not m.skill.company_wide)
    ]
    if len(contenders) > 1 and len(tied) > 1:
        options = "\n".join(f"• {m.skill.summary}" for m in tied[:3])
        return ChatAnswer(
            text="I can read that a few ways — which did you mean?\n" + options,
            kind=CLARIFY,
            data={"candidates": [m.skill.name for m in tied[:3]]},
            debug=debug,
        )

    skill = match.skill

    # --- may they run it? ---------------------------------------------------
    if skill.company_wide and not actor.may_see_company:
        return ChatAnswer(
            text=missing_permission_message(needs_company=True, needs_salary=False),
            kind=DENIED, skill=skill.name, debug=debug,
        )
    # Everyone may see their own pay; another person's needs the salary right.
    if skill.salary and employee_named and not actor.may_see_salary:
        return ChatAnswer(
            text=missing_permission_message(needs_company=False, needs_salary=True),
            kind=DENIED, skill=skill.name, debug=debug,
        )
    if skill.about_employee and employee is None:
        return ChatAnswer(
            text=(
                "Your account isn't linked to an employee record, so I don't have "
                "attendance, leave or payslips for you. Ask about a named employee, "
                "or contact HR to get your account linked."
            ),
            kind=DENIED, skill=skill.name, debug=debug,
        )

    # --- run it -------------------------------------------------------------
    context = SkillContext(
        question=text,
        db=db,
        actor=actor,
        period=period,
        employee=employee,
        employee_named=employee_named,
        llm=llm_module.get_llm(),
    )

    try:
        result = skill.handler(context)
    except Exception:  # noqa: BLE001 - one broken skill must not 500 the chat
        logger.exception("chatbot.skill_failed skill=%s user_id=%s", skill.name, actor.user_id)
        return ChatAnswer(
            text="Something went wrong looking that up. Please try again, "
                 "or check the relevant page.",
            kind=ERROR, skill=skill.name, debug=debug,
        )

    if result is None:
        # The skill ran and found nothing to report. That is an answer, not a
        # failure — but it is worth trying the model rather than dead-ending.
        return _general(text, period, employee, employee_named, debug,
                        prefix=f"I couldn't find any records for that.")

    logger.info(
        "chatbot.answered skill=%s user_id=%s named=%s followup=%s ms=%.0f",
        skill.name, actor.user_id, employee_named, is_followup,
        (time.monotonic() - started) * 1000,
    )
    return ChatAnswer(
        text=result.text,
        kind=ANSWER,
        skill=skill.name,
        data=result.data,
        sources=list(result.sources),
        employee=_employee_payload(employee) if skill.about_employee else None,
        period=_period_payload(period),
        debug=debug,
    )


def _general(
    text: str,
    period,
    employee,
    employee_named: bool,
    debug: dict[str, Any] | None,
    prefix: str = "",
) -> ChatAnswer:
    """No skill claimed it — answer from the model, with no employee data attached."""
    produced = llm_module.general_answer(text)
    if produced is None:
        return ChatAnswer(
            text=(
                prefix
                or "I don't have a lookup for that yet. Ask me about attendance, "
                   "leave, payroll, employees, holidays or documents — or raise a "
                   "query with HR."
            ).strip(),
            kind=GENERAL, debug=debug,
        )
    answer, flagged = produced
    return ChatAnswer(
        text=f"{prefix} {answer}".strip() if prefix else answer,
        kind=GENERAL,
        data={"unverified": flagged},
        debug=debug,
    )
