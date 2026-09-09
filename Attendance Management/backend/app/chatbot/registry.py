"""The skill catalogue.

A *skill* is one read-only lookup that answers one kind of question. Adding a
topic to the chatbot means writing a function and decorating it — no routing
graph to rewire, no validator branch, no node to register. That is the whole
reason this module exists: the assistant this replaces needed five coordinated
edits to answer a new kind of question, which is why it never grew past one.

Matching is deterministic keyword scoring. It costs no tokens, is trivially
testable, and — unlike asking a 1.5B model to pick a tool — cannot invent a
skill that does not exist. The model is used later, for phrasing and for
questions no skill claims.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from sqlalchemy.orm import Session

from app.chatbot.access import Actor
from app.chatbot.period import Period
from app.chatbot.resolver import ResolvedEmployee


@dataclass
class SkillContext:
    """Everything a skill is allowed to read.

    The database session and the actor are here rather than passed separately so
    that a skill signature never grows: a new shared input is added once, here.
    """

    question: str
    db: Session
    actor: Actor
    period: Period
    #: The employee the question named, or the actor themselves when it named
    #: nobody. None when the question is company-wide or the actor has no record.
    employee: Optional[ResolvedEmployee] = None
    #: True when the question actually named someone, rather than defaulting to
    #: the asker. Skills use it to choose between "your" and "Priya's".
    employee_named: bool = False
    llm: Optional[Callable[..., str]] = None

    @property
    def low(self) -> str:
        return self.question.lower()

    def subject(self) -> str:
        """"you" / "Priya Sharma" — how to refer to the employee in the answer."""
        if not self.employee:
            return "you"
        return self.employee.full_name if self.employee_named else "you"

    def possessive(self) -> str:
        """"your" / "Priya Sharma's"."""
        if not self.employee or not self.employee_named:
            return "your"
        return f"{self.employee.full_name}'s"

    def is_self(self) -> bool:
        return not self.employee_named


@dataclass
class SkillResult:
    """What a skill produced. `text` is the answer; everything else is extra."""

    text: str
    data: dict[str, Any] | None = None
    #: Which tables or services were read. Surfaced to Admin/HR so a figure can
    #: be traced back to where it came from.
    sources: tuple[str, ...] = ()


Handler = Callable[[SkillContext], Optional[SkillResult]]


@dataclass(frozen=True)
class Skill:
    name: str
    summary: str
    handler: Handler
    #: term -> weight. Negative weights are how a skill says "not me": without
    #: them "how many employees are absent today" scores for headcount as
    #: loudly as for attendance.
    keywords: Mapping[str, float]
    #: Reads data about one employee, so a named person should steer towards it.
    about_employee: bool = False
    #: Reads across every employee. Requires `Actor.may_see_company`.
    company_wide: bool = False
    #: Reads pay. Requires `Actor.may_see_salary` unless it is the actor's own.
    salary: bool = False
    #: Phrases of which at least one MUST appear, or the skill does not compete
    #: at all. Weights alone cannot express "only when explicitly asked for":
    #: the per-employee breakdowns each carry their domain word, so "show leave"
    #: outscored the leave balance simply because "leave" is in both. A gate is
    #: the honest way to say "I answer this question and no other".
    requires: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    topic: str = "general"


_REGISTRY: dict[str, Skill] = {}

#: A skill must beat this before it is trusted to answer. Below it, the question
#: goes to the model instead of to a lookup that only half matched.
MIN_SCORE = 2.0

#: Naming a person is strong evidence for a per-employee skill and against a
#: company-wide one — "Priya's attendance" is not a company roll-call.
_NAMED_BONUS = 2.5
_NAMED_PENALTY = 3.0


def skill(
    *,
    name: str,
    summary: str,
    keywords: Mapping[str, float],
    about_employee: bool = False,
    company_wide: bool = False,
    salary: bool = False,
    requires: Sequence[str] = (),
    examples: Sequence[str] = (),
    topic: str = "general",
):
    """Register a function as a skill. The decorated function is returned as-is."""

    def decorate(handler: Handler) -> Handler:
        if name in _REGISTRY:
            raise ValueError(f"duplicate skill: {name}")
        # A decorator applies to whatever function follows it, so inserting a
        # helper between `@skill(...)` and its handler silently registers the
        # helper — which then receives a SkillContext and fails at runtime, in
        # production, as a generic "something went wrong". Every handler takes
        # `ctx`, so checking the parameter name turns that into an import error.
        first = next(iter(inspect.signature(handler).parameters), None)
        if first != "ctx":
            raise TypeError(
                f"skill {name!r} is registered on {handler.__name__}(), whose first "
                f"parameter is {first!r}, not 'ctx'. Is a helper defined between "
                f"the @skill decorator and its handler?"
            )
        _REGISTRY[name] = Skill(
            name=name,
            summary=summary,
            handler=handler,
            keywords=dict(keywords),
            about_employee=about_employee,
            company_wide=company_wide,
            salary=salary,
            requires=tuple(requires),
            examples=tuple(examples),
            topic=topic,
        )
        return handler

    return decorate


def all_skills() -> list[Skill]:
    return sorted(_REGISTRY.values(), key=lambda s: (s.topic, s.name))


def get(name: str) -> Skill | None:
    return _REGISTRY.get(name)


def _present(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}s?(?![a-z0-9])", f" {text.lower()} "))


def has_any(text: str, terms: Sequence[str]) -> bool:
    return any(_present(text, term) for term in terms)


def score(text: str, keywords: Mapping[str, float]) -> float:
    """Sum the weights of every keyword present, matched on word boundaries.

    The boundaries matter: without them "pl" fires inside "please" and "hr"
    inside "hrs".

    A trailing "s" is tolerated, so a skill declares "holiday" and still matches
    "holidays". People pluralise freely — "what are our weekly offs?" was
    reaching no skill at all before this — and making every author remember both
    forms is how a catalogue this size develops holes. Declare the singular;
    declaring both would double-count the plural.
    """
    low = f" {text.lower()} "
    total = 0.0
    for term, weight in keywords.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}s?(?![a-z0-9])", low):
            total += weight
    return total


@dataclass
class Match:
    skill: Skill
    score: float


#: How strongly the previous turn's skill is favoured during a follow-up.
#:
#: Deliberately small. It has to be enough to win when the new question carries
#: no domain word at all ("same for last month", "and Neha?"), and no more than
#: that: at 3.0 the previous skill also beat a domain word the user *did* type,
#: so "what about her leave?" kept answering with attendance.
FOLLOWUP_BOOST = 1.5


def rank(
    question: str,
    *,
    employee_named: bool,
    may_see_company: bool,
    boost_skill: str | None = None,
) -> list[Match]:
    """Every skill that scored anything, best first.

    `boost_skill` is the skill that answered the previous turn. During a
    follow-up it competes even when the new question mentions no domain at all,
    which is what makes "and Neha?" or "same for last month" mean anything.
    """
    matches: list[Match] = []
    for candidate in _REGISTRY.values():
        # A gated skill answers one shape of question. If that shape is not
        # asked for, it must not compete on its domain word alone.
        boosted = candidate.name == boost_skill
        if candidate.requires and not has_any(question, candidate.requires) and not boosted:
            continue
        value = score(question, candidate.keywords)
        if boosted:
            value += FOLLOWUP_BOOST
        elif value <= 0:
            continue
        if employee_named:
            value += _NAMED_BONUS if candidate.about_employee else 0.0
            value -= _NAMED_PENALTY if candidate.company_wide else 0.0
        elif candidate.company_wide and may_see_company:
            # No person named and the asker can see the company: a company-wide
            # reading is the more likely one.
            value += 1.0
        matches.append(Match(candidate, round(value, 2)))
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def best(
    question: str,
    *,
    employee_named: bool,
    may_see_company: bool,
    boost_skill: str | None = None,
) -> Match | None:
    matches = rank(
        question,
        employee_named=employee_named,
        may_see_company=may_see_company,
        boost_skill=boost_skill,
    )
    if matches and matches[0].score >= MIN_SCORE:
        return matches[0]
    return None


def catalogue() -> list[dict[str, Any]]:
    """The capability list, for the UI and for the "what can you do" answer."""
    return [
        {
            "name": s.name,
            "topic": s.topic,
            "summary": s.summary,
            "examples": list(s.examples),
            "company_wide": s.company_wide,
            "salary": s.salary,
        }
        for s in all_skills()
    ]
