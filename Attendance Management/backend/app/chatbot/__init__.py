"""The HRMS chatbot.

A question-answering layer over the HRMS, aimed first at Admin and HR: ask about
any employee's attendance, leave or payslips, about company-wide totals, or
about how the system works.

    from app.chatbot import Actor, ask
    answer = ask(question="how many people are absent today?",
                 actor=Actor.from_user(current_user), db=db)

Design in one paragraph. A *skill* is one read-only lookup registered with a
decorator (`app/chatbot/skills/`). A question is matched to a skill by keyword
scoring, the actor's role decides whether that skill may run, and the skill
reads the database and writes a sentence. Anything no skill claims goes to the
local chat model with **no employee data attached**, so the bot always replies
while every figure it states still comes from a query. Adding a new topic is one
function in one file — there is no workflow graph to rewire.

This package is independent of `app/assistant/`, which is a separate,
attendance-only demonstration of a hand-rolled graph runtime. Neither imports
the other.
"""
from app.chatbot.access import Actor
from app.chatbot.service import ChatAnswer, ask

__all__ = ["Actor", "ChatAnswer", "ask"]
