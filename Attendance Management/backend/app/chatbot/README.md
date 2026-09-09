# HRMS Chatbot

Answers questions about the HRMS. Built for **Admin and HR first** — they can ask
about any employee and about the company as a whole — with Managers and
Employees automatically narrowed to what they may already see in the UI.

Endpoint: `POST /api/chatbot/ask` · `GET /api/chatbot/capabilities`
Chat model: the Resume Analyzer's local pipeline (`CHAT_MODEL`, default
`Qwen/Qwen2.5-1.5B-Instruct`) — reused, not loaded a second time.

> This package is **independent of `app/assistant/`**, which is a separate,
> attendance-only demonstration of a hand-rolled graph runtime. Neither imports
> the other, and changing one cannot break the other.

## How a question is answered

```
question
  -> resolver   who is it about?     (matched against the employees table, scoped by role)
  -> period     when is it about?    (today / last month / August 2026)
  -> registry   which skill claims it?  (keyword scoring, no tokens spent)
  -> access     may this actor run it?  (company-wide and salary are gated)
  -> skill      one read-only lookup -> one sentence
  -> llm        optional rephrase, rejected if any figure changed
```

Anything no skill claims goes to the chat model **with no employee data
attached**, so the bot always replies — but every figure it states came from a
query, never from the model.

## Adding a topic

One function. No routing to rewire.

```python
# app/chatbot/skills/mything.py
from app.chatbot.registry import SkillContext, SkillResult, skill

@skill(
    name="mything.example",
    topic="mything",
    summary="What this answers, shown in /capabilities.",
    keywords={"widget": 4.0, "widgets report": 5.0, "payslip": -3.0},
    about_employee=True,     # needs a resolved employee; a named person steers here
    company_wide=False,      # True -> Admin/HR only
    salary=False,            # True -> another person's pay needs Admin/HR
    examples=("How many widgets does Priya have?",),
)
def example(ctx: SkillContext) -> SkillResult | None:
    rows = ctx.db.query(...).filter(...).all()   # ctx.employee, ctx.period, ctx.actor
    if not rows:
        return None                              # -> falls through to the model
    return SkillResult(text="…", data={...}, sources=("widgets",))
```

Then add the module name to `_MODULES` in `skills/__init__.py`.

**Totals vs rows.** `payroll.company_run` answers "what did payroll total?" and
`breakdown.payroll` answers "payroll for each employee" — different questions,
so they are different skills. When two skills score within `_TIE_MARGIN` the
service asks which was meant instead of guessing; "one by one for each employee"
is equally valid for attendance, leave and payroll, and picking one silently
would answer a question nobody asked.

**Keyword weights.** Scoring sums the weights of the terms present; a trailing
`s` is tolerated, so declare the singular only (declaring both double-counts).
Negative weights are how a skill says "not me" — without
`"absent": -3.5` on `workforce.headcount`, "how many employees are absent today"
scores for headcount as loudly as for attendance.

**`requires=` is a gate, not a weight.** A skill that names one shape of
question — the per-employee breakdowns — must declare the phrases that shape
needs, and it then does not compete at all without one. Weights cannot express
this: each breakdown carries its own domain word, so "show leave" outscored the
leave *balance* purely because "leave" appears in both, and the breakdowns
quietly swallowed their whole domain. Use `requires` whenever a skill answers a
narrower question than its vocabulary suggests.

**Tuning is guarded.** `tests/test_chatbot_routing.py` asserts ~85 real
phrasings across pay, leave, attendance, holidays, employee details and
breakdowns land in the right area. Run it after any weight edit — a change for
one skill silently changes every other skill's chances:

```
pytest tests/test_chatbot_routing.py -q
```

## Conversation context

The endpoint is stateless; the UI echoes back what the previous turn settled —
`last_skill`, `last_employee_id`, `last_period`. A follow-up states only what
changed, so everything it leaves out is carried:

| Turn | Carried | Result |
|---|---|---|
| "Priya's attendance last month" | — | attendance, Priya, last month |
| "what about her leave?" | person, period | **leave**, Priya, last month |
| "and her payslip?" | person, period | **payslip**, Priya, last month |
| "and Neha?" | skill, period | payslip, **Neha**, last month |
| "same for last month" | skill, person | same lookup, **last month** |

`last_employee_id` is **never trusted**: it is re-loaded through
`resolver.load`, which applies the caller's own scope, so echoing back an id
they may not see resolves to nothing.

The previous skill competes via `FOLLOWUP_BOOST`, kept deliberately small. It
must win when the new turn names no domain ("and Neha?") and lose when it does
("what about her **leave**?"). At 3.0 it beat a domain word the user typed;
1.5 is the value the tests pin.

## Rules the code keeps

- **Figures come from the database, never the model.** The model rephrases (and
  the rewrite is discarded if any number changed) and answers questions no skill
  claims, where it is given no employee data at all.
- **Read-only.** No skill writes. `run_payroll_for_period`,
  `apply_leave_request` and `approve_leave_request` are deliberately unreachable
  — a question must never trigger a payroll run or an approval.
- **One authorisation boundary.** Skills never check a role. `Actor.employee_filter()`
  returns a SQLAlchemy criterion, applied inside the query, so an out-of-scope
  name is indistinguishable from a name that does not exist.
- **Policy is not re-derived.** Attendance goes through `monthly_attendance_summary`
  and leave through `paid_leave_summary`, so the chatbot cannot drift from the pages.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `CHATBOT_LLM_ENABLED` | `true` | `false` makes it fully deterministic: skills are unaffected, unmatched questions get the capability list instead of a generated answer. |
| `CHAT_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Read by `backend/main.py`; shared with the Resume Analyzer. |
| `PRELOAD_CHAT_MODEL` | unset | `1` warms the model at startup. Worth setting on a server, or the first question pays the load. |
