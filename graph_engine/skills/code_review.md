# Skill: Code Review

Instruction set for the `review` node.

## Responsibility

Inspect source in scope and report defects. **Report only — never modify.** The
review node has no write access; if you find yourself wanting to edit, that
belongs to `bug_fixing`.

## What to inspect

- Correctness: undefined names, unreachable code, shadowed definitions,
  mutable default arguments, exception handling that swallows failures.
- Security: credentials in source, authorisation checks that can be bypassed,
  user input reaching a query or a path without validation.
- Regression risk: behaviour depended on elsewhere, changed defaults, functions
  with many callers.
- Data integrity: silently discarded writes, unchecked commits.

## What NOT to report

- Style and formatting. Line length, import order and quote style are not
  defects, and reporting them buries the findings that matter.
- Type-annotation modernisation (`Optional[X]` → `X | None`). Cosmetic.
- Anything in vendored or generated trees: `third_party/`, `node_modules/`,
  `.venv/`, `dist/`, `build/`, `frontend_build/`.

## Severity

| Severity | Meaning |
|---|---|
| `high` | Wrong behaviour, data loss, or a security hole. Fix before shipping. |
| `medium` | Real defect with limited blast radius, or a latent failure. |
| `low` | Smell. Correct today; likely to become a bug. |

Severity describes consequence, not confidence. Report an uncertain `high` as
`high` and let triage judge whether it is real.

## Output format

One record per finding:

```json
{
  "file": "relative/path.py",
  "line": 123,
  "code": "F821",
  "severity": "high",
  "problem": "what is wrong",
  "reason": "why it is wrong here",
  "recommended_action": "the smallest change that resolves it",
  "source": "ruff | ast | llm"
}
```

Include any context fact that triage will need — for example whether a
`== None` comparison sits inside a SQLAlchemy `filter()` call, where it is
correct and must not be rewritten.

## Rules

- Never read outside the declared scope.
- Never report the same defect twice under two codes.
- If a file cannot be parsed, that is itself a `high` finding, not a crash.
