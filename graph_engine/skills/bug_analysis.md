# Skill: Bug Analysis (Triage)

Instruction set for the `bug_analysis` node.

## Responsibility

Turn review findings into a decision. For each finding: is it real, what is the
root cause, how urgent, and is it safe to fix automatically?

**Do not fix every finding.** A reviewer reports; a triager decides. Fixing
everything reported is how an autonomous system introduces bugs.

## False positives to dismiss

Dismissal is a result. Record it with a reason — never silently drop a finding.

| Pattern | Why it is not a bug |
|---|---|
| `col == None` inside `filter()` / `where()` | Idiomatic SQLAlchemy. `is None` returns a plain bool and silently drops the filter — rewriting it *creates* a bug. |
| Unused import in `__init__.py` | Re-export. Removing it breaks importers. |
| `except Exception: pass` in a cleanup / best-effort path | Often deliberate; the alternative is failing a request because logging failed. Check whether the surrounding code documents it. |
| Unused argument in an interface implementation | Required by the contract. |
| Finding in a test fixture that deliberately constructs a bad state | The bad state is the point. |

## Root cause

State the cause, not the symptom.

- Symptom: "test asserts True but got False".
- Root cause: "the default was changed in `config.py` without updating the test
  that pins it".

## Priority

Order by `severity` first, then by blast radius (how many callers), then by how
cheap the fix is. A `high` in a single leaf function outranks a `medium` in a
widely-called helper only when the `high` is actually reachable.

## Auto-fixable

Mark `auto_fixable: true` **only** when all of these hold:

1. The fix is mechanical — a known transformation, not a design decision.
2. It is local: one file, a handful of lines.
3. It cannot change behaviour for any caller.
4. It is covered by an existing test, or a test can be added in the same change.

Mark `auto_fixable: false` for anything touching: authorisation, database
migrations, money or payroll arithmetic, attendance status rules, concurrency, or
any public function signature. These need a human.

## Output

```json
{
  "file": "...", "line": 1, "code": "...", "severity": "high",
  "root_cause": "...", "priority": 1,
  "auto_fixable": true, "fix_strategy": "remove_unused_import",
  "dismissed": false, "dismiss_reason": null
}
```
