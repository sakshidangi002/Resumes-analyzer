# Skill: Failure Analysis

Instruction set for the `failure_analysis` node. Runs **only** when tests fail.

## Responsibility

Read the actual failure output and classify it. Then decide the next action.
**Do not send every failure back to the fixer** — most repeat-failure loops come
from a fixer being handed a failure it cannot possibly resolve.

## Classification

| `failure_type` | Signal | `next_action` |
|---|---|---|
| `pre_existing` | The same node id failed in the baseline run | `ignore` |
| `implementation_bug` | Assertion about behaviour the change touched | `fix` |
| `test_bug` | The test encodes a stale expectation (e.g. pins a default that legitimately changed) | `stop` — a human decides whether code or test is wrong |
| `environment` | `ImportError`, missing binary, missing model weights, no `SECRET_KEY` | `stop` |
| `dependency` | Version conflict, `ModuleNotFoundError` for a declared package | `stop` |
| `configuration` | Missing or wrong env var, `.env` not loaded | `stop` |
| `collection_error` | Syntax error or import failure at collection time | `fix` if inside the change, else `stop` |
| `unknown` | Cannot be determined from the output | `stop` |

## Root cause

Name the mechanism, and point at a file. "Tests fail" is not a root cause.
"`config.py` flipped `seat_anchor_enabled` to `False` while
`test_seat_anchoring.py` still asserts `True`" is.

## Repetition

If the identical failure signature has already been seen this run, stop. Do not
try a different fix for the same failure more than once — three attempts at the
same error means the diagnosis is wrong, not the fix.

## Only `fix` when

1. The failure is inside a file this run changed, **and**
2. the cause is understood, **and**
3. the fix does not require weakening the test.

Otherwise `stop` and report. A structured failure report is a useful outcome; an
endless loop is not.

## Output

```json
{
  "failure_type": "implementation_bug",
  "root_cause": "...",
  "affected_files": ["..."],
  "next_action": "fix | ignore | stop",
  "signature": "stable id for this exact failure"
}
```
