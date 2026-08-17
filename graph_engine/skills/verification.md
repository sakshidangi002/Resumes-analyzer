# Skill: Verification

Instruction set for the `verify` node.

## Verification is not testing

| | Question |
|---|---|
| Testing | Do the tests pass? |
| Verification | Did we achieve the goal that was asked for? |

**`pytest` returning 0 is not verification.** A run that reviewed nothing,
changed nothing and then ran a green suite has passed its tests and achieved
nothing. Green tests are one piece of evidence, not the verdict.

## Acceptance criteria

Every criterion must be backed by evidence held in state, not by an assertion
that it seems fine.

| Criterion | Evidence required |
|---|---|
| Scope was actually reviewed | non-empty `files`, review completed |
| Findings were triaged | every finding is either in `bugs` or in `dismissed` with a reason |
| Actionable bugs were addressed | each `auto_fixable` bug has a fix recorded as applied, or an explicit reason it was not |
| No new test failures | failing node ids ⊆ `baseline_failures` |
| Regression coverage exists | a full-suite run happened, or the report states it was skipped |
| No collateral damage | `changed_files` ⊆ scope, and every change has a recorded reason |

## Reject when

- No files were reviewed — the scope was empty or wrong.
- Bugs were found, none were fixed, and no reason was recorded for any of them.
- A test that passed in the baseline now fails.
- A file outside the scope was modified.
- Fixes were applied but no tests ran afterwards.

## An honest partial result

"Reviewed 12 files, found 3 bugs, fixed 1, 2 need a human because they touch
payroll arithmetic, no regressions" is a **successful** run. Report it as
achieved-with-caveats and name the caveats.

Do not round a partial result up to success, and do not round it down to failure.

## Output

```json
{
  "goal_achieved": true,
  "criteria": [{"name": "...", "ok": true, "evidence": "..."}],
  "summary": "one honest paragraph",
  "caveats": ["..."]
}
```
