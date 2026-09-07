# Known issues

Issues found but deliberately NOT fixed in the ticket that found them, because
fixing them needs a product decision or belongs to a different ticket.

---

## KI-001 — `seat_anchor_enabled`: the test and the config disagree

**Found during:** T-01 (health endpoint). Unrelated to that change.

**Status:** open — needs a product decision, not a code fix.

`tests/test_seat_anchoring.py::test_seat_anchoring_is_enabled_by_default`
asserts:

```python
assert get_settings().seat_anchor_enabled is True
```

but `app/core/config.py:436` declares:

```python
seat_anchor_enabled: bool = False
```

and no `.env` in the repository sets it. The test fails on a clean checkout.

**Why this is not a one-line fix.** One of the two is wrong about intended
product behaviour, and it is not obvious which:

* If seat anchoring is *meant* to be on by default, the config default is the
  bug, and turning it on changes live identification behaviour on the monitor
  cameras — it lets a person be labelled from their seat position when Re-ID
  and face both fail. That is a behaviour change to the identification path and
  must not be made casually.
* If it is *meant* to be off by default, the test is asserting a policy that was
  never adopted, and should be deleted or inverted.

Flipping the config to make a red test green would silently enable a feature on
the identification path. Deleting the test would discard whatever intent it was
written to protect.

**Decide first:** should seat anchoring be on by default? Then change the losing
side. Note that seat anchoring only ever *labels* a box on a MONITOR camera —
it can never write attendance (monitor cameras are hard-blocked) — so the blast
radius is display and occupancy, not payroll.

**Do not** bundle this into an unrelated ticket.

---

## KI-002 — `health_snapshot()` reads a ThreadPoolExecutor private

**Found during:** T-01. Accepted for now.

**Status:** open — technical debt, low severity.

`CameraManager.health_snapshot()` reads the attendance writer's queue depth via
`_attendance_executor._work_queue.qsize()`. `_work_queue` is a CPython stdlib
internal, not a supported interface.

This follows existing precedent — `_submit_attendance` already reads the same
attribute for its overflow guard — so T-01 did not make the situation worse.
But it is still fragile in a direction that matters: if a CPython change makes
the attribute disappear, the guarded `except` reports the depth as unknown,
which `/health` treats as healthy. It fails *open*.

**The fix:** own the number instead of inspecting someone else's. Wrap
submission in a small counter (increment on submit, decrement when the task
starts running) and read that. Deferred from T-01 because it modifies the
attendance write path, which was out of that ticket's scope.

Worth folding into a ticket that already touches attendance submission.
