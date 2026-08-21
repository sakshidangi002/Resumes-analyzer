"""`verify` — did we achieve the goal? Not "did pytest exit 0".

A run that reviewed nothing, changed nothing and then ran a green suite has
passing tests and zero accomplishment. So every criterion here must be backed by
evidence already in state, and `goal_achieved` is the conjunction of the
criteria — never a summary judgement.

The node also distinguishes *achieved* from *achieved with caveats*. "Found 3
bugs, fixed 1, 2 need a human because they touch payroll" is a successful run,
and reporting it as a failure would be as dishonest as reporting it as clean.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from graph_engine.context import EngineContext
from graph_engine.skills import skill_for_node

logger = logging.getLogger(__name__)


def _criterion(name: str, ok: bool, evidence: str) -> dict:
    return {"name": name, "ok": ok, "evidence": evidence}


def verify(state: Mapping[str, Any], ctx: EngineContext) -> Mapping[str, Any]:
    skill_name, _ = skill_for_node("verify")

    files = state.get("files") or []
    findings = state.get("review_findings") or []
    bugs = state.get("bugs") or []
    dismissed = state.get("dismissed") or []
    fixes = state.get("fixes") or []
    changed = state.get("changed_files") or []
    tests = state.get("test_results") or {}
    baseline = set(state.get("baseline_failures") or [])

    criteria: list[dict] = []
    caveats: list[str] = []

    # 1. The scope was actually inspected.
    criteria.append(_criterion(
        "scope_reviewed", bool(files),
        f"{len(files)} Python file(s) inspected in {state.get('scope')!r}",
    ))

    # 2. Every finding was accounted for — triaged, dismissed, or already fixed.
    #
    # The "already fixed" arm matters on the drain loop: bug_analysis drops bugs
    # whose fix has landed so it does not re-propose them, which means they
    # appear in neither `bugs` nor `dismissed`. Counting only those two made a
    # successful multi-pass run look like it had lost track of its findings.
    def _key(item: dict) -> tuple:
        return (item.get("file"), item.get("line"), item.get("code"))

    accounted = {_key(b) for b in bugs} | {_key(d) for d in dismissed} | {_key(f) for f in fixes}
    unaccounted = [f for f in findings if _key(f) not in accounted]
    resolved = len(findings) - len(unaccounted)
    criteria.append(_criterion(
        "findings_triaged", not unaccounted,
        f"{resolved} of {len(findings)} finding(s) accounted for "
        f"({len(bugs)} open bug(s), {len(dismissed)} dismissed, "
        f"{sum(1 for f in fixes if f.get('applied'))} fixed)",
    ))

    # 3. Actionable bugs were addressed, or have a recorded reason they were not.
    actionable = [b for b in bugs if b.get("auto_fixable")]
    fix_keys = {(f["file"], f["line"]) for f in fixes}
    unaddressed = [b for b in actionable if (b["file"], b["line"]) not in fix_keys]
    criteria.append(_criterion(
        "actionable_bugs_addressed", not unaddressed,
        f"{len(actionable)} actionable bug(s); {len(unaddressed)} with no recorded outcome",
    ))
    if unaddressed:
        # Say what was dropped and why. A bounded run that stays silent about
        # its bound reads as "we covered everything" when it did not.
        reason = ("the fix budget was spent" if int(state.get("fix_budget", 0) or 0) <= 0
                  else "the run stopped before reaching them")
        caveats.append(
            f"{len(unaddressed)} actionable bug(s) were never attempted because "
            f"{reason} — rerun with a larger --fix-budget to continue"
        )
    not_fixable = [b for b in bugs if not b.get("auto_fixable")]
    if not_fixable:
        caveats.append(
            f"{len(not_fixable)} bug(s) need a human: "
            + "; ".join(sorted({b.get("not_fixable_reason") or "no mechanical fix"
                                for b in not_fixable}))
        )
    skipped_fixes = [f for f in fixes if not f["applied"]]
    if skipped_fixes:
        caveats.append(
            f"{len(skipped_fixes)} fix(es) not applied: "
            + "; ".join(sorted({f["reason"] for f in skipped_fixes}))
        )

    # A fix the engine *declined* (dry run, budget, dirty file) is a legitimate
    # outcome. A fix that *errored* means the engine broke, and reporting that as
    # an achieved goal would hide a defect in the engine itself.
    errored = [f for f in fixes if f.get("outcome") == "error"]
    criteria.append(_criterion(
        "no_engine_errors", not errored,
        "no fix strategy errored" if not errored
        else f"{len(errored)} fix(es) errored: "
             + "; ".join(sorted({f["reason"][:80] for f in errored})),
    ))

    # 4. No test regressed relative to the baseline.
    if tests:
        failing = {f["nodeid"] for f in (tests.get("failures") or [])}
        new_failures = sorted(failing - baseline)
        criteria.append(_criterion(
            "no_new_test_failures", not new_failures,
            f"{len(failing)} failing; {len(new_failures)} new vs baseline"
            + (f": {new_failures[:5]}" if new_failures else ""),
        ))
        if failing & baseline:
            caveats.append(
                f"{len(failing & baseline)} pre-existing failure(s) left untouched: "
                f"{sorted(failing & baseline)[:3]}"
            )
    else:
        # No tests ran. That is only acceptable when nothing was changed.
        criteria.append(_criterion(
            "no_new_test_failures", not changed,
            "no test run recorded" + (" but files were changed" if changed else
                                      " — nothing was changed, so nothing to regress"),
        ))

    # 5. Regression breadth.
    stage = tests.get("stage")
    if changed:
        criteria.append(_criterion(
            "regression_coverage", stage == "regression",
            f"last test stage was {stage!r}",
        ))
        if stage != "regression":
            caveats.append(
                "full-suite regression did not run; a targeted pass is weaker evidence"
            )

    # 6. Nothing outside the scope was touched.
    prefixes = [s.replace("\\", "/").rstrip("/") for s in ctx.config.scopes if s]
    outside = [
        c for c in changed
        if prefixes and not any(c.replace("\\", "/").startswith(p) for p in prefixes)
    ]
    criteria.append(_criterion(
        "no_collateral_damage", not outside,
        f"{len(changed)} file(s) changed, all within scope" if not outside
        else f"changed outside scope: {outside}",
    ))
    unexplained = [f["file"] for f in fixes if f["applied"] and not f.get("reason")]
    if unexplained:
        criteria.append(_criterion(
            "changes_explained", False, f"no reason recorded for: {unexplained}",
        ))

    goal_achieved = all(c["ok"] for c in criteria)

    if goal_achieved and caveats:
        summary = (
            f"Goal achieved with caveats. Reviewed {len(files)} file(s), "
            f"found {len(findings)} finding(s) → {len(bugs)} bug(s) after triage, "
            f"applied {sum(1 for f in fixes if f['applied'])} fix(es), "
            f"{len(changed)} file(s) changed. No new test failures."
        )
    elif goal_achieved:
        summary = (
            f"Goal achieved. Reviewed {len(files)} file(s), {len(bugs)} bug(s) after triage, "
            f"{sum(1 for f in fixes if f['applied'])} fix(es) applied, no new test failures."
        )
    else:
        failed = [c["name"] for c in criteria if not c["ok"]]
        summary = f"Goal not achieved. Unmet criteria: {', '.join(failed)}."

    verification = {
        "goal_achieved": goal_achieved,
        "criteria": criteria,
        "caveats": caveats,
        "summary": summary,
    }
    logger.info(
        "graph_engine.verify achieved=%s unmet=%s caveats=%d",
        goal_achieved, [c["name"] for c in criteria if not c["ok"]], len(caveats),
    )
    return {
        "verification": verification,
        "_trace": {
            "skill": skill_name,
            "goal_achieved": goal_achieved,
            "criteria": {c["name"]: c["ok"] for c in criteria},
            "caveat_count": len(caveats),
        },
    }
