"""`result_validator` — decide whether the retrieved data can support an answer.

Entirely deterministic. These are arithmetic invariants of
`monthly_attendance_summary`, and checking them with Python is both cheaper and
more trustworthy than asking a language model whether some numbers look right.

What it enforces:

* the critical tool actually returned something;
* the payload has the fields and numeric types the response generator reads;
* the day buckets are internally consistent (they must reconcile against the
  calendar, and cannot exceed the working days they are drawn from);
* the period is meaningful for this employee (not before their joining date);
* non-critical tool failures degrade the answer rather than failing it.

A failure here does not produce a message — it produces a *kind*, which
`failure_classifier` turns into a route.
"""
from __future__ import annotations

from typing import Any, Mapping

from app.assistant import failures
from app.assistant.runtime import RunContext

#: Without this the question cannot be answered at all. The other two tools
#: enrich the answer and are allowed to fail.
CRITICAL_TOOL = "monthly_summary"

_REQUIRED_FIELDS = (
    "month", "year", "total_calendar_days", "working_days", "present",
    "half_day", "leave", "absent", "holiday", "weekly_off", "attendance_percentage",
)


def _fail(code: str, detail: str) -> dict:
    return {"code": code, "detail": detail}


def result_validator(state: Mapping[str, Any], ctx: RunContext) -> Mapping[str, Any]:
    results = state.get("tool_results") or {}
    errors = state.get("tool_errors") or {}
    problems: list[dict] = []
    kind = failures.VALIDATION_FAILURE

    # 1. Did the critical read happen at all?
    summary = results.get(CRITICAL_TOOL)
    if summary is None:
        tool_error = errors.get(CRITICAL_TOOL) or {}
        kind = tool_error.get("kind") or failures.MISSING_DATA
        problems.append(_fail("critical_tool_missing", f"{CRITICAL_TOOL} produced no result"))
        return _result(False, problems, kind, state)

    # 2. Structural validity.
    if not isinstance(summary, dict):
        return _result(
            False, [_fail("bad_shape", "summary is not a mapping")],
            failures.TOOL_FAILURE, state,
        )
    missing = [f for f in _REQUIRED_FIELDS if f not in summary]
    if missing:
        problems.append(_fail("missing_fields", ",".join(missing)))
    non_numeric = [
        f for f in _REQUIRED_FIELDS
        if f in summary and not isinstance(summary[f], (int, float))
    ]
    if non_numeric:
        problems.append(_fail("non_numeric_fields", ",".join(non_numeric)))

    # 3. Arithmetic consistency — only if the shape held up.
    if not problems:
        total = summary["total_calendar_days"]
        working = summary["working_days"]
        counted = summary["present"] + summary["half_day"] + summary["leave"] + summary["absent"]

        if working != total - summary["holiday"] - summary["weekly_off"]:
            problems.append(_fail(
                "working_days_mismatch",
                f"working={working} total={total} "
                f"holiday={summary['holiday']} weekly_off={summary['weekly_off']}",
            ))
        # Days that required attendance can never exceed the working days they
        # are drawn from. Fewer is normal: the current month has days that
        # haven't elapsed yet.
        if counted > working:
            problems.append(_fail("bucket_overflow", f"counted={counted} working={working}"))
        if not 0 <= summary["attendance_percentage"] <= 100:
            problems.append(_fail("percentage_out_of_range", str(summary["attendance_percentage"])))
        if not 1 <= summary["month"] <= 12:
            problems.append(_fail("bad_month", str(summary["month"])))

    # 4. Business rules from the employee context (when that branch succeeded).
    context = results.get("employee_period_context") or {}
    if context.get("employee_found") is False:
        return _result(
            False, [_fail("employee_not_found", "no employee row")],
            failures.MISSING_DATA, state,
        )
    if context.get("joined_after_period"):
        return _result(
            False,
            [_fail("period_before_joining", str(context.get("date_of_joining")))],
            failures.BUSINESS_RULE,
            state,
        )

    ok = not problems
    return _result(ok, problems, None if ok else kind, state)


def _result(
    ok: bool, problems: list[dict], kind: str | None, state: Mapping[str, Any]
) -> dict:
    update: dict[str, Any] = {
        "validation": {"ok": ok, "failures": problems},
        "_trace": {
            "ok": ok,
            "failure_codes": [p["code"] for p in problems],
            "kind": kind,
            # Which enrichments were unavailable — the response degrades, not fails.
            "degraded": sorted(
                set(("late_mark_count", "employee_period_context"))
                - set((state.get("tool_results") or {}).keys())
            ),
        },
    }
    if not ok:
        update["error"] = {
            "kind": kind or failures.VALIDATION_FAILURE,
            "message": failures.user_message(kind or failures.VALIDATION_FAILURE),
            "node": "result_validator",
        }
    return update
