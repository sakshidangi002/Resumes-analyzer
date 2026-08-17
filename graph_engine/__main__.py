"""CLI: `python -m graph_engine --goal "..." --scope "..."`.

Defaults are read-only. `--apply-fixes` is required before a single byte is
written, so an accidental invocation reviews and reports instead of editing.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from graph_engine.config import (
    ALL_SCOPES,
    DEFAULT_FIX_BUDGET,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_ITERATIONS,
    EngineConfig,
)
from graph_engine.graph import run_engine


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="graph_engine",
        description="Run the review -> triage -> fix -> test -> verify workflow graph.",
    )
    parser.add_argument("--goal", required=True, help="What this run is meant to achieve.")
    parser.add_argument("--scope", action="append", default=[], metavar="PATH",
                        help="Repo-relative directory or file the run may read and "
                             "change. Repeat for several: --scope a --scope b")
    parser.add_argument("--all", action="store_true",
                        help="Whole application: " + ", ".join(ALL_SCOPES))
    parser.add_argument("--apply-fixes", action="store_true",
                        help="Actually modify files. Without this the run is read-only.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Permit edits to files with uncommitted modifications.")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--fix-budget", type=int, default=DEFAULT_FIX_BUDGET)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--skip-regression", action="store_true",
                        help="Targeted tests only. Faster, and weaker evidence.")
    parser.add_argument("--json", action="store_true", help="Emit the final state as JSON.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _print_report(result: dict) -> None:
    verification = result.get("verification") or {}
    metrics = result.get("metrics") or {}
    tests = result.get("test_results") or {}

    print()
    print("=" * 78)
    print(f"  run {result.get('run_id')}   status={result.get('status')}   "
          f"stop_reason={result.get('stop_reason')}")
    print("=" * 78)
    print(f"  scopes         : {result.get('scope')}")
    print(f"  files reviewed : {len(result.get('files') or [])}")
    print(f"  findings       : {len(result.get('review_findings') or [])}")
    print(f"  bugs (triaged) : {len(result.get('bugs') or [])}")
    print(f"  dismissed      : {len(result.get('dismissed') or [])}")

    fixes = result.get("fixes") or []
    print(f"  fixes applied  : {sum(1 for f in fixes if f['applied'])} / {len(fixes)} attempted")
    # Applied and errored first: on a whole-application run the declines are
    # dozens of identical lines and would bury the two rows that matter.
    ranked = sorted(fixes, key=lambda f: (f["outcome"] == "declined", f["file"]))
    for entry in ranked[:15]:
        mark = {"applied": "OK  ", "error": "ERR ", "declined": "SKIP"}[entry["outcome"]]
        print(f"      [{mark}] {entry['file']}:{entry['line']} {entry.get('code')} "
              f"-> {entry['reason']}")
    if len(ranked) > 15:
        print(f"      ... and {len(ranked) - 15} more (use --json for the full list)")

    if tests:
        print(f"  tests          : stage={tests.get('stage')} exit={tests.get('exit_code')} "
              f"passed={tests.get('passed')} failed={tests.get('failed')} "
              f"({tests.get('duration_s')}s)")
    analysis = result.get("failure_analysis")
    if analysis:
        print(f"  failure        : {analysis['failure_type']} -> {analysis['next_action']} "
              f"({analysis['decision_reason']})")

    print()
    print(f"  VERIFICATION   : {'ACHIEVED' if verification.get('goal_achieved') else 'NOT ACHIEVED'}")
    for criterion in verification.get("criteria", []):
        print(f"      [{'x' if criterion['ok'] else ' '}] {criterion['name']}: {criterion['evidence']}")
    for caveat in verification.get("caveats", []):
        print(f"      ! {caveat}")
    if verification.get("summary"):
        print(f"\n  {verification['summary']}")

    print(f"\n  metrics: {metrics}")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which cannot encode the arrows and
    # dashes in findings and skill text. Without this the whole run is lost to a
    # UnicodeEncodeError at the final print.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # not a real TTY / already wrapped
            pass

    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    scopes = tuple(ALL_SCOPES) if args.all else tuple(args.scope)
    if not scopes:
        parser_error = "either --scope PATH (repeatable) or --all is required"
        print(f"graph_engine: error: {parser_error}", file=sys.stderr)
        return 2

    config = EngineConfig(
        goal=args.goal,
        scopes=scopes,
        apply_fixes=args.apply_fixes,
        allow_dirty_files=args.allow_dirty,
        max_iterations=args.max_iterations,
        fix_budget=args.fix_budget,
        timeout_seconds=args.timeout,
        skip_regression=args.skip_regression,
    )
    result = run_engine(config)

    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "trace"},
                         indent=2, default=str))
    else:
        _print_report(result)

    # 0 = goal achieved, 1 = ran but did not achieve it, 2 = stopped on a
    # precondition. Distinct codes so CI can treat them differently.
    if result.get("status") == "done":
        return 0
    return 2 if result.get("status") == "stopped" else 1


if __name__ == "__main__":
    sys.exit(main())
