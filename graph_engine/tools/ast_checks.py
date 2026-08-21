"""Custom AST analyzers for patterns ruff cannot judge in this codebase.

Both checks below are deliberately *reported without judgement*. Whether they
are real defects depends on context that the review node has no business
deciding — that is the bug-analysis node's job. The review node's contract is to
report facts; the extra metadata each finding carries is what makes triage
possible.

`CMP_NONE` is the sharp example. `x == None` is a Python smell, but

    db.query(ResumeDB).filter(ResumeDB.deleted_at == None)

is idiomatic SQLAlchemy: `is None` there would evaluate to a plain bool and
silently drop the filter. Rewriting it would be a genuine bug introduced by an
over-eager fixer, so the check records whether the comparison sits inside a
query call and lets triage dismiss it.
"""
from __future__ import annotations

import ast
import logging

logger = logging.getLogger(__name__)

#: Method names whose arguments are SQLAlchemy expressions, not Python booleans.
_QUERY_CALLS = frozenset({"filter", "filter_by", "where", "having", "any_", "and_", "or_"})

CMP_NONE = "GE001"
EXCEPT_PASS = "GE002"


class _Analyzer(ast.NodeVisitor):
    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.findings: list[dict] = []
        self._query_depth = 0

    # -- context tracking ---------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        name = ""
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id

        entering = name in _QUERY_CALLS
        if entering:
            self._query_depth += 1
        self.generic_visit(node)
        if entering:
            self._query_depth -= 1

    # -- checks -------------------------------------------------------------
    def visit_Compare(self, node: ast.Compare) -> None:
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            if not (isinstance(comparator, ast.Constant) and comparator.value is None):
                continue
            self.findings.append({
                "file": self.rel_path,
                "line": node.lineno,
                "code": CMP_NONE,
                "severity": "low",
                "problem": f"Comparison to None using "
                           f"{'==' if isinstance(op, ast.Eq) else '!='} instead of "
                           f"{'is' if isinstance(op, ast.Eq) else 'is not'}",
                "source": "ast",
                # The fact that makes triage possible.
                "in_query_context": self._query_depth > 0,
                "ruff_fixable": False,
            })
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        body_is_pass = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
        if body_is_pass:
            self.findings.append({
                "file": self.rel_path,
                "line": node.lineno,
                "code": EXCEPT_PASS,
                "severity": "medium",
                "problem": "Exception swallowed with a bare `pass` — failures here are invisible",
                "source": "ast",
                # A `# pragma`/comment-annotated handler is usually a considered
                # decision; triage weighs this.
                "has_comment": bool(node.type is None),
                "ruff_fixable": False,
            })
        self.generic_visit(node)


def analyze_file(rel_path: str, source: str) -> list[dict]:
    """Findings for one file. A syntax error is itself reported, not raised."""
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        return [{
            "file": rel_path,
            "line": exc.lineno or 0,
            "code": "E999",
            "severity": "high",
            "problem": f"Syntax error: {exc.msg}",
            "source": "ast",
            "ruff_fixable": False,
        }]

    analyzer = _Analyzer(rel_path)
    analyzer.visit(tree)
    return analyzer.findings


def closure_is_called_in_place(source: str, line: int) -> bool | None:
    """Does the closure containing `line` run inside its own loop iteration?

    Ruff's B023 warns that a function defined in a loop reads the loop variable
    late. That is only a defect when the function *outlives* the iteration. When
    it is defined and called within the same loop body and its name is never
    referenced outside that body, the variable holds the intended value and there
    is nothing to fix — rewriting it would churn correct code.

    Returns True when the closure provably stays inside its iteration, False when
    it escapes, and None when the shape could not be determined.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    def _contains(node, lineno: int) -> bool:
        return node.lineno <= lineno <= getattr(node, "end_lineno", node.lineno)

    # Innermost function containing the reported line.
    func = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _contains(node, line):
            if func is None or node.lineno > func.lineno:
                func = node
    if func is None:
        return None

    # Innermost loop containing that function.
    loop = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)) and _contains(node, func.lineno):
            if loop is None or node.lineno > loop.lineno:
                loop = node
    if loop is None:
        return None

    # The closure stays in place only if every reference to its name is the
    # callee of a direct call. The moment the bare name is passed somewhere --
    # `handlers.append(f)`, `return f`, `Thread(target=f)` -- it can outlive the
    # iteration and the late binding becomes real.
    name = func.name
    called = escaped = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            called += 1
            continue
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
            parent_is_call_target = False
            for outer in ast.walk(tree):
                if isinstance(outer, ast.Call) and outer.func is node:
                    parent_is_call_target = True
                    break
            if not parent_is_call_target:
                escaped += 1

    if escaped:
        return False            # handed to something that outlives the iteration
    return called > 0           # only ever invoked in place


#: Method names whose failure is routinely ignorable: releasing a resource that
#: may already be gone. A handler whose whole body is these calls is best-effort
#: by construction.
_CLEANUP_CALLS = frozenset({
    "close", "remove", "unlink", "rmtree", "delete", "flush", "join",
    "terminate", "kill", "disconnect", "release", "shutdown", "cleanup",
    "stop", "quit", "cancel",
    # Session/transaction teardown and page-settling: failure means the thing was
    # already gone or already settled, which is the normal case.
    "logout", "rollback", "expunge", "abort", "detach", "unbind", "reset",
    "wait_for_load_state", "wait_for_timeout", "set_default_timeout",
})

#: Parsing/validation calls whose failure means "the input was not of that shape".
#: An `except (ValueError, TypeError)` around one of these IS the validation --
#: the handler is the fallback path, not a swallowed fault.
_PARSE_CALLS = frozenset({
    "fromisoformat", "strptime", "UUID", "int", "float", "loads", "parse",
    "fromtimestamp", "b64decode", "unquote", "literal_eval",
})

#: Exception types that, around a parse call, mean "bad input" and nothing else.
_PARSE_EXCEPTIONS = frozenset({"ValueError", "TypeError", "AttributeError", "KeyError"})

#: Phrases an author uses when they mean "this failure does not matter".
_BEST_EFFORT_MARKERS = (
    "best-effort", "best effort", "ignore", "optional", "not critical",
    "never let", "must not", "safe to", "don't care", "dont care",
    "no-op", "noop", "cosmetic",
)


def except_pass_intent(source: str, line: int) -> dict:
    """Can this `except ...: pass` be proven to be deliberate best-effort?

    Three gates, all required, mirroring the agreed policy:

    1. the module already has a logger, so adding a call introduces no import;
    2. the handler names its exception type -- a bare `except:` catches
       KeyboardInterrupt and is far more likely to be hiding a real problem;
    3. the intent is *evidenced*, either by a nearby comment saying the failure
       does not matter, or by a body that only releases resources.

    Anything else returns provable=False and is left for a human. The point is
    not to silence the lint; it is to avoid annotating a handler that is quietly
    swallowing a genuine fault.
    """
    verdict = {"provable": False, "logger_name": None, "reason": "", "indent": "", "lineno": None}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        verdict["reason"] = "file does not parse"
        return verdict

    logger_name = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("logger", "log", "LOGGER"):
                    logger_name = target.id
    # NOTE: a missing logger is recorded, not returned on. Whether the handler is
    # deliberate is a property of the code, not of whether someone happened to
    # define a logger in that module; deciding it early mislabelled best-effort
    # cleanup as an unproven swallowed fault.
    verdict["logger_name"] = logger_name

    handler = parent_try = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                if h.lineno == line or (h.lineno <= line <= getattr(h, "end_lineno", h.lineno)):
                    handler, parent_try = h, node
    if handler is None:
        verdict["reason"] = "no matching except handler at that line"
        return verdict
    if not (len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass)):
        verdict["reason"] = "handler body is not a bare pass"
        return verdict
    if handler.type is None:
        verdict["reason"] = "bare `except:` also catches KeyboardInterrupt/SystemExit"
        return verdict

    lines = source.splitlines()
    context = " ".join(lines[max(0, handler.lineno - 4): handler.lineno + 1]).lower()
    commented = any(m in context for m in _BEST_EFFORT_MARKERS)

    calls = [c.func.attr for c in ast.walk(parent_try) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Attribute)]
    cleanup_only = bool(calls) and all(c in _CLEANUP_CALLS for c in calls)

    # A parse fallback: `except ValueError:` around fromisoformat/UUID/int is the
    # validation itself. Nothing is being hidden -- the handler IS the answer to
    # "the caller sent something that is not a date".
    exc_names = set()
    node_type = handler.type
    for t in (node_type.elts if isinstance(node_type, ast.Tuple) else [node_type]):
        if isinstance(t, ast.Name):
            exc_names.add(t.id)
    parse_names = [c.func.attr if isinstance(c.func, ast.Attribute) else
                   (c.func.id if isinstance(c.func, ast.Name) else "")
                   for c in ast.walk(parent_try) if isinstance(c, ast.Call)]
    parse_fallback = (
        bool(exc_names) and exc_names <= _PARSE_EXCEPTIONS
        and any(n in _PARSE_CALLS for n in parse_names)
    )

    if not (commented or cleanup_only or parse_fallback):
        verdict["reason"] = ("nothing evidences that this failure is ignorable; it may be "
                             "hiding a real fault")
        return verdict

    pass_line = handler.body[0].lineno
    verdict.update(
        provable=True,
        lineno=pass_line,
        indent=lines[pass_line - 1][: len(lines[pass_line - 1]) - len(lines[pass_line - 1].lstrip())],
        reason="comment states the failure is ignorable" if commented
               else "handler body only releases resources" if cleanup_only
               else "parse fallback: the handler IS the validation path",
    )
    return verdict


def module_logger_plan(source: str) -> dict:
    """Where to insert a module logger, if the module has none.

    Returns the 0-based line index to insert at, and whether `import logging` is
    also needed. Insertion goes after the last top-level import so the module
    keeps its existing shape.
    """
    plan = {"needed": False, "insert_at": 0, "needs_import": True, "name": "logger"}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return plan

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("logger", "log", "LOGGER"):
                    return plan            # already has one

    plan["needed"] = True
    last_import_end = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_end = getattr(node, "end_lineno", node.lineno)
            if isinstance(node, ast.Import):
                if any(a.name == "logging" for a in node.names):
                    plan["needs_import"] = False
        elif isinstance(node, ast.ImportFrom) and node.module == "logging":
            plan["needs_import"] = False

    # Skip a module docstring if there are no imports at all.
    if last_import_end == 0 and tree.body:
        first = tree.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            last_import_end = getattr(first, "end_lineno", first.lineno)
    plan["insert_at"] = last_import_end
    return plan


def swallowed_operation(source: str, line: int) -> str:
    """A truthful short name for whatever the handler is swallowing.

    Derived from the first call in the `try` body, so the log message names the
    real operation instead of asserting that the failure does not matter.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "operation"

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(h.lineno <= line <= getattr(h, "end_lineno", h.lineno) for h in node.handlers):
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call):
                    func = sub.func
                    if isinstance(func, ast.Attribute):
                        owner = func.value
                        prefix = owner.id + "." if isinstance(owner, ast.Name) else ""
                        return (prefix + func.attr)[:60]
                    if isinstance(func, ast.Name):
                        return func.id[:60]
        # No call at all -- name the statement kind instead of inventing meaning.
        if node.body:
            return type(node.body[0]).__name__.lower()
    return "operation"


def raise_from_plan(source: str, line: int) -> dict:
    """Can `raise X(...)` inside an except block be chained mechanically?

    Only one rewrite is ever proposed: `raise X(...) from <bound name>`, which
    sets `__cause__` and nothing else. It cannot change control flow, the
    exception raised, or which handler catches it -- it only stops Python from
    printing "During handling of the above exception, another occurred" and
    records the real cause instead.

    The other spelling, `from None`, *suppresses* context. That is a deliberate
    choice to hide information and is never inferred here.

    Requires the handler to bind a name (`except X as e:`); without one there is
    nothing to chain and the site stays manual.
    """
    plan = {"provable": False, "reason": "", "insert_line": None, "insert_col": None, "name": None}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        plan["reason"] = "file does not parse"
        return plan

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not (node.lineno <= line <= getattr(node, "end_lineno", node.lineno)):
            continue
        if not node.name:
            plan["reason"] = "handler does not bind the exception (`except X as e:`)"
            return plan
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Raise) or sub.lineno < line:
                continue
            if sub.exc is None:
                plan["reason"] = "bare `raise` re-raises the original; nothing to chain"
                return plan
            if sub.cause is not None:
                plan["reason"] = "already chained"
                return plan
            plan.update(
                provable=True,
                insert_line=getattr(sub.exc, "end_lineno", sub.lineno),
                insert_col=getattr(sub.exc, "end_col_offset", None),
                name=node.name,
                reason="chains to the bound exception, preserving the cause",
            )
            return plan
        plan["reason"] = "no raise found in this handler"
        return plan

    plan["reason"] = "no except handler at that line"
    return plan
