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
