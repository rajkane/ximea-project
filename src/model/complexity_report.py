"""Complexity report tool (same logic as `src/tests/test_complexity_gate.py`).

Usage:
  python -m src.model.complexity_report

Defaults:
- MAX_COMPLEXITY=10
- output: /tmp/complexity_report.txt

Optional env overrides:
- COMPLEXITY_MAX
- COMPLEXITY_OUT
"""

from __future__ import annotations

import ast
import os
from typing import Iterable


def _iter_py_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.py') and not fn.startswith('test_'):
                yield os.path.join(dirpath, fn)


class _ComplexityVisitor(ast.NodeVisitor):
    """Very small cyclomatic complexity approximation (dependency-free)."""

    def __init__(self) -> None:
        self.complexity = 1

    def generic_visit(self, node) -> None:
        if isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.Try,
                ast.With,
                ast.BoolOp,
                ast.IfExp,
                ast.ExceptHandler,
            ),
        ):
            self.complexity += 1
        super().generic_visit(node)


def _function_complexity(fn_node: ast.AST) -> int:
    v = _ComplexityVisitor()
    v.visit(fn_node)
    return v.complexity


def _write_report(out_path: str, max_complexity: int, offenders: list[tuple[int, str, str, int]]) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"MAX_COMPLEXITY={max_complexity}\n")
        f.write(f"offenders={len(offenders)}\n")
        for c, p, name, ln in offenders:
            f.write(f"- {c:>3} {p}:{ln} {name}\n")


def main() -> int:
    max_complexity = int(os.getenv('COMPLEXITY_MAX', '10'))
    out_path = os.getenv('COMPLEXITY_OUT', '/tmp/complexity_report.txt')

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    offenders: list[tuple[int, str, str, int]] = []

    for path in _iter_py_files(src_root):
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()
        tree = ast.parse(src, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                c = _function_complexity(node)
                if c > max_complexity:
                    offenders.append((c, path, node.name, node.lineno))

    offenders.sort(reverse=True, key=lambda x: x[0])
    _write_report(out_path, max_complexity, offenders)

    print(out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
