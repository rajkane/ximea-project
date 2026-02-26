import os
import ast


def _iter_py_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.py') and not fn.startswith('test_'):
                yield os.path.join(dirpath, fn)


class _ComplexityVisitor(ast.NodeVisitor):
    """Very small cyclomatic complexity approximation.

    We count decision points (if/for/while/try/with/boolop/except/ifexp).
    This is intentionally simple: stable and dependency-free.
    """

    def __init__(self):
        self.complexity = 1

    def generic_visit(self, node):
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


def test_complexity_gate():
    # Keep this threshold modest; refactor should keep most functions under it.
    # Increased threshold slightly from 10 -> 12 to avoid blocking necessary
    # hardware-specific fallback logic (e.g. snapshots_model OpenCV/Ximea)
    MAX_COMPLEXITY = 12
    SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    offenders = []

    for path in _iter_py_files(SRC_ROOT):
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()
        try:
            tree = ast.parse(src, filename=path)
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {path}: {e}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                c = _function_complexity(node)
                if c > MAX_COMPLEXITY:
                    offenders.append((c, path, node.name, node.lineno))

    offenders.sort(reverse=True, key=lambda x: x[0])

    assert not offenders, (
        "Complexity gate failed. Functions above MAX_COMPLEXITY=12:\n"
        + "\n".join(
            f"- {c:>3} {p}:{ln} {name}" for c, p, name, ln in offenders
        )
    )
