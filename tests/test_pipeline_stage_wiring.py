"""Guard the stage call sites in ``koala.pipeline.run`` against dead wiring.

The stage functions receive most of their inputs through
``name=locals().get('name')`` keyword arguments. That only works for names
bound inside ``run`` itself; a module-level helper or an undefined name is
silently passed as ``None`` and fails much later (or only for configurations
the regression gate does not exercise). This test fails as soon as such a
name appears.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "koala" / "pipeline.py"

# Names the callee rebinds with a local import before use.
_REBOUND_BY_CALLEE = {"prepare_laplace_metric"}


def _run_locals(tree):
    run = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    bound = set()
    for node in ast.walk(run):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.FunctionDef):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return bound


def test_locals_get_arguments_are_bound_inside_run():
    source = PIPELINE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    bound = _run_locals(tree)
    requested = set(re.findall(r"locals\(\)\.get\('([A-Za-z_][A-Za-z_0-9]*)'\)", source))
    unbound = sorted(requested - bound - _REBOUND_BY_CALLEE)
    assert not unbound, (
        "These stage inputs are passed as locals().get(...) but are never bound "
        f"inside koala.pipeline.run, so the stages receive None: {unbound}"
    )
