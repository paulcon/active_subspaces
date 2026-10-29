"""Import rules of spec 12 §5.6 [A-11] (V12.6): AST scan and fresh-interpreter imports."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import active_subspaces as asub

PKG_DIR = Path(asub.__file__).parent
FOUNDATIONS = {"_version", "_codes", "errors", "_validation", "_random", "spaces", "results"}
ALLOWED = {
    "models": FOUNDATIONS,
    "registry": FOUNDATIONS | {"models"},
    "_reference_data": set(),
    "test_functions": FOUNDATIONS | {"models", "registry", "_reference_data"},
}
THIRD_PARTY = {
    "numpy",
    "concurrent",
    "__future__",
    "collections",
    "dataclasses",
    "math",
    "re",
    "types",
    "typing",
}


def _imports(path: Path) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0
            mod = node.module or ""
            if mod == "active_subspaces":
                out.extend((node.lineno, f"active_subspaces.{a.name}") for a in node.names)
            else:
                out.append((node.lineno, mod))
        elif isinstance(node, ast.Import):
            out.extend((node.lineno, a.name) for a in node.names)
    return out


def test_test_functions_imports():
    for name, allowed in ALLOWED.items():
        for line, mod in _imports(PKG_DIR / f"{name}.py"):
            if mod.startswith("active_subspaces"):
                assert mod.startswith("active_subspaces."), f"{name}.py:{line} imports {mod}"
                assert mod.split(".")[1] in allowed, f"{name}.py:{line} imports {mod}"
            else:
                assert mod.split(".")[0] in THIRD_PARTY, f"{name}.py:{line} imports {mod}"


def _run(code: str) -> str:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(PKG_DIR.parent), os.environ.get("PYTHONPATH", "")]),
    }
    proc = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_lazy_test_functions_fresh_interpreter():
    out = _run(
        "import sys, active_subspaces as a\n"
        "print('active_subspaces.test_functions' in sys.modules)\n"
        "print(a.test_functions.builtin_models().names()[0])\n"
        "print('active_subspaces.test_functions' in sys.modules)\n"
    )
    assert out.split() == ["False", "borehole", "True"]
    assert (
        _run("import active_subspaces.test_functions as tf; print(len(tf.builtin_models()))")
        == "9"
    )
