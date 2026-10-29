"""Tests for spec 15 §4.3–4.4 and §8: namespace, ``__all__``, layering, import side effects."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import active_subspaces as asub

PKG_DIR = Path(asub.__file__).parent
# Layer 0, in order: each module may import only the ones before it (spec 15 §4.4).
LAYER0 = ("_version", "_codes", "errors", "_validation", "_random", "spaces", "results")


def _package_imports(path: Path) -> list[tuple[int, str]]:
    """Every package-internal import in ``path`` (module level or in a function body), as
    (line, module).
    """
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}:{node.lineno}: relative import [P-02]"
            mod = node.module or ""
            if mod == "active_subspaces":
                out.extend((node.lineno, f"active_subspaces.{a.name}") for a in node.names)
            elif mod.startswith("active_subspaces."):
                out.append((node.lineno, mod))
        elif isinstance(node, ast.Import):
            out.extend(
                (node.lineno, a.name) for a in node.names if a.name.startswith("active_subspaces")
            )
    return out


def test_layer0_import_order():
    for i, name in enumerate(LAYER0):
        allowed = {f"active_subspaces.{m}" for m in LAYER0[:i]}
        for line, mod in _package_imports(PKG_DIR / f"{name}.py"):
            assert mod in allowed, (
                f"{name}.py:{line} imports {mod}; layer 0 allows only {sorted(allowed)}"
            )


def _core_modules() -> list[types.ModuleType]:
    lazy = set(getattr(asub, "_LAZY_SUBMODULES", ()))
    return [
        getattr(asub, n)
        for n in asub.__all__
        if n not in lazy and isinstance(getattr(asub, n), types.ModuleType)
    ]


def test_all_is_sorted_tuple_without_private_names():
    for mod in [asub, *_core_modules()]:
        assert isinstance(mod.__all__, tuple), mod.__name__
        assert list(mod.__all__) == sorted(mod.__all__), mod.__name__
        assert len(set(mod.__all__)) == len(mod.__all__), mod.__name__
        for name in mod.__all__:
            assert hasattr(mod, name), f"{mod.__name__}.{name}"
            assert not name.startswith("_") or (mod is asub and name == "__version__"), (
                f"{mod.__name__}.{name}"
            )


def test_all_is_complete():
    """Every public function, class and module-level object defined in a public module is in
    its __all__.
    """
    for mod in _core_modules():
        for name, obj in vars(mod).items():
            if name.startswith("_") or isinstance(obj, types.ModuleType):
                continue
            owner = getattr(obj, "__module__", None)
            if inspect.isclass(obj) or inspect.isfunction(obj):
                if owner != mod.__name__:
                    continue
            elif owner is not None or type(obj).__module__ == "__future__":
                continue  # imported typing constructs, __future__ features
            assert name in mod.__all__, (
                f"{mod.__name__}.{name} is public-looking but not in __all__"
            )


def test_top_level_all():
    expected = {"__version__"}
    for mod in _core_modules():
        expected.add(mod.__name__.rsplit(".", 1)[1])
        expected.update(getattr(mod, "_toplevel", mod.__all__))
    expected.update(getattr(asub, "_LAZY_SUBMODULES", ()))
    assert set(asub.__all__) == expected
    for name in asub.__all__:
        assert hasattr(asub, name)


def test_foundation_names_at_top_level():
    for name in (
        "Uniform",
        "Gaussian",
        "InputSpace",
        "Result",
        "QoIResult",
        "Provenance",
        "load",
        "from_dict",
        "ActiveSubspacesDeprecationWarning",
        "errors",
        "results",
        "spaces",
    ):
        assert name in asub.__all__
    assert asub.spaces.__all__ == ("Gaussian", "InputSpace", "Uniform")


def test_import_side_effects():
    code = (
        "import logging, sys, active_subspaces\n"
        "bad = [m for m in ('matplotlib', 'active_subspaces.plotting', "
        "'active_subspaces.experimental',"
        " 'active_subspaces.test_functions') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "h = logging.getLogger('active_subspaces').handlers\n"
        "assert len(h) == 1 and isinstance(h[0], logging.NullHandler), h\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PKG_DIR.parent), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert proc.stderr == ""


def test_no_top_level_name_collisions():
    owners: dict[str, list[str]] = {}
    for mod in _core_modules():
        for name in getattr(mod, "_toplevel", mod.__all__):
            owners.setdefault(name, []).append(mod.__name__)
    assert {n: o for n, o in owners.items() if len(o) > 1} == {}


def test_lazy_submodules_and_experimental_not_exposed():
    assert set(asub._LAZY_SUBMODULES) == {"plotting", "test_functions"}
    assert set(asub._LAZY_SUBMODULES) <= set(asub.__all__)
    assert set(asub._LAZY_SUBMODULES) <= set(dir(asub))
    assert "experimental" not in asub.__all__
    assert asub.plotting.__name__ == "active_subspaces.plotting"
    assert asub.test_functions.__name__ == "active_subspaces.test_functions"


def test_py_typed_marker():
    assert (PKG_DIR / "py.typed").is_file()


# Removed 2016 top-level names: spec 15 §4.3, spec 18 §5.4 and §8 item 3 [D-168].
REMOVED_2016 = (
    "utils",
    "domains",
    "integrals",
    "optimizers",
    "response_surfaces",
    "base",
    "sdr",
    "gradfree",
    "ActiveSubspaceReducedModel",
)


def test_removed_2016_table_keys():
    assert set(asub._REMOVED_2016) == set(REMOVED_2016)
    assert "subspaces" not in asub._REMOVED_2016
    assert "gradients" not in asub._REMOVED_2016


def test_removed_2016_names_raise_attribute_error_with_guide():
    before = set(sys.modules)
    for key in REMOVED_2016:
        try:
            getattr(asub, key)
        except AttributeError as exc:
            msg = str(exc)
        else:
            raise AssertionError(f"ac.{key} resolved")
        assert f"no attribute {key!r}" in msg
        assert "2016 API (v0.1)" in msg
        assert asub._REMOVED_2016[key] in msg
        assert "https://active-subspaces.readthedocs.io/en/stable/migration.html" in msg
        assert not hasattr(asub, key)
        assert key not in asub.__all__
        assert key not in dir(asub)
    assert set(sys.modules) == before  # nothing imported


def test_unknown_attribute_message_is_plain():
    with pytest.raises(AttributeError) as excinfo:
        _ = asub.no_such_name
    assert str(excinfo.value) == "module 'active_subspaces' has no attribute 'no_such_name'"


def test_removed_2016_submodule_import_fails_normally():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("active_subspaces.utils")
