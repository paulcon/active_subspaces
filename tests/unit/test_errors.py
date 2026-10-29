"""Tests for spec 06: error and warning hierarchy, codes, warnings, logging."""

from __future__ import annotations

import ast
import logging
import os
import pickle
import re
import warnings
from pathlib import Path

import pytest

import active_subspaces as asub
from active_subspaces import errors
from active_subspaces._codes import CODES

ERRORS = [
    (asub.InputError, ValueError, "input.nonfinite"),
    (asub.SchemaError, ValueError, "schema.invalid"),
    (asub.UnsupportedError, NotImplementedError, "unsupported.space"),
    (asub.NumericalError, RuntimeError, "numerical.singular"),
    (asub.SolverError, RuntimeError, "solver.failed"),
    (asub.ModelError, RuntimeError, "model.raised"),
]
WARNINGS = [
    (asub.ValidityWarning, "validity.assumption"),
    (asub.AccuracyWarning, "accuracy.curvature_clipped"),
    (asub.ConvergenceWarning, "convergence.max_iter"),
    (asub.ExperimentalWarning, "experimental.module"),
]
PKG_DIR = Path(errors.__file__).parent


@pytest.mark.parametrize(("cls", "builtin", "code"), ERRORS)
def test_error_hierarchy(cls, builtin, code):
    exc = cls("message", code=code, details={"param": "X"})
    assert isinstance(exc, asub.ActiveSubspacesError)
    assert isinstance(exc, builtin)
    assert exc.code == code
    assert exc.details == {"param": "X"}
    with pytest.raises(builtin):
        raise exc
    assert code in str(exc)
    assert "message" in str(exc)


@pytest.mark.parametrize(("cls", "builtin", "code"), ERRORS)
def test_error_pickles(cls, builtin, code):
    exc = cls("boom", code=code, details={"row": 3, "got": [1, 2]})
    back = pickle.loads(pickle.dumps(exc))
    assert type(back) is cls
    assert back.code == code
    assert back.details == {"row": 3, "got": [1, 2]}
    assert back.message == "boom"


@pytest.mark.parametrize(("cls", "code"), WARNINGS)
def test_warning_hierarchy_and_pickle(cls, code):
    w = cls("careful", code=code, details={"k": 1})
    assert isinstance(w, asub.ActiveSubspacesWarning)
    assert isinstance(w, UserWarning)
    back = pickle.loads(pickle.dumps(w))
    assert type(back) is cls
    assert back.code == code
    assert back.details == {"k": 1}


def test_details_default_empty():
    assert asub.InputError("x", code="input.shape").details == {}


def test_code_table_is_static_and_complete():
    """Spec 06 §4.3: one static table of 44 codes, code -> (class name, meaning), checked
    against the classes.
    """
    assert len(CODES) == 44  # convergence.noise_not_detected removed [D-174]
    by_name = {name: getattr(errors, name) for name in errors.__all__}
    for code, (class_name, meaning) in CODES.items():
        family, _, _rest = code.partition(".")
        assert re.fullmatch(r"[a-z]+\.[a-z][a-z0-9_]*", code), code
        assert class_name in by_name, code
        cls = by_name[class_name]
        assert cls.family == family
        assert errors._family_class(code) is cls
        assert isinstance(meaning, str)
        assert meaning
        assert "\n" not in meaning
    assert errors._registered_codes() == {c: m for c, (_, m) in CODES.items()}
    assert not any(c.startswith("deprecated.") for c in CODES)  # none in v1.0 [D-152]
    with pytest.raises(TypeError):
        CODES["input.new_thing"] = ("InputError", "x")  # type: ignore[index]


def test_codes_module_imports_nothing_from_the_package():
    tree = ast.parse((PKG_DIR / "_codes.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert not (node.module or "").startswith("active_subspaces")
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("active_subspaces") for a in node.names)


def test_no_runtime_registration_api():
    assert not hasattr(errors, "_register_codes")
    assert not hasattr(errors, "registered_codes")
    assert not hasattr(errors, "_CODES")


def test_deprecation_warning_class():
    cls = asub.ActiveSubspacesDeprecationWarning
    assert issubclass(cls, asub.ActiveSubspacesWarning)
    assert issubclass(cls, DeprecationWarning)
    assert cls.family == "deprecated"
    assert "ActiveSubspacesDeprecationWarning" in asub.__all__
    w = cls("old", code="deprecated.parameter", details={"param": "x"})
    back = pickle.loads(pickle.dumps(w))
    assert type(back) is cls
    assert back.code == "deprecated.parameter"
    assert back.details == {"param": "x"}
    with pytest.warns(DeprecationWarning, match="old"):
        errors._warn(cls, "old", code="deprecated.parameter")


def test_errors_all_is_sorted_tuple_of_thirteen_classes():
    assert isinstance(errors.__all__, tuple)
    assert list(errors.__all__) == sorted(errors.__all__)
    assert len(errors.__all__) == 13


def _library_class_names() -> dict[str, str]:
    names = {}
    for name in errors.__all__:
        cls = getattr(errors, name)
        if cls.family:
            names[name] = cls.family
    return names


def test_every_raise_and_warn_uses_a_literal_registered_code():
    """AST scan of src/: library errors and _warn calls use literal codes from CODES of the right
    family,
    and every ``code=`` keyword literal anywhere in the package is in CODES."""
    classes = _library_class_names()
    problems = []
    for path in sorted(PKG_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            where = f"{path.relative_to(PKG_DIR)}:{node.lineno}"
            kw = {k.arg: k.value for k in node.keywords}
            code_node = kw.get("code")
            if (
                isinstance(code_node, ast.Constant)
                and isinstance(code_node.value, str)
                and code_node.value not in CODES
            ):
                problems.append(f"{where}: unregistered code {code_node.value!r}")
            func = node.func
            fname = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if fname in classes:
                family = classes[fname]
            elif fname == "_warn" and node.args:
                first = node.args[0]
                cname = first.id if isinstance(first, ast.Name) else getattr(first, "attr", None)
                family = classes.get(cname)
            else:
                continue
            if code_node is None:
                problems.append(f"{where}: no code")
                continue
            if not (isinstance(code_node, ast.Constant) and isinstance(code_node.value, str)):
                problems.append(f"{where}: code is not a literal")
                continue
            if family is not None and code_node.value.split(".")[0] != family:
                problems.append(
                    f"{where}: code {code_node.value!r} does not match class family {family!r}"
                )
    assert not problems, "\n".join(problems)


_DEEP_SRC = """
def outer(fn):
    return middle(fn)

def middle(fn):
    return fn()

def inner():
    _warn(AccuracyWarning, "degraded", code="accuracy.curvature_clipped", details={"x": 1})
"""


def _deep_namespace():
    ns = {"_warn": errors._warn, "AccuracyWarning": asub.AccuracyWarning}
    exec(compile(_DEEP_SRC, str(PKG_DIR / "_fake_deep_module.py"), "exec"), ns)
    return ns


@pytest.mark.parametrize("skip_prefixes", [True, False])
def test_warn_stacklevel_points_outside_package(monkeypatch, skip_prefixes):
    if skip_prefixes and not errors._USE_SKIP_PREFIXES:
        pytest.skip("skip_file_prefixes needs Python 3.12+")
    monkeypatch.setattr(errors, "_USE_SKIP_PREFIXES", skip_prefixes)
    ns = _deep_namespace()
    with pytest.warns(asub.AccuracyWarning) as rec:
        ns["outer"](ns["inner"])
    assert len(rec) == 1
    w = rec[0]
    assert os.path.abspath(w.filename) == os.path.abspath(__file__)
    assert w.message.code == "accuracy.curvature_clipped"
    assert w.message.details == {"x": 1}


def test_warnings_filterable_by_class():
    ns = _deep_namespace()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=asub.AccuracyWarning)
        ns["outer"](ns["inner"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=asub.ActiveSubspacesWarning)
        with pytest.raises(asub.AccuracyWarning):
            ns["outer"](ns["inner"])


def test_logging_only_null_handler():
    logger = logging.getLogger("active_subspaces")
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.NullHandler)
    assert logger.level == logging.NOTSET
    assert logger.propagate


def test_public_all():
    for name in errors.__all__:
        assert hasattr(asub, name)
        assert name in asub.__all__
    assert len(set(asub.__all__)) == len(asub.__all__)
    for name in asub.__all__:
        assert hasattr(asub, name)
