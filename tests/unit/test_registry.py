"""Unit tests of the model registry (spec 12 §4.5, §5.5; V12.23)."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces.errors import InputError
from active_subspaces.models import Model
from active_subspaces.registry import ModelRegistry

NINE = (
    "borehole",
    "otl_circuit",
    "piston",
    "quadratic_gaussian_10",
    "quadratic_uniform_10",
    "ridge_sin_gaussian_10",
    "ridge_sin_uniform_10",
    "robot_arm",
    "wing_weight",
)


def _value(X):
    return X.sum(axis=1)


def _model(name, m=2):
    return Model(space=asub.Uniform.reference(m), value=_value, name=name)


def test_registry():
    from active_subspaces import test_functions as tf

    reg = tf.builtin_models()
    again = tf.builtin_models()
    assert reg is not again
    assert reg.names() == NINE
    assert tuple(reg) == NINE
    assert len(reg) == 9
    assert all(reg[n] is again[n] and reg.get(n) is reg[n] for n in NINE)
    assert "borehole" in reg
    assert "nope" not in reg
    assert 3 not in reg
    assert reg["borehole"] is tf.BOREHOLE
    d = reg.describe()
    json.dumps(d, allow_nan=False)
    assert [x["name"] for x in d] == list(NINE)
    bh = d[0]
    assert bh["dim"] == 8
    assert bh["qoi_units"] == "m^3/yr"
    assert bh["references"] == ["SurjanovicBingham"]
    assert bh["has_grad"] is True
    assert bh["has_value_and_grad"] is False
    assert bh["cost"] == 1e-6
    assert bh["reference_results"]["type"] == "ReferenceResults"
    assert bh["space"]["names"] == ["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw"]
    assert d[NINE.index("quadratic_uniform_10")]["has_value_and_grad"] is True


def test_registry_construction_and_errors():
    a, b = _model("a"), _model("b")
    reg = ModelRegistry([b, a])
    assert reg.names() == ("a", "b")
    assert ModelRegistry().names() == ()
    with pytest.raises(InputError) as e:
        ModelRegistry([a, _model("a")])
    assert e.value.code == "input.duplicate_model"
    for bad in ([Model(space=asub.Uniform.reference(2), value=_value)], ["a"], [3]):
        with pytest.raises(InputError) as e:
            ModelRegistry(bad)
        assert e.value.code == "input.parameter"
    with pytest.raises(InputError) as e:
        ModelRegistry(a)
    assert e.value.code == "input.parameter"
    with pytest.raises(InputError) as e:
        reg.get("c")
    assert e.value.code == "input.unknown_model"
    assert e.value.details["known"] == ["a", "b"]
    with pytest.raises(InputError) as e:
        reg["c"]
    assert e.value.code == "input.unknown_model"
    with pytest.raises(InputError):
        reg.get(None)  # type: ignore[arg-type]


def test_with_models_is_functional():
    a, b = _model("a"), _model("b")
    reg = ModelRegistry([a])
    reg2 = reg.with_models(b)
    assert reg.names() == ("a",)
    assert reg2.names() == ("a", "b")
    with pytest.raises(InputError) as e:
        reg2.with_models(_model("a"))
    assert e.value.code == "input.duplicate_model"
    with pytest.raises(InputError) as e:
        reg.with_models(b, _model("b"), replace=True)
    assert e.value.code == "input.duplicate_model"
    a2 = _model("a", m=3)
    reg3 = reg2.with_models(a2, replace=True)
    assert reg3["a"] is a2
    assert reg2["a"] is a
    with pytest.raises(InputError):
        reg.with_models(Model(space=asub.Uniform.reference(2), value=_value))
    with pytest.raises(InputError):
        reg.with_models(b, replace="yes")  # type: ignore[arg-type]


def test_registry_is_immutable():
    reg = ModelRegistry([_model("a")])
    with pytest.raises(AttributeError):
        reg._models = {}  # type: ignore[misc]
    with pytest.raises(AttributeError):
        reg.extra = 1  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        reg._models["b"] = _model("b")  # type: ignore[index]
    with pytest.raises(TypeError):
        reg["b"] = _model("b")  # type: ignore[index]
    assert not hasattr(reg, "register")
    assert reg.names() == ("a",)


def test_registry_pickles():
    from active_subspaces import test_functions as tf

    reg = tf.builtin_models()
    back = pickle.loads(pickle.dumps(reg))
    assert back.names() == reg.names()
    X = tf.PISTON.space.sample(4, rng=0)
    assert np.array_equal(back["piston"].value(X), tf.piston(X))
