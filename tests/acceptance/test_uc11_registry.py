"""UC-11: a registry test function, used as the MCP server will (spec 01 §5; spec 17
V01.57–V01.60).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import active_subspaces as asub

REG = asub.test_functions.builtin_models()
NAMES = REG.names()


def test_catalogue_and_spaces():
    """V01.57 [A-11], [D-12], [D-21], [D-124]: the 2016 problems are registered; every space is
    homogeneous.
    """
    assert {"borehole", "otl_circuit", "piston", "robot_arm", "wing_weight"} <= set(NAMES)
    for name in NAMES:
        assert REG.get(name).space.kind in ("uniform", "gaussian")


@pytest.mark.parametrize("name", NAMES)
def test_gradients(name, stat_rng):
    """V01.58: analytic gradients equal ``grads_to_physical`` of central differences in normalized
    coordinates
    (step 1e-5) at 20 interior points; F6 relative to max|∇f|."""
    model = REG.get(name)
    sp = model.space
    child = stat_rng("UC-11/gradient_points").spawn(len(NAMES))[
        NAMES.index(name)
    ]  # per-case streams (spec 17 §5.2)
    x = sp.sample(20, rng=child, normalized=True)
    if sp.kind == "uniform":
        x = 0.9 * x
    h, m = 1e-5, sp.dim
    g_norm = np.empty((20, m))
    for j in range(m):
        e = np.zeros(m)
        e[j] = h
        fp = asub.evaluate(model, X=sp.from_normalized(x + e))
        fm = asub.evaluate(model, X=sp.from_normalized(x - e))
        g_norm[:, j] = (fp - fm) / (2 * h)
    G = asub.evaluate_grads(model, X=sp.from_normalized(x))
    fd = sp.grads_to_physical(g_norm)
    assert np.max(np.abs(fd - G)) <= 1e-6 * np.max(np.abs(G))


@pytest.mark.parametrize("name", NAMES)
def test_batch_contract(name):
    """V01.59 [A-06], [D-23]: ``value`` maps (k, m) -> (k,) and ``grad`` (k, m) -> (k, m), k =
    1 and 7.
    """
    model = REG.get(name)
    for k in (1, 7):
        X = model.space.sample(k, rng=k)
        assert np.shape(model.value(X)) == (k,)
        if model.grad is not None:
            assert np.shape(model.grad(X)) == (k, model.space.dim)


def _wing(stat_rng):
    model = REG.get("wing_weight")
    X = model.space.sample(500, rng=stat_rng("UC-11/sample"))
    return model, X


def test_workers(stat_rng):
    """V01.59 [D-26], [D-15], [D-125]: a thread pool with ``batch_size=50`` gives the same
    array, bitwise.
    """
    model, X = _wing(stat_rng)
    with ThreadPoolExecutor(4) as ex:
        par = asub.evaluate_grads(model, X=X, executor=ex, batch_size=50)
    assert np.array_equal(par, asub.evaluate_grads(model, X=X, batch_size=50))


def test_json(stat_rng):
    """V01.60 [D-19], [D-20]: JSON round trip (``binary=True``); provenance records the space
    and the model name.
    """
    model, X = _wing(stat_rng)
    G = asub.evaluate_grads(model, X=X)
    assert G.shape == (500, 10)
    sub = asub.active_subspace(
        grads=G, space=model.space, nboot=100, rng=stat_rng("UC-11/bootstrap")
    )
    payload = json.dumps(sub.to_dict(binary=True), allow_nan=False)
    sub_back = asub.from_dict(json.loads(payload))
    assert sub_back == sub
    assert sub.provenance.to_dict()["input_space"] == model.space.to_dict()
    fd = asub.finite_difference_gradients(model, X=X[:20])
    assert fd.provenance.parameters["model"] == "wing_weight"


def test_json_provenance_input_space_attribute(stat_rng):
    """V01.60 (literal form of spec 01):
    ``sub.provenance.input_space == model.space.to_dict()`` [D-20].
    """
    model, X = _wing(stat_rng)
    sub = asub.active_subspace(grads=asub.evaluate_grads(model, X=X[:50]), space=model.space)
    assert sub.provenance.input_space == model.space.to_dict()
    back = asub.from_dict(json.loads(json.dumps(sub.to_dict(binary=True))))
    assert back.provenance.input_space == model.space.to_dict()


def test_failures(stat_rng):
    """V01.60 [D-25], [D-17]: a raising row -> ``ModelError`` ``model.raised`` with ``__cause__``;
    a NaN -> ``ModelError`` ``model.nonfinite``."""
    model, X = _wing(stat_rng)

    def raising(Z):
        if np.any(np.all(Z == X[3], axis=1)):
            raise RuntimeError("solver diverged")
        return model.value(Z)

    def nan(Z):
        out = np.array(model.value(Z), dtype=float)
        out[np.all(Z == X[3], axis=1)] = np.nan
        return out

    with pytest.raises(asub.ModelError) as e:
        asub.evaluate(asub.Model(space=model.space, value=raising), X=X[:10])
    assert e.value.code == "model.raised"
    assert isinstance(e.value.__cause__, RuntimeError)
    with pytest.raises(asub.ModelError) as e:
        asub.evaluate(asub.Model(space=model.space, value=nan), X=X[:10])
    assert e.value.code == "model.nonfinite"


def test_unknown_name():
    """V01.60: ``get("nope")`` -> ``InputError`` ``input.unknown_model`` with the known names
    in ``details``.
    """
    with pytest.raises(asub.InputError) as e:
        REG.get("nope")
    assert e.value.code == "input.unknown_model"
    assert set(NAMES) <= set(e.value.details["known"])
