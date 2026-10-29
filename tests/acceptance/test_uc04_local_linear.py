"""UC-04: local linear gradients from data (spec 01 §5; spec 17 V01.22–V01.25)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import active_subspaces as asub
from helpers.tolerances import assert_close
from helpers.usecases import HandMap, ka1, physical_model, space

_, K3 = ka1(3, [1.0, 0.5, 0.1])
G3 = np.array([1.0, -2.0, 0.5])
G10 = np.array([1.0, -2.0, 0.5, 0.0, 3.0, -1.0, 0.2, 0.0, 1.5, -0.7])


def _X(stat_rng):
    return space("U3").sample(300, rng=stat_rng("UC-04/sample"))


def test_alignment(stat_rng):
    """V01.22 [C-03], [D-32]: rows of ``lg.grads`` are aligned with X. T = median(e)/median(e'), e
    against Kx_i,
    e' against Kx_{i-1} (rows shifted by one). Construction 6, calibrated bound 0.5 (max 0.19)."""
    _, f = physical_model("U3", lambda x: 0.5 * np.einsum("ij,jk,ik->i", x, K3, x))
    X = _X(stat_rng)
    lg = asub.local_linear_gradients(X=X, f=f(X), space=space("U3"))
    assert lg.grads.shape == (300, 3)
    gx = HandMap.of("U3").x(X) @ K3
    e = np.linalg.norm(lg.grads - gx, axis=1)
    e_shift = np.linalg.norm(lg.grads - np.roll(gx, 1, axis=0), axis=1)
    assert np.median(e) <= 0.5 * np.median(e_shift)


def test_linear_exact(stat_rng):
    """V01.23: KA-3, every row equals g; ``grads_physical`` equals ∇_ξ f (rel. 1e-9)."""
    _, f = physical_model("U3", lambda x: 2.0 + x @ G3)
    X = _X(stat_rng)
    lg = asub.local_linear_gradients(X=X, f=f(X), space=space("U3"))
    assert_close(lg.grads, np.tile(G3, (300, 1)), rtol=1e-9)
    assert_close(lg.grads_physical, np.tile(HandMap.of("U3").grad(G3), (300, 1)), rtol=1e-9)


def test_small_M(stat_rng):
    """V01.24 [C-19]: m = 10, M = 15 runs and recovers g exactly (rel. 1e-8)."""
    sp = asub.Uniform.reference(10)
    X = sp.sample(15, rng=stat_rng("small"))
    lg = asub.local_linear_gradients(X=X, f=1.0 + X @ G10, space=sp)
    assert_close(lg.grads, np.tile(G10, (15, 1)), rtol=1e-8)


def test_too_few(stat_rng):
    """V01.24: m = 10, M = 10 -> ``input.too_few_samples``."""
    sp = asub.Uniform.reference(10)
    X = sp.sample(10, rng=stat_rng("too_few"))
    with pytest.raises(asub.InputError) as e:
        asub.local_linear_gradients(X=X, f=X @ G10, space=sp)
    assert e.value.code == "input.too_few_samples"


def test_deterministic_and_composes(stat_rng):
    """V01.25: no ``rng`` parameter; two calls give == results; a GradientResult is accepted as
    ``grads=`` [D-88].
    """
    assert "rng" not in inspect.signature(asub.local_linear_gradients).parameters
    _, f = physical_model("U3", lambda x: 0.5 * np.einsum("ij,jk,ik->i", x, K3, x))
    X = _X(stat_rng)
    a = asub.local_linear_gradients(X=X, f=f(X), space=space("U3"))
    b = asub.local_linear_gradients(X=X, f=f(X), space=space("U3"))
    assert a == b
    sub = asub.active_subspace(grads=a, space=space("U3"))
    assert sub.space == space("U3")
