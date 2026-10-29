"""UC-08: minimization over the active variable, with the lift (spec 01 §5; spec 17 V01.40–V01.44).

KA-7: f(x) = (q_1^T x − 0.3)² + β (q_2^T x − 0.2)², β = 0.1, M = 1000.
``accuracy.curvature_clipped`` is allowed [D-49]; no other warning.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import active_subspaces as asub
from helpers.usecases import HandMap, allow_warnings, ka1, physical_model, space

Q, _ = ka1(5, np.ones(5))
Q1, Q2 = Q[:, 0], Q[:, 1]
BETA = 0.1
B = np.vstack([Q1, Q2])  # f = ‖D(Bx − t)‖²
D = np.array([1.0, np.sqrt(BETA)])
T = np.array([0.3, 0.2])


def phi(x):
    return (x @ Q1 - 0.3) ** 2 + BETA * (x @ Q2 - 0.2) ** 2


def dphi(x):
    return np.outer(2 * (x @ Q1 - 0.3), Q1) + np.outer(2 * BETA * (x @ Q2 - 0.2), Q2)


def _run(name, stat_rng):
    model, _ = physical_model(name, phi, dphi)
    X = space(name).sample(1000, rng=stat_rng("UC-08/sample"))
    f, G = asub.evaluate_with_grads(model, X=X)
    sub = asub.active_subspace(grads=G, space=space(name), rng=stat_rng("UC-08/bootstrap"))
    dom = asub.active_domain(sub)
    surr = asub.fit_surrogate(subspace=sub, X=X, f=f, kind="poly", degree=2)
    with allow_warnings("accuracy.curvature_clipped"):
        res = asub.minimize(surr, domain=dom, X=X, f=f, rng=stat_rng("UC-08/minimize"))
    return sub, res


def _f_at(name, x_phys):
    return float(phi(HandMap.of(name).x(np.atleast_2d(x_phys)))[0])


def _fiber_excess(name, sub, res):
    """f(x*) − min_z f(Ŵ_1 y* + Ŵ_2 z), in closed form by least squares (KA-7)."""
    x0 = sub.W1 @ np.atleast_1d(res.y_star)
    Mz = D[:, None] * (B @ sub.W2)
    rz = D * (B @ x0 - T)
    z, *_ = np.linalg.lstsq(Mz, -rz, rcond=None)
    fiber_min = float(np.sum((Mz @ z + rz) ** 2))
    return _f_at(name, res.x_star_physical) - fiber_min


def test_dimension_and_minimum(stat_rng):
    """V01.40: n == 1; T = f(x*) evaluated by the test, construction 6, provisional bound 1e-3
    (spec 17 G-4).
    """
    sub, res = _run("G5", stat_rng)
    assert sub.n == 1
    assert _f_at("G5", res.x_star_physical) <= 1e-3


def test_lift_fiber_minimizer(stat_rng):
    """V01.41 [C-16], [D-49]: the lift returns the fiber minimizer: excess over min_z f(Ŵ_1 y* +
    Ŵ_2 z) <= 1e-8
    (F8; about 0.015 for the 2016 lift)."""
    sub, res = _run("G5", stat_rng)
    assert abs(_fiber_excess("G5", sub, res)) <= 1e-8


def test_named_fields(stat_rng):
    """V01.42 [A-10]: fields by name, not a tuple; ``f_model is None`` without a model; x_star
    is normalized [D-11].
    """
    _, res = _run("G5", stat_rng)
    assert not isinstance(res, tuple)
    assert res.f_model is None
    assert np.ndim(res.f_star) == 0
    assert np.isfinite(res.f_star)
    assert res.x_star.shape == res.x_star_physical.shape == (5,)
    assert np.max(np.abs(HandMap.of("G5").x(res.x_star_physical) - res.x_star)) <= 1e-12 * max(
        1.0, np.max(np.abs(res.x_star))
    )
    assert np.size(res.y_star) == 1
    assert {"f_star", "f_model", "active", "lift"} <= {f.name for f in dataclasses.fields(res)}


def test_bounded(stat_rng):
    """V01.43: on U5, x* lies in the box; f(x*) <= 1e-2 (provisional, G-4); the unconstrained fiber
    minimizer is
    interior, so the fiber excess is <= 1e-8 (F8) [C-16]."""
    sub, res = _run("U5", stat_rng)
    assert space("U5").contains(res.x_star_physical[None])[0]
    assert _f_at("U5", res.x_star_physical) <= 1e-2
    assert abs(_fiber_excess("U5", sub, res)) <= 1e-8


@pytest.mark.parametrize("name", ["G5", "U5"])
def test_determinism(stat_rng, name):
    """V01.44 [D-15]: the same seeds give == results."""
    assert _run(name, stat_rng)[1] == _run(name, stat_rng)[1]
