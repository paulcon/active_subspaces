"""UC-07: integration exploiting the active subspace (spec 01 §5; spec 17 V01.35–V01.39).

Bounded: ``U6``, φ = (a^T x)^4 with a from KA-5 (a ridge, KA-2): E f = 4.732/15.
Gaussian: ``G6``, φ = (â^T x)² + 0.05‖x‖², E f = 1.3; G(y) is quadratic, so only the z draws err.
``accuracy.correlated_draws`` is allowed in the bounded case [D-46], [D-48]; no other warning.
"""

from __future__ import annotations

import itertools

import numpy as np

import active_subspaces as asub
from helpers.usecases import allow_warnings, physical_model, space

A = np.array([0.3, -0.5, 0.2, 0.7, 0.1, -0.4])
A_HAT = A / np.linalg.norm(A)
MU_B = 4.732 / 15
MU_G = 1.3

MODEL_B, _ = physical_model("U6", lambda x: (x @ A) ** 4, lambda x: np.outer(4 * (x @ A) ** 3, A))
MODEL_G, _ = physical_model(
    "G6",
    lambda x: (x @ A_HAT) ** 2 + 0.05 * np.sum(x**2, axis=1),
    lambda x: np.outer(2 * (x @ A_HAT), A_HAT) + 0.1 * x,
)


def _bounded(stat_rng):
    X50 = space("U6").sample(50, rng=stat_rng("bounded_points"))
    f50, G50 = asub.evaluate_with_grads(MODEL_B, X=X50)
    sub = asub.active_subspace(grads=G50, space=space("U6"))
    return X50, f50, sub, asub.active_domain(sub)


def _gaussian_domain():
    """Exact C from the 2-point Gauss–Hermite tensor rule (64 nodes): W_1 = ±â, n = 1."""
    nodes = np.array(list(itertools.product([-1.0, 1.0], repeat=6)))
    G = asub.evaluate_grads(MODEL_G, X=space("G6").from_normalized(nodes))
    sub = asub.active_subspace(grads=G, space=space("G6"), quad_weights=np.full(64, 1 / 64))
    assert sub.n == 1
    return asub.active_domain(sub)


def _integrate_bounded(dom, stat_rng):
    with allow_warnings("accuracy.correlated_draws"):
        return asub.integrate(
            MODEL_B, domain=dom, n_nodes=3, n_mc=10, rng=stat_rng("UC-07/bounded")
        )


def test_bounded_ridge(stat_rng):
    """V01.35 [N-10], [A-07], [D-47], [D-53], [D-33]: ``active_domain`` replaces the hand-built
    domain and map;
    estimate = 4.732/15 (rel. 1e-10; 3 Gauss nodes for the
    box-spline density, deterministic weights); inactive_std_error <= 1e-12 |estimate|;
    total_std_error None."""
    *_, dom = _bounded(stat_rng)
    assert dom.kind == "interval"
    res = _integrate_bounded(dom, stat_rng)
    assert abs(res.estimate - MU_B) <= 1e-10 * MU_B
    assert res.inactive_std_error <= 1e-12 * abs(res.estimate)
    assert res.total_std_error is None


def test_surrogate_path(stat_rng):
    """V01.36: ``integrate_surrogate`` of the degree-4 fit on the 50 points = 4.732/15 (rel.
    1e-9); no inactive error.
    """
    X50, f50, sub, dom = _bounded(stat_rng)
    surr4 = asub.fit_surrogate(subspace=sub, X=X50, f=f50, kind="poly", degree=4)
    res_s = asub.integrate_surrogate(surr4, domain=dom, n_nodes=3)
    assert abs(res_s.estimate - MU_B) <= 1e-9 * MU_B
    assert res_s.inactive_std_error is None


def test_gaussian_non_ridge(stat_rng):
    """V01.37 [D-48]: T = |μ̂ − 1.3| / inactive_std_error (ddof 1). Construction 1 with the t
    quantile 5.05
    (Welch–Satterthwaite ν ≈ 202 over the four Hermite nodes; spec 17 G-3); se > 0."""
    res_g = asub.integrate(
        MODEL_G, domain=_gaussian_domain(), n_nodes=4, n_mc=100, rng=stat_rng("UC-07/gaussian")
    )
    assert res_g.inactive_std_error > 0
    assert abs(res_g.estimate - MU_G) <= 5.05 * res_g.inactive_std_error


def test_replicates(stat_rng):
    """V01.38: T = |μ̂ − 1.3| / total_std_error over 20 replicates; construction 1, t_19 = 7.07
    (bound 7.1).
    """
    res_r = asub.integrate(
        MODEL_G,
        domain=_gaussian_domain(),
        n_nodes=4,
        n_mc=100,
        n_replicates=20,
        rng=stat_rng("UC-07/replicates"),
    )
    assert np.isfinite(res_r.total_std_error)
    assert res_r.total_std_error > 0
    assert abs(res_r.estimate - MU_G) <= 7.1 * res_r.total_std_error


def test_determinism(stat_rng):
    """V01.39 [D-15]: the same seeds give == results."""
    *_, dom = _bounded(stat_rng)
    assert _integrate_bounded(dom, stat_rng) == _integrate_bounded(dom, stat_rng)
    dom_g = _gaussian_domain()
    a = asub.integrate(MODEL_G, domain=dom_g, n_nodes=4, n_mc=100, rng=stat_rng("UC-07/gaussian"))
    b = asub.integrate(MODEL_G, domain=dom_g, n_nodes=4, n_mc=100, rng=stat_rng("UC-07/gaussian"))
    assert a == b
