"""Known-answer tests of active_subspaces.integration (spec 10 §8; spec 17 V10.7, V10.8)."""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import integration as I


def _ridge_domain(a, space):
    return asub.active_domain(
        asub.subspace_from_basis(np.asarray(a, dtype=float)[:, None], space=space)
    )


@pytest.mark.parametrize("m", [2, 4])
def test_exact_ridge_uniform(m, seed):
    """V10.7: f = (a^T x)^2, W1 = a/||a||, N_q = 2: mu = ||a||^2 / 3; the inactive error
    vanishes.
    """
    a = np.array([1.0, -2.0, 0.5, 0.75][:m])
    dom = _ridge_domain(a, asub.Uniform.reference(m))
    if m > 2:  # hit-and-run chains with n_mc < 20 warn [D-114]
        with pytest.warns(asub.AccuracyWarning, match=r"accuracy\.correlated_draws"):
            res = I.integrate(lambda X: (X @ a) ** 2, domain=dom, n_nodes=2, n_mc=5, rng=seed)
    else:
        res = I.integrate(lambda X: (X @ a) ** 2, domain=dom, n_nodes=2, n_mc=5, rng=seed)
    mu = float(a @ a) / 3
    assert res.estimate == pytest.approx(mu, rel=1e-13)
    # every draw on a fiber gives the same value up to rounding of a^T (W1 y + W2 z)
    assert res.inactive_std_error <= 1e-12 * mu


def test_exact_ridge_gaussian(seed):
    """V10.8: f = exp(a^T x) on N(0, I), N_q = 20: mu = exp(||a||^2 / 2)."""
    a = np.array([0.6, -0.3, 0.5])
    dom = _ridge_domain(a, asub.Gaussian.reference(3))
    res = I.integrate(lambda X: np.exp(X @ a), domain=dom, n_nodes=20, n_mc=2, rng=seed)
    assert res.estimate == pytest.approx(np.exp(a @ a / 2), rel=1e-10)


def test_quartic_ridge_uniform():
    """Box-spline rule with N_q = 3 is exact to degree 5: E[(w^T x)^4] = sum w^4/5 + (1 - sum
    w^4)/3 for ||w|| = 1.
    """
    w = np.array([3.0, -1.0, 2.0, 0.5, 1.0])
    w /= np.linalg.norm(w)
    dom = _ridge_domain(w, asub.Uniform.reference(5))
    w = np.asarray(dom.subspace.W1)[:, 0]
    s4 = float(np.sum(w**4))
    exact = s4 / 5 + (1 - s4) / 3
    res = I.integrate_surrogate(lambda Y: Y[:, 0] ** 4 + Y[:, 0] ** 5, domain=dom, n_nodes=3)
    assert res.estimate == pytest.approx(exact, rel=1e-12)
    assert I.integrate_surrogate(
        lambda Y: Y[:, 0] ** 4, domain=dom, n_nodes=2
    ).estimate != pytest.approx(exact, rel=1e-6)


def test_tensor_polynomial_gaussian(seed):
    """Gaussian n = 2, k = 3 per dimension: E[y1^4 + y1 y2^2 + y2^4 y1^2] = 3 + 0 + 3 exactly."""
    m = 4
    Q = np.linalg.qr(np.random.default_rng(4).standard_normal((m, m)))[0]
    dom = asub.active_domain(asub.subspace_from_basis(Q[:, :2], space=asub.Gaussian.reference(m)))
    W1 = np.asarray(dom.subspace.W1)

    def f(X):
        Y = X @ W1
        return Y[:, 0] ** 4 + Y[:, 0] * Y[:, 1] ** 2 + Y[:, 1] ** 4 * Y[:, 0] ** 2

    res = I.integrate(f, domain=dom, n_nodes=[3, 3], n_mc=2, rng=seed)
    assert res.estimate == pytest.approx(6.0, rel=1e-12)
    res = I.integrate_surrogate(
        lambda Y: Y[:, 0] ** 4 + Y[:, 0] * Y[:, 1] ** 2 + Y[:, 1] ** 4 * Y[:, 0] ** 2,
        domain=dom,
        n_nodes=9,
    )
    assert res.estimate == pytest.approx(6.0, rel=1e-12)


def test_physical_gaussian_space(gaussian_space, seed):
    """A correlated Gaussian space: integrate a ridge of the normalized x through physical
    points.
    """
    space = gaussian_space(3)
    a = np.array([0.4, 0.2, -0.5])
    sub = asub.subspace_from_basis(a[:, None], space=space)
    dom = asub.active_domain(sub)
    w = np.asarray(sub.W1)[:, 0]
    res = I.integrate(
        lambda Xp: (space.to_normalized(Xp) @ w) ** 3 + 2.0,
        domain=dom,
        n_nodes=2,
        n_mc=2,
        rng=seed,
    )
    assert res.estimate == pytest.approx(2.0, rel=1e-12)


def test_constant_on_zonotope(seed):
    """The Delaunay-centroid rule is exact for constants [D-105]."""
    dom = asub.active_domain(
        asub.subspace_from_basis(
            np.random.default_rng(2).standard_normal((5, 2)), space=asub.Uniform.reference(5)
        )
    )
    res = I.integrate_surrogate(
        lambda Y: np.full(Y.shape[0], 2.5), domain=dom, n_nodes=3, n_weight_samples=5000, rng=seed
    )
    assert res.estimate == pytest.approx(2.5, rel=1e-14)


def test_interval_rule_moments():
    """The n = 1 rule integrates y^d exactly (d <= 2 N_q - 1) against the box spline of w."""
    w = np.array([1.0, 0.5])
    w /= np.linalg.norm(w)
    dom = _ridge_domain(w, asub.Uniform.reference(2))
    rule = I.active_quadrature_rule(dom, n_nodes=4)
    w = np.asarray(dom.subspace.W1)[:, 0]
    # exact moments of w1 x1 + w2 x2 by a fine tensor Gauss-Legendre rule in x
    t, v = np.polynomial.legendre.leggauss(20)
    T1, T2 = np.meshgrid(t, t, indexing="ij")
    V = np.outer(v, v) / 4
    Y = w[0] * T1 + w[1] * T2
    for d in range(8):
        exact = float(np.sum(V * Y**d))
        got = float(np.sum(rule.weights * rule.nodes[:, 0] ** d))
        assert got == pytest.approx(exact, rel=1e-12, abs=1e-14)
