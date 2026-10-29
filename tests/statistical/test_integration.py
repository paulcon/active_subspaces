"""Statistical tests of active_subspaces.integration (spec 10 §8; spec 17 V10.5, V10.9-V10.11).

Fixed seeds [D-78], [D-159]; bounds have false-failure probability <= 1e-6 at a random seed (spec
17 §3.3).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.spatial
import scipy.stats

import active_subspaces as asub
from active_subspaces import integration as I

ALPHA = 1e-6


def _bonferroni(k: int) -> float:
    """z_k = max(5, Phi^{-1}(1 - alpha / (2k))), rounded up to two decimals (spec 17 §3.3,
    D-165).
    """
    return max(5.0, float(np.ceil(scipy.stats.norm.isf(ALPHA / (2 * k)) * 100) / 100))


def test_zonotope_weights_square(seed):
    """V10.5: on Y = [-1, 1]^2 (uniform density) each weight is its simplex area / 4 (binomial
    se).
    """
    sub = asub.subspace_from_basis(np.eye(3)[:, :2], space=asub.Uniform.reference(3))
    dom = asub.active_domain(sub)
    N_X = 100_000
    rule = I.active_quadrature_rule(dom, n_nodes=5, n_weight_samples=N_X, rng=seed)
    assert rule.weight_deficit == 0.0
    assert abs(rule.weights.sum() - 1) <= 1e-12
    T = scipy.spatial.Delaunay(np.vstack([dom.vertices, rule.design]))
    P = T.points[T.simplices]
    E1, E2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    area = 0.5 * np.abs(E1[:, 0] * E2[:, 1] - E1[:, 1] * E2[:, 0])
    centroids = P.mean(axis=1)
    # empty cells (slivers between maximin points on the boundary) are dropped [D-105]
    kept = np.array([np.any(np.all(rule.nodes == c, axis=1)) for c in centroids])
    assert np.array_equal(rule.nodes, centroids[kept])
    weights = np.zeros(T.nsimplex)
    weights[kept] = rule.weights
    p = area / 4
    se = np.sqrt(p * (1 - p) / N_X)
    z = _bonferroni(T.nsimplex)
    assert np.all(np.abs(weights - p) <= z * se)


def _calibration(space_kind: str, m: int, n_nodes: int, n_mc: int, thin=None, n_seeds: int = 200):
    Q = np.linalg.qr(np.random.default_rng(7).standard_normal((m, m)))[0]
    space = asub.Uniform.reference(m) if space_kind == "uniform" else asub.Gaussian.reference(m)
    sub = asub.subspace_from_basis(Q[:, :1], space=space)
    dom = asub.active_domain(sub)
    w1, w2 = np.asarray(sub.W1)[:, 0], np.asarray(sub.W2)[:, 0]
    mu = 1 / 3 if space_kind == "uniform" else 1.0
    t, t_raw = [], []
    for s in range(n_seeds):
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", asub.AccuracyWarning
            )  # thin=1 warns accuracy.correlated_draws
            r = I.integrate(
                lambda X: (X @ w1) ** 2 + X @ w2,
                domain=dom,
                n_nodes=n_nodes,
                n_mc=n_mc,
                thin=thin,
                rng=s,
            )
        t.append((r.estimate - mu) / r.inactive_std_error)
        t_raw.append((r.estimate - mu) * np.sqrt(r.autocorrelation_time) / r.inactive_std_error)
    return np.array(t), np.array(t_raw)


@pytest.mark.slow
@pytest.mark.parametrize("space_kind", ["uniform", "gaussian"])
def test_error_calibration(space_kind):
    """V10.9: f = (w1^T x)^2 + w2^T x, n = 1, m - n = 1: standardized errors have mean 0, variance
    1.

    N_q (N_MC - 1) = 125 >= 100, so the Student-t inflation is below the resolution;
    bounds |mean| <= 0.356 and s^2 in [0.58, 1.57] (spec 17 V10.9).
    """
    t, _ = _calibration(space_kind, m=2, n_nodes=5, n_mc=26)
    assert abs(t.mean()) <= 0.356
    assert 0.58 <= t.var(ddof=1) <= 1.57


@pytest.mark.slow
def test_error_calibration_hit_and_run():
    """V10.10: m - n = 3, thin = 1: calibrated with the pooled Geyer tau [D-114], not without it
    [N-09].

    Provisional band (spec 17 V10.10, G-14): the variance interval is widened to [0.5, 2.0] for the
    variability of tau_hat; without the correction the variance exceeds 2.0.
    """
    t, t_raw = _calibration("uniform", m=4, n_nodes=5, n_mc=51, thin=1)
    assert abs(t.mean()) <= 0.356
    assert 0.5 <= t.var(ddof=1) <= 2.0
    assert t_raw.var(ddof=1) > 2.0


def test_replicates(seed):
    """V10.11: f = prod(1 + c_j x_j) on the cube (mu = 1), n = 2: |mu_hat - 1| <= t_{R-1}
    sigma_tot; R = 1 gives None.
    """
    m = 4
    c = np.array([0.5, -0.4, 0.3, 0.2])
    sub = asub.subspace_from_basis(
        np.random.default_rng(3).standard_normal((m, 2)), space=asub.Uniform.reference(m)
    )
    dom = asub.active_domain(sub)

    def f(X):
        return np.prod(1 + c * X, axis=1)

    R = 20
    with pytest.warns(
        asub.AccuracyWarning, match=r"accuracy\.correlated_draws"
    ):  # n_mc < 20 [D-114]
        res = I.integrate(
            f, domain=dom, n_nodes=5, n_mc=5, n_replicates=R, n_weight_samples=10_000, rng=seed
        )
    bound = scipy.stats.t.isf(ALPHA / 2, R - 1)
    assert bound == pytest.approx(7.07, abs=0.01)
    assert res.total_std_error > 0
    assert abs(res.estimate - 1.0) <= bound * res.total_std_error
    with pytest.warns(asub.AccuracyWarning, match=r"accuracy\.correlated_draws"):
        one = I.integrate(f, domain=dom, n_nodes=5, n_mc=5, n_weight_samples=10_000, rng=seed)
    assert one.total_std_error is None
