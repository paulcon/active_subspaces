"""Statistical checks for ``active_subspaces.reduced_model`` (spec 11 §8; spec 17 V11.6, V11.8).

Seeds come from the ``stat_rng`` fixture (spec 17 §5.2), fixed by the test's name [D-78], [D-159].
"""

from __future__ import annotations

import math

import numpy as np
import scipy.stats

import active_subspaces as asub
from active_subspaces import integration as I
from active_subspaces import reduced_model as R


class _Exp:
    """f(x) = exp(c^T x) on the reference cube (picklable)."""

    def __init__(self, c: np.ndarray) -> None:
        self.c = c

    def __call__(self, X):
        return np.exp(X @ self.c)


def test_probability_gaussian_linear(stat_rng):
    """V11.6: f = a^T xi on Gaussian.reference(m), exact linear surrogate.

    Statistic p_hat; construction 1 of spec 17 §3.3, |p_hat - p| <= 5 se with se from the result;
    guards [C-06], D-116.
    """
    m = 5
    sp = asub.Gaussian.reference(m)
    a = np.array([1.0, -0.5, 0.25, 2.0, 0.0])
    X = sp.sample(100, rng=stat_rng("data"))
    rm = R.build_reduced_model(
        space=sp, X=X, f=X @ a, grads=np.tile(a, (100, 1)), surrogate_kind="poly", degree=1, rng=0
    )
    assert rm.n == 1
    na = float(np.linalg.norm(a))
    for k, (lower, upper) in enumerate([(-1.0, 1.5), (0.5, np.inf), (-np.inf, -2.0)]):
        res = R.probability(rm, lower, upper, rng=stat_rng(f"draw{k}"))
        p = scipy.stats.norm.cdf(upper / na) - scipy.stats.norm.cdf(lower / na)
        assert res.n_samples == 10_000
        assert res.std_error > 0
        # three comparisons: Bonferroni z_3 = 5.1 (spec 17 §3.3 construction 2)
        assert abs(res.p - p) <= 5.1 * res.std_error
        lo, hi = res.confidence_interval()
        assert lo <= res.p <= hi


def test_average_exp(stat_rng):
    """V11.8: mean of exp(c^T x) on the cube, prod_j sinh(c_j)/c_j, through a reduced model with
    real inactive variation.

    Statistic mu_hat over R = 20 replicates; construction 1 with the t_19 quantile 7.07 on
    total_std_error.
    The rule error in y (10 box-spline Gauss nodes on exp(c_1 y)) is below 1e-20, far under 1e-3 se
    (spec 17 G-18).
    """
    m = 4
    sp = asub.Uniform.reference(m)
    c = np.array([0.8, 0.4, 0.3, 0.2])
    X = sp.sample(60, rng=stat_rng("data"))
    sub = asub.subspace_from_basis(np.eye(m)[:, :1], space=sp)
    model = asub.Model(space=sp, value=_Exp(c))
    rm = R.build_reduced_model(
        X=X, f=model.value(X), subspace=sub, surrogate_kind="poly", degree=4
    )
    mu = float(np.prod(np.sinh(c) / c))
    # default n_mc = 50 >= 20: no correlated_draws warning [D-114, D-171]
    res = R.average(rm, n_nodes=10, model=model, n_replicates=20, rng=stat_rng("average"))
    assert res.total_std_error is not None
    assert res.total_std_error > 0
    assert abs(res.estimate - mu) <= 7.07 * res.total_std_error
    # model=None integrates the surrogate, bitwise as spec 10 does
    s = R.average(rm, n_nodes=10, rng=3)
    assert s == I.integrate_surrogate(rm.surrogate, domain=rm.domain, n_nodes=10, rng=3)
    assert math.isfinite(s.estimate)
