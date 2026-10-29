"""Regression against the 2016 dev reduced model (spec 11 §7; spec 17 V11.18, [C-06], [C-07])."""

from __future__ import annotations

import math

import numpy as np

import active_subspaces as asub
from active_subspaces import reduced_model as R


def _probability_2016(g: np.ndarray, lb: float, ub: float) -> tuple[float, float, float]:
    """dev ``ActiveSubspaceReducedModel.probability`` on given surrogate values
    (base.py:497-500).
    """
    M = g.shape[0]
    c = np.all(np.vstack((g > lb, g < ub)).T, axis=1)
    p = np.sum(c) / float(M)
    plb, pub = p + 2.58 * np.sqrt(p * (1 - p) / M), p - 2.58 * np.sqrt(p * (1 - p) / M)
    return p, plb, pub


def test_probability_swap():
    """[C-06] ``swap``: the 2016 'lower' bound exceeds its 'upper' bound; v1 returns an ordered
    interval with the same p.
    """
    m = 3
    sp = asub.Gaussian.reference(m)
    gen = np.random.default_rng(0)
    X = gen.standard_normal((50, m))
    a = np.array([1.0, 0.5, -0.2])
    rm = R.build_reduced_model(
        space=sp, X=X, f=X @ a, grads=np.tile(a, (50, 1)), surrogate_kind="poly", degree=1, rng=0
    )
    N = 10_000
    res = R.probability(rm, -0.5, 1.0, n_samples=N, rng=7)
    x = sp.sample(N, rng=7, normalized=True)
    g = rm.surrogate.evaluate(x @ rm.W1)
    p, plb, pub = _probability_2016(g, -0.5, 1.0)
    assert res.p == p  # same estimate (open and closed intervals agree for a continuous g)
    assert plb > pub  # the 2016 bounds are swapped
    lo, hi = res.confidence_interval(0.99)
    assert lo < res.p < hi
    assert not (lo == plb and hi == pub)
    # the sorted 2016 pair is the normal interval, which the Wilson interval matches to O(1/N) here
    assert abs(lo - pub) <= 1e-3
    assert abs(hi - plb) <= 1e-3


def test_model_path_r_squared_scale():
    """[C-07]: 2016's 1 - ||r||^2 / var(f) omits 1/M; v1's holdout R^2 is the weighted
    coefficient of determination.
    """
    gen = np.random.default_rng(1)
    f = gen.standard_normal(75)
    g = f + 0.05 * gen.standard_normal(75)
    legacy = 1 - np.sum((f - g) ** 2) / np.var(f)
    v1 = R._r_squared(f, g, np.full(75, 1 / 75))
    assert math.isclose(1 - v1, (1 - legacy) / 75, rel_tol=1e-12)
