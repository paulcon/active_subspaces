"""Statistical checks for ``active_subspaces.surrogates`` (spec 09 §8; spec 17 V09.20)."""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub


@pytest.mark.slow
def test_prediction_coverage():
    """The 95% prediction interval covers a new observation at about 95% [D-200].

    2000 data sets with Gaussian errors; the binomial standard error of the rate is 0.005, so the
    band [0.93, 0.97] is four standard errors wide on each side.
    """
    gen = np.random.default_rng(20261029)
    space = asub.Uniform.reference(1)
    X = np.linspace(-1.0, 1.0, 15)[:, None]
    x_new = np.array([[0.37]])
    hits = 0
    R = 2000
    for _ in range(R):
        f = 2.0 - X[:, 0] + 0.5 * X[:, 0] ** 2 + 0.3 * gen.standard_normal(15)
        s = asub.fit_surrogate(X=X, f=f, space=space, subspace=None, kind="poly", degree=2)
        res = asub.predict_interval(s, x_new)
        y_new = 2.0 - 0.37 + 0.5 * 0.37**2 + 0.3 * gen.standard_normal()
        hits += bool(res.lower[0] <= y_new <= res.upper[0])
    assert 0.93 <= hits / R <= 0.97
