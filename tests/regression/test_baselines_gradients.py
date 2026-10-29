"""v1 gradient estimators against the D-37 baselines ``gradients/*`` (spec 17 §5.5; [C-19])."""

from __future__ import annotations

import numpy as np

import active_subspaces as asub
from helpers.baselines import assert_matches_baseline, load_baseline


def fd_function(X: np.ndarray) -> np.ndarray:
    """``tools/capture_baselines.py``: f = exp(0.7 x1 - 0.3 x2) + x3 sin(x4)."""
    X = np.atleast_2d(X)
    return np.exp(0.7 * X[:, 0] - 0.3 * X[:, 1]) + X[:, 2] * np.sin(X[:, 3])


def test_finite_difference():
    """Forward differences, h = 1e-6, interior points: F8 against 2016."""
    b = load_baseline("gradients", "finite_difference")
    res = asub.finite_difference_gradients(
        fd_function, X=b.inputs["X"], space=asub.Uniform.reference(4), h=float(b.inputs["h"])
    )
    assert_matches_baseline(b, "grads", np.asarray(res.grads))


def test_local_linear_small():
    """[C-19] ``raises``: with M = 15 <= 1.7 m, 2016 raised ValueError; v1 caps p at the
    distinct points and returns.
    """
    b = load_baseline("gradients", "local_linear_small")
    res = asub.local_linear_gradients(
        X=b.inputs["X"], f=b.inputs["f"], space=asub.Uniform.reference(10)
    )
    assert_matches_baseline(b, "grads", res)
    assert res.grads.shape == (15, 10)
    assert np.all(np.isfinite(res.grads))
    assert res.p == 15
