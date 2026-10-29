"""v1 lift against the D-37 baseline ``optimization/lift`` (spec 17 §5.5; V10.13; [C-16], [D-49],
[D-163]).

The ``repro:C-16`` problem: m = 4, n = 1, f = (x - x0)^T A (x - x0), M = 400, minimum 0 at the
interior x0.
2016 ``MinVariableMap`` gave f(x*) = 0.138 (unbounded) and 0.032 (bounded); v1 must do better
(``improves``) and reproduce its own recorded value at the default p = 3 (``v1_reference``, F8).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import active_subspaces as asub
from helpers.baselines import assert_matches_baseline, load_baseline


def c16_function(X: np.ndarray, A: np.ndarray, x0: np.ndarray) -> np.ndarray:
    D = np.atleast_2d(X) - x0
    return np.einsum("ij,jk,ik->i", D, A, D)


@pytest.mark.parametrize("kind", ["gaussian", "uniform"])
def test_c16_lift(kind):
    b = load_baseline("optimization", "lift")
    A, x0, X, f = b.inputs["A"], b.inputs["x0"], b.inputs["X"], b.inputs["f"]
    space = asub.Gaussian.reference(4) if kind == "gaussian" else asub.Uniform.reference(4)
    sub = asub.active_subspace(grads=b.inputs["grads"], space=space, n=1)
    dom = asub.active_domain(sub)
    y = np.asarray(sub.W1).T @ x0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no clipping (or any other) warning
        full = asub.lift(y, domain=dom, X=X, f=f, lift_dim=4)
        default = asub.lift(y, domain=dom, X=X, f=f)
    assert (
        float(c16_function(full.x, A, x0)[0]) <= 1e-10
    )  # exact quadratic, full lift: the QP optimum
    assert default.lift_dim == 3
    assert default.n_clipped == 0
    value = np.float64(c16_function(default.x, A, x0)[0])
    assert_matches_baseline(b, f"f_xstar_{kind}", value)
    assert_matches_baseline(b, f"v1_f_xstar_{kind}", value)
