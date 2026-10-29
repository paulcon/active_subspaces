"""Regression of the lift against 2016 (spec 10 §8; spec 17 V10.13; [C-16], [D-49], [D-107],
[D-163]).

The ``repro:C-16`` problem: m = 4, n = 1, f = (x - x0)^T A (x - x0), M = 400, true minimum 0 at the
interior x0. 2016 ``MinVariableMap`` gave f(x*) = 0.138 (unbounded) and 0.032 (bounded).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import optimization as O

F_2016 = {"gaussian": 0.138, "uniform": 0.032}
# : v1 value at the default p = 3 [D-163]. No baseline file exists yet (spec 15 §5.13); recorded
# here.
F_V1_DEFAULT = 0.02519323115526065


def _c16_case():
    rng = np.random.default_rng(0)
    m = 4
    Q, _ = np.linalg.qr(rng.normal(size=(m, m)))
    A = Q @ np.diag([10.0, 1.0, 0.5, 0.2]) @ Q.T
    x0 = np.array([0.3, -0.2, 0.1, 0.25])

    def f(X):
        return np.einsum("ij,jk,ik->i", X - x0, A, X - x0)

    X = rng.uniform(-1, 1, (400, m))
    return f, X, x0, 2 * (X - x0) @ A


@pytest.mark.parametrize("kind", ["uniform", "gaussian"])
def test_c16_lift(kind):
    f, X, x0, G = _c16_case()
    space = asub.Uniform.reference(4) if kind == "uniform" else asub.Gaussian.reference(4)
    sub = asub.active_subspace(grads=G, space=space, n=1)
    dom = asub.active_domain(sub)
    y = np.asarray(sub.W1).T @ x0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no clipping (or any other) warning
        full = O.lift(y, domain=dom, X=X, f=f(X), lift_dim=4)
        default = O.lift(y, domain=dom, X=X, f=f(X))
    assert full.n_clipped == 0
    assert default.n_clipped == 0
    assert float(f(full.x[None, :])[0]) <= 1e-10
    assert default.lift_dim == 3
    value = float(f(default.x[None, :])[0])
    assert value < F_2016[kind]
    assert value == pytest.approx(F_V1_DEFAULT, rel=1e-8)
