"""Maximin design against the 2016 algorithm (V09.11).

The 2016 ``maximin_design`` (legacy ``utils/designs.py``) is ported inline, with
the global reseed with 42 replaced by a local generator per seed: SLSQP on the
non-smooth objective ``min squared distance`` over points and vertices, three
``N(0, I)`` starts, hull constraints from ``ConvexHull.equations``. A seed for
which 2016 raised (a failed start) is left out of its sample.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, distance_matrix

import active_subspaces as asub


def _obj(y, vert):
    n = vert.shape[1]
    Y = y.reshape(-1, n)
    N = Y.shape[0]
    d0 = ((distance_matrix(Y, Y) + 1e5 * np.eye(N)) ** 2).min()
    d1 = (distance_matrix(Y, vert) ** 2).min()
    return -min(d0, d1)


def _grad(y, vert):
    Nv, n = vert.shape
    Y = y.reshape(-1, n)
    N = Y.shape[0]
    D0 = ((distance_matrix(Y, Y) + 1e5 * np.eye(N)) ** 2).ravel()
    D1 = (distance_matrix(Y, vert) ** 2).ravel()
    g = np.zeros(N * n)
    if D0.min() < D1.min():
        k = int(D0.argmin())
        i, j = k // N, k % N
        g[i * n : (i + 1) * n] = 2 * (Y[i] - Y[j])
        g[j * n : (j + 1) * n] = 2 * (Y[j] - Y[i])
    else:
        k = int(D1.argmin())
        i, j = k // Nv, k % Nv
        g[i * n : (i + 1) * n] = 2 * (Y[i] - vert[j])
    return -g


def _maximin_2016(vert, N, seed):
    n = vert.shape[1]
    C = ConvexHull(vert)
    A = np.kron(np.eye(N), C.equations[:, :n])
    b = np.kron(np.ones(N), C.equations[:, n])
    cons = ({"type": "ineq", "fun": lambda x: -(A @ x + b), "jac": lambda x: -A},)
    gen = np.random.default_rng(seed)
    best = None
    for _ in range(3):
        y0 = gen.normal(size=(N, n)).ravel()
        res = minimize(
            _obj,
            y0,
            args=(vert,),
            jac=_grad,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 100, "ftol": 1e-4},
        )
        if not res.success:
            return None
        if best is None or res.fun < best.fun:
            best = res
    return float(np.sqrt(-_obj(best.x, vert)))


@pytest.mark.slow
def test_maximin_vs_2016():
    sp = asub.Uniform(-np.ones(5), np.ones(5))
    sub = asub.subspace_from_basis(np.random.default_rng(3).normal(size=(5, 2)), space=sp)
    dom = asub.active_domain(sub)
    N = 10
    v1 = [asub.active_design(domain=dom, n_points=N, rng=s).min_distance for s in range(20)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = [_maximin_2016(dom.vertices, N, s) for s in range(20)]
    old = [d for d in old if d is not None]
    assert len(old) >= 10
    assert np.median(v1) >= np.median(old)
