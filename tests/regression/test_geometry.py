"""Regression against the 2016 zonotope vertices (spec 08 §8, D-37).

The 2016 ``domains.zonotope_vertices`` (randomized sign-pattern search) is
ported below with a local ``RandomState`` and run on the 2016 test fixture
(``np.random.seed(42); df = np.random.normal(size=(10, 3))``, n = 2). Its
vertex order is random, so the vertices are compared as sets.
"""

from __future__ import annotations

import math

import numpy as np

import active_subspaces as asub


def _unique_rows(S):
    T = S.view(np.dtype((np.void, S.dtype.itemsize * S.shape[1])))
    return np.unique(T).view(S.dtype).reshape(-1, S.shape[1])


def _zonotope_vertices_2016(W1, rs, Nsamples=10000, maxcount=100000):
    m, n = W1.shape
    totalverts = 2 * sum(math.comb(m - 1, i) for i in range(n))  # nzv(m, n)
    Z = rs.normal(size=(Nsamples, n))
    X = _unique_rows(np.sign(np.dot(Z, W1.transpose())))
    X = _unique_rows(np.vstack((X, -X)))
    count = 0
    while X.shape[0] < totalverts and count <= maxcount:
        Z = rs.normal(size=(Nsamples, n))
        X0 = _unique_rows(np.sign(np.dot(Z, W1.transpose())))
        X = _unique_rows(np.vstack((X, X0, -X0)))
        count += 1
    return np.dot(X, W1), X


def test_vertices_match_2016_fixture():
    df = np.random.RandomState(42).normal(size=(10, 3))
    evals, evecs = np.linalg.eigh(df.T @ df / 10)
    W_2016 = evecs[:, np.argsort(evals)[::-1]]
    W1_2016 = W_2016[:, :2]
    V_2016, X_2016 = _zonotope_vertices_2016(W1_2016, np.random.RandomState(0))
    assert V_2016.shape == (6, 2)

    space = asub.Uniform.reference(3)
    sub = asub.active_subspace(space=space, X=np.zeros((10, 3)), grads=df, n=2)
    dom = asub.active_domain(sub)
    signs = np.sign(np.sum(W1_2016 * sub.W1, axis=0))
    V = V_2016 * signs
    assert dom.n_vertices == 6
    D = np.max(np.abs(dom.vertices[:, None, :] - V[None, :, :]), axis=-1)
    assert np.all(D.min(axis=0) < 1e-10)
    assert np.all(D.min(axis=1) < 1e-10)
    assert {tuple(r) for r in dom.vertex_preimages} == {tuple(r) for r in X_2016}
