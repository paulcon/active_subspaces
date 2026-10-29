"""Property tests for ``active_subspaces.subspaces`` (spec 07 §3.2, §8; spec 17 V07.10)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import subspaces as S


@pytest.mark.parametrize("estimator", sorted(S.ESTIMATORS))
def test_sign_rule_and_permutation(estimator):
    m, M = 4, 80
    gen = np.random.default_rng(21)
    space = asub.Gaussian.reference(m)
    X = gen.standard_normal((M, m))
    B = np.diag([3.0, -1.5, 0.7, 0.2]) + 0.1
    f = (
        0.5 * np.einsum("ij,jk,ik->i", X, B, X)
        + X @ np.array([1.0, 0.3, -0.2, 0.1])
        + 0.2 * X[:, 0] ** 3
    )
    G = X @ B + np.array([1.0, 0.3, -0.2, 0.1]) + np.outer(0.6 * X[:, 0] ** 2, np.eye(m)[0])
    perm = np.array([2, 0, 3, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asub.ValidityWarning)
        r = S.active_subspace(space=space, X=X, f=f, grads=G, estimator=estimator, n=1)
        rp = S.active_subspace(
            space=space, X=X[:, perm], f=f, grads=G[:, perm], estimator=estimator, n=1
        )
    W = r.W
    j = np.argmax(np.abs(W), axis=0)
    assert np.all(W[j, np.arange(m)] > 0)
    k = 1 if estimator == "ols" else m  # OLS: only w_1 is defined
    lam = np.abs(r.eigenvalues[: k + (k < m)])
    assert np.all(np.diff(lam) < -1e-8 * lam[0])  # distinct eigenvalues, so no ties
    assert np.allclose(
        rp.eigenvalues, r.eigenvalues, rtol=0, atol=1e-10 * np.abs(r.eigenvalues).max()
    )
    assert np.allclose(rp.W[:, :k], W[perm, :k], rtol=0, atol=1e-8)
