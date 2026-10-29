"""Regressions against 2016 behaviour (spec 07 §8; spec 17 V07.8, V07.14; [C-01], [C-17],
[N-01]).
"""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import subspaces as S
from active_subspaces.errors import ValidityWarning


def _C(res) -> np.ndarray:
    return (res.W * res.eigenvalues) @ res.W.T


def test_ladle_ridge():
    """[C-17]: on an exact 1-D ridge 2016's ladle returned 2 (argmin + 1); v1 returns 1, and so
    does RS.
    """
    m, M = 10, 300
    a = np.zeros(m)
    a[:2] = [2.0, 1.0]
    a /= np.linalg.norm(a)
    space = asub.Uniform.reference(m)
    for seed in range(10):
        X = np.random.default_rng(seed).uniform(-1, 1, (M, m))
        G = np.outer(np.exp(X @ a), a)
        res = S.active_subspace(space=space, grads=G, dimension_rule="li", nboot=100, rng=seed)
        assert res.n == 1, seed
        assert res.criterion.shape == (m,)  # k = 0..k_n with k_n = m - 1
        assert S.with_dimension(res, dimension_rule="rs").n == 1, seed


def test_qphd_bounded():
    """[C-01]: 2016 raised TypeError for inputs inside [-1, 1]; v1 returns gg^T + H Sigma H
    [D-40].
    """
    X = np.random.default_rng(0).uniform(-1, 1, (200, 4))
    c = np.array([1.0, 0.5, 0.1, 0.0])
    f = (X @ c) ** 2  # g = 0, H = 2 c c^T
    with pytest.warns(ValidityWarning):
        res = S.active_subspace(space=asub.Uniform.reference(4), X=X, f=f, estimator="qphd", n=1)
    H = 2 * np.outer(c, c)
    C = H @ (np.eye(4) / 3) @ H
    assert np.linalg.norm(_C(res) - C, 2) <= 1e-10 * np.linalg.norm(C, 2)
    assert res.eigenvalues[0] == pytest.approx(4 / 3 * (c @ c) ** 2, rel=1e-10)


def _save_2016(
    X: np.ndarray, f: np.ndarray, weights: np.ndarray, H: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """2016 dev ``save_subspace`` for Gaussian data (gamma = 1); returns (C, slice labels)."""
    _M, m = X.shape
    Z = X - X.mean(axis=0)
    bins = np.percentile(f, np.linspace(0, 100, H + 1))
    ind = np.digitize(f, bins)
    ind[ind == 0] = 1
    ind[ind == len(bins)] = H
    C = np.zeros((m, m))
    for i in range(H):
        in_slice = ind == i + 1
        if np.any(in_slice):
            Zt = Z[in_slice] - Z[in_slice].mean(axis=0)
            sw = weights[in_slice] / weights[in_slice].sum()
            V = (
                np.eye(m) - (Zt.T @ (sw[:, None] * Zt)) / (1 - np.sum(sw**2))
                if in_slice.sum() > 1
                else np.eye(m)
            )
            C += V @ V
    return C, ind


def test_save_vs_2016():
    """[N-01], [D-42]: same slices as 2016 (S = 10, M divisible by S), and C_v1 = C_2016 / S."""
    M, m, H = 200, 4, 10
    gen = np.random.default_rng(3)
    X = gen.standard_normal((M, m))
    f = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.01 * gen.standard_normal(M)
    assert np.unique(f).size == M
    C16, ind = _save_2016(X, f, np.full(M, 1 / M), H)
    slices = S._slices(f, H)
    assert len(slices) == H
    for h, idx in enumerate(slices):
        assert np.array_equal(np.sort(idx), np.flatnonzero(ind == h + 1))
    res = S.active_subspace(space=asub.Gaussian.reference(m), X=X, f=f, estimator="save", n=1)
    assert np.linalg.norm(_C(res) - C16 / H) <= 1e-12 * np.linalg.norm(C16 / H)
