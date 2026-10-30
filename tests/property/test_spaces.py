"""Property tests for ``active_subspaces.spaces`` (spec 03 §8; spec 17 V03.9)."""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub


@pytest.mark.parametrize("seed", range(10))
def test_physical_signs(seed):
    """q_j^T A^{-T} w_j > 0 for every column, so physical directions keep the signs of W;
    negating column j of W negates column j of the output only [D-197].
    """
    gen = np.random.default_rng(seed)
    m = int(gen.integers(2, 8))
    k = int(gen.integers(1, m + 1))
    L = np.tril(gen.standard_normal((m, m))) + m * np.eye(m)
    spaces = [
        asub.Uniform(-gen.uniform(0.1, 10.0, m), gen.uniform(0.1, 1e3, m)),
        asub.Gaussian(gen.standard_normal(m), L @ L.T),
    ]
    if m >= 3:  # V03.9: the product, with a full-covariance Gaussian block
        mg = int(gen.integers(2, m))
        Lg = np.tril(gen.standard_normal((mg, mg))) + mg * np.eye(mg)
        spaces.append(
            asub.Product(
                [
                    asub.Uniform(-gen.uniform(0.1, 10.0, m - mg), gen.uniform(0.1, 1e3, m - mg)),
                    asub.Gaussian(gen.standard_normal(mg), Lg @ Lg.T),
                ]
            )
        )
    W = np.linalg.qr(gen.standard_normal((m, k)))[0]
    for space in spaces:
        P = space.directions_to_physical(W)
        target = np.linalg.solve(np.asarray(space.scale).T, W)
        assert np.all(np.einsum("ij,ij->j", P, target) > 0)
        j = int(gen.integers(0, k))
        Wf = W.copy()
        Wf[:, j] *= -1.0
        Pf = space.directions_to_physical(Wf)
        np.testing.assert_array_equal(Pf[:, j], -P[:, j])
        others = [i for i in range(k) if i != j]
        np.testing.assert_array_equal(Pf[:, others], P[:, others])
