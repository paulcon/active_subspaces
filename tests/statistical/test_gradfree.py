"""Statistical check of spec 14 B: ridge recovery (V14.17, provisional S(pot); calibrate in
Phase 4).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import active_subspaces as asub

with warnings.catch_warnings():
    warnings.simplefilter("ignore", asub.ExperimentalWarning)
    from active_subspaces.experimental import sketched_active_subspace


@pytest.mark.parametrize(
    ("method", "rank", "bound"), [("projection", None, 0.1), ("alternating", 1, 0.05)]
)
def test_ridge_recovery(method, rank, bound, stat_rng):
    """T: projector distance between span(W[:, 0]) and span(a) for f = sin(a^T x), m = 10, k = 3, M
    = 2000,
    Gaussian design. Construction 6, provisional bounds of spec 14 §8 (0.1 projection, 0.05
    alternating)."""
    ss = stat_rng("ridge")
    child = ss.spawn(2)[0 if method == "projection" else 1]
    gen = np.random.default_rng(child)
    a = gen.standard_normal(10)
    a /= np.linalg.norm(a)
    res = sketched_active_subspace(
        lambda X: np.sin(X @ a),
        space=asub.Uniform.reference(10),
        k=3,
        M=2000,
        method=method,
        rank=rank,
        rng=gen,
    )
    w = res.W[:, 0]
    dist = np.linalg.norm(np.outer(w, w) - np.outer(a, a), 2)
    assert dist <= bound, dist
