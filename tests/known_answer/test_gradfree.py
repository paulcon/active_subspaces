"""Known answers for spec 14 B: full sketches, SVD identity and signs [C-24], rank-deficient
Bernoulli (V14.16, V14.19, V14.20).
"""

from __future__ import annotations

import warnings

import numpy as np

import active_subspaces as asub
from active_subspaces import _random

with warnings.catch_warnings():
    warnings.simplefilter("ignore", asub.ExperimentalWarning)
    from active_subspaces.experimental import (
        gradfree,
        sketched_active_subspace,
        subspace_from_sketches,
    )


def _sign_rule_holds(W: np.ndarray) -> bool:
    idx = np.argmax(np.abs(W), axis=0)
    return bool(np.all(W[idx, np.arange(W.shape[1])] > 0))


def test_full_sketches():
    """k = m, orthogonal design: the projection estimator recovers G = grad f; W, s are its SVD
    with s = sigma / M.
    """
    m, M = 5, 40
    gen = np.random.default_rng(1)
    space = asub.Uniform.reference(m)
    X = gen.uniform(-1, 1, (M, m))
    B = gen.standard_normal((m, m))
    grads = X @ B + 0.3  # rows: grad f(x_i), exact
    Z = gen.standard_normal((M, m, m))
    E = np.linalg.qr(Z)[0]
    D = np.einsum("imk,im->ik", E, grads)
    res = subspace_from_sketches(sketches=D, directions=E, space=space)
    G = grads.T
    Gr = res.W @ np.diag(M * res.s) @ res.V.T
    assert np.linalg.norm(Gr - G) <= 1e-10 * np.linalg.norm(G)
    U, sig, _ = np.linalg.svd(G, full_matrices=False)
    np.testing.assert_allclose(res.s, sig / M, rtol=1e-10)
    np.testing.assert_allclose(np.abs(res.W.T @ U), np.eye(m), atol=1e-8)


def test_physical_directions_map():
    """subspace_from_sketches maps physical directions with A^{-1}: full sketches give G = A^T
    grad_xi f (dense Gaussian).
    """
    m, M = 4, 30
    gen = np.random.default_rng(2)
    Bc = gen.standard_normal((m, m))
    space = asub.Gaussian(gen.standard_normal(m), Bc @ Bc.T + m * np.eye(m))
    grads_phys = gen.standard_normal((M, m))
    E = gen.standard_normal((M, m, m))
    D = np.einsum("imk,im->ik", E, grads_phys)
    res = subspace_from_sketches(sketches=D, directions=E, space=space)
    G_norm = space.grads_to_normalized(grads_phys).T
    Gr = res.W @ np.diag(M * res.s) @ res.V.T
    assert np.linalg.norm(Gr - G_norm) <= 1e-10 * np.linalg.norm(G_norm)


def test_svd_identity():
    """[C-24]: G = W diag(M s) V^T to 1e-12 with a zero in the first row and mixed-sign first
    entries; W follows the sign rule.
    """
    m, M = 4, 6
    gen = np.random.default_rng(3)
    G = gen.standard_normal((m, M))
    G[0, :] = 0.0
    G[0, 2] = 1.0
    G[:, 0] *= -1
    E = np.tile(np.eye(m), (M, 1, 1))
    res = subspace_from_sketches(
        sketches=G.T.copy(), directions=E, space=asub.Uniform.reference(m)
    )
    Gr = res.W @ np.diag(M * res.s) @ res.V.T
    assert np.linalg.norm(Gr - G) <= 1e-12 * np.linalg.norm(G)
    assert _sign_rule_holds(res.W)
    # first-row signs of W are mixed (so a first-row rule would flip differently) and the identity
    # still holds
    assert len(set(np.sign(res.W[0, : min(m, M)]).tolist())) > 1 or np.any(res.W[0] == 0)
    # the private SVD helper applies the same flips to V
    W, sig, V = gradfree._svd(G)
    assert np.linalg.norm(W @ np.diag(sig) @ V.T - G) <= 1e-12 * np.linalg.norm(G)
    # a column whose largest entry is negative in the raw SVD is flipped together with its V column
    U0, _s0, Vt0 = np.linalg.svd(G, full_matrices=False)
    flips = np.sign(U0[np.argmax(np.abs(U0), axis=0), np.arange(U0.shape[1])])
    np.testing.assert_array_equal(W, U0 * flips)
    np.testing.assert_array_equal(V, Vt0.T * flips)


def test_rank_deficient_bernoulli():
    """m = 3, k = 2, M = 200 Bernoulli: runs, and each g_i equals P_i grad f(x_i) to 1e-12."""
    m, k, M = 3, 2, 200
    space = asub.Uniform.reference(m)
    a = np.array([1.0, -0.5, 2.0])
    res = sketched_active_subspace(
        lambda X: X @ a + 0.1 * X[:, 0] ** 2, space=space, k=k, M=M, design="bernoulli", rng=4
    )
    assert res.W.shape == (3, 3)
    gen, _ = _random.generator(4)
    child = gen.spawn(2)
    X = space.sample(M, rng=child[0], normalized=True)
    E = 2.0 * child[1].integers(0, 2, size=(M, m, k)) - 1.0
    deficient = [i for i in range(M) if np.linalg.matrix_rank(E[i]) < k]
    assert len(deficient) > 20
    grads = a + np.outer(0.2 * X[:, 0], [1.0, 0.0, 0.0])
    D = np.einsum("imk,im->ik", E, grads)
    G = gradfree._projection(D, E)
    for i in range(M):
        Q = np.linalg.svd(E[i], full_matrices=False)[0][:, : np.linalg.matrix_rank(E[i])]
        Pg = Q @ (Q.T @ grads[i])
        np.testing.assert_allclose(G[:, i], Pg, rtol=0, atol=1e-12 * np.linalg.norm(grads[i]))
