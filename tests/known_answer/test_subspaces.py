"""Known answers for ``active_subspaces.subspaces`` (spec 07 §8; spec 17 V07.1-V07.6, V07.9,
V07.12, V07.13, V07.16).
"""

from __future__ import annotations

import contextlib
import itertools
import math

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import subspaces as S
from active_subspaces.errors import AccuracyWarning, ValidityWarning


def _orth(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return np.linalg.qr(A.reshape(A.shape[0], -1))[0]


def _dist(U: np.ndarray, V: np.ndarray) -> float:
    """Subspace distance ||(I - P_U) V||_2 for orthonormal bases of equal dimension (spec 17
    §3.1).
    """
    U, V = _orth(U), _orth(V)
    return float(np.linalg.norm(V - U @ (U.T @ V), 2))


def _C(res) -> np.ndarray:
    return (res.W * res.eigenvalues) @ res.W.T


def _tensor_rule(nodes: np.ndarray, weights: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(list(itertools.product(nodes, repeat=m)))
    w = np.array([math.prod(t) for t in itertools.product(weights, repeat=m)])
    return X, w / w.sum()


def _sym(m: int, seed: int) -> np.ndarray:
    B = np.random.default_rng(seed).standard_normal((m, m))
    return (B + B.T) / 2


@pytest.fixture(params=["uniform", "gaussian"])
def space(request, uniform_space, gaussian_space):
    return uniform_space(5, seed=3) if request.param == "uniform" else gaussian_space(5, seed=3)


# --- V07.1 ---------------------------------------------------------------------------------------


def test_as_ridge(space):
    a = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    X = space.sample(50, rng=0)
    G = np.cos(X @ a)[:, None] * a[None, :]  # f = sin(a^T xi)
    res = S.active_subspace(space=space, grads=G)
    A = np.asarray(space.scale)
    assert res.n == 1
    assert _dist(res.W1, A.T @ a) <= 1e-12
    assert _dist(res.physical_directions(), a) <= 1e-12


# --- V07.2, V07.3 --------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["uniform", "gaussian"])
def test_as_quadratic_exact_rule(kind):
    m = 3
    B = _sym(m, 0)
    if kind == "uniform":
        X, w = _tensor_rule(*np.polynomial.legendre.leggauss(2), m)
        space, Sigma = asub.Uniform.reference(m), np.eye(m) / 3
    else:
        X, w = _tensor_rule(*np.polynomial.hermite_e.hermegauss(3), m)
        space, Sigma = asub.Gaussian.reference(m), np.eye(m)
    res = S.active_subspace(space=space, X=X, grads=X @ B, quad_weights=w)  # f = x^T B x / 2
    C = B @ Sigma @ B
    assert np.linalg.norm(_C(res) - C) <= 1e-13 * np.linalg.norm(C)
    lam = np.linalg.eigvalsh(C)[::-1]
    assert np.allclose(res.eigenvalues, lam, rtol=0, atol=1e-13 * lam[0])


def test_nas_asx_nasx_by_hand():
    m, M = 3, 6
    gen = np.random.default_rng(4)
    X = gen.standard_normal((M, m))
    G = gen.standard_normal((M, m))
    G[2] = 0.0
    w = gen.uniform(0.5, 1.0, M)
    w /= w.sum()
    space = asub.Gaussian.reference(m)
    Gh = np.array([g / np.linalg.norm(g) if np.linalg.norm(g) > 0 else g for g in G])
    Xh = X / np.linalg.norm(X, axis=1, keepdims=True)
    expected = {
        "nas": sum(w[i] * np.outer(Gh[i], Gh[i]) for i in range(M)),
        "asx": sum(w[i] * (np.outer(G[i], X[i]) + np.outer(X[i], G[i])) / 2 for i in range(M)),
        "nasx": sum(
            w[i] * (np.outer(Gh[i], Xh[i]) + np.outer(Xh[i], Gh[i])) / 2 for i in range(M)
        ),
    }
    for name, C in expected.items():
        res = S.active_subspace(space=space, X=X, grads=G, quad_weights=w, estimator=name, n=1)
        assert np.linalg.norm(_C(res) - C) <= 1e-12 * np.linalg.norm(C), name
        assert np.all(np.diff(np.abs(res.eigenvalues)) <= 0)
    assert np.trace(expected["nas"]) == pytest.approx(1 - w[2])


# --- V07.4 ---------------------------------------------------------------------------------------


def test_ols_linear():
    m = 4
    a = np.array([0.5, -1.0, 2.0, 0.25])
    space = asub.Gaussian.reference(m)
    X = space.sample(20, rng=1)
    res = S.active_subspace(space=space, X=X, f=3.0 + X @ a, estimator="ols")
    assert res.eigenvalues[0] == pytest.approx(a @ a, rel=1e-12)
    assert np.all(res.eigenvalues[1:] == 0)
    assert np.allclose(res.W1[:, 0], a / np.linalg.norm(a), atol=1e-12)
    # physical slope on a Uniform space: the normalized slope is A^T a
    uspace = asub.Uniform([0, 1, 2, 3], [1, 3, 6, 4])
    Xu = uspace.sample(20, rng=2)
    with pytest.warns(ValidityWarning):
        ru = S.active_subspace(space=uspace, X=Xu, f=Xu @ a, estimator="ols")
    u = np.asarray(uspace.scale).T @ a
    assert ru.eigenvalues[0] == pytest.approx(u @ u, rel=1e-12)
    assert _dist(ru.physical_directions(), a) <= 1e-12


# --- V07.5 ---------------------------------------------------------------------------------------


def test_qphd_exact(space):
    m = space.dim
    gen = np.random.default_rng(5)
    g, H = gen.standard_normal(m), _sym(m, 6)
    X = space.sample(40, rng=7)
    x = space.to_normalized(X)
    f = 1.5 + x @ g + 0.5 * np.einsum("ij,jk,ik->i", x, H, x)
    Sigma = np.asarray(space.reference_covariance)
    C = np.outer(g, g) + H @ Sigma @ H
    if space.kind == "uniform":
        with pytest.warns(ValidityWarning):
            res = S.active_subspace(space=space, X=X, f=f, estimator="qphd", n=2)
    else:
        res = S.active_subspace(space=space, X=X, f=f, estimator="qphd", n=2)
    lam = np.linalg.eigvalsh(C)[::-1]
    assert np.linalg.norm(_C(res) - C, 2) <= 1e-10 * lam[0]
    assert np.allclose(res.eigenvalues, lam, rtol=0, atol=1e-10 * lam[0])


# --- V07.6 ---------------------------------------------------------------------------------------


def test_phd_scale():
    m = 3
    B = _sym(m, 8)
    X, w = _tensor_rule(*np.polynomial.hermite_e.hermegauss(3), m)
    f = np.einsum("ij,jk,ik->i", X, B, X)
    res = S.active_subspace(
        space=asub.Gaussian.reference(m), X=X, f=f, quad_weights=w, estimator="phd", n=1
    )
    C = 2 * B
    assert np.linalg.norm(_C(res) - C) <= 1e-12 * np.linalg.norm(C)
    lam = np.linalg.eigvalsh(C)
    lam = lam[np.argsort(-np.abs(lam), kind="stable")]
    assert np.allclose(res.eigenvalues, lam, rtol=0, atol=1e-12 * np.abs(lam).max())
    assert (res.eigenvalues < 0).any()  # signed [D-31]


# --- V07.9 ---------------------------------------------------------------------------------------


def test_ss_by_hand_and_duplicate_row():
    space = asub.Gaussian.reference(2)
    X = np.array([[0.0, 0.0], [1.0, 0.5], [-0.5, 2.0], [0.3, -1.0], [2.0, 1.0]])
    f = np.array([0.0, 1.0, -2.0, 0.5, 3.0])
    w = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    C = np.zeros((2, 2))
    for i in range(5):
        for j in range(5):
            if i != j:
                dx = X[j] - X[i]
                C += w[i] * w[j] * (f[i] - f[j]) ** 2 / (dx @ dx) * np.outer(dx, dx)
    with pytest.warns(ValidityWarning):
        r1 = S.active_subspace(space=space, X=X, f=f, quad_weights=w, estimator="ss", n=1)
    assert np.linalg.norm(_C(r1) - C) <= 1e-12 * np.linalg.norm(C)
    X2 = np.vstack([X, X[2]])
    f2 = np.append(f, f[2])
    w2 = np.append(w, w[2] / 2)
    w2[2] /= 2
    with pytest.warns(ValidityWarning):
        r2 = S.active_subspace(space=space, X=X2, f=f2, quad_weights=w2, estimator="ss", n=1)
    assert np.linalg.norm(_C(r2) - C) <= 1e-12 * np.linalg.norm(C)


# --- V07.12, V07.13 ------------------------------------------------------------------------------


def test_evg_gap():
    lam = np.array([1.97, 0.051, 1.2e-6, 1e-7])
    m = lam.size
    G = 2.0 * np.diag(np.sqrt(lam))  # weights 1/4: C = diag(lam)
    res = S.active_subspace(space=asub.Gaussian.reference(m), grads=G)
    assert res.n == 1
    assert res.dimension_options == (("gap", "linear"),)
    assert np.allclose(res.criterion, lam[:-1] - lam[1:], rtol=1e-12, atol=0)
    rlog = S.with_dimension(res, dimension_rule=S.EVG(gap="log"))
    assert rlog.n == 2
    assert np.allclose(rlog.criterion, np.log(lam[:-1] / lam[1:]), rtol=1e-12, atol=0)
    ks, vals, k_hat = S.dimension_criterion(res, rule=S.EVG(gap="log"))
    assert ks.tolist() == [1, 2, 3]
    assert k_hat == 2
    assert np.array_equal(vals, rlog.criterion)


def _ladle_by_definition(lam: np.ndarray, dets: np.ndarray) -> np.ndarray:
    """Luo & Li (2016) eqs. (3)-(8): g_n(k) = f_n(k) + phi_n(k) on k = 0..k_n."""
    p = lam.size
    kn = p - 1 if p <= 10 else math.floor(p / math.log(p))
    f0 = [0.0] + [
        float(np.mean([1.0 - dets[b, k - 1] for b in range(dets.shape[0])]))
        for k in range(1, kn + 1)
    ]
    fsum = 1.0 + sum(f0)
    lsum = 1.0 + sum(abs(lam[i]) for i in range(kn + 1))
    return np.array([f0[k] / fsum + abs(lam[k]) / lsum for k in range(kn + 1)])


@pytest.mark.parametrize(("m", "kn"), [(10, 9), (11, 4)])
def test_ladle_formula(m, kn, make_provenance):
    gen = np.random.default_rng(m)
    B = 7
    lam = np.sort(gen.uniform(0, 2, m))[::-1]
    dets = gen.uniform(0, 1, (B, m - 1))
    hand = S.SubspaceResult(
        space=asub.Gaussian.reference(m),
        estimator="as",
        estimator_options=(),
        eigenvalues=lam,
        eigenvectors=np.eye(m),
        n=1,
        dimension_rule="fixed",
        dimension_options=(),
        criterion=None,
        n_samples=20,
        nboot=B,
        bootstrap_eigenvalues=np.zeros((B, m)),
        bootstrap_distances=np.zeros((B, m - 1)),
        bootstrap_dets=dets,
        provenance=make_provenance(),
    )
    ks, vals, k_hat = S.dimension_criterion(hand, rule="li")
    g = _ladle_by_definition(lam, dets)
    assert ks.tolist() == list(range(kn + 1))
    assert np.allclose(vals, g, rtol=0, atol=1e-14)
    assert k_hat == int(np.argmin(g))
    with pytest.warns(AccuracyWarning) if k_hat == 0 else contextlib.nullcontext():
        r = S.with_dimension(hand, dimension_rule="li")
    assert r.n == max(k_hat, 1)
    assert np.array_equal(r.criterion, vals)


def test_ladle_and_rs_known_rank():
    """Exact rank-2 structure f = sin(a1^T x) + (a2^T x)^2 / 2 with exact gradients: LI and RS
    choose 2.
    """
    m, M = 6, 300
    gen = np.random.default_rng(12)
    a1, a2 = np.linalg.qr(gen.standard_normal((m, 2)))[0].T
    space = asub.Gaussian.reference(m)
    for seed in range(3):
        X = space.sample(M, rng=seed)
        G = np.cos(X @ a1)[:, None] * a1 + (X @ a2)[:, None] * a2
        res = S.active_subspace(space=space, grads=G, dimension_rule="li", nboot=50, rng=seed)
        assert res.n == 2
        assert S.with_dimension(res, dimension_rule="rs").n == 2
        assert _dist(res.W1, np.column_stack([a1, a2])) <= 1e-12


# --- V07.16 --------------------------------------------------------------------------------------


def test_subspace_from_basis(space):
    m = space.dim
    a = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    A = np.asarray(space.scale)
    res = S.subspace_from_basis(a[:, None], space=space)
    assert res.n == 1
    assert res.estimator == "basis"
    assert res.dimension_rule == "fixed"
    assert _dist(res.W1, A.T @ a) <= 1e-12
    assert _dist(res.physical_directions(), a) <= 1e-12
    assert np.allclose(res.W.T @ res.W, np.eye(m), atol=1e-13)
    j = np.argmax(np.abs(res.W), axis=0)
    assert np.all(res.W[j, np.arange(m)] > 0)
    V = np.random.default_rng(2).standard_normal((m, 2))
    r2 = S.subspace_from_basis(space.directions_to_physical(V), space=space)
    assert r2.n == 2
    assert _dist(r2.W1, V) <= 1e-12
