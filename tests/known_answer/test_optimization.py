"""Known-answer tests of active_subspaces.optimization (spec 10 §8; spec 17 V10.14)."""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import optimization as O


def _domain(m, n, seed, space):
    B = np.random.default_rng(seed).standard_normal((m, n))
    return asub.active_domain(asub.subspace_from_basis(B, space=space))


def test_linear_on_zonotope(seed):
    """V10.14: g = c^T y on a zonotope has g* = -||W1 c||_1 (the LP optimum over the cube); its
    vertex lifts.
    """
    m = 5
    dom = _domain(m, 2, 3, asub.Uniform.reference(m))
    W1 = np.asarray(dom.subspace.W1)
    c = np.array([0.7, -1.2])
    res = O.minimize_active(
        lambda Y: Y @ c,
        domain=dom,
        grad=lambda Y: np.tile(c, (Y.shape[0], 1)),
        n_starts=5,
        rng=seed,
    )
    assert res.value == pytest.approx(-np.sum(np.abs(W1 @ c)), rel=1e-8)
    X = np.random.default_rng(seed).uniform(-1, 1, (60, m))
    lifted = O.lift(res.y_star, domain=dom, X=X, f=X @ (W1 @ c) + 0.1 * np.sum(X**2, axis=1))
    assert lifted.projection_distance <= 1e-6 * (1 + np.linalg.norm(res.y_star))
    assert np.allclose(
        lifted.x, -np.sign(W1 @ c), atol=1e-6
    )  # the fiber of the vertex is one cube vertex


@pytest.mark.parametrize("kind", ["uniform", "gaussian"])
def test_ridge_quadratic(kind, seed):
    """f = (a^T x - t)^2 + 0.05 ||x||^2 with W1 = a / ||a||: the minimizer x* = t a / (||a||^2
    + 0.05).
    """
    m = 4
    space = asub.Uniform.reference(m) if kind == "uniform" else asub.Gaussian.reference(m)
    a = np.array([1.0, -0.5, 0.25, 0.5])
    t = 0.4
    sub = asub.subspace_from_basis(a[:, None], space=space)
    dom = asub.active_domain(sub)
    w = np.asarray(sub.W1)[:, 0]
    s = float(w @ a)  # +-||a||

    def f(X):
        return (X @ a - t) ** 2 + 0.05 * np.sum(X**2, axis=1)

    def g(
        Y,
    ):  # the exact conditional minimum over z: f restricted to y = W1^T x is minimized at z = 0
        return (s * Y[:, 0] - t) ** 2 + 0.05 * Y[:, 0] ** 2

    gen = np.random.default_rng(seed)
    X = gen.uniform(-1, 1, (80, m)) if kind == "uniform" else gen.standard_normal((80, m))
    res = O.minimize(g, domain=dom, X=X, f=f(X), model=f, lift_dim=m, rng=seed)
    x_true = t * a / (a @ a + 0.05)
    assert res.x_star == pytest.approx(x_true, abs=1e-6)
    assert res.f_model == pytest.approx(float(f(x_true[None, :])[0]), abs=1e-10)
    assert res.f_star == pytest.approx(res.f_model, abs=1e-10)


def test_lift_minimizes_over_fiber(seed):
    """[C-16]: with an exact quadratic and p = m, x* minimizes f over the fiber {x in cube :
    W1^T x = y}.
    """
    m = 5
    dom = _domain(m, 2, 7, asub.Uniform.reference(m))
    W1, W2 = np.asarray(dom.subspace.W1), np.asarray(dom.subspace.W2)
    gen = np.random.default_rng(seed)
    B = gen.standard_normal((m, m))
    A = B @ B.T + 0.5 * np.eye(m)
    x0 = gen.uniform(-1.5, 1.5, m)  # outside the cube: the fiber minimum is constrained

    def f(X):
        return np.einsum("ij,jk,ik->i", X - x0, A, X - x0)

    X = gen.uniform(-1, 1, (60, m))
    y = np.array([0.3, -0.2])
    res = O.lift(y, domain=dom, X=X, f=f(X), lift_dim=m)
    # direct solve of the fiber QP: min_z f(W1 y + W2 z) s.t. -1 <= W1 y + W2 z <= 1
    H = 2 * W2.T @ A @ W2
    gz = 2 * W2.T @ A @ (W1 @ y - x0)
    s = W1 @ y
    ref = asub.solve_qp(H, gz, A_ub=np.vstack([W2, -W2]), b_ub=np.concatenate([1 - s, 1 + s]))
    x_ref = s + W2 @ ref.x
    assert float(f(res.x[None, :])[0]) - float(f(x_ref[None, :])[0]) <= 1e-8
    assert np.allclose(res.x @ W1, y, atol=1e-10)
