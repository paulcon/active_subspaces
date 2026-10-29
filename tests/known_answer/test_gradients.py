"""Known answers for spec 07 §3.8 / §8 "Gradients" (V07.24, V07.26, V07.27)."""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import gradients as gr

U = np.finfo(np.float64).eps


def _uniform(m, seed):
    gen = np.random.default_rng(seed)
    lower = gen.uniform(-5, 5, m)
    return asub.Uniform(lower, lower + 10.0 ** gen.uniform(-1, 1, m))


def _gaussian(m, seed):
    gen = np.random.default_rng(seed)
    B = gen.standard_normal((m, m))
    return asub.Gaussian(gen.standard_normal(m), B @ B.T + m * np.eye(m))


# ---------------------------------------------------------------------------
# Local linear (V07.24, V07.26)
# ---------------------------------------------------------------------------

_LL_CASES = {
    "m1_M2": ("u", 1, 2),
    "m1_M10": ("u", 1, 10),
    "m3_Mm+1": ("u", 3, 4),
    "C-19_M15_m10": ("u", 10, 15),
    "m5_M60": ("u", 5, 60),
    "gauss_m4_M30": ("g", 4, 30),
    "gauss_m10_M11": ("g", 10, 11),
}


@pytest.mark.parametrize("case", list(_LL_CASES))
def test_local_linear_linear(case):
    kind, m, M = _LL_CASES[case]
    s = _uniform(m, M) if kind == "u" else _gaussian(m, M)
    gen = np.random.default_rng(100 + M)
    X = s.sample(M, rng=gen)
    a = gen.standard_normal(m)  # physical gradient
    f = X @ a - 2.0
    r = gr.local_linear_gradients(X=X, f=f, space=s)
    g = s.grads_to_normalized(a[None, :])[0]
    assert r.grads.shape == (M, m)
    np.testing.assert_allclose(
        r.grads, np.broadcast_to(g, (M, m)), rtol=0, atol=1e-10 * np.linalg.norm(g)
    )


def test_local_linear_linear_with_duplicates():
    s = _uniform(3, 1)
    gen = np.random.default_rng(5)
    X = s.sample(12, rng=gen)
    X = np.vstack([X, X[[0, 0, 4, 7]]])  # a bootstrap-like sample with repeated rows
    a = np.array([1.0, -2.0, 0.5])
    r = gr.local_linear_gradients(X=X, f=X @ a, space=s)
    g = s.grads_to_normalized(a[None, :])[0]
    np.testing.assert_allclose(
        r.grads, np.broadcast_to(g, r.grads.shape), atol=1e-10 * np.linalg.norm(g)
    )


def test_local_linear_convergence():
    # Quadratic f; the gradient error falls as the sampling density increases (fixed designs, M ->
    # 4M).
    # The error is O(neighborhood radius) = O(M^{-1/2}) in m = 2. With the default p = m + 1 the
    # fits interpolate, so a few nearly collinear triples dominate the mean; the median is used
    # there.
    s = asub.Uniform.reference(2)
    B = np.array([[2.0, 0.5], [0.5, -1.0]])

    def fun(X):
        return 0.5 * np.einsum("ij,jk,ik->i", X, B, X) + X[:, 0]

    for p, stat in ((None, np.median), (6, np.mean)):
        errs = []
        for M in (100, 400, 1600):
            X = s.sample(M, rng=2026)
            r = gr.local_linear_gradients(X=X, f=fun(X), space=s, p=p)
            exact = r.points @ B + np.array([1.0, 0.0])
            errs.append(stat(np.linalg.norm(r.grads - exact, axis=1)))
        assert errs[1] < 0.75 * errs[0], (p, errs)
    assert errs[2] < 0.75 * errs[1], (p, errs)


# ---------------------------------------------------------------------------
# Finite differences (V07.27)
# ---------------------------------------------------------------------------


def _quadratic(space, seed):
    """f(x) = c + g^T x + x^T H x / 2 in normalized coordinates, as a physical model."""
    gen = np.random.default_rng(seed)
    m = space.dim
    Hm = gen.standard_normal((m, m))
    H = Hm + Hm.T
    g = gen.standard_normal(m)

    def fun(Z):
        x = space.to_normalized(Z)
        return 1.5 + x @ g + 0.5 * np.einsum("ij,jk,ik->i", x, H, x)

    return fun, g, H


@pytest.mark.parametrize("kind", ["u", "g"])
def test_forward_error_on_quadratic(kind):
    m = 4
    s = _uniform(m, 3) if kind == "u" else _gaussian(m, 3)
    fun, g, H = _quadratic(s, 7)
    Xn = s.sample(25, rng=9, normalized=True)
    if kind == "u":
        Xn[0] = 1.0  # backward at x_j = 1
        Xn[1] = -1.0
    else:
        Xn[0] = [3.0, -4.0, 1.5, 8.0]  # |x_j| > 1: steps grow
    X = s.from_normalized(Xn)
    h = 1e-3
    r = gr.finite_difference_gradients(fun, X=X, space=s, h=h)
    exact = r.points @ H + g
    S = r.steps
    if kind == "u":
        assert (S[0] < 0).all()
        assert (S[1] > 0).all()
    else:
        assert np.all(np.abs(S[0]) > h)
    # (f(x + s e_j) - f(x)) / s = d_j f + s H_jj / 2 exactly for a quadratic
    predicted = exact + 0.5 * S * np.diag(H)[None, :]
    scale = np.max(np.abs(r.values)) + np.max(np.abs(exact)) * np.max(np.abs(Xn))
    tol = 1e3 * U * scale / np.abs(S)
    assert np.all(np.abs(r.grads - predicted) <= tol)
    assert np.all(np.abs(r.grads - exact) <= 0.5 * np.abs(S) * np.abs(np.diag(H)) + tol)


@pytest.mark.parametrize("kind", ["u", "g"])
def test_central_exact_on_quadratic(kind):
    m = 3
    s = _uniform(m, 4) if kind == "u" else _gaussian(m, 4)
    fun, g, H = _quadratic(s, 8)
    Xn = s.sample(20, rng=10, normalized=True)
    if kind == "u":
        Xn[0] = [1.0, -1.0, 0.0]
    X = s.from_normalized(Xn)
    r = gr.finite_difference_gradients(fun, X=X, space=s, h=1e-3, scheme="central")
    exact = r.points @ H + g
    central = r.central_mask
    assert central.sum() > 0
    np.testing.assert_allclose(
        r.grads[central], exact[central], rtol=0, atol=1e-8 * np.max(np.abs(exact))
    )
    if kind == "u":
        assert not central[0, 0]
        assert not central[0, 1]
        assert central[0, 2]
        assert r.steps[0, 0] < 0
        assert r.steps[0, 1] > 0
        # fallback entries carry the one-sided O(h) error
        fb = ~central
        pred = exact[fb] + 0.5 * r.steps[fb] * np.broadcast_to(np.diag(H), r.steps.shape)[fb]
        np.testing.assert_allclose(r.grads[fb], pred, rtol=0, atol=1e-8 * np.max(np.abs(exact)))
    else:
        assert central.all()


def test_error_orders():
    # Forward error O(h), central O(h^2), on a smooth non-polynomial function.
    s = asub.Uniform.reference(3)
    c = np.array([0.7, -0.4, 0.9])

    def fun(X):
        return np.exp(X @ c)

    X = np.array([[0.1, 0.2, -0.3], [-0.4, 0.0, 0.5]])
    exact = np.exp(X @ c)[:, None] * c[None, :]
    for scheme, order in (("forward", 1), ("central", 2)):
        e1 = np.abs(
            gr.finite_difference_gradients(fun, X=X, space=s, h=1e-2, scheme=scheme).grads - exact
        )
        e2 = np.abs(
            gr.finite_difference_gradients(fun, X=X, space=s, h=5e-3, scheme=scheme).grads - exact
        )
        ratio = e1 / e2
        assert np.all(np.abs(ratio - 2.0**order) < 0.1 * 2.0**order), (scheme, ratio)


def test_linear_physical_gradient_recovered():
    s = _uniform(5, 12)
    a = np.arange(1.0, 6.0)
    X = s.sample(10, rng=3)
    for scheme in ("forward", "central"):
        r = gr.finite_difference_gradients(lambda Z: Z @ a, X=X, space=s, h=1e-4, scheme=scheme)
        np.testing.assert_allclose(r.grads_physical, np.broadcast_to(a, (10, 5)), rtol=1e-8)
