"""Known answers for ``active_subspaces.reduced_model`` (spec 11 §8; spec 17 V11.1, V11.2,
V11.9, V11.10).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import reduced_model as R
from active_subspaces import test_functions as tf
from active_subspaces.errors import AccuracyWarning, ValidityWarning


def _space(m: int) -> asub.Uniform:
    gen = np.random.default_rng(42)
    lower = gen.uniform(-3, 3, m)
    return asub.Uniform(lower, lower + 10.0 ** gen.uniform(-1, 1, m))


class _Square:
    """Physical ridge f(xi) = (a^T xi + c)^2 and its gradient, as batch callables."""

    def __init__(self, a: np.ndarray, c: float = 0.0) -> None:
        self.a, self.c = a, c

    def value(self, X):
        return (X @ self.a + self.c) ** 2

    def grad(self, X):
        return 2.0 * (X @ self.a + self.c)[:, None] * self.a[None, :]


def _direction_error(rm, a: np.ndarray) -> float:
    w = rm.subspace.physical_directions()[:, 0]
    return float(np.linalg.norm(np.outer(w, w) - np.outer(a, a) / (a @ a), 2))


def test_exact_ridge_data_path():
    """V11.1: (a^T xi)^2, m = 6, M = 200, given gradients, poly degree 2."""
    m = 6
    sp = _space(m)
    a = np.random.default_rng(1).standard_normal(m)
    fn = _Square(a)
    X = sp.sample(200, rng=2)
    rm = R.build_reduced_model(
        space=sp, X=X, f=fn.value(X), grads=fn.grad(X), surrogate_kind="poly", degree=2, rng=3
    )
    assert rm.n == 1
    assert rm.gradient_source == "given"
    assert rm.r_squared_kind == "loo"
    assert _direction_error(rm, a) <= 1e-10
    Xt = sp.sample(100, rng=4)
    ft = fn.value(Xt)
    assert np.max(np.abs(R.predict(rm, Xt) - ft)) <= 1e-8 * np.max(np.abs(ft))
    Gt = fn.grad(Xt)
    assert np.max(np.abs(R.predict_gradient(rm, Xt) - Gt)) <= 1e-8 * np.max(np.abs(Gt))
    assert rm.r_squared >= 1 - 1e-10


def test_exact_ridge_local_linear():
    """Without gradients the default estimator uses local linear gradients [D-32]; exact for a
    linear f.
    """
    m = 4
    sp = _space(m)
    a = np.array([1.0, -0.5, 0.25, 0.0])
    X = sp.sample(80, rng=0)
    rm = R.build_reduced_model(space=sp, X=X, f=X @ a, n=1, surrogate_kind="poly", degree=1, rng=0)
    assert rm.gradient_source == "local_linear"
    assert rm.grads.shape == (80, m)
    np.testing.assert_allclose(
        rm.grads, np.tile(np.asarray(sp.scale) @ a, (80, 1)), rtol=1e-8, atol=1e-10
    )
    assert _direction_error(rm, a) <= 1e-8
    assert rm.r_squared >= 1 - 1e-10


def test_model_path():
    """V11.2: the same f as a Model; M = floor(6 * 7 * ln 6) = 75; holdout R^2; finite
    differences without a gradient.
    """
    m = 6
    sp = _space(m)
    a = np.random.default_rng(1).standard_normal(m)
    fn = _Square(a, c=0.5)
    model = asub.Model(space=sp, value=fn.value, grad=fn.grad)
    rm = R.build_reduced_model(model=model, surrogate_kind="poly", degree=2, rng=5)
    assert rm.X.shape == (75, m)
    assert rm.source == "model"
    assert rm.gradient_source == "model"
    assert rm.r_squared_kind == "holdout"
    assert rm.r_squared >= 1 - 1e-10
    assert rm.n == 1
    assert _direction_error(rm, a) <= 1e-10
    np.testing.assert_allclose(rm.f, fn.value(rm.X_physical), rtol=1e-12)
    value_only = asub.Model(space=sp, value=fn.value)
    rm_fd = R.build_reduced_model(model=value_only, surrogate_kind="poly", degree=2, rng=5)
    assert rm_fd.gradient_source == "finite_difference"
    assert rm_fd.n == rm.n
    assert rm_fd.r_squared >= 1 - 1e-10
    np.testing.assert_array_equal(rm_fd.X, rm.X)  # same sample stream on both paths


def test_registry_function_end_to_end():
    """A registry known-answer problem (``ridge_sin_uniform_10``): exact direction, mean 0,
    exact R^2 to the surrogate error.
    """
    model = tf.builtin_models()["ridge_sin_uniform_10"]
    rr = model.reference_results
    rm = R.build_reduced_model(model=model, rng=17)
    assert rm.n == 1
    assert rm.gradient_source == "model"
    W = rm.W1
    assert (
        np.linalg.norm(W @ W.T - np.outer(rr.eigenvectors[:, 0], rr.eigenvectors[:, 0]), 2)
        <= 1e-10
    )
    assert rm.eigenvalues[0] > 0
    assert np.all(np.abs(rm.eigenvalues[1:]) <= 1e-12 * rm.eigenvalues[0])
    assert rm.r_squared >= 0.999
    with pytest.warns(
        asub.AccuracyWarning, match=r"accuracy\.correlated_draws"
    ):  # n_mc < 20 [D-114]
        res = R.average(rm, n_nodes=10, model=model, n_mc=4, rng=1)
    assert abs(res.estimate - rr.mean) <= 1e-10
    assert rm.provenance.parameters["model"] == "ridge_sin_uniform_10"


def test_minimum_convex_ridge():
    """V11.9: on a convex quadratic ridge the minimum is found with named fields [A-10]."""
    m = 5
    sp = _space(m)
    coeffs = np.array([1.0, 0.5, -0.4, 0.2, 0.1])
    a = coeffs / ((sp.upper - sp.lower) / 2)  # a^T xi + c = coeffs^T x - 0.3 in normalized x
    c = -(a @ np.asarray(sp.shift)) - 0.3  # so min f = 0 is attained inside the box
    fn = _Square(a, c)
    X = sp.sample(150, rng=6)
    f = fn.value(X)
    rm = R.build_reduced_model(
        space=sp, X=X, f=f, grads=fn.grad(X), surrogate_kind="poly", degree=2, rng=7
    )
    model = asub.Model(space=sp, value=fn.value)
    with warnings.catch_warnings():
        # an exact ridge has no curvature along z; the lift clips it [D-49]
        warnings.filterwarnings("ignore", message=r"lift: .*curvature", category=AccuracyWarning)
        res = R.minimum(rm, model=model, rng=8)
    assert isinstance(res, asub.optimization.MinimizationResult)
    assert bool(sp.contains(res.x_star_physical[None, :])[0])
    assert res.x_star.shape == (m,)
    assert res.y_star.shape == (1,)
    scale = float(np.ptp(f))
    assert abs(res.f_star - 0.0) <= 1e-6 * scale
    assert abs(res.f_model - 0.0) <= 1e-6 * scale


def _lingrad_2016(X: np.ndarray, f: np.ndarray) -> np.ndarray:
    """dev ``sdr._lingrad``."""
    A = np.hstack((np.ones((X.shape[0], 1)), X))
    u = np.linalg.lstsq(A, f, rcond=None)[0]
    return u[1:] / np.linalg.norm(u[1:])


def _quadratic_fit(X: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares quadratic about the origin of X: gradient b and Hessian H at 0 (dev
    ``PolynomialApproximation(2)``).
    """
    M, m = X.shape
    cols = [np.ones(M), *X.T]
    pairs = [(i, j) for i in range(m) for j in range(i, m)]
    cols += [X[:, i] * X[:, j] for i, j in pairs]
    coef = np.linalg.lstsq(np.column_stack(cols), f, rcond=None)[0]
    b = coef[1 : m + 1]
    H = np.zeros((m, m))
    for k, (i, j) in enumerate(pairs):
        c = coef[m + 1 + k]
        if i == j:
            H[i, i] = 2 * c
        else:
            H[i, j] = H[j, i] = c
    return b, H


def test_sdr_equivalence():
    """V11.10 [D-117]: linear_gradient_check is OLS; quadratic_model_check is QPHD with C = A
    C_dev A.
    """
    m = 5
    gen = np.random.default_rng(3)
    Xn = gen.standard_normal((300, m))
    f = Xn @ np.array([1.0, -2.0, 0.5, 0.0, 0.3]) + 0.1 * np.sin(Xn[:, 0])
    w = _lingrad_2016(Xn, f)
    res = asub.active_subspace(space=asub.Gaussian.reference(m), X=Xn, f=f, estimator="ols", n=1)
    w1 = res.W1[:, 0]
    assert min(np.linalg.norm(w - w1), np.linalg.norm(w + w1)) <= 1e-12

    sp = _space(m)
    X = sp.sample(400, rng=4)
    Q = gen.standard_normal((m, m))
    xc = (X - sp.shift) / ((sp.upper - sp.lower) / 2)  # a quadratic with a linear part
    f = 0.5 * np.einsum("ij,jk,ik->i", xc, Q + Q.T, xc) + xc @ np.arange(1.0, m + 1)
    with pytest.warns(ValidityWarning):
        res = asub.active_subspace(space=sp, X=X, f=f, estimator="qphd", n=1)
    C = (res.W * res.eigenvalues) @ res.W.T
    b, H = _quadratic_fit(X - sp.shift, f)  # dev: centered physical data
    gamma = (sp.upper - sp.lower) ** 2 / 12.0
    C_dev = np.outer(b, b) + H @ np.diag(gamma) @ H
    A = np.asarray(sp.scale)
    expected = A @ C_dev @ A
    assert np.linalg.norm(C - expected, 2) <= 1e-10 * np.linalg.norm(expected, 2)
