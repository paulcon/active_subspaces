"""Known-answer tests of the test problems (spec 12 §3.2-3.3, §8; V12.1, V12.5, V12.7, V12.10)."""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import test_functions as tf


def _gradient_cases():
    cases = [(name, tf.builtin_models()[name]) for name in tf.builtin_models()]
    for space in (asub.Uniform.reference(10), asub.Gaussian.reference(10)):
        for profile in ("linear", "sin", "square"):
            cases.append(
                (
                    f"ridge_{profile}_{space.kind}",
                    tf.ridge(np.arange(1, 11), profile=profile, space=space),
                )
            )
    cases.append(
        (
            "quadratic_scaled",
            tf.quadratic([2.0, 1.0, 0.25], space=asub.Uniform([0, 1, 2], [3, 5, 9])),
        )
    )
    gs = asub.Gaussian(
        np.array([1.0, -1.0, 0.0]), np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 0.5]])
    )
    cases.append(
        ("ridge_square_correlated", tf.ridge([1.0, 2.0, -1.0], profile="square", space=gs))
    )
    return cases


_CASES = _gradient_cases()


@pytest.mark.parametrize(("name", "model"), _CASES, ids=[c[0] for c in _CASES])
def test_gradients(name, model):
    space = model.space
    m = space.dim
    X = space.sample(100, rng=0)
    G = model.grad(X)
    if isinstance(space, asub.Uniform):
        steps = 1e-6 * (space.upper - space.lower)
    else:
        steps = np.full(m, 1e-6)
    FD = np.empty_like(G)
    for j in range(m):
        E = np.zeros(m)
        E[j] = steps[j]
        FD[:, j] = (model.value(X + E) - model.value(X - E)) / (2.0 * steps[j])
    scale = np.max(np.abs(G), axis=1, keepdims=True)
    assert np.all(scale > 0)
    assert np.max(np.abs(FD - G) / scale) <= 1e-6


def test_exact_nulls():
    space = tf.BOREHOLE.space
    X = space.sample(1000, rng=0)
    g = space.grads_to_normalized(tf.borehole_grad(X))
    null = g[:, 3] + g[:, 5]
    assert np.all(np.abs(null) <= 1e-12 * np.linalg.norm(g, axis=1))

    rspace = tf.ROBOT_ARM.space
    Xr = rspace.sample(1000, rng=0)
    Gr = tf.robot_arm_grad(Xr)
    assert np.all(Gr[:, 0] == 0.0)
    assert np.all(np.any(Gr[:, 1:] != 0.0, axis=1))
    Xr[:10, 4:] = 0.0  # L = 0: u = v = 0
    Xr[10, :] = [0.5, np.pi, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]  # folded arm: u = v = 0 up to rounding
    Gr = tf.robot_arm_grad(Xr)
    assert np.array_equal(Gr[:10], np.zeros((10, 8)))
    assert np.all(np.isfinite(Gr))
    assert np.all(Gr[:, 0] == 0.0)
    assert np.array_equal(tf.robot_arm(Xr[:10]), np.zeros(10))
    corner = np.array([[0.0] * 8, [2 * np.pi] * 4 + [0.0] * 4])
    f, G = asub.evaluate_with_grads(tf.ROBOT_ARM, X=corner)
    assert np.array_equal(f, [0.0, 0.0])
    assert np.array_equal(G, np.zeros((2, 8)))


def test_quadratic_factory():
    lam = np.array([1.0, 1e-1] + [1e-4 * 2.0 ** -(j - 3) for j in range(3, 11)])
    for space, sigma2 in ((asub.Uniform.reference(10), 1 / 3), (asub.Gaussian.reference(10), 1.0)):
        for W in (None, np.linalg.qr(np.random.default_rng(3).standard_normal((10, 10)))[0]):
            model = tf.quadratic(lam, directions=W, space=space)
            H = model.grad(np.eye(10))  # reference space: physical == normalized; rows are H e_j
            assert np.array_equal(H, H.T)
            Wd = tf.dct_basis(10) if W is None else W
            C = sigma2 * H @ H
            target = Wd @ np.diag(lam) @ Wd.T
            assert np.max(np.abs(C - target)) <= 1e-14 * np.max(np.abs(target))
            rr = model.reference_results
            assert rr.mean == pytest.approx(0.5 * sigma2 * np.trace(H), rel=1e-14)


def test_dct_basis():
    for m in range(1, 51):
        W = tf.dct_basis(m)
        assert W.shape == (m, m)
        assert np.max(np.abs(W.T @ W - np.eye(m))) <= 1e-14
    W4 = tf.dct_basis(4)
    assert np.allclose(W4[:, 0], 0.5, rtol=0, atol=1e-15)
    assert W4[0, 1] == pytest.approx(np.sqrt(0.5) * np.cos(np.pi / 8), rel=1e-15)
