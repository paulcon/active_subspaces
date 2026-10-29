"""Known answers for the private Gauss rules (spec 10 §8, spec 17 V10.1-V10.3)."""

from __future__ import annotations

from fractions import Fraction
from math import comb

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss

from active_subspaces import _quadrature as q


def _double_factorial(n: int) -> int:
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


@pytest.mark.parametrize("k", range(1, 41))
@pytest.mark.parametrize("family", ["hermite", "legendre"])
def test_gauss_1d(family, k):
    t, v = q.gauss_hermite(k) if family == "hermite" else q.gauss_legendre(k)
    assert t.shape == (k,)
    assert v.shape == (k,)
    assert np.all(np.diff(t) > 0)
    assert np.all(v > 0)
    assert abs(v.sum() - 1.0) <= 1e-14
    for d in range(2 * k):
        moment = float(np.sum(v * t**d))
        if d % 2 == 0:
            exact = _double_factorial(d - 1) if family == "hermite" else 1.0 / (d + 1)
            assert abs(moment - exact) <= 1e-12 * exact, d
        else:
            assert abs(moment) <= 1e-12 * float(np.sum(v * np.abs(t) ** d)), d
    if k % 2 == 1:
        assert t[k // 2] == 0.0
    np.testing.assert_array_equal(t, -t[::-1])
    np.testing.assert_array_equal(v, v[::-1])
    if family == "hermite":
        x, w = hermegauss(k)
        w = w / w.sum()
    else:
        x, w = leggauss(k)
        w = w / 2.0
    np.testing.assert_allclose(t, x, rtol=0, atol=1e-13)
    np.testing.assert_allclose(v, w, rtol=0, atol=1e-13)


def test_gauss_from_recurrence_mass():
    # Legendre on [-1, 1] with mass 2 (Lebesgue measure).
    k = 6
    j = np.arange(k, dtype=float)
    beta = j**2 / (4 * j**2 - 1)
    beta[0] = 2.0
    t, v = q.gauss_from_recurrence(np.zeros(k), beta)
    x, w = leggauss(k)
    np.testing.assert_allclose(t, x, atol=1e-14)
    np.testing.assert_allclose(v, w, atol=1e-14)
    assert abs(v.sum() - 2.0) <= 1e-14


def test_gauss_from_recurrence_shifted():
    # Legendre on [0, 1]: alpha_j = 1/2, beta_j = j^2 / (4 (4 j^2 - 1)), mass 1.
    k = 5
    j = np.arange(k, dtype=float)
    beta = j**2 / (4 * (4 * j**2 - 1))
    beta[0] = 1.0
    t, v = q.gauss_from_recurrence(np.full(k, 0.5), beta)
    for d in range(2 * k):
        assert abs(np.sum(v * t**d) - 1.0 / (d + 1)) <= 1e-14


def test_tensor_rule():
    rules = [q.gauss_hermite(3), q.gauss_legendre(2), q.gauss_hermite(4)]
    nodes, weights = q.tensor_rule(rules)
    assert nodes.shape == (24, 3)
    assert weights.shape == (24,)
    # C order: last coordinate fastest.
    k = (3, 2, 4)
    for idx, (a, b, c) in enumerate(np.ndindex(*k)):
        np.testing.assert_array_equal(nodes[idx], [rules[0][0][a], rules[1][0][b], rules[2][0][c]])
        assert weights[idx] == rules[0][1][a] * rules[1][1][b] * rules[2][1][c]

    # Exact for prod y_i^{d_i}, d_i <= 2 k_i - 1.
    def exact_h(d):
        return 0.0 if d % 2 else float(_double_factorial(d - 1))

    def exact_l(d):
        return 0.0 if d % 2 else 1.0 / (d + 1)

    for d0 in range(6):
        for d1 in range(4):
            for d2 in range(8):
                val = np.sum(weights * nodes[:, 0] ** d0 * nodes[:, 1] ** d1 * nodes[:, 2] ** d2)
                exact = exact_h(d0) * exact_l(d1) * exact_h(d2)
                scale = np.sum(
                    weights * np.abs(nodes[:, 0] ** d0 * nodes[:, 1] ** d1 * nodes[:, 2] ** d2)
                )
                assert abs(val - exact) <= 1e-12 * scale


def test_tensor_rule_one_dimension():
    t, v = q.gauss_legendre(4)
    nodes, weights = q.tensor_rule([(t, v)])
    np.testing.assert_array_equal(nodes[:, 0], t)
    np.testing.assert_array_equal(weights, v)


def test_lanczos_gauss_reproduces_known_rule():
    # The 5-point Gauss rule of the 40-point Legendre rule (a discrete measure
    # exact to degree 79) is the 5-point Legendre rule.
    t40, v40 = q.gauss_legendre(40)
    t, v = q.lanczos_gauss(t40, v40, 5)
    t5, v5 = q.gauss_legendre(5)
    np.testing.assert_allclose(t, t5, atol=1e-14)
    np.testing.assert_allclose(v, v5, atol=1e-14)


def test_lanczos_gauss_full_support():
    # k equal to the number of atoms returns the measure itself.
    gen = np.random.default_rng(3)
    t = np.sort(gen.uniform(-2, 3, 7))
    p = gen.uniform(0.1, 1.0, 7)
    nodes, weights = q.lanczos_gauss(t, p, 7)
    np.testing.assert_allclose(nodes, t, atol=1e-13)
    np.testing.assert_allclose(weights, p, rtol=1e-12)


def _box_spline_moments(w: np.ndarray, D: int) -> list[Fraction]:
    """Exact moments of w^T x, x ~ U[-1, 1]^m, from the cumulants kappa_{2j} = 2^{2j} B_{2j} /
    (2j) sum w_i^{2j}.
    """
    wf = [Fraction(float(x)) for x in w]
    # Bernoulli numbers B_0..B_D (B_1 = -1/2 convention; only even ones are used).
    B = [Fraction(0)] * (D + 1)
    B[0] = Fraction(1)
    for n in range(1, D + 1):
        B[n] = -sum(comb(n + 1, j) * B[j] for j in range(n)) / (n + 1)
    kappa = [Fraction(0)] * (D + 1)
    for n in range(2, D + 1, 2):
        kappa[n] = Fraction(2**n) * B[n] / n * sum(x**n for x in wf)
    mu = [Fraction(1)] + [Fraction(0)] * D
    for n in range(1, D + 1):
        mu[n] = sum(comb(n - 1, j - 1) * kappa[j] * mu[n - j] for j in range(1, n + 1))
    return mu


def _direct_moments(w: np.ndarray, D: int) -> list[Fraction]:
    mu = [Fraction(1)] + [Fraction(0)] * D
    for x in w:
        x = Fraction(float(x))
        mi = [Fraction(0) if d % 2 else x**d / (d + 1) for d in range(D + 1)]
        mu = [sum(comb(d, e) * mu[e] * mi[d - e] for e in range(d + 1)) for d in range(D + 1)]
    return mu


def test_box_spline_oracle_agrees_with_direct_convolution():
    w = np.random.default_rng(0).standard_normal(4)
    assert _box_spline_moments(w, 12) == _direct_moments(w, 12)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8, 13, 20])
@pytest.mark.parametrize("m", [1, 2, 5, 50])
def test_box_spline(m, k):
    gen = np.random.default_rng(1000 * m + k)
    w = gen.standard_normal(m)
    w /= np.linalg.norm(w)
    t, p = q.box_spline_gauss(w, k)
    assert t.shape == (k,)
    assert p.shape == (k,)
    assert np.all(np.diff(t) > 0)
    assert np.all(p > 0)
    assert np.all(np.abs(t) < np.abs(w).sum())
    assert abs(p.sum() - 1.0) <= 1e-14
    np.testing.assert_array_equal(t, -t[::-1])
    np.testing.assert_array_equal(p, p[::-1])
    exact = _box_spline_moments(w, 2 * k - 1)
    for d in range(2 * k):
        moment = float(np.sum(p * t**d))
        if d % 2 == 0:
            if d <= 2 * k - 2:
                assert abs(moment - float(exact[d])) <= 1e-12 * float(exact[d]), d
        else:
            assert abs(moment) <= 1e-12 * float(np.sum(p * np.abs(t) ** d)), d


@pytest.mark.parametrize("k", [1, 4, 9, 20])
def test_box_spline_one_input_is_scaled_legendre(k):
    for w0 in (0.7, -1.3):
        t, p = q.box_spline_gauss([w0], k)
        s, v = q.gauss_legendre(k)
        np.testing.assert_allclose(t, abs(w0) * s, rtol=0, atol=1e-14)
        np.testing.assert_allclose(p, v, rtol=0, atol=1e-14)


def test_box_spline_zero_entries_change_nothing():
    w = np.array([0.3, -0.5, 0.2, 0.8])
    w_padded = np.array([0.0, 0.3, 0.0, -0.5, 0.2, 0.0, 0.8, 0.0])
    for k in (1, 3, 7):
        t, p = q.box_spline_gauss(w, k)
        tz, pz = q.box_spline_gauss(w_padded, k)
        np.testing.assert_array_equal(t, tz)
        np.testing.assert_array_equal(p, pz)


def test_box_spline_two_equal_inputs():
    # y = (x_1 + x_2) / 2 has the triangular density on [-1, 1]; E[y^2] = 1/6, E[y^4] = 1/15.
    t, p = q.box_spline_gauss([0.5, 0.5], 3)
    assert abs(np.sum(p * t**2) - 1 / 6) <= 1e-15
    assert abs(np.sum(p * t**4) - 1 / 15) <= 1e-15
    # The two-point Gauss rule of the triangular density: nodes +-1/sqrt(6).
    t2, p2 = q.box_spline_gauss([0.5, 0.5], 2)
    np.testing.assert_allclose(t2, [-1 / np.sqrt(6), 1 / np.sqrt(6)], atol=1e-15)
    np.testing.assert_allclose(p2, [0.5, 0.5], atol=1e-15)
