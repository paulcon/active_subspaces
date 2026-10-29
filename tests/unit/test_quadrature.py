"""Contracts of the private Gauss rules (spec 10 §3.4, §4.5)."""

from __future__ import annotations

import numpy as np
import pytest

from active_subspaces import _quadrature as q
from active_subspaces.errors import InputError, NumericalError


def test_private_module():
    assert q.__all__ == ()


def test_return_types():
    for t, v in (
        q.gauss_hermite(4),
        q.gauss_legendre(3),
        q.box_spline_gauss([0.6, 0.8], 5),
        q.lanczos_gauss([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], 2),
    ):
        assert t.dtype == np.float64
        assert v.dtype == np.float64
        assert t.ndim == 1
        assert t.shape == v.shape
        assert t.flags.c_contiguous
        assert t.flags.writeable
    nodes, weights = q.tensor_rule([q.gauss_hermite(2), q.gauss_hermite(3)])
    assert nodes.shape == (6, 2)
    assert weights.shape == (6,)
    assert nodes.dtype == np.float64


def test_single_point_rules():
    t, v = q.gauss_hermite(1)
    assert t.tolist() == [0.0]
    assert v.tolist() == [1.0]
    t, v = q.gauss_from_recurrence([0.25], [3.0])
    assert t.tolist() == [0.25]
    assert v.tolist() == [3.0]
    t, v = q.box_spline_gauss([0.3, -0.4], 1)
    assert t.tolist() == [0.0]
    assert v.tolist() == [1.0]


def test_symmetrization():
    # Without symmetrize the middle node of an odd rule is only near 0; with it, exactly 0.
    k = 7
    j = np.arange(k, dtype=float)
    beta = j**2 / (4 * j**2 - 1)
    beta[0] = 1.0
    t, v = q.gauss_from_recurrence(np.zeros(k), beta, symmetric=True)
    assert t[3] == 0.0
    np.testing.assert_array_equal(t, -t[::-1])
    np.testing.assert_array_equal(v, v[::-1])
    t0, v0 = q.gauss_from_recurrence(np.zeros(k), beta)
    np.testing.assert_allclose(t0, t, atol=1e-15)
    np.testing.assert_allclose(v0, v, atol=1e-15)


def test_validation():
    for bad in (0, -1, 2.5, True):
        with pytest.raises(InputError):
            q.gauss_hermite(bad)
        with pytest.raises(InputError):
            q.gauss_legendre(bad)
    with pytest.raises(InputError):
        q.gauss_from_recurrence([0.0, 0.0], [1.0, -0.5])
    with pytest.raises(InputError):
        q.gauss_from_recurrence([0.0, 0.0], [1.0])
    with pytest.raises(InputError):
        q.tensor_rule([])
    with pytest.raises(InputError):
        q.tensor_rule([(np.zeros(2), np.ones(3))])
    with pytest.raises(InputError):
        q.box_spline_gauss([0.0, 0.0], 3)
    with pytest.raises(InputError):
        q.box_spline_gauss([np.nan, 1.0], 3)
    with pytest.raises(InputError):
        q.lanczos_gauss([0.0, 1.0], [1.0, 1.0], 3)
    with pytest.raises(InputError):
        q.lanczos_gauss([0.0, 1.0], [1.0, -1.0], 1)


def test_lanczos_breakdown():
    # Three atoms but only two distinct points: no 3-point Gauss rule exists.
    with pytest.raises(NumericalError) as exc:
        q.lanczos_gauss([0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 3)
    assert exc.value.code == "numerical.singular"


def test_lanczos_ignores_atom_order():
    gen = np.random.default_rng(7)
    t = gen.uniform(-1, 1, 30)
    p = gen.uniform(0, 1, 30)
    perm = gen.permutation(30)
    a = q.lanczos_gauss(t, p, 6)
    b = q.lanczos_gauss(t[perm], p[perm], 6)
    np.testing.assert_allclose(a[0], b[0], atol=1e-14)
    np.testing.assert_allclose(a[1], b[1], atol=1e-14)
    assert abs(a[1].sum() - p.sum()) <= 1e-13


def test_inputs_not_modified():
    w = np.array([0.5, -0.25, 0.0, 1.0])
    w0 = w.copy()
    q.box_spline_gauss(w, 4)
    np.testing.assert_array_equal(w, w0)
    t, p = np.linspace(-1, 1, 9), np.ones(9)
    t0, p0 = t.copy(), p.copy()
    q.lanczos_gauss(t, p, 4)
    np.testing.assert_array_equal(t, t0)
    np.testing.assert_array_equal(p, p0)


def test_deterministic_bitwise():
    gen = np.random.default_rng(11)
    w = gen.standard_normal(9)
    a = q.box_spline_gauss(w, 10)
    _ = [np.empty(int(n)) for n in gen.integers(1, 50, 5)]  # perturb allocation alignment
    b = q.box_spline_gauss(w.copy(), 10)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_box_spline_scale_and_sign_symmetry():
    w = np.array([0.2, -0.7, 0.4])
    t, p = q.box_spline_gauss(w, 6)
    t2, p2 = q.box_spline_gauss(-w, 6)
    np.testing.assert_allclose(t2, t, atol=1e-15)
    np.testing.assert_allclose(p2, p, atol=1e-15)
    t3, p3 = q.box_spline_gauss(4.0 * w, 6)
    np.testing.assert_allclose(t3, 4.0 * t, atol=1e-14)
    np.testing.assert_allclose(p3, p, atol=1e-14)
