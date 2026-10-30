"""Tests for spec 03: input spaces and coordinate maps."""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle

import numpy as np
import pytest
import scipy.linalg

import active_subspaces as asub
from active_subspaces import _validation as v
from active_subspaces.spaces import _orient_columns as orient_columns
from active_subspaces.spaces import _space_from_dict as space_from_dict


def _proj_dist(A, B):
    """Spectral-norm distance between the orthogonal projectors onto span(A) and span(B)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.linalg.norm(Qa @ Qa.T - Qb @ Qb.T, 2)


def _wide_uniform(m, seed):
    gen = np.random.default_rng(seed)
    lower = gen.uniform(-1, 1, m) * 10.0 ** gen.uniform(-6, 6, m)
    widths = 10.0 ** gen.uniform(-6, 6, m)
    return asub.Uniform(lower, lower + widths)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([0.0, 1.0], [1.0, 1.0]),  # lower == upper
        ([0.0, 2.0], [1.0, 1.0]),  # lower > upper
        ([0.0], [1.0, 2.0]),  # length mismatch
        ([0.0, np.nan], [1.0, 2.0]),
        ([0.0, -np.inf], [1.0, 2.0]),
        ([[0.0, 1.0]], [[1.0, 2.0]]),  # 2-D
        ([], []),
        (["a"], ["b"]),
    ],
)
def test_uniform_construction_errors(lower, upper):
    with pytest.raises(asub.InputError) as e:
        asub.Uniform(lower, upper)
    assert e.value.code == "input.parameter"


@pytest.mark.parametrize(
    "cov",
    [
        [[1.0, 0.5], [0.4, 1.0]],  # not symmetric
        [[1.0, 2.0], [2.0, 1.0]],  # indefinite
        [[1.0, 0.0], [0.0, np.nan]],
        [1.0, 0.0],  # nonpositive variance
        [1.0, 2.0, 3.0],
        np.ones((2, 3)),
    ],
)
def test_gaussian_construction_errors(cov):
    with pytest.raises(asub.InputError) as e:
        asub.Gaussian([0.0, 0.0], cov)
    assert e.value.code == "input.parameter"


@pytest.mark.parametrize(
    ("names", "units"),
    [
        (("a",), None),
        (("a", "a"), None),
        (("a", ""), None),
        ("ab", None),
        (None, ("m",)),
        (None, ("m", 3)),
    ],
)
def test_label_errors(names, units):
    with pytest.raises(asub.InputError) as e:
        asub.Uniform([0, 0], [1, 1], names=names, units=units)
    assert e.value.code == "input.parameter"


def test_defaults_and_attributes():
    s = asub.Uniform([0, 1], [2, 5])
    assert s.dim == 2
    assert s.kind == "uniform"
    assert s.names == ("x1", "x2")
    assert s.units == (None, None)
    np.testing.assert_array_equal(s.scale, np.diag([1.0, 2.0]))
    np.testing.assert_array_equal(s.shift, [1.0, 3.0])
    np.testing.assert_array_equal(s.reference_covariance, np.eye(2) / 3)
    g = asub.Gaussian([1.0, 2.0], [[4.0, 2.0], [2.0, 5.0]], names=("p", "q"), units=("Pa", None))
    assert g.kind == "gaussian"
    assert g.names == ("p", "q")
    assert g.units == ("Pa", None)
    np.testing.assert_allclose(g.scale @ g.scale.T, g.cov)
    assert np.allclose(np.triu(g.scale, 1), 0)  # Cholesky, lower triangular [D-62]
    np.testing.assert_array_equal(g.shift, [1.0, 2.0])
    np.testing.assert_array_equal(g.reference_covariance, np.eye(2))
    for arr in (
        s.lower,
        s.upper,
        s.scale,
        s.shift,
        s.reference_covariance,
        g.mean,
        g.cov,
        g.scale,
    ):
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            arr.flags.writeable = True


def test_immutable_and_owned():
    lo = np.array([0.0, 0.0])
    s = asub.Uniform(lo, [1.0, 1.0])
    lo[0] = -5.0
    assert s.lower[0] == 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.lower = np.zeros(2)
    with pytest.raises(ValueError, match="read-only"):
        s.lower[0] = 3.0


def test_equality_and_hash():
    a = asub.Uniform([0, 0], [1, 1])
    assert a == asub.Uniform([0.0, 0.0], [1.0, 1.0])
    assert a != asub.Uniform([0, 0], [1, 2])
    assert a != asub.Uniform([0, 0], [1, 1], names=("a", "b"))
    assert a != asub.Uniform([0, 0], [1, 1], units=("m", None))
    assert a != asub.Gaussian.reference(2)
    with pytest.raises(TypeError):
        hash(a)
    assert asub.Gaussian([0, 0], [1, 2]) != asub.Gaussian([0, 0], np.diag([1.0, 2.0]))


def test_repr_is_short():
    s = asub.Uniform(np.zeros(1000), np.ones(1000))
    assert len(repr(s)) < 400
    assert "m=1000" in repr(s)


def test_reference_spaces_identity():
    X = np.random.default_rng(0).uniform(-1, 1, (5, 3))
    for s in (asub.Uniform.reference(3), asub.Gaussian.reference(3)):
        for fn in (s.to_normalized, s.from_normalized, s.grads_to_normalized, s.grads_to_physical):
            Y = fn(X)
            np.testing.assert_array_equal(Y, X)
            assert not np.shares_memory(Y, X)
    with pytest.raises(asub.InputError):
        asub.Uniform.reference(0)


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_uniform_round_trip_wide_ranges(seed):
    s = _wide_uniform(6, seed)
    X = s.sample(50, rng=seed)
    back = s.from_normalized(s.to_normalized(X))
    np.testing.assert_allclose(back, X, rtol=1e-14, atol=0)


@pytest.mark.parametrize("seed", range(3))
def test_gaussian_round_trip(gaussian_space, seed):
    s = gaussian_space(5, seed)
    X = s.sample(40, rng=seed)
    np.testing.assert_allclose(s.from_normalized(s.to_normalized(X)), X, rtol=1e-13, atol=1e-13)
    G = np.random.default_rng(seed).standard_normal((7, 5))
    np.testing.assert_allclose(
        s.grads_to_physical(s.grads_to_normalized(G)), G, rtol=1e-12, atol=1e-13
    )


def _grad_check(space, seed):
    m = space.dim
    gen = np.random.default_rng(seed)
    c = gen.standard_normal(m) / np.diag(space.scale)

    def f(xi):
        return np.sin(xi @ c) + 0.5 * (xi @ c) ** 2

    def grad(xi):
        t = xi @ c
        return (np.cos(t) + t)[:, None] * c

    x0 = space.sample(4, rng=seed, normalized=True) * 0.5
    xi0 = space.from_normalized(x0)
    Gn = space.grads_to_normalized(grad(xi0))
    h = 1e-6
    fd = np.empty_like(x0)
    for j in range(m):
        e = np.zeros(m)
        e[j] = h
        fd[:, j] = (f(space.from_normalized(x0 + e)) - f(space.from_normalized(x0 - e))) / (2 * h)
    np.testing.assert_allclose(Gn, fd, rtol=1e-6, atol=1e-7)


def test_gradient_map_matches_finite_differences(uniform_space, gaussian_space):
    _grad_check(uniform_space(4, 1), 1)
    _grad_check(gaussian_space(4, 2), 2)


def test_gaussian_diagonal_shorthand_matches_full():
    var = np.array([0.5, 2.0, 9.0])
    a = asub.Gaussian([1.0, 2.0, 3.0], var)
    b = asub.Gaussian([1.0, 2.0, 3.0], np.diag(var))
    X = a.sample(10, rng=3)
    np.testing.assert_allclose(a.to_normalized(X), b.to_normalized(X), rtol=1e-15)
    np.testing.assert_allclose(a.scale, b.scale, rtol=1e-15)
    W = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 2)))[0]
    np.testing.assert_allclose(
        a.directions_to_physical(W), b.directions_to_physical(W), atol=1e-14
    )


def test_map_input_validation():
    s = asub.Uniform([0, 0], [1, 1])
    with pytest.raises(asub.InputError) as e:
        s.to_normalized(np.ones((3, 3)))
    assert e.value.code == "input.shape"
    with pytest.raises(asub.InputError) as e:
        s.grads_to_physical([[np.nan, 0.0]])
    assert e.value.code == "input.nonfinite"
    with pytest.raises(asub.InputError) as e:
        s.from_normalized(np.ones((2, 2), dtype=complex))
    assert e.value.code == "input.dtype"
    one = asub.Uniform([0.0], [2.0])
    assert one.to_normalized(np.array([0.0, 2.0])).shape == (2, 1)


# ---------------------------------------------------------------------------
# Physical view of active directions [D-11 amended]
# ---------------------------------------------------------------------------


def test_physical_view_recovers_ridge_direction():
    s = asub.Uniform([0.0, -100.0, 5.0, 0.0], [0.01, 100.0, 6.0, 1e4])
    a = np.array([30.0, 0.01, 1.0, 1e-4])
    X = s.sample(400, rng=7)
    G_phys = np.cos(X @ a)[:, None] * a  # f(ξ) = sin(aᵀξ)
    Gn = v.gradients(G_phys, s)
    C = Gn.T @ Gn / len(Gn)
    _, W = np.linalg.eigh(C)
    W1 = W[:, -1:]
    P = s.directions_to_physical(W1)
    assert _proj_dist(P, a[:, None]) < 1e-12
    np.testing.assert_allclose(P.T @ P, np.eye(1), atol=1e-14)
    assert _proj_dist(s.scale @ W1, a[:, None]) > 1e-3  # A W1 is the wrong object


def test_directions_to_physical_orthonormal_and_oriented(gaussian_space):
    """Physical directions keep the signs of W: q_j^T A^{-T} w_j > 0 [D-197]."""
    s = gaussian_space(5, 4)
    W = np.linalg.qr(np.random.default_rng(2).standard_normal((5, 3)))[0]
    P = s.directions_to_physical(W)
    assert P.shape == (5, 3)
    np.testing.assert_allclose(P.T @ P, np.eye(3), atol=1e-13)
    expected = np.linalg.solve(s.scale.T, W)
    assert np.all(np.einsum("ij,ij->j", P, expected) > 0)
    assert _proj_dist(P, expected) < 1e-12
    flipped = W.copy()
    flipped[:, 1] *= -1.0
    P2 = s.directions_to_physical(flipped)
    np.testing.assert_array_equal(P2[:, 1], -P[:, 1])
    np.testing.assert_array_equal(P2[:, [0, 2]], P[:, [0, 2]])
    for bad in (np.ones((4, 2)), np.ones(5), np.ones((5, 6)), np.ones((5, 0))):
        with pytest.raises(asub.InputError):
            s.directions_to_physical(bad)


def test_orient_columns():
    W = np.array([[0.1, -0.2], [-0.9, 0.2], [0.3, 0.0]])
    out = orient_columns(W)
    np.testing.assert_array_equal(out[:, 0], -W[:, 0])
    np.testing.assert_array_equal(out[:, 1], -W[:, 1])  # tie: first largest-magnitude entry
    assert not np.shares_memory(out, W)


def test_orient_columns_near_tie():
    """A tie broken only by rounding uses the first index, on every platform [D-81]."""
    a = 0.566966564981637
    W = np.array([[0.1], [-a], [0.2], [a * (1 + 4e-15)]])
    np.testing.assert_array_equal(orient_columns(W), -W)  # pivot row 1, not the argmax row 3
    np.testing.assert_array_equal(orient_columns(W[::-1])[::-1], W)  # reversed: pivot row 0
    W[3, 0] = a * (1 + 1e-6)  # a real gap is not a tie
    np.testing.assert_array_equal(orient_columns(W), W)


def test_square_root_invariance(gaussian_space):
    s = gaussian_space(4, 9)
    L = s.scale
    Q = np.linalg.qr(np.random.default_rng(3).standard_normal((4, 4)))[0]
    LQ = L @ Q
    c = np.array([1.0, -2.0, 0.5, 3.0])
    X = s.sample(300, rng=5)
    G = np.column_stack([np.cos(X @ c)] * 4) * c + 0.1 * X  # physical gradients
    G1, G2 = G @ L, G @ LQ
    e1, W1 = np.linalg.eigh(G1.T @ G1)
    e2, W2 = np.linalg.eigh(G2.T @ G2)
    np.testing.assert_allclose(e1, e2, rtol=1e-10)
    for n in (1, 2):
        P1 = np.linalg.solve(L.T, W1[:, -n:])
        P2 = np.linalg.solve(LQ.T, W2[:, -n:])
        assert _proj_dist(P1, P2) < 1e-10
        assert _proj_dist(s.directions_to_physical(W1[:, -n:]), P2) < 1e-10


# ---------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------


def test_uniform_support():
    s = asub.Uniform([0.0, 10.0], [1.0, 30.0])
    at_bounds = np.array([[0.0, 10.0], [1.0, 30.0], [0.5, 20.0]])
    assert s.contains(at_bounds).all()
    v.points(at_bounds, s)
    tiny = np.array([[1.0 + 0.5e-12, 30.0 + 0.5e-12 * 20]])
    assert s.contains(tiny).all()
    outside = np.array([[0.5, 30.0 + 1e-9 * 20.0]])
    assert not s.contains(outside).any()
    with pytest.raises(asub.InputError) as e:
        v.points(outside, s)
    assert e.value.code == "input.out_of_space"
    assert e.value.details["coordinate"] == 1
    below = np.array([[0.0 - 1e-9, 20.0]])
    assert not s.contains(below).any()
    assert s.contains(np.array([[1.0, -1.0]]), normalized=True).all()
    assert not s.contains(np.array([[1.0 + 1e-9, 0.0]]), normalized=True).any()


def test_gaussian_support_everything():
    g = asub.Gaussian.reference(2)
    assert g.contains(np.array([[1e8, -1e8]])).all()
    with pytest.raises(asub.InputError):
        g.contains(np.array([[np.inf, 0.0]]))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_moments_normalized():
    M = 20000
    for s, cov in (
        (asub.Uniform([0, 0, 0], [1, 2, 3]), np.eye(3) / 3),
        (asub.Gaussian([5, 5, 5], [1, 2, 3]), np.eye(3)),
    ):
        Z = s.sample(M, rng=11, normalized=True)
        assert Z.shape == (M, 3)
        se_mean = np.sqrt(np.diag(cov) / M)
        assert np.all(np.abs(Z.mean(axis=0)) < 5 * se_mean)
        np.testing.assert_allclose(np.cov(Z.T), cov, atol=6 * np.sqrt(2.0 / M) * cov.max() * 1.5)


def test_sample_physical(gaussian_space):
    s = asub.Uniform([0, 10], [1, 30])
    X = s.sample(1000, rng=1)
    assert s.contains(X).all()
    g = gaussian_space(3, 1)
    Xg = g.sample(40000, rng=2)
    np.testing.assert_allclose(
        Xg.mean(axis=0), g.mean, atol=5 * np.sqrt(np.diag(g.cov) / 40000).max()
    )
    np.testing.assert_allclose(np.cov(Xg.T), g.cov, rtol=0.05, atol=0.05 * np.abs(g.cov).max())


def test_sample_determinism_and_rng_forms():
    s = asub.Uniform([0, 0], [1, 1])
    a = s.sample(5, rng=42)
    np.testing.assert_array_equal(a, s.sample(5, rng=42))
    np.testing.assert_array_equal(a, s.sample(5, rng=np.random.default_rng(42)))
    np.testing.assert_array_equal(a, s.sample(5, rng=np.random.SeedSequence(42)))
    expected = np.random.default_rng(42).uniform(-1, 1, (5, 2)) * 0.5 + 0.5
    np.testing.assert_array_equal(a, expected)
    for bad in (-1, True, 1.5, np.random.RandomState(0), "seed"):
        with pytest.raises(asub.InputError) as e:
            s.sample(5, rng=bad)
        assert e.value.code == "input.parameter"
    with pytest.raises(asub.InputError):
        s.sample(0, rng=1)
    assert s.sample(np.int64(3), rng=1).shape == (3, 2)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "space",
    [
        asub.Uniform([0.1, -3.0], [0.2, 7.0], names=("a", "b"), units=("m", None)),
        asub.Gaussian([1.0, 2.0], [[2.0, 0.3], [0.3, 1.0]]),
        asub.Gaussian([1.0, 2.0, 3.0], [1.0, 1e-8, 1e8]),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_space_round_trip(space, binary):
    d = space.to_dict(binary=binary)
    text = json.dumps(d, allow_nan=False)
    back = type(space).from_dict(json.loads(text))
    assert back == space
    assert space_from_dict(json.loads(text)) == space
    assert pickle.loads(pickle.dumps(space)) == space
    assert (
        not pickle.loads(pickle.dumps(space)).lower.flags.writeable
        if hasattr(space, "lower")
        else True
    )


def test_space_from_dict_errors():
    d = asub.Uniform([0, 0], [1, 1]).to_dict()
    with pytest.raises(asub.SchemaError) as e:
        asub.Uniform.from_dict({**d, "extra": 1})
    assert e.value.code == "schema.invalid"
    bad = dict(d)
    bad["lower"] = asub.Uniform([2, 2], [3, 3]).to_dict()["lower"]
    with pytest.raises(asub.SchemaError):
        asub.Uniform.from_dict(bad)
    with pytest.raises(asub.SchemaError):
        asub.Gaussian.from_dict(d)
    with pytest.raises(asub.SchemaError):
        space_from_dict({"type": "Mixed"})


# ---------------------------------------------------------------------------
# Product space [D-198] (03 §8, V03.8)
# ---------------------------------------------------------------------------


def _hydrology_like():
    lo = np.arange(1.0, 19.0)
    hi = lo + np.linspace(0.5, 50.0, 18)
    return asub.Product(
        [
            asub.Uniform(lo[:16], hi[:16]),
            asub.Gaussian([-5.11], [0.52**2]),
            asub.Uniform(lo[16:], hi[16:]),
        ]
    )


def test_product_maps_support_and_sampling():
    s = _hydrology_like()
    assert (s.dim, s.kind) == (19, "product")
    assert s.names == tuple(f"x{j}" for j in range(1, 20))
    d = s.reference_covariance.diagonal()
    np.testing.assert_allclose(d[16], 1.0)
    np.testing.assert_allclose(np.delete(d, 16), 1.0 / 3.0)
    assert np.count_nonzero(s.reference_covariance - np.diag(d)) == 0
    A = scipy.linalg.block_diag(*(b.scale for b in s.blocks))
    np.testing.assert_array_equal(s.scale, A)
    X = s.sample(50, rng=4)
    x = s.to_normalized(X)
    np.testing.assert_allclose(x, np.linalg.solve(A, (X - s.shift).T).T, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(s.from_normalized(x), X, rtol=1e-14)
    G = np.random.default_rng(1).standard_normal((50, 19))
    np.testing.assert_allclose(s.grads_to_normalized(G), G @ A, rtol=1e-14)
    np.testing.assert_allclose(s.grads_to_physical(G @ A), G, rtol=1e-13)
    # contains: any value in the Gaussian column; bounds on the others
    Y = X.copy()
    Y[:, 16] = 1e6
    assert s.contains(Y).all()
    Y[0, 3] = s.blocks[0].upper[3] + 1e-9 * (s.blocks[0].upper[3] - s.blocks[0].lower[3])
    assert not s.contains(Y)[0]
    # sampling: block b from child stream b, in block order
    ch = np.random.default_rng(9).spawn(3)
    expected = np.hstack(
        [
            ch[0].uniform(-1, 1, (7, 16)),
            ch[1].standard_normal((7, 1)),
            ch[2].uniform(-1, 1, (7, 2)),
        ]
    )
    np.testing.assert_array_equal(s.sample(7, rng=9, normalized=True), expected)


def test_uniform_mask_read_only():
    """03 §4: the boundary mask is a read-only array, on every kind."""
    u, g = asub.Uniform([0.0, 0.0], [1.0, 2.0]), asub.Gaussian([0.0], [1.0])
    for s in (u, g, asub.Product([u, g])):
        mask = s._uniform_mask
        assert not mask.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            mask[0] = not mask[0]
    np.testing.assert_array_equal(asub.Product([u, g, u])._uniform_mask, [1, 1, 0, 1, 1])


def _full_cov_product():
    cov = np.array([[2.0, 0.6], [0.6, 1.0]])
    u = asub.Uniform([1.0, -2.0], [3.0, 6.0])
    g = asub.Gaussian([0.5, -1.0], cov)
    return asub.Product([u, g]), cov


def test_product_full_covariance_block_maps():
    """03 §3.1: A is block-diagonal, diag((u - l) / 2) on a uniform block and the Cholesky
    factor on a Gaussian block; the maps and gradient maps follow, written out by hand.
    """
    s, cov = _full_cov_product()
    L = np.linalg.cholesky(cov)
    A = scipy.linalg.block_diag(np.diag([1.0, 4.0]), L)
    b = np.array([2.0, 2.0, 0.5, -1.0])
    np.testing.assert_allclose(s.scale, A, rtol=1e-14)
    np.testing.assert_allclose(s.shift, b, rtol=1e-14)
    X = s.sample(30, rng=3)
    x = np.hstack([(X[:, :2] - b[:2]) / [1.0, 4.0], np.linalg.solve(L, (X[:, 2:] - b[2:]).T).T])
    np.testing.assert_allclose(s.to_normalized(X), x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(s.from_normalized(x), X, rtol=1e-13, atol=1e-13)
    G = np.random.default_rng(2).standard_normal((30, 4))
    Gn = np.hstack([G[:, :2] * [1.0, 4.0], G[:, 2:] @ L])
    np.testing.assert_allclose(s.grads_to_normalized(G), Gn, rtol=1e-13)
    Gp = np.hstack([G[:, :2] / [1.0, 4.0], np.linalg.solve(L.T, G[:, 2:].T).T])
    np.testing.assert_allclose(s.grads_to_physical(G), Gp, rtol=1e-13)
    np.testing.assert_allclose(s.grads_to_physical(s.grads_to_normalized(G)), G, rtol=1e-13)


def test_product_full_covariance_directions_to_physical():
    """03 §3.1, D-197: an orthonormal basis of span(A^{-T} W) whose columns keep the signs of
    A^{-T} W, with A block-diagonal and the Gaussian block's factor a Cholesky factor.
    """
    s, cov = _full_cov_product()
    L = np.linalg.cholesky(cov)
    A = scipy.linalg.block_diag(np.diag([1.0, 4.0]), L)
    W = np.linalg.qr(np.random.default_rng(5).standard_normal((4, 2)))[0]
    T = np.linalg.solve(A.T, W)
    Q, R = np.linalg.qr(T)
    Q = Q * np.sign(np.diag(R))
    P = s.directions_to_physical(W)
    np.testing.assert_allclose(P, Q, atol=1e-13)
    np.testing.assert_allclose(P.T @ P, np.eye(2), atol=1e-13)
    assert np.all(np.einsum("ij,ij->j", P, T) > 0)


def test_product_guards():
    """03 §5.3, V03.8: geometry, inverse map, integration, optimization, designs and the
    reduced model raise ``unsupported.space`` for a mixed product [D-13].
    """
    s = asub.Product([asub.Uniform([0.0, 1.0], [2.0, 5.0]), asub.Gaussian([0.5], [0.25])])
    a = np.array([1.0, 0.1, 1.0])
    X = s.sample(40, rng=0)
    f = np.sin(X @ a)
    G = np.cos(X @ a)[:, None] * a
    sub = asub.active_subspace(space=s, X=X, f=f, grads=G, n=1)

    def code(call):
        with pytest.raises(asub.UnsupportedError) as e:
            call()
        return e.value.code

    assert code(lambda: asub.active_domain(sub)) == "unsupported.space"
    assert code(lambda: asub.inverse_map(sub, Y=np.zeros((2, 1)), rng=0)) == "unsupported.space"
    assert (
        code(lambda: asub.build_reduced_model(space=s, X=X, f=f, grads=G, rng=0))
        == "unsupported.space"
    )
    # an ActiveDomain cannot have kind "product", so the domain guards of integration,
    # optimization and designs are reached with a domain whose kind is forced
    uni = asub.Uniform([0.0, 1.0, 0.0], [2.0, 5.0, 1.0])
    Xu = uni.sample(40, rng=0)
    dom = asub.active_domain(
        asub.active_subspace(
            space=uni, X=Xu, f=np.sin(Xu @ a), grads=np.cos(Xu @ a)[:, None] * a, n=1
        )
    )
    bad = copy.copy(dom)
    object.__setattr__(bad, "kind", "product")
    assert code(lambda: asub.integrate(lambda Z: Z[:, 0], domain=bad, n_nodes=3)) == (
        "unsupported.space"
    )
    assert code(lambda: asub.active_design(domain=bad, n_points=3)) == "unsupported.space"
    assert (
        code(lambda: asub.minimize(lambda Y: Y[:, 0], domain=bad, X=Xu, f=np.sin(Xu @ a)))
        == "unsupported.space"
    )


def test_product_construction_and_serialization():
    u = asub.Uniform([0.0, 0.0], [1.0, 2.0], names=["a", "b"])
    g = asub.Gaussian([0.0], [1.0])
    s = asub.Product([u, g])
    assert s.names == ("a", "b", "x3")
    assert asub.spaces._space_from_dict(s.to_dict()) == s
    assert asub.spaces._space_from_dict(json.loads(json.dumps(s.to_dict(binary=True)))) == s
    assert pickle.loads(pickle.dumps(s)) == s
    assert s != asub.Product([g, u])
    assert asub.Product([u]) != u
    for bad in ([], [s], ["x"], [u, asub.Uniform([0.0], [1.0], names=["a"])]):
        with pytest.raises(asub.InputError):
            asub.Product(bad)
    with pytest.raises(asub.SchemaError):
        asub.Product.from_dict({"type": "Product", "blocks": [s.to_dict()]})
    assert asub.spaces._poincare_constant(s) == 1.0
    assert asub.spaces._poincare_constant(asub.Product([u])) == 4.0 / np.pi**2


def test_reference_names():
    assert asub.Uniform.reference(2, names=["a", "b"]) == asub.Uniform(
        [-1, -1], [1, 1], names=("a", "b")
    )
    assert asub.Gaussian.reference(2, names=["a", "b"], units=["m", None]) == asub.Gaussian(
        [0, 0], [1, 1], names=("a", "b"), units=("m", None)
    )
    assert asub.Uniform.reference(2, units=["m", "s"]).units == ("m", "s")
    assert asub.Uniform.reference(2) == asub.Uniform([-1, -1], [1, 1])
    for cls in (asub.Uniform, asub.Gaussian):
        for bad in (["a"], ["a", "a"], "ab"):
            with pytest.raises(asub.InputError) as e:
                cls.reference(2, names=bad)
            assert e.value.code == "input.parameter"
