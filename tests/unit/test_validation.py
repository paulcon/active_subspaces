"""Tests for spec 02: the array contract and the private validation helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import _validation as v


@pytest.fixture
def space():
    return asub.Uniform([0.0, -1.0, 10.0], [2.0, 3.0, 20.0])


@pytest.fixture
def X(space):
    return space.sample(6, rng=0)


def _code(excinfo):
    return excinfo.value.code


# ---------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((4, 3), dtype=bool),
        np.ones((4, 3), dtype=complex),
        np.array([["a"] * 3] * 4),
        np.array([[object()] * 3] * 4, dtype=object),
        np.array(["2020-01-01"] * 3, dtype="datetime64[D]").reshape(1, 3),
    ],
)
def test_points_rejects_dtypes(space, bad):
    with pytest.raises(asub.InputError) as e:
        v.points(bad, space)
    assert _code(e) == "input.dtype"
    assert isinstance(e.value, ValueError)


@pytest.mark.parametrize("shape", [(5,), (5, 2), (5, 4), (2, 5, 3), (0, 3)])
def test_points_rejects_shapes(space, shape):
    with pytest.raises(asub.InputError) as e:
        v.points(np.ones(shape), space)
    assert _code(e) == "input.shape"


def test_points_shape_message(space):
    with pytest.raises(asub.InputError, match=r"X: expected shape \(M, 3\), got \(5, 2\)") as e:
        v.points(np.ones((5, 2)), space)
    assert e.value.details == {"param": "X", "expected": ["M", 3], "got": [5, 2]}


def test_points_nonfinite_reports_row(space, X):
    X = X.copy()
    X[3, 1] = np.nan
    with pytest.raises(asub.InputError) as e:
        v.points(X, space)
    assert _code(e) == "input.nonfinite"
    assert e.value.details["row"] == 3
    X[3, 1] = np.inf
    with pytest.raises(asub.InputError):
        v.points(X, space)


def test_points_out_of_space_reports_row_and_coordinate(space, X):
    X = X.copy()
    X[4, 2] = 25.0
    with pytest.raises(asub.InputError) as e:
        v.points(X, space)
    assert _code(e) == "input.out_of_space"
    assert e.value.details["row"] == 4
    assert e.value.details["coordinate"] == 2


def test_out_of_space_details(space):
    names = ("a", "b", "c")
    sp = asub.Uniform(space.lower, space.upper, names=names)
    X = sp.sample(10, rng=1)
    X[2, 1] = 3.5
    X[5, 1] = -2.0
    X[7, 2] = 21.0
    with pytest.raises(asub.InputError) as e:
        v.points(X, sp, "X")
    d = e.value.details
    assert d["n_outside"] == 3
    assert (d["row"], d["column"], d["name"]) == (2, 1, "b")
    assert (d["side"], d["bound"], d["value"]) == ("upper", 3.0, 3.5)
    assert d["coordinate"] == 1
    assert all(type(d[k]) is t for k, t in [("row", int), ("value", float), ("bound", float)])
    assert "3 of 10 rows" in str(e.value)
    assert "np.float64" not in str(e.value)
    X = sp.sample(4, rng=1)
    X[1, 0] = -0.5
    with pytest.raises(asub.InputError) as e:
        v.points(X, sp, "X")
    assert (e.value.details["side"], e.value.details["bound"]) == ("lower", 0.0)
    assert "lower 0" in str(e.value)


def test_out_of_space_message_near_bound():
    """03 §5.2, §8: a point u + 1e-9 (u - l) is reported with repr digits, not as the bound."""
    sp = asub.Uniform([0.0, 0.0], [1.0, 2e6])
    X = np.array([[1.0 + 1e-9, 5.0]])
    with pytest.raises(asub.InputError) as e:
        v.points(X, sp, "X")
    assert e.value.code == "input.out_of_space"
    assert "= 1.000000001 > upper 1.0" in str(e.value)
    assert e.value.details["value"] == X[0, 0]


def test_out_of_space_details_product():
    sp = asub.Product(
        [asub.Gaussian([0.0], [1.0], names=["g"]), asub.Uniform([0.0], [1.0], names=["u"])]
    )
    X = np.array([[50.0, 0.5], [0.0, 1.5], [0.0, -0.5]])
    with pytest.raises(asub.InputError) as e:
        v.points(X, sp, "X")
    d = e.value.details
    assert d["n_outside"] == 2
    assert (d["row"], d["column"], d["name"], d["side"], d["bound"]) == (1, 1, "u", "upper", 1.0)
    assert "np.float64" not in str(e.value)


def test_points_normalizes_without_mutation(space, X):
    before = X.copy()
    Xn = v.points(X, space)
    np.testing.assert_array_equal(X, before)
    assert X.flags.writeable
    assert not np.shares_memory(Xn, X)
    np.testing.assert_allclose(Xn, space.to_normalized(X))
    assert np.all(np.abs(Xn) <= 1)


@pytest.mark.parametrize("make", ["uniform", "gaussian"])
def test_points_physical(make, space, X):
    """Spec 02 §8: normalize=False runs the same checks and returns a new array equal to the
    coerced input.
    """
    if make == "gaussian":
        space = asub.Gaussian(
            [1.0, -2.0, 0.5], [[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]]
        )
        X = space.sample(6, rng=1)
    before = X.copy()
    Xp = v.points(X, space, normalize=False)
    assert Xp.dtype == np.float64
    assert Xp.flags.c_contiguous
    assert Xp.flags.writeable
    np.testing.assert_array_equal(Xp, np.asarray(X, dtype=np.float64))
    assert not np.shares_memory(Xp, X)
    np.testing.assert_array_equal(X, before)
    np.testing.assert_array_equal(space.to_normalized(Xp), v.points(X, space))
    Xi = np.round(X).astype(np.int64)
    if make == "gaussian":
        np.testing.assert_array_equal(v.points(Xi, space, normalize=False), Xi.astype(np.float64))
    Xf = np.asfortranarray(X)
    assert v.points(Xf, space, normalize=False).flags.c_contiguous


def test_points_physical_same_checks(space, X):
    for bad, code in (
        (np.ones((4, 2)), "input.shape"),
        (np.ones((4, 3), dtype=bool), "input.dtype"),
        (np.where(np.arange(18).reshape(6, 3) == 4, np.nan, X), "input.nonfinite"),
        (X + np.array([0.0, 100.0, 0.0]), "input.out_of_space"),
    ):
        with pytest.raises(asub.InputError) as e:
            v.points(bad, space, normalize=False)
        assert _code(e) == code


def test_points_physical_one_dimensional():
    s = asub.Uniform([0.0], [4.0])
    Xp = v.points(np.array([0.0, 2.0, 4.0]), s, normalize=False)
    assert Xp.shape == (3, 1)
    np.testing.assert_array_equal(Xp[:, 0], [0.0, 2.0, 4.0])


def test_check_space_is_duck_typed():
    class FakeSpace:
        kind = "uniform"
        dim = 2

    fake = FakeSpace()
    assert v.check_space(fake) is fake
    for bad in (
        "not a space",
        None,
        3,
        type("K", (), {"kind": "uniform", "dim": True})(),
        type("D", (), {"kind": 1, "dim": 2})(),
    ):
        with pytest.raises(asub.InputError) as e:
            v.check_space(bad, "space")
        assert _code(e) == "input.parameter"


def test_points_reference_space_returns_new_array():
    s = asub.Uniform.reference(2)
    X = np.array([[0.5, -0.5]])
    Xn = v.points(X, s)
    np.testing.assert_array_equal(Xn, X)
    assert not np.shares_memory(Xn, X)


def test_points_accepts_lists_and_ints(space):
    Xn = v.points([[1, 1, 15]], space)
    assert Xn.dtype == np.float64
    np.testing.assert_allclose(Xn, [[0.0, 0.0, 0.0]])
    Xn32 = v.points(np.array([[1, 1, 15]], dtype=np.float32), space)
    assert Xn32.dtype == np.float64


def test_points_one_dimensional_space():
    s = asub.Uniform([0.0], [4.0])
    Xn = v.points(np.array([0.0, 2.0, 4.0]), s)
    assert Xn.shape == (3, 1)
    np.testing.assert_allclose(Xn[:, 0], [-1, 0, 1])
    G = v.gradients(np.array([1.0, 2.0, 3.0]), s)
    assert G.shape == (3, 1)


def test_points_rejects_non_space():
    with pytest.raises(asub.InputError) as e:
        v.points(np.ones((2, 2)), "not a space")
    assert _code(e) == "input.parameter"


# ---------------------------------------------------------------------------
# values, gradients
# ---------------------------------------------------------------------------


def test_values_shapes():
    f = np.arange(5.0)
    a = v.values(f)
    b = v.values(f.reshape(-1, 1))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (5,)
    assert not np.shares_memory(a, f)
    with pytest.raises(asub.InputError) as e:
        v.values(np.ones((5, 2)))
    assert _code(e) == "input.vector_qoi"
    with pytest.raises(asub.InputError) as e:
        v.values(np.ones((5, 1)), M=4)
    assert _code(e) == "input.size_mismatch"
    with pytest.raises(asub.InputError) as e:
        v.values(np.array([1.0, np.inf]))
    assert _code(e) == "input.nonfinite"
    with pytest.raises(asub.InputError) as e:
        v.values(np.ones((2, 2, 1)))
    assert _code(e) == "input.shape"


def test_gradients_normalized(space):
    G = np.arange(12.0).reshape(4, 3)
    Gn = v.gradients(G, space, M=4)
    np.testing.assert_allclose(Gn, G * np.array([1.0, 2.0, 5.0]))
    with pytest.raises(asub.InputError) as e:
        v.gradients(G, space, M=5)
    assert _code(e) == "input.size_mismatch"
    with pytest.raises(asub.InputError) as e:
        v.gradients(G[:, :2], space)
    assert _code(e) == "input.shape"


def test_gradients_no_support_check(space):
    Gn = v.gradients(np.full((2, 3), 1e6), space)
    assert np.all(np.isfinite(Gn))


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def test_quad_weights_default():
    w = v.quad_weights(None, 4)
    np.testing.assert_array_equal(w, np.full(4, 0.25))


@pytest.mark.parametrize("delta", [1e-11, -1e-11])
def test_quad_weights_tolerance_accepts(delta):
    w = np.full(4, 0.25)
    w[0] += delta
    np.testing.assert_array_equal(v.quad_weights(w, 4), w)


@pytest.mark.parametrize("delta", [1e-9, -1e-9])
def test_quad_weights_tolerance_rejects(delta):
    w = np.full(4, 0.25)
    w[0] += delta
    with pytest.raises(asub.InputError) as e:
        v.quad_weights(w, 4)
    assert _code(e) == "input.weights"


def test_quad_weights_errors():
    with pytest.raises(asub.InputError) as e:
        v.quad_weights([1.5, -0.5], 2)
    assert _code(e) == "input.weights"
    with pytest.raises(asub.InputError) as e:
        v.quad_weights([0.5, 0.5], 3)
    assert _code(e) == "input.size_mismatch"
    with pytest.raises(asub.InputError) as e:
        v.quad_weights([[0.5, 0.5]], 2)
    assert _code(e) == "input.shape"
    with pytest.raises(asub.InputError) as e:
        v.quad_weights([np.nan, 1.0], 2)
    assert _code(e) == "input.nonfinite"


def test_sample_weights():
    np.testing.assert_array_equal(v.sample_weights(None, 3), np.ones(3))
    w = np.array([0.0, 2.0, 7.0])
    out = v.sample_weights(w, 3)
    np.testing.assert_array_equal(out, w)  # not normalized
    assert not np.shares_memory(out, w)
    with pytest.raises(asub.InputError) as e:
        v.sample_weights([1.0, -1.0, 1.0], 3)
    assert _code(e) == "input.weights"
    with pytest.raises(asub.InputError) as e:
        v.sample_weights([0.0, 0.0], 2)
    assert _code(e) == "input.weights"


# ---------------------------------------------------------------------------
# scalars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, np.int64(2), np.intp(2), np.int32(2), np.uint8(2)])
def test_dimension_accepts_integers(n):
    out = v.dimension(n, 5)
    assert out == 2
    assert type(out) is int


@pytest.mark.parametrize("n", [True, 2.0, "2", None, np.float64(2)])
def test_dimension_rejects_non_integers(n):
    with pytest.raises(asub.InputError) as e:
        v.dimension(n, 5)
    assert _code(e) == "input.parameter"


@pytest.mark.parametrize("n", [0, 5, -1])
def test_dimension_range(n):
    with pytest.raises(asub.InputError) as e:
        v.dimension(n, 5)
    assert _code(e) == "input.parameter"
    assert v.dimension(5, 5, hi=5) == 5


def test_real_boolean_choice_min_samples():
    assert v.real(np.float32(0.5), "t", lo=0, hi=1) == 0.5
    assert type(v.real(3, "t")) is float
    for bad in (True, "1", np.nan, np.inf):
        with pytest.raises(asub.InputError):
            v.real(bad, "t")
    with pytest.raises(asub.InputError):
        v.real(0.0, "t", lo=0, lo_open=True)
    assert v.real(np.inf, "t", finite=False) == np.inf
    assert v.boolean(np.bool_(True), "b") is True
    with pytest.raises(asub.InputError):
        v.boolean(1, "b")
    assert v.choice("a", "c", ("a", "b")) == "a"
    with pytest.raises(asub.InputError) as e:
        v.choice("z", "c", ("a", "b"))
    assert e.value.details["expected"] == ["a", "b"]
    v.min_samples(5, 5, "OLS")
    with pytest.raises(asub.InputError) as e:
        v.min_samples(4, 5, "OLS")
    assert _code(e) == "input.too_few_samples"
    assert e.value.details["minimum"] == 5


# ---------------------------------------------------------------------------
# digest
# ---------------------------------------------------------------------------


def test_digest_stable_and_layout_independent():
    a = np.random.default_rng(1).standard_normal((7, 3))
    d1 = v.digest(a)
    assert d1 == v.digest(a.copy())
    assert d1 == v.digest(np.asfortranarray(a))
    assert d1 == v.digest(a.tolist())
    assert d1["shape"] == [7, 3]
    assert d1["dtype"] == "float64"
    assert len(d1["sha256"]) == 64
    b = a.copy()
    b[0, 0] = np.nextafter(b[0, 0], np.inf)
    assert v.digest(b)["sha256"] != d1["sha256"]
    assert v.digest(np.arange(3))["sha256"] == v.digest(np.arange(3.0))["sha256"]


# ---------------------------------------------------------------------------
# data(): whole-call validation and order
# ---------------------------------------------------------------------------


def test_data_happy_path(space, X):
    f = np.arange(6.0).reshape(-1, 1)
    G = np.ones((6, 3))
    d = v.data(space, X=X, f=f, grads=G, use_quad_weights=True)
    assert d.M == 6
    assert d.m == 3
    np.testing.assert_allclose(d.X, space.to_normalized(X))
    np.testing.assert_array_equal(d.f, np.arange(6.0))
    np.testing.assert_allclose(d.grads, space.grads_to_normalized(G))
    np.testing.assert_array_equal(d.quad_weights, np.full(6, 1 / 6))
    assert d.sample_weights is None
    assert set(d.digests) == {"X", "f", "grads"}
    assert d.digests["X"] == v.digest(X)  # physical, not normalized
    assert d.digests["f"] == v.digest(np.arange(6.0))


def test_data_requires(space):
    with pytest.raises(asub.InputError) as e:
        v.data(space, X=np.ones((2, 3)), require=("grads",))
    assert _code(e) == "input.parameter"
    with pytest.raises(asub.InputError):
        v.data(space)


def test_data_check_order(space, X):
    bad_shape_X = np.ones((6, 2))
    # dtype before shape
    with pytest.raises(asub.InputError) as e:
        v.data(space, X=bad_shape_X, f=np.ones(6, dtype=bool))
    assert _code(e) == "input.dtype"
    # shape before size
    with pytest.raises(asub.InputError) as e:
        v.data(space, X=X[:5], grads=np.ones((6, 2)))
    assert _code(e) == "input.shape"
    # size before finiteness
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(asub.InputError) as e:
        v.data(space, X=Xn, f=np.ones(5))
    assert _code(e) == "input.size_mismatch"
    # finiteness before support
    Xo = X.copy()
    Xo[0, 0] = 100.0
    with pytest.raises(asub.InputError) as e:
        v.data(space, X=Xo, f=np.array([np.nan] + [1.0] * 5))
    assert _code(e) == "input.nonfinite"
    # support before weights
    with pytest.raises(asub.InputError) as e:
        v.data(space, X=Xo, quad_weights=np.ones(6))
    assert _code(e) == "input.out_of_space"
    with pytest.raises(asub.InputError) as e:
        v.data(space, X=X, quad_weights=np.ones(6))
    assert _code(e) == "input.weights"


def test_data_does_not_mutate_or_alias(space, X):
    f = np.arange(6.0)
    G = np.ones((6, 3))
    w = np.full(6, 1 / 6)
    copies = [a.copy() for a in (X, f, G, w)]
    d = v.data(space, X=X, f=f, grads=G, quad_weights=w, sample_weights=w)
    for a, c in zip((X, f, G, w), copies, strict=False):
        np.testing.assert_array_equal(a, c)
        assert a.flags.writeable
    for out in (d.X, d.f, d.grads, d.quad_weights, d.sample_weights):
        for a in (X, f, G, w):
            assert not np.shares_memory(out, a)


def test_validation_never_warns(space, X):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v.data(
            space,
            X=X,
            f=np.ones(6),
            grads=np.ones((6, 3)),
            quad_weights=None,
            sample_weights=np.ones(6),
        )
