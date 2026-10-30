"""Unit tests for spec 07 §3.8, §4.6, §5.2: gradients from data."""

from __future__ import annotations

import concurrent.futures

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import gradients as gr
from active_subspaces import models
from active_subspaces.errors import InputError, ModelError
from active_subspaces.results import _InvariantError


def _codes(excinfo):
    return excinfo.value.code


def _linear_data(space, M, seed=0):
    gen = np.random.default_rng(seed)
    X = space.sample(M, rng=gen)
    a = gen.standard_normal(space.dim)
    return X, X @ a + 1.0, a


# ---------------------------------------------------------------------------
# GradientResult
# ---------------------------------------------------------------------------


def _ll_result(make_provenance, **over):
    s = asub.Uniform.reference(2)
    kw = {
        "space": s,
        "method": "local_linear",
        "points": np.zeros((3, 2)),
        "grads": np.ones((3, 2)),
        "values": np.zeros(3),
        "centers": np.arange(3),
        "p": 3,
        "scheme": None,
        "steps": None,
        "central_mask": None,
        "n_evaluations": 0,
        "provenance": make_provenance(),
    }
    kw.update(over)
    return gr.GradientResult(**kw)


def test_gradient_result_invariants(make_provenance):
    r = _ll_result(make_provenance)
    assert r.centers.dtype == np.int64
    assert r.qoi == "f"
    assert r.m == 2
    bad = [
        {"method": "ols"},
        {"grads": np.ones((2, 2))},
        {"values": np.zeros(2)},
        {"centers": None},
        {"p": 2},
        {"scheme": "forward"},
        {"n_evaluations": 4},
        {"points": np.full((3, 2), np.nan)},
    ]
    for over in bad:
        with pytest.raises(_InvariantError):
            _ll_result(make_provenance, **over)
    with pytest.raises(_InvariantError):
        _ll_result(
            make_provenance,
            method="finite_difference",
            centers=None,
            p=None,
            scheme="forward",
            steps=np.ones((3, 2)),
            central_mask=np.ones((3, 2), dtype=bool),
            n_evaluations=9,
        )


def test_gradient_result_physical_views(make_provenance, uniform_space):
    s = uniform_space(3, seed=4)
    G = np.arange(6.0).reshape(2, 3)
    P = np.array([[0.1, -0.2, 0.3], [0.0, 0.5, -1.0]])
    r = _ll_result(
        make_provenance, space=s, points=P, grads=G, values=np.zeros(2), centers=np.arange(2), p=4
    )
    np.testing.assert_array_equal(r.points_physical, s.from_normalized(P))
    np.testing.assert_array_equal(r.grads_physical, s.grads_to_physical(G))
    for arr in (r.points_physical, r.grads_physical):
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            arr[0, 0] = 1.0
        with pytest.raises(ValueError, match="WRITEABLE"):
            arr.flags.writeable = True


def test_exports():
    assert gr.__all__ == (
        "GradientResult",
        "finite_difference_gradients",
        "finite_difference_gradients_multi",
        "local_linear_gradients",
    )
    assert gr._toplevel == gr.__all__


# ---------------------------------------------------------------------------
# Local linear: neighbor sets and validation
# ---------------------------------------------------------------------------


def test_neighbor_sets():
    # Reference normalized points on a line; equidistant neighbors are broken by row index [C-02].
    Xn = np.array([[0.0], [0.25], [-0.25], [0.5], [-0.5]])
    sets = gr._neighbor_sets(Xn, np.arange(5), 2)
    assert [s.tolist() for s in sets] == [[0, 1], [0, 1], [0, 2], [1, 3], [2, 4]]
    sets = gr._neighbor_sets(Xn, np.array([1, 0]), 3)
    assert [s.tolist() for s in sets] == [[0, 1, 3], [0, 1, 2]]
    # The center is always included, even when p = M.
    sets = gr._neighbor_sets(Xn, np.arange(5), 5)
    assert all(s.tolist() == [0, 1, 2, 3, 4] for s in sets)


def test_neighbor_sets_count_distinct_locations():
    # Row 5 duplicates row 2: p counts locations, every row at a chosen location enters [D-82
    # amendment].
    Xn = np.array([[0.0], [0.25], [-0.25], [0.5], [-0.5], [-0.25]])
    sets = gr._neighbor_sets(Xn, np.array([0, 2, 5]), 3)
    assert [s.tolist() for s in sets] == [[0, 1, 2, 5], [0, 2, 4, 5], [0, 2, 4, 5]]
    # Ties against a duplicated location use its first row index; -0.0 equals 0.0.
    Xn = np.array([[0.0], [0.5], [-0.5], [0.5], [-0.0]])
    sets = gr._neighbor_sets(Xn, np.array([0]), 2)
    assert sets[0].tolist() == [0, 1, 3, 4]


def test_duplicates_enter_with_multiplicity():
    s = asub.Uniform.reference(2)
    X = np.array([[0.0, 0.0], [0.3, 0.1], [-0.2, 0.4], [0.1, -0.5], [0.3, 0.1], [0.9, 0.9]])
    f = np.array(
        [1.0, 2.0, -1.0, 0.5, 2.6, 7.0]
    )  # rows 1 and 4 share a location, with different values
    r = gr.local_linear_gradients(X=X, f=f, space=s, p=4, centers=[0])
    rows = [0, 1, 2, 3, 4]
    A = np.column_stack([np.ones(5), X[rows] - X[0]])
    ref = np.linalg.lstsq(A, f[rows], rcond=None)[0][1:]
    np.testing.assert_allclose(r.grads[0], ref, rtol=1e-12, atol=1e-14)


def test_default_p_and_alignment():
    s = asub.Uniform.reference(4)
    X, f, _ = _linear_data(s, 30)
    r = gr.local_linear_gradients(X=X, f=f, space=s)
    assert r.p == 6  # max(m + 1, floor(1.7 m)) = 6
    np.testing.assert_array_equal(r.centers, np.arange(30))  # [C-03]: aligned with X
    np.testing.assert_array_equal(r.points, s.to_normalized(X))
    np.testing.assert_array_equal(r.values, f)
    assert r.method == "local_linear"
    assert r.n_evaluations == 0
    assert r.scheme is None
    assert r.steps is None
    # p defaults to M_d when there are few distinct locations: here 15 rows, 15 locations, m = 10
    # [C-19].
    s10 = asub.Uniform.reference(10)
    X, f, _ = _linear_data(s10, 15)
    assert gr.local_linear_gradients(X=X, f=f, space=s10).p == 15


def test_centers_kept_in_order_with_duplicates():
    s = asub.Uniform.reference(2)
    X, f, _ = _linear_data(s, 12)
    full = gr.local_linear_gradients(X=X, f=f, space=s)
    sub = gr.local_linear_gradients(X=X, f=f, space=s, centers=np.array([5, 1, 5], dtype=np.int32))
    np.testing.assert_array_equal(sub.centers, [5, 1, 5])
    assert sub.centers.dtype == np.int64
    np.testing.assert_array_equal(sub.grads, full.grads[[5, 1, 5]])
    np.testing.assert_array_equal(sub.values, f[[5, 1, 5]])
    np.testing.assert_array_equal(sub.points, full.points[[5, 1, 5]])


@pytest.mark.parametrize(
    "centers", [[0.0, 1.0], [[0, 1]], [], [0, 12], [-1], np.array([True, False]), "0"]
)
def test_bad_centers(centers):
    s = asub.Uniform.reference(2)
    X, f, _ = _linear_data(s, 12)
    with pytest.raises(asub.InputError) as e:
        gr.local_linear_gradients(X=X, f=f, space=s, centers=centers)
    assert _codes(e) == "input.parameter"


def test_p_validation():
    s = asub.Uniform.reference(3)
    X, f, _ = _linear_data(s, 10)
    for p in (3, 11, 2.0, True, "5"):
        with pytest.raises(asub.InputError) as e:
            gr.local_linear_gradients(X=X, f=f, space=s, p=p)
        assert _codes(e) == "input.parameter"
    assert gr.local_linear_gradients(X=X, f=f, space=s, p=np.int64(4)).p == 4  # [P-04]
    assert gr.local_linear_gradients(X=X, f=f, space=s, p=10).p == 10  # p = M allowed [C-19]
    # p is bounded by the number of distinct locations
    X2 = np.vstack([X[:6], X[:4]])
    with pytest.raises(asub.InputError) as e:
        gr.local_linear_gradients(X=X2, f=X2 @ np.ones(3), space=s, p=7)
    assert _codes(e) == "input.parameter"


def test_too_few_samples():
    s = asub.Uniform.reference(10)
    X, f, _ = _linear_data(s, 10)
    with pytest.raises(asub.InputError) as e:
        gr.local_linear_gradients(X=X, f=f, space=s)
    assert _codes(e) == "input.too_few_samples"
    # 20 rows but only 5 distinct locations
    X5 = np.tile(X[:5], (4, 1))
    with pytest.raises(asub.InputError) as e:
        gr.local_linear_gradients(X=X5, f=X5.sum(axis=1), space=s)
    assert _codes(e) == "input.too_few_samples"


def test_input_validation():
    s = asub.Uniform.reference(2)
    X, f, _ = _linear_data(s, 8)
    with pytest.raises(asub.InputError) as e:
        gr.local_linear_gradients(X=X, f=f[:7], space=s)
    assert _codes(e) == "input.size_mismatch"
    Xbad = X.copy()
    Xbad[0, 0] = 2.0
    with pytest.raises(asub.InputError) as e:
        gr.local_linear_gradients(X=Xbad, f=f, space=s)
    assert _codes(e) == "input.out_of_space"
    with pytest.raises(asub.InputError) as e:
        gr.local_linear_gradients(X=X, f=np.column_stack([f, f]), space=s)
    assert _codes(e) == "input.vector_qoi"
    with pytest.raises(asub.InputError) as e:
        gr.local_linear_gradients(X=X, f=f, space="uniform")
    assert _codes(e) == "input.parameter"
    for q in ("", 3):
        with pytest.raises(asub.InputError) as e:
            gr.local_linear_gradients(X=X, f=f, space=s, qoi=q)
        assert _codes(e) == "input.parameter"
    with pytest.raises(TypeError):
        gr.local_linear_gradients(X, f, s)  # keyword-only


def test_rank_deficient_warns_once():
    s = asub.Uniform.reference(2)
    t = np.linspace(-1, 1, 7)
    X = np.column_stack([t, 0.5 * t])  # collinear: every local system has rank 2 < 3
    with pytest.warns(asub.AccuracyWarning) as rec:
        r = gr.local_linear_gradients(X=X, f=t, space=s)
    assert len(rec) == 1
    w = rec[0].message
    assert w.code == "accuracy.rank_deficient"
    assert w.details["centers"] == 7
    # minimum-norm solution: the gradient lies along the line direction
    d = np.array([1.0, 0.5]) / np.linalg.norm([1.0, 0.5])
    np.testing.assert_allclose(r.grads - np.outer(r.grads @ d, d), 0.0, atol=1e-12)


def test_m1_and_1d_inputs():
    s = asub.Uniform([0.0], [4.0])
    X = np.array([0.5, 1.0, 3.0, 2.0])
    r = gr.local_linear_gradients(X=X, f=3.0 * X - 1.0, space=s)
    assert r.p == 2
    np.testing.assert_allclose(r.grads_physical, 3.0, rtol=1e-12)


def test_local_linear_deterministic_and_no_mutation(gaussian_space):
    s = gaussian_space(3, seed=2)
    X, f, _ = _linear_data(s, 25)
    f = f + np.sin(X[:, 0])
    X.flags.writeable = False
    X0, f0 = X.copy(), f.copy()
    a = gr.local_linear_gradients(X=X, f=f, space=s)
    b = gr.local_linear_gradients(X=X, f=f, space=s)
    assert a == b
    np.testing.assert_array_equal(a.grads, b.grads)
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(f, f0)


def test_local_linear_provenance():
    s = asub.Uniform.reference(2)
    X, f, _ = _linear_data(s, 8)
    r = gr.local_linear_gradients(X=X, f=f, space=s, centers=[1, 2], qoi="drag")
    pv = r.provenance
    assert pv.function == "active_subspaces.local_linear_gradients"
    assert pv.rng is None
    assert r.qoi == "drag"
    assert pv.parameters["p_used"] == 3
    assert pv.parameters["qoi"] == "drag"
    assert set(pv.parameters["X"]) == {"shape", "dtype", "sha256"}


def test_round_trip(assert_roundtrip, assert_readonly, uniform_space):
    s = uniform_space(3, seed=1)
    X, f, _ = _linear_data(s, 12)
    r = gr.local_linear_gradients(X=X, f=f, space=s, centers=[0, 3, 3])
    assert_roundtrip(r)
    assert_readonly(r)
    fd = gr.finite_difference_gradients(
        lambda Z: Z.sum(axis=1), X=X, space=s, scheme="central", h=0.3
    )
    assert_roundtrip(fd)
    assert_readonly(fd)


# ---------------------------------------------------------------------------
# Finite differences: validation and bookkeeping
# ---------------------------------------------------------------------------


def _quad_fun(space):
    def fun(Z):
        x = space.to_normalized(Z)
        return 0.5 * (x**2).sum(axis=1) + x[:, 0]

    return fun


def test_fd_space_handling(uniform_space):
    s = uniform_space(2, seed=3)
    X = s.sample(4, rng=0)
    fun = _quad_fun(s)
    with pytest.raises(asub.InputError) as e:
        gr.finite_difference_gradients(fun, X=X)
    assert _codes(e) == "input.parameter"
    mdl = models.Model(space=s, value=fun, name="quad", qoi="q")
    other = asub.Uniform(s.lower, s.upper + 1.0)
    with pytest.raises(asub.InputError) as e:
        gr.finite_difference_gradients(mdl, X=X, space=other)
    assert _codes(e) == "input.parameter"
    assert e.value.details["reason"] == "space_mismatch"
    r = gr.finite_difference_gradients(mdl, X=X, space=asub.Uniform(s.lower, s.upper))
    assert r.space == s
    assert r.qoi == "q"
    assert r.provenance.parameters["model"] == "quad"
    assert r.provenance.parameters["batch_size"] is None


@pytest.mark.parametrize("h", [0.0, -1e-6, 0.51, np.nan, np.inf, True, [1e-6, 0.6], [0.0, 1e-3]])
def test_fd_bad_h(h):
    s = asub.Uniform.reference(2)
    with pytest.raises(asub.InputError) as e:
        gr.finite_difference_gradients(lambda Z: Z[:, 0], X=np.zeros((1, 2)), space=s, h=h)
    assert _codes(e) == "input.parameter"


def test_fd_zero_dim_array_h_is_a_scalar():
    s = asub.Uniform.reference(2)
    X = np.array([[0.1, -0.3], [0.5, 0.2]])
    r0 = gr.finite_difference_gradients(
        lambda Z: Z @ np.array([1.0, 2.0]), X=X, space=s, h=np.array(1e-4)
    )
    r1 = gr.finite_difference_gradients(lambda Z: Z @ np.array([1.0, 2.0]), X=X, space=s, h=1e-4)
    assert r0 == r1
    assert r0.provenance.parameters["h"] == 1e-4
    with pytest.raises(asub.InputError) as e:
        gr.finite_difference_gradients(lambda Z: Z[:, 0], X=X, space=s, h=np.array(0.6))
    assert _codes(e) == "input.parameter"


def test_nonfinite_gradients_from_finite_data_raise():
    """Spec 06 §5: an overflow to +-inf in the estimate raises numerical.nonfinite, never
    returns it.
    """
    s = asub.Uniform([0.0, 0.0], [1.0, 1.0])
    X = s.sample(20, rng=0)
    f = np.where(np.arange(20) % 2 == 0, 1e308, -1e308)
    with pytest.raises(asub.NumericalError) as e:
        gr.local_linear_gradients(X=X, f=f, space=s)
    assert _codes(e) == "numerical.nonfinite"
    assert e.value.details["what"] == "grads"
    Xc = np.array([[0.5 - 1e-9, 0.3]])
    with pytest.raises(asub.NumericalError) as e:
        gr.finite_difference_gradients(
            lambda Z: np.where(Z[:, 0] >= 0.5, 1e308, -1e308), X=Xc, space=s
        )
    assert _codes(e) == "numerical.nonfinite"
    assert e.value.details["row"] == 0


def test_fd_bad_h_shape_and_scheme():
    s = asub.Uniform.reference(2)
    with pytest.raises(asub.InputError) as e:
        gr.finite_difference_gradients(
            lambda Z: Z[:, 0], X=np.zeros((1, 2)), space=s, h=[1e-6] * 3
        )
    assert _codes(e) == "input.shape"
    with pytest.raises(asub.InputError) as e:
        gr.finite_difference_gradients(
            lambda Z: Z[:, 0], X=np.zeros((1, 2)), space=s, scheme="backward"
        )
    assert _codes(e) == "input.parameter"
    with pytest.raises(asub.InputError) as e:
        gr.finite_difference_gradients(lambda Z: Z[:, 0], X=np.full((1, 2), 3.0), space=s)
    assert _codes(e) == "input.out_of_space"


def test_fd_bookkeeping_uniform():
    s = asub.Uniform([0.0, -2.0, 10.0], [1.0, 2.0, 30.0])
    Xn = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.9999995], [0.2, 0.9, -0.5]])
    X = s.from_normalized(Xn)

    def fun(Z):
        return Z.sum(axis=1)

    h = 1e-6
    fw = gr.finite_difference_gradients(fun, X=X, space=s, h=h)
    assert fw.scheme == "forward"
    assert not fw.central_mask.any()
    assert fw.n_evaluations == (3 + 1) * 3
    expected = np.full((3, 3), h)
    expected[1, 0] = -h  # x_j = 1: backward
    expected[1, 2] = -h  # x_j + h > 1: backward
    np.testing.assert_array_equal(fw.steps, expected)
    ce = gr.finite_difference_gradients(fun, X=X, space=s, h=0.2, scheme="central")
    mask = np.abs(Xn) + 0.2 <= 1.0
    np.testing.assert_array_equal(ce.central_mask, mask)
    assert ce.n_evaluations == 3 + 9 + int(mask.sum())
    # fallback entries take the forward rule: backward where x + h > 1, forward otherwise
    np.testing.assert_array_equal(ce.steps, np.where((~mask) & (Xn + 0.2 > 1.0), -0.2, 0.2))


def test_fd_bookkeeping_gaussian(gaussian_space):
    s = gaussian_space(2, seed=5)
    Xn = np.array([[0.5, -3.0], [10.0, 0.0]])
    X = s.from_normalized(Xn)
    h = np.array([1e-4, 2e-4])
    r = gr.finite_difference_gradients(lambda Z: Z[:, 0], X=X, space=s, h=h, scheme="central")
    assert r.central_mask.all()
    assert r.n_evaluations == (2 * 2 + 1) * 2
    np.testing.assert_allclose(r.steps, h * np.maximum(1.0, np.abs(Xn)), rtol=1e-12)
    assert list(r.provenance.parameters["h"]) == [1e-4, 2e-4]


def test_fd_values_and_value_only_model():
    s = asub.Uniform([1.0, 1.0], [2.0, 5.0])
    X = s.sample(5, rng=1)

    def fun(Z):
        return Z[:, 0] * Z[:, 1]

    r = gr.finite_difference_gradients(models.Model(space=s, value=fun), X=X)
    np.testing.assert_array_equal(r.values, fun(X))
    np.testing.assert_array_equal(r.points, s.to_normalized(X))
    assert r.method == "finite_difference"
    assert r.centers is None
    assert r.p is None


def test_fd_model_errors():
    s = asub.Uniform.reference(2)
    X = np.zeros((3, 2))

    def boom(Z):
        raise RuntimeError("no")

    with pytest.raises(asub.ModelError) as e:
        gr.finite_difference_gradients(boom, X=X, space=s)
    assert _codes(e) == "model.raised"
    with pytest.raises(asub.ModelError) as e:
        gr.finite_difference_gradients(lambda Z: np.where(Z[:, 0] > 0, np.nan, 0.0), X=X, space=s)
    assert _codes(e) == "model.nonfinite"


def test_fd_never_leaves_space(uniform_space):
    s = uniform_space(4, seed=7)
    gen = np.random.default_rng(3)
    X = s.sample(20, rng=gen)
    X[:5] = s.lower  # corners
    X[5:10] = s.upper
    X[10, 2] = s.upper[2]
    seen = []

    def strict(Z):
        assert np.all((Z >= s.lower) & (Z <= s.upper))
        assert s.contains(Z).all()
        seen.append(Z.shape[0])
        return np.exp(s.to_normalized(Z).sum(axis=1))

    for scheme in ("forward", "central"):
        for h in (1e-6, 0.5):
            gr.finite_difference_gradients(strict, X=X, space=s, h=h, scheme=scheme)
    assert seen


def test_bare_callable_and_executor():
    s = asub.Uniform([0.0, 1.0, -3.0], [2.0, 4.0, 3.0])
    X = s.sample(9, rng=11)

    def fun(Z):
        return np.sin(Z).sum(axis=1) + Z[:, 0] * Z[:, 2]

    for scheme in ("forward", "central"):
        a = gr.finite_difference_gradients(fun, X=X, space=s, scheme=scheme, h=1e-4)
        b = gr.finite_difference_gradients(
            models.Model(space=s, value=fun), X=X, scheme=scheme, h=1e-4
        )
        assert a == b
        for workers in (1, 3):
            with concurrent.futures.ThreadPoolExecutor(workers) as ex:
                c = gr.finite_difference_gradients(
                    fun, X=X, space=s, scheme=scheme, h=1e-4, executor=ex, batch_size=7
                )
            d = gr.finite_difference_gradients(
                fun, X=X, space=s, scheme=scheme, h=1e-4, batch_size=7
            )
            assert c == d == a
            assert d.provenance.parameters["batch_size"] == 7


def test_fd_no_mutation():
    s = asub.Uniform.reference(2)
    X = np.array([[0.1, 0.9], [-1.0, 1.0]])
    X.flags.writeable = False
    X0 = X.copy()

    def scribble(Z):
        Z[:] = 0.0
        return np.ones(Z.shape[0])

    gr.finite_difference_gradients(scribble, X=X, space=s, scheme="central", h=0.01)
    np.testing.assert_array_equal(X, X0)


# ---------------------------------------------------------------------------
# GradientResult as grads= (D-88)
# ---------------------------------------------------------------------------


def test_gradient_result_as_grads(gaussian_space):
    subspaces = pytest.importorskip("active_subspaces.subspaces")
    s = gaussian_space(3, seed=9)
    X = s.sample(40, rng=2)

    def fun(Z):
        return np.exp(0.3 * Z.sum(axis=1))

    g = gr.finite_difference_gradients(fun, X=X, space=s, scheme="central", h=1e-5)
    a = subspaces.active_subspace(grads=g)
    ref = subspaces.active_subspace(grads=g.grads, X=g.points, space=asub.Gaussian.reference(3))
    assert a.space == s
    assert a.qoi == g.qoi
    for name in ("eigenvalues", "eigenvectors", "n", "criterion", "estimator", "dimension_rule"):
        x, y = getattr(a, name), getattr(ref, name)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y)
        else:
            assert x == y
    exact = subspaces.active_subspace(grads=g, space=s)
    assert exact == a
    with pytest.raises(asub.InputError) as e:
        subspaces.active_subspace(grads=g, space=gaussian_space(3, seed=10))
    assert _codes(e) == "input.parameter"
    assert e.value.details["reason"] == "space_mismatch"


def test_c03_no_weights_and_every_sample():
    # [C-03]: the local fit takes no weights (they had no effect in 2016); gradients at every row.
    import inspect

    params = inspect.signature(gr.local_linear_gradients).parameters
    assert set(params) == {"X", "f", "space", "p", "centers", "qoi"}
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    s = asub.Uniform.reference(1)
    X = np.linspace(-1, 1, 9)[:, None]
    r = gr.local_linear_gradients(X=X, f=X[:, 0] ** 2, space=s, p=3)
    assert r.grads.shape == (9, 1)  # m = 1 works and is aligned with X
    np.testing.assert_allclose(r.grads[1:-1, 0], 2 * X[1:-1, 0], atol=1e-12)  # symmetric stencils


def _neighbor_sets_argsort(Xn, centers, p):
    """The stable-argsort neighbor rule that ``_neighbor_sets`` replaced (reference for the
    test below).
    """
    loc, label = gr._distinct_locations(Xn)
    Md = loc.shape[0]
    out = []
    for i in centers:
        d2 = np.einsum("jk,jk->j", loc - Xn[i], loc - Xn[i])
        chosen = np.zeros(Md, dtype=bool)
        chosen[np.argsort(d2, kind="stable")[:p]] = True
        out.append(np.flatnonzero(chosen[label]))
    return out


@pytest.mark.parametrize("blocks", [False, True])
def test_neighbor_sets_match_stable_argsort(monkeypatch, blocks):
    # Grid points give many exact distance ties; duplicates, -0.0, and every p from 1 to past M_d.
    if blocks:
        monkeypatch.setattr(
            gr, "_BLOCK_ELEMENTS", 7
        )  # several centers per block, partial last block
    gen = np.random.default_rng(20261029)
    for trial in range(40):
        m = int(gen.integers(1, 4))
        M = int(gen.integers(2, 40))
        Xn = gen.integers(-2, 3, size=(M, m)) / 2.0
        if trial % 2:
            Xn = np.vstack([Xn, Xn[gen.integers(0, M, 5)]])  # duplicated rows
        Xn[gen.random(Xn.shape) < 0.1] = -0.0
        Md = gr._distinct_locations(Xn)[0].shape[0]
        centers = gen.permutation(Xn.shape[0])[: int(gen.integers(1, Xn.shape[0] + 1))]
        for p in sorted({1, 2, max(1, Md // 2), Md - 1, Md, Md + 2} - {0}):
            new = gr._neighbor_sets(Xn, centers, p)
            ref = _neighbor_sets_argsort(Xn, centers, p)
            assert len(new) == len(ref) == centers.shape[0]
            for a, b in zip(new, ref, strict=True):
                assert a.dtype == b.dtype
                np.testing.assert_array_equal(a, b)


_MULTI_FIELDS = ("points", "grads", "values", "steps", "central_mask")


def _multi_model(P: int):
    """A (k, P) toy model and a per-column scalar view of it; the first counts its calls."""
    w = np.arange(1.0, P + 1.0)
    calls: list[int] = []

    def multi(Z):
        calls.append(Z.shape[0])
        return np.sin(Z @ np.ones(Z.shape[1]))[:, None] * w + Z[:, :1] ** 2 * w[::-1]

    def column(p):
        return lambda Z: multi(Z)[:, p]

    return multi, column, calls


@pytest.mark.parametrize("scheme", ["forward", "central"])
def test_finite_difference_multi(scheme):
    """D-199: one evaluation call; each output equals the scalar call bitwise; labels; shape."""
    s = asub.Uniform([0.0, 1.0, -3.0], [2.0, 4.0, 3.0])
    X = np.vstack([s.sample(8, rng=3), s.lower, s.upper])  # bounds force one-sided entries
    M, m = X.shape
    P = 21
    multi, column, calls = _multi_model(P)
    res = gr.finite_difference_gradients_multi(
        multi, X=X, space=s, outputs=P, h=1e-4, scheme=scheme
    )
    assert isinstance(res, tuple)
    assert len(res) == P
    if scheme == "forward":
        assert calls == [M * (m + 1)]
    else:
        assert len(calls) == 1
        assert calls[0] == res[0].n_evaluations > M * (m + 1)
    assert [r.qoi for r in res] == [f"f{p}" for p in range(P)]
    for workers in (None, 3):
        if workers is None:
            got = res
        else:
            with concurrent.futures.ThreadPoolExecutor(workers) as ex:
                got = gr.finite_difference_gradients_multi(
                    multi,
                    X=X,
                    space=s,
                    outputs=P,
                    h=1e-4,
                    scheme=scheme,
                    executor=ex,
                    batch_size=7,
                )
        for p in range(P):
            ref = gr.finite_difference_gradients(column(p), X=X, space=s, h=1e-4, scheme=scheme)
            r = got[p]
            for name in _MULTI_FIELDS:
                a, b = getattr(r, name), getattr(ref, name)
                assert a.dtype == b.dtype, (p, name)
                assert a.tobytes() == b.tobytes(), (p, name)
            assert r.n_evaluations == ref.n_evaluations
            assert r.method == ref.method
            assert r.scheme == ref.scheme
            assert r.space == ref.space
            assert r.qoi == f"f{p}"
            assert got[0].steps is not None
            prm = r.provenance.parameters
            assert prm["outputs"] == P
            assert prm["output"] == p
            assert prm["qoi_names"] == [f"f{q}" for q in range(P)]
            assert r.provenance.function == "active_subspaces.finite_difference_gradients_multi"


def test_finite_difference_multi_names_and_errors():
    """D-199: qoi_names are used; (k,) models raise model.shape; bad arguments input.parameter."""
    s = asub.Uniform.reference(2)
    X = s.sample(4, rng=0)

    def two(Z):
        return np.stack([Z[:, 0], Z[:, 1] ** 2], axis=1)

    model = models.Model(space=s, value=two, qoi="ignored")
    res = gr.finite_difference_gradients_multi(model, X=X, outputs=2, qoi_names=("a", "b"))
    assert [r.qoi for r in res] == ["a", "b"]
    with pytest.raises(ModelError) as e:
        gr.finite_difference_gradients_multi(lambda Z: Z[:, 0], X=X, space=s, outputs=2)
    assert e.value.code == "model.shape"
    with pytest.raises(ModelError) as e:
        gr.finite_difference_gradients_multi(two, X=X, space=s, outputs=3)
    assert e.value.code == "model.shape"
    for bad in (["a"], ["a", "a"], ["a", ""], ["a", 1], "ab", 5):
        with pytest.raises(InputError) as e:
            gr.finite_difference_gradients_multi(two, X=X, space=s, outputs=2, qoi_names=bad)
        assert e.value.code == "input.parameter"
    for bad in (0, -1, 1.5, True, "2", None):
        with pytest.raises(InputError) as e:
            gr.finite_difference_gradients_multi(two, X=X, space=s, outputs=bad)
        assert e.value.code == "input.parameter"


def test_finite_difference_product_boundary_rule():
    """On a Product the one-sided rule applies to uniform coordinates only [D-198]."""
    sp = asub.Product([asub.Uniform([0.0], [1.0]), asub.Gaussian([0.0], [1.0])])
    X = np.array([[1.0, 0.0], [0.5, 30.0]])  # at the uniform upper bound; far in the Gaussian
    seen = []

    def model(Z):
        seen.append(Z.copy())
        return Z[:, 0] ** 2 + 3.0 * Z[:, 1]

    res = gr.finite_difference_gradients(model, X=X, space=sp, h=1e-4, scheme="central")
    assert not res.central_mask[0, 0]  # uniform coordinate at its bound: one-sided
    assert res.steps[0, 0] < 0
    assert res.central_mask[1, 1]  # Gaussian coordinate: central, whatever its value
    allz = np.vstack(seen)
    assert allz[:, 0].max() <= 1.0
    np.testing.assert_allclose(res.grads_physical[:, 1], 3.0, rtol=1e-6)
