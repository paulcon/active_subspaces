"""Unit tests for ``active_subspaces.reduced_model`` (spec 11 §8; spec 17 V11.3-V11.5, V11.7,
V11.11-V11.17).
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import math
import warnings

import numpy as np
import pytest
import scipy.stats

import active_subspaces as asub
from active_subspaces import _random
from active_subspaces import integration as I
from active_subspaces import optimization as O
from active_subspaces import reduced_model as R
from active_subspaces import surrogates as S
from active_subspaces import test_functions as tf
from active_subspaces.errors import (
    AccuracyWarning,
    InputError,
    ModelError,
    ValidityWarning,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _space(m: int = 4) -> asub.Uniform:
    return asub.Uniform(np.linspace(-2.0, 1.0, m), np.linspace(0.0, 7.0, m) + 1.0)


def _ridge_data(space, M: int = 60, seed: int = 0, offset: float = 0.0):
    """Physical ridge f = (a^T xi + offset)^2 with exact physical gradients."""
    m = space.dim
    a = np.linspace(1.0, 0.2, m)
    X = space.sample(M, rng=seed)
    t = X @ a + offset
    return X, t**2, 2.0 * t[:, None] * a[None, :], a


@pytest.fixture(scope="module")
def data_rm():
    sp = _space()
    X, f, G, _ = _ridge_data(sp)
    return R.build_reduced_model(
        space=sp, X=X, f=f, grads=G, surrogate_kind="poly", degree=2, nboot=10, rng=5
    )


def _ridge_model(space):
    """A ``tf.ridge`` model (picklable) on ``space``: f = (a^T x)^2 / 2 in normalized
    coordinates.
    """
    return tf.ridge(np.linspace(1.0, 0.3, space.dim), profile="square", space=space)


@pytest.fixture(scope="module")
def model_rm():
    sp = _space()
    return R.build_reduced_model(
        model=_ridge_model(sp), surrogate_kind="poly", degree=2, nboot=10, rng=7
    )


def _short_chains():
    """Hit-and-run chains with n_mc < 20 warn accuracy.correlated_draws (D-114, amended
    2026-10-29).
    """
    return pytest.warns(AccuracyWarning, match=r"accuracy\.correlated_draws")


@contextlib.contextmanager
def _clipping_allowed():
    """An exact ridge has no curvature along z, so the lift may clip it [D-49]; nothing else
    may warn.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"lift: .*curvature", category=AccuracyWarning)
        yield


def _code(excinfo) -> str:
    return excinfo.value.code


# ---------------------------------------------------------------------------
# V11.3 [C-07]: the R^2 helper
# ---------------------------------------------------------------------------


def test_r_squared():
    gen = np.random.default_rng(0)
    f = gen.standard_normal(30)
    w = gen.uniform(0.5, 1.5, 30)
    w /= w.sum()
    g = f + 0.1 * gen.standard_normal(30)
    assert R._r_squared(f, f, w) == 1.0
    fbar = float(np.sum(w * f))
    assert abs(R._r_squared(f, np.full(30, fbar), w)) <= 1e-14
    r2 = R._r_squared(f, g, w)
    dup = R._r_squared(np.r_[f, f], np.r_[g, g], np.r_[w, w] / 2)
    assert dup == pytest.approx(r2, rel=1e-14, abs=1e-14)  # no factor M [C-07]
    assert R._r_squared(np.full(5, 0.1), np.zeros(5), np.full(5, 0.2)) is None


# ---------------------------------------------------------------------------
# V11.4 [C-08], [D-118]: default M
# ---------------------------------------------------------------------------


def test_default_M():
    assert [R._default_n_samples(m) for m in (2, 6, 10, 100)] == [12, 75, 151, 2790]
    model = asub.Model(space=asub.Uniform.reference(1), value=lambda X: X[:, 0])
    with pytest.raises(InputError) as e:
        R.build_reduced_model(model=model, rng=0)
    assert _code(e) == "input.parameter"


def test_model_path_draws_default_M(model_rm):
    assert model_rm.X.shape == (R._default_n_samples(4), 4)
    assert model_rm.provenance.parameters["n_samples"] == R._default_n_samples(4)
    np.testing.assert_array_equal(
        model_rm.quad_weights, np.full(model_rm.X.shape[0], 1 / model_rm.X.shape[0])
    )


# ---------------------------------------------------------------------------
# V11.5 [C-05]: support check
# ---------------------------------------------------------------------------


def test_out_of_space():
    m = 3
    sp = asub.Uniform.reference(m)
    gen = np.random.default_rng(1)
    X = gen.uniform(-5, 5, (40, m))
    with pytest.raises(InputError) as e:
        R.build_reduced_model(space=sp, X=X, f=X.sum(1), grads=np.ones((40, m)), nboot=0)
    assert _code(e) == "input.out_of_space"
    X = gen.uniform(-1, 1, (40, m))
    X[:3] = [[1, -1, 1], [-1, -1, -1], [1, 1, 1]]  # on the bounds: accepted
    G = np.tile([1.0, 0.5, 0.0], (40, 1))
    rm = R.build_reduced_model(
        space=sp,
        X=X,
        f=X @ [1.0, 0.5, 0.0],
        grads=G,
        nboot=0,
        n=1,
        surrogate_kind="poly",
        degree=1,
    )
    assert rm.X.shape == (40, m)


# ---------------------------------------------------------------------------
# V11.7 [C-06], [D-119]: probability interval
# ---------------------------------------------------------------------------


def _wilson_direct(p, N, c):
    z = scipy.stats.norm.ppf((1 + c) / 2)
    center = (p + z * z / (2 * N)) / (1 + z * z / N)
    half = z / (1 + z * z / N) * math.sqrt(p * (1 - p) / N + z * z / (4 * N * N))
    return center - half, center + half


def test_probability_interval(data_rm):
    f = data_rm.f
    q = np.quantile(f, [0.2, 0.7])
    for lower, upper in [
        (q[0], q[1]),
        (-np.inf, q[0]),
        (q[1], np.inf),
        (-np.inf, np.inf),
        (-10.0, -1.0),
    ]:
        res = R.probability(data_rm, lower, upper, n_samples=2000, rng=3)
        lo, hi = res.confidence_interval()
        assert 0.0 <= lo <= res.p <= hi <= 1.0
        assert res.std_error == pytest.approx(math.sqrt(res.p * (1 - res.p) / 2000), rel=1e-15)
    res = R.probability(data_rm, -np.inf, np.inf, n_samples=500, rng=3)
    assert res.p == 1.0
    assert res.confidence_interval()[0] < 1.0
    res = R.probability(data_rm, -10.0, -1.0, n_samples=500, rng=3)  # f >= 0 on the surrogate
    assert res.p == 0.0
    assert res.std_error == 0.0
    lo, hi = res.confidence_interval()
    assert lo == 0.0
    assert hi > 0.0
    for p, N, c in [
        (0.0, 500, 0.99),
        (0.3, 1000, 0.99),
        (0.999, 10_000, 0.95),
        (1.0, 50, 0.5),
        (1e-4, 10**5, 0.999),
    ]:
        lo, hi = R._wilson(p, N, c)
        dlo, dhi = _wilson_direct(p, N, c)
        assert abs(lo - min(max(dlo, 0.0), p)) <= 1e-15
        assert abs(hi - max(min(dhi, 1.0), p)) <= 1e-15
    with pytest.raises(InputError):
        res.confidence_interval(1.0)


def test_probability_counts_closed_interval(data_rm):
    gen = np.random.default_rng(9)
    x = data_rm.space.sample(300, rng=gen, normalized=True)
    g = data_rm.surrogate.evaluate(x @ data_rm.W1)
    lower, upper = (
        float(np.sort(g)[50]),
        float(np.sort(g)[200]),
    )  # both attained: closed at both ends
    res = R.probability(data_rm, lower, upper, n_samples=300, rng=np.random.default_rng(9))
    assert res.p == np.count_nonzero((g >= lower) & (g <= upper)) / 300 == 151 / 300


# ---------------------------------------------------------------------------
# V11.11: composition
# ---------------------------------------------------------------------------


def test_composition(data_rm):
    rm = data_rm
    assert rm.domain.subspace == rm.subspace == rm.surrogate.subspace
    assert rm.space == rm.subspace.space
    assert rm.qoi == rm.subspace.qoi == rm.surrogate.qoi == "f"
    assert rm.m == 4
    assert rm.n == rm.subspace.n == rm.active_dim
    np.testing.assert_array_equal(rm.W1, rm.subspace.W1)
    np.testing.assert_array_equal(rm.X_physical, rm.space.from_normalized(rm.X))
    assert not rm.X_physical.flags.writeable

    X = rm.X_physical
    sub2 = asub.with_dimension(rm.subspace, 2)
    rm2 = R.build_reduced_model(X=X, f=rm.f, subspace=sub2, surrogate_kind="poly", degree=2, rng=1)
    assert rm2.n == 2
    assert rm2.subspace == sub2
    assert rm2.subspace.nboot == rm.subspace.nboot
    assert rm2.provenance.parameters["n"] is None
    rm3 = R.build_reduced_model(
        X=X, f=rm.f, subspace=rm.subspace, n=2, surrogate_kind="poly", rng=1
    )
    assert rm3.n == 2
    assert rm3.subspace.eigenvalues is not None
    np.testing.assert_array_equal(rm3.subspace.eigenvectors, rm.subspace.eigenvectors)

    other = asub.Uniform(rm.space.lower, rm.space.upper + 1.0)
    with pytest.raises(InputError) as e:
        R.build_reduced_model(space=other, X=X, f=rm.f, subspace=sub2)
    assert _code(e) == "input.parameter"
    assert e.value.details["reason"] == "space_mismatch"
    with pytest.raises(InputError) as e:
        R.build_reduced_model(space=rm.space, X=X, f=rm.f, n=4)
    assert _code(e) == "input.parameter"
    with pytest.raises(InputError) as e:
        R.build_reduced_model(X=X, f=rm.f, subspace=asub.with_dimension(rm.subspace, 4))
    assert _code(e) == "input.parameter"


def test_invariants(data_rm, model_rm):
    for rm in (data_rm, model_rm):
        d = rm.to_dict()
        assert rm.source in ("data", "model")
        bad = dict(
            d, fields=dict(d["fields"], r_squared_kind="holdout" if rm.source == "data" else "loo")
        )
        with pytest.raises(asub.SchemaError):
            asub.from_dict(bad)
    d = data_rm.to_dict()
    with pytest.raises(asub.SchemaError):
        asub.from_dict(dict(d, fields=dict(d["fields"], gradient_source="none")))  # grads present
    with pytest.raises(asub.SchemaError):
        asub.from_dict(dict(d, fields=dict(d["fields"], r_squared=1.5)))
    assert data_rm.source == "data"
    assert data_rm.r_squared_kind == "loo"
    assert data_rm.gradient_source == "given"
    assert model_rm.source == "model"
    assert model_rm.r_squared_kind == "holdout"
    assert model_rm.gradient_source == "model"


def test_readonly(data_rm, assert_readonly):
    assert_readonly(data_rm)


# ---------------------------------------------------------------------------
# V11.12 [D-104]: Surrogate protocol
# ---------------------------------------------------------------------------


def test_surrogate_protocol(data_rm):
    rm = data_rm
    assert isinstance(rm, asub.Surrogate)
    assert R.predict is S.predict
    assert R.predict_gradient is S.predict_gradient
    assert asub.predict is R.predict
    Xt = rm.space.sample(20, rng=2)
    np.testing.assert_array_equal(R.predict(rm, Xt), S.predict(rm.surrogate, Xt))
    np.testing.assert_array_equal(R.predict_gradient(rm, Xt), S.predict_gradient(rm.surrogate, Xt))
    np.testing.assert_array_equal(rm.predict(Xt), R.predict(rm, Xt))
    np.testing.assert_array_equal(rm.predict_gradient(Xt), R.predict_gradient(rm, Xt))
    Y = rm.space.to_normalized(Xt) @ rm.W1
    np.testing.assert_array_equal(rm.evaluate(Y), rm.surrogate.evaluate(Y))
    np.testing.assert_array_equal(rm.gradient(Y), rm.surrogate.gradient(Y))
    out = R.predict(rm, Xt)
    assert out.flags.writeable
    # spec 10 accepts rm itself
    a = I.integrate_surrogate(rm, domain=rm.domain, n_nodes=5, rng=1)
    b = I.integrate_surrogate(rm.surrogate, domain=rm.domain, n_nodes=5, rng=1)
    assert a.estimate == b.estimate
    with _clipping_allowed():
        mn = O.minimize(rm, domain=rm.domain, X=rm.X_physical, f=rm.f, rng=1)
    assert np.isfinite(mn.f_star)
    with pytest.raises(InputError) as e:
        R.predict(rm, np.full((1, 4), 100.0))
    assert _code(e) == "input.out_of_space"


# ---------------------------------------------------------------------------
# V11.13: "basis" subspace
# ---------------------------------------------------------------------------


def test_basis_subspace():
    sp = _space(3)
    X, f, _, a = _ridge_data(sp, M=40)
    sub = asub.subspace_from_basis(a[:, None], space=sp)
    rm = R.build_reduced_model(X=X, f=f, subspace=sub, surrogate_kind="rbf", rng=0)
    assert rm.subspace.estimator == "basis"
    assert rm.n == 1
    assert np.all(np.isnan(rm.eigenvalues))
    assert rm.gradient_source == "none"
    assert rm.grads is None
    model = asub.Model(space=sp, value=lambda X: (X @ a) ** 2)
    rm2 = R.build_reduced_model(model=model, subspace=sub, n_samples=20, n_design=4, n_mc=2, rng=0)
    assert rm2.gradient_source == "none"
    assert np.all(np.isnan(rm2.eigenvalues))


# ---------------------------------------------------------------------------
# V11.14 [D-121]: envelope warning
# ---------------------------------------------------------------------------


def _flat_data(m: int = 7, M: int = 100):
    gen = np.random.default_rng(0)
    X = gen.uniform(-1, 1, (M, m))
    G = gen.standard_normal((M, m)) * np.r_[np.ones(5), np.full(m - 5, 1e-3)]
    return X, X.sum(axis=1), G


def test_envelope_warning():
    sp = asub.Uniform.reference(7)
    X, f, G = _flat_data()
    with pytest.warns(AccuracyWarning) as rec:
        rm = R.build_reduced_model(
            space=sp, X=X, f=f, grads=G, nboot=0, surrogate_kind="poly", degree=1
        )
    w = [r.message for r in rec if getattr(r.message, "code", None) == "accuracy.large_dimension"]
    assert len(w) == 1
    assert rm.n == 5
    assert dict(w[0].details) == {"n": 5, "envelope": 4, "n_design": 3125}
    rm = R.build_reduced_model(
        space=sp, X=X, f=f, grads=G, n=5, nboot=0, surrogate_kind="poly", degree=1
    )
    assert rm.n == 5  # explicit n: no warning (filterwarnings=error)
    # Gaussian envelope is 6: n = 5 does not warn
    rm = R.build_reduced_model(
        space=asub.Gaussian.reference(7),
        X=X,
        f=f,
        grads=G,
        nboot=0,
        surrogate_kind="poly",
        degree=1,
    )
    assert rm.n == 5


# ---------------------------------------------------------------------------
# V11.15: determinism
# ---------------------------------------------------------------------------


def _build_model(space, **kw):
    return R.build_reduced_model(
        model=_ridge_model(space), surrogate_kind="rbf", nboot=5, n_samples=30, n_design=6, **kw
    )


def test_determinism_seed_and_replay():
    sp = _space()
    a = _build_model(sp, rng=11)
    b = _build_model(sp, rng=11)
    c = _build_model(sp, rng=np.random.default_rng(11))
    d = _build_model(sp, rng=np.random.SeedSequence(11))
    assert a == b == c == d
    e = _build_model(sp, rng=_random.replay(a))
    assert e == a
    assert _build_model(sp, rng=12) != a


def test_determinism_executors():
    sp = _space()
    ref = _build_model(sp, rng=3, batch_size=7)
    for workers in (1, 2, 7):
        with concurrent.futures.ThreadPoolExecutor(workers) as ex:
            assert _build_model(sp, rng=3, batch_size=7, executor=ex) == ref
    with concurrent.futures.ProcessPoolExecutor(2) as ex:
        assert _build_model(sp, rng=3, batch_size=7, executor=ex) == ref


def test_determinism_finite_difference_executor():
    sp = _space()
    mdl = _ridge_model(sp)
    value_only = asub.Model(space=sp, value=mdl.value)
    kw = {
        "model": value_only,
        "surrogate_kind": "poly",
        "nboot": 0,
        "n_samples": 20,
        "n_design": 5,
        "n_mc": 2,
        "batch_size": 11,
        "rng": 4,
    }
    ref = R.build_reduced_model(**kw)
    assert ref.gradient_source == "finite_difference"
    assert ref.provenance.parameters["gradient_source"] == "finite_difference"
    with concurrent.futures.ThreadPoolExecutor(3) as ex:
        assert R.build_reduced_model(executor=ex, **kw) == ref


@pytest.mark.parametrize("kind", ["uniform", "gaussian"])
def test_probability_chunked_draw(kind, monkeypatch):
    sp = (
        _space()
        if kind == "uniform"
        else asub.Gaussian(np.zeros(4), np.diag([1.0, 2.0, 3.0, 4.0]))
    )
    X, f, G, _ = _ridge_data(sp, M=50)
    rm = R.build_reduced_model(
        space=sp, X=X, f=f, grads=G, n=1, nboot=0, surrogate_kind="poly", rng=0
    )
    lower, upper = np.quantile(f, [0.3, 0.6])
    full = R.probability(rm, lower, upper, n_samples=1001, rng=8)
    x = sp.sample(1001, rng=8, normalized=True)
    g = rm.surrogate.evaluate(x @ rm.W1)
    assert full.p == np.count_nonzero((g >= lower) & (g <= upper)) / 1001
    monkeypatch.setattr(R, "_CHUNK_ELEMENTS", 4 * 37)  # 37 rows per chunk
    chunked = R.probability(rm, lower, upper, n_samples=1001, rng=8)
    assert chunked == full


def test_method_equals_function(model_rm):
    rm = model_rm
    model = _ridge_model(rm.space)
    assert rm.probability(0.0, 0.2, n_samples=500, rng=1) == R.probability(
        rm, 0.0, 0.2, n_samples=500, rng=1
    )
    assert rm.average(n_nodes=4, rng=1) == R.average(rm, n_nodes=4, rng=1)
    with _short_chains():
        assert rm.average(n_nodes=4, model=model, n_mc=4, rng=1) == R.average(
            rm, n_nodes=4, model=model, n_mc=4, rng=1
        )
    with _clipping_allowed():
        a = rm.minimum(model=model, n_starts=3, rng=1)
    with _clipping_allowed():
        b = R.minimum(rm, model=model, n_starts=3, rng=1)
    assert a == b
    assert a.f_model is not None


def test_operations_equal_spec10(model_rm):
    rm = model_rm
    model = _ridge_model(rm.space)
    assert R.average(rm, n_nodes=5, n_replicates=2, rng=4) == I.integrate_surrogate(
        rm.surrogate, domain=rm.domain, n_nodes=5, n_replicates=2, rng=4
    )
    assert R.average(rm, n_nodes=5, model=model, rng=4) == I.integrate(
        model, domain=rm.domain, n_nodes=5, n_mc=50, rng=4
    )  # default n_mc [D-171]
    with _clipping_allowed():
        a = R.minimum(rm, n_starts=4, rng=2)
    with _clipping_allowed():
        b = O.minimize(
            rm.surrogate,
            domain=rm.domain,
            X=rm.X_physical,
            f=rm.f,
            space=rm.space,
            n_starts=4,
            rng=2,
        )
    assert a == b
    assert a.f_model is None
    assert a.provenance.function == "active_subspaces.minimize"
    assert (
        R.average(rm, n_nodes=5, rng=4).provenance.function
        == "active_subspaces.integrate_surrogate"
    )


def test_logging(caplog, data_rm):
    sp = _space()
    X, f, G, _ = _ridge_data(sp)
    with caplog.at_level(logging.INFO, logger="active_subspaces.reduced_model"):
        rm = R.build_reduced_model(
            space=sp, X=X, f=f, grads=G, surrogate_kind="poly", nboot=0, rng=0
        )
    msgs = [r.getMessage() for r in caplog.records if r.name == "active_subspaces.reduced_model"]
    assert any(f"active dimension n = {rm.n}" in s for s in msgs)


# ---------------------------------------------------------------------------
# V11.16: serialization
# ---------------------------------------------------------------------------


def test_round_trip(data_rm, model_rm, assert_roundtrip):
    assert_roundtrip(data_rm)
    assert_roundtrip(model_rm)
    gsp = asub.Gaussian.reference(3)
    gen = np.random.default_rng(2)
    X = gen.standard_normal((50, 3))
    rm = R.build_reduced_model(
        space=gsp,
        X=X,
        f=X @ [1.0, 0.5, 0.1],
        estimator="ols",
        nboot=0,
        surrogate_kind="rbf",
        rng=0,
    )
    assert rm.grads is None
    assert rm.gradient_source == "none"
    assert_roundtrip(rm)
    for lower, upper in [(-np.inf, 0.0), (0.0, np.inf), (-np.inf, np.inf), (-0.5, 0.5)]:
        res = R.probability(rm, lower, upper, n_samples=200, rng=1)
        assert_roundtrip(res)
        d = res.to_dict()
        if math.isinf(lower):
            assert d["fields"]["lower"] == "-Infinity"
        if math.isinf(upper):
            assert d["fields"]["upper"] == "Infinity"
    back = asub.from_dict(res.to_dict())
    assert back.reduced_model == rm
    assert back.confidence_interval() == res.confidence_interval()
    np.testing.assert_array_equal(R.predict(back.reduced_model, X), R.predict(rm, X))


# ---------------------------------------------------------------------------
# V11.17: one test per row of spec 11 §6
# ---------------------------------------------------------------------------


def _bad_value(X):
    raise RuntimeError("boom")


def _errors_cases():
    sp = _space()
    X, f, G, a = _ridge_data(sp, M=30)
    model = asub.Model(space=sp, value=lambda X: X @ a)
    other = asub.Uniform(sp.lower, sp.upper + 1.0)
    return sp, X, f, G, model, other


@pytest.mark.parametrize(
    "row",
    [
        "no_source",
        "both_sources",
        "X_without_f",
        "f_without_X",
        "grads_without_X",
        "model_params_with_data",
        "space_missing",
        "one_input",
        "n_equals_m",
        "subspace_n_equals_m",
        "subspace_with_estimator",
        "model_space_mismatch",
        "subspace_space_mismatch",
        "qoi_mismatch",
        "bad_scalars",
        "lower_nan",
        "lower_ge_upper",
        "data_codes",
        "too_few_samples",
        "model_raises",
        "validity",
        "loo_undefined",
        "average_model_only",
        "average_space_mismatch",
    ],
)
def test_errors(row, data_rm):
    sp, X, f, G, model, other = _errors_cases()

    def param(call, reason=None, code="input.parameter"):
        with pytest.raises(InputError) as e:
            call()
        assert _code(e) == code
        if reason is not None:
            assert e.value.details["reason"] == reason

    B = R.build_reduced_model
    if row == "no_source":
        param(lambda: B(space=sp))
    elif row == "both_sources":
        param(lambda: B(space=sp, X=X, f=f, model=model))
        param(lambda: B(model=model, quad_weights=np.full(30, 1 / 30)))
    elif row == "X_without_f":
        param(lambda: B(space=sp, X=X))
    elif row == "f_without_X":
        param(lambda: B(space=sp, f=f))
    elif row == "grads_without_X":
        param(lambda: B(space=sp, grads=G))
    elif row == "model_params_with_data":
        for kw in (
            {"n_samples": 10},
            {"n_design": 5},
            {"n_mc": 2},
            {"fd_step": 1e-4},
            {"batch_size": 3},
        ):
            param(lambda kw=kw: B(space=sp, X=X, f=f, **kw))
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            param(lambda: B(space=sp, X=X, f=f, executor=ex))
    elif row == "space_missing":
        param(lambda: B(X=X, f=f, grads=G))
        param(lambda: B(model=lambda X: X[:, 0]))
    elif row == "one_input":
        sp1 = asub.Uniform.reference(1)
        param(lambda: B(space=sp1, X=np.zeros((5, 1)), f=np.zeros(5)))
    elif row == "n_equals_m":
        param(lambda: B(space=sp, X=X, f=f, grads=G, n=4))
        param(lambda: B(model=model, n=4))
    elif row == "subspace_n_equals_m":
        sub = asub.with_dimension(data_rm.subspace, 4)
        param(lambda: B(X=data_rm.X_physical, f=data_rm.f, subspace=sub))
        param(lambda: B(X=data_rm.X_physical, f=data_rm.f, subspace=data_rm.subspace, n=4))
    elif row == "subspace_with_estimator":
        for kw in ({"estimator": "as"}, {"dimension_rule": "evg"}, {"nboot": 0}):
            param(
                lambda kw=kw: B(X=data_rm.X_physical, f=data_rm.f, subspace=data_rm.subspace, **kw)
            )
    elif row == "model_space_mismatch":
        param(lambda: B(model=model, space=other), "space_mismatch")
        param(
            lambda: B(
                model=model, subspace=asub.subspace_from_basis(np.eye(4)[:, :1], space=other)
            ),
            "space_mismatch",
        )
    elif row == "subspace_space_mismatch":
        param(
            lambda: B(space=other, X=data_rm.X_physical, f=data_rm.f, subspace=data_rm.subspace),
            "space_mismatch",
        )
    elif row == "qoi_mismatch":
        param(
            lambda: B(X=data_rm.X_physical, f=data_rm.f, subspace=data_rm.subspace, qoi="g"),
            "qoi_mismatch",
        )
        named = asub.Model(space=sp, value=model.value, qoi="h")
        param(lambda: B(model=named, subspace=data_rm.subspace), "qoi_mismatch")
    elif row == "bad_scalars":
        for kw in (
            {"n_samples": 0},
            {"n_design": 0},
            {"n_mc": 0},
            {"n_mc": 1},
            {"fd_step": 0.0},
            {"fd_step": 0.6},
            {"batch_size": 0},
            {"surrogate_kind": "gp"},
            {"degree": -1},
            {"n": 0},
            {"nboot": -1},
            {"rng": -1},
            {"estimator": "pca"},
        ):
            param(lambda kw=kw: B(model=model, **kw))
    elif row == "lower_nan":
        param(lambda: R.probability(data_rm, float("nan"), 1.0))
        param(lambda: R.probability(data_rm, 0.0, float("nan")))
    elif row == "lower_ge_upper":
        param(lambda: R.probability(data_rm, 1.0, 1.0))
        param(lambda: R.probability(data_rm, np.inf, np.inf))
        param(lambda: R.probability(data_rm, 0.0, 1.0, n_samples=0))
    elif row == "data_codes":
        param(lambda: B(space=sp, X=X[:, :3], f=f, grads=G), code="input.shape")
        param(lambda: B(space=sp, X=X, f=f[:-1], grads=G), code="input.size_mismatch")
        bad = f.copy()
        bad[2] = np.nan
        param(lambda: B(space=sp, X=X, f=bad, grads=G), code="input.nonfinite")
        param(
            lambda: B(space=sp, X=X, f=f, grads=G, quad_weights=np.full(30, 0.5)),
            code="input.weights",
        )
        param(lambda: B(space=sp, X=X, f=np.c_[f, f], grads=G), code="input.vector_qoi")
        gr = asub.local_linear_gradients(X=X, f=f, space=sp)
        param(
            lambda: B(space=other, X=np.tile(other.lower, (30, 1)), f=f, grads=gr),
            "space_mismatch",
        )
    elif row == "too_few_samples":
        param(lambda: B(space=sp, X=X[:4], f=f[:4]), code="input.too_few_samples")
    elif row == "model_raises":
        with pytest.raises(ModelError) as e:
            B(model=asub.Model(space=sp, value=_bad_value), rng=0)
        assert e.value.code == "model.raised"
    elif row == "validity":
        with pytest.warns(ValidityWarning) as rec:
            B(space=sp, X=X, f=f, estimator="ols", nboot=0, surrogate_kind="poly", rng=0)
        assert any(getattr(r.message, "code", None) == "validity.assumption" for r in rec)
    elif row == "loo_undefined":
        sp2 = asub.Uniform.reference(2)
        X3 = np.array([[-0.5, 0.1], [0.0, -0.2], [0.7, 0.3]])
        G3 = np.tile([1.0, 0.0], (3, 1))
        with pytest.warns(AccuracyWarning) as rec:
            rm = B(
                space=sp2,
                X=X3,
                f=X3[:, 0] ** 2,
                grads=G3,
                n=1,
                nboot=0,
                surrogate_kind="poly",
                degree=2,
            )
        assert any(getattr(r.message, "code", None) == "accuracy.loo_undefined" for r in rec)
        assert rm.r_squared is None
        assert rm.r_squared_kind == "loo"
    elif row == "average_model_only":
        for kw in ({"n_mc": 3}, {"batch_size": 2}):
            param(lambda kw=kw: R.average(data_rm, n_nodes=3, **kw), "model_only")
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            param(lambda: R.average(data_rm, n_nodes=3, executor=ex), "model_only")
    elif row == "average_space_mismatch":
        m_other = asub.Model(space=other, value=model.value)
        param(lambda: R.average(data_rm, n_nodes=3, model=m_other), "space_mismatch")
        param(lambda: R.minimum(data_rm, model=m_other), "space_mismatch")
        param(lambda: R.probability(data_rm.surrogate, 0.0, 1.0))


@pytest.mark.parametrize("n_mc", [0, 1, 2])
def test_n_mc_checked_before_evaluation(n_mc):
    """V11.19, D-115 amended 2026-10-29: n_mc >= 2 on the model path, checked before any model
    evaluation.
    """
    calls = [0]

    def f(X):
        calls[0] += X.shape[0]
        return np.sum(X, axis=1) ** 2

    kw = {
        "model": f,
        "space": asub.Uniform.reference(3),
        "n_mc": n_mc,
        "n_samples": 20,
        "n_design": 5,
        "nboot": 0,
    }
    if n_mc < 2:
        with pytest.raises(InputError) as e:
            R.build_reduced_model(**kw, rng=0)
        assert e.value.code == "input.parameter"
        assert e.value.details["param"] == "n_mc"
        assert calls[0] == 0
    else:
        rm = R.build_reduced_model(**kw, surrogate_kind="poly", rng=0)
        assert calls[0] > 0
        assert rm.provenance.parameters["n_mc"] == 2
