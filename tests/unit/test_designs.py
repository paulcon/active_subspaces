"""Contracts of spec 09 designs and conditional samples (V09.10, V09.15)."""

from __future__ import annotations

import concurrent.futures

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import _random
from active_subspaces import designs as de


def _ref(m):
    return asub.Uniform(-np.ones(m), np.ones(m))


@pytest.fixture(scope="module")
def zono():
    sp = asub.Uniform(-np.ones(5), 2 * np.ones(5))
    sub = asub.subspace_from_basis(np.random.default_rng(3).normal(size=(5, 2)), space=sp)
    return asub.active_domain(sub)


@pytest.fixture(scope="module")
def square():
    sub = asub.subspace_from_basis(np.eye(3)[:, :2], space=_ref(3))
    return asub.active_domain(sub)


@pytest.fixture(scope="module")
def interval():
    sp = asub.Uniform(np.zeros(3), np.array([1.0, 2.0, 4.0]))
    return asub.active_domain(asub.subspace_from_basis(np.array([[1.0], [0.5], [0.2]]), space=sp))


@pytest.fixture(scope="module")
def gauss():
    sp = asub.Gaussian(
        np.array([1.0, -1.0, 0.0]), np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]])
    )
    return asub.active_domain(asub.subspace_from_basis(np.array([[1.0], [1.0], [0.0]]), space=sp))


def _facet_slack(dom, Y):
    return np.max(Y @ dom.facet_normals.T - dom.facet_offsets)


# ---------------------------------------------------------------------------
# V09.10 maximin contract
# ---------------------------------------------------------------------------


def test_maximin_feasible_and_deterministic(zono, seed):
    d1 = asub.active_design(domain=zono, n_points=8, rng=seed)
    assert d1.method == "maximin_slsqp"
    assert d1.points.shape == (8, 2)
    assert _facet_slack(zono, d1.points) <= 0.0
    assert np.all(asub.in_active_domain(zono, Y=d1.points))
    dpp, dpv = de._sq_dists(d1.points, zono.vertices)
    assert d1.min_distance == pytest.approx(np.sqrt(min(dpp.min(), dpv.min())), rel=1e-14)
    d2 = asub.active_design(domain=zono, n_points=8, rng=seed)
    assert np.array_equal(d1.points, d2.points)
    assert d1 == d2
    d3 = asub.active_design(domain=zono, n_points=8, rng=_random.replay(d1))
    assert np.array_equal(d1.points, d3.points)


def test_maximin_single_point_is_centre(square):
    d = asub.active_design(domain=square, n_points=1, rng=0)
    np.testing.assert_allclose(d.points, [[0.0, 0.0]], atol=1e-6)
    assert d.min_distance == pytest.approx(np.sqrt(2.0), abs=1e-6)


def test_maximin_all_starts_fail(square, monkeypatch):
    monkeypatch.setattr(de, "_slsqp", lambda x0, cons, jac, bounds=None: x0 + 100.0)
    with pytest.raises(asub.SolverError) as ei:
        asub.active_design(domain=square, n_points=3, rng=0)
    assert ei.value.code == "solver.failed"
    assert ei.value.details["failed"] == 3


def test_maximin_one_failed_start(square, monkeypatch):
    real = de._maximin_local
    calls = []

    def flaky(Y0, V, nu, bt):
        calls.append(1)
        Y = real(Y0, V, nu, bt)
        return Y + 10.0 if len(calls) == 2 else Y

    monkeypatch.setattr(de, "_maximin_local", flaky)
    with pytest.warns(asub.ConvergenceWarning) as rec:
        d = asub.active_design(domain=square, n_points=4, n_starts=3, rng=0)
    assert [w.message.code for w in rec] == ["convergence.failed_starts"]
    assert rec[0].message.details == {"failed": 1, "n_starts": 3}
    assert _facet_slack(square, d.points) <= 0.0


def test_maximin_pull_in(square, monkeypatch):
    """A usable design outside by round-off is pulled into the domain radially (§5.3 step 6)."""

    def outside(Y0, V, nu, bt):
        Y = Y0.copy()
        Y[0] = V[0] * (1 + 5e-10)
        return Y

    monkeypatch.setattr(de, "_maximin_local", outside)
    d = asub.active_design(domain=square, n_points=3, n_starts=1, rng=0)
    assert _facet_slack(square, d.points) <= 0.0
    np.testing.assert_allclose(d.points[0], square.vertices[0], rtol=1e-12)


def test_rejection_exhausted(monkeypatch, square):
    monkeypatch.setattr(de, "_MAX_BATCHES", 0)
    with pytest.raises(asub.SolverError) as ei:
        asub.active_design(domain=square, n_points=2, rng=0)
    assert ei.value.code == "solver.failed"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_active_design_validation(square):
    with pytest.raises(asub.InputError) as ei:
        asub.active_design(domain=square.subspace, n_points=3)
    assert ei.value.code == "input.parameter"
    for bad in (0, 2.5, True):
        with pytest.raises(asub.InputError) as ei:
            asub.active_design(domain=square, n_points=bad)
        assert ei.value.code == "input.parameter"
    with pytest.raises(asub.InputError):
        asub.active_design(domain=square, n_points=3, n_starts=0)
    with pytest.raises(asub.InputError):
        asub.active_design(domain=square, n_points=3, rng=-1)


def test_unsupported_kind(square):
    weird = square.from_dict(square.to_dict())
    object.__setattr__(weird, "kind", "mixed")
    with pytest.raises(asub.UnsupportedError) as ei:
        asub.active_design(domain=weird, n_points=2)
    assert ei.value.code == "unsupported.space"


# ---------------------------------------------------------------------------
# Conditional samples
# ---------------------------------------------------------------------------


def _model(space):
    return asub.Model(space=space, value=lambda X: np.sin(X[:, 0]) + X[:, -1] ** 2, name="toy")


@pytest.mark.parametrize("which", ["interval", "zono", "gauss"])
def test_conditional_samples_layout(which, request, seed):
    dom = request.getfixturevalue(which)
    sp = dom.subspace.space
    cs = asub.conditional_samples(domain=dom, model=_model(sp), n_design=4, n_mc=3, rng=seed)
    n_design = cs.Y.shape[0] - cs.is_vertex.sum()
    assert n_design == 4
    assert cs.is_vertex.sum() == (0 if which == "gauss" else dom.n_vertices)
    assert np.all(np.diff(cs.index) >= 0)
    np.testing.assert_array_equal(np.bincount(cs.index), cs.counts)
    # X normalized; model values at the physical points
    Xp = sp.from_normalized(cs.X)
    if isinstance(sp, asub.Uniform):
        Xp = np.clip(Xp, sp.lower, sp.upper)
    np.testing.assert_allclose(cs.f_values, np.sin(Xp[:, 0]) + Xp[:, -1] ** 2, rtol=1e-14)
    np.testing.assert_allclose(
        cs.X[cs.index < n_design] @ dom.subspace.W1, cs.Y[cs.index[cs.index < n_design]], atol=1e-9
    )
    for k in range(cs.Y.shape[0]):
        fk = cs.f_values[cs.index == k]
        assert cs.means[k] == pytest.approx(fk.mean(), rel=1e-12)
        assert cs.variances[k] == pytest.approx(
            fk.var(ddof=1) if fk.size > 1 else 0.0, rel=1e-10, abs=1e-15
        )
    assert cs.provenance.parameters["model"] == "toy"
    assert cs.provenance.parameters["batch_size"] is None


def test_conditional_samples_determinism_and_executor(zono, seed):
    model = _model(zono.subspace.space)
    a = asub.conditional_samples(domain=zono, model=model, n_design=3, n_mc=4, rng=seed)
    b = asub.conditional_samples(
        domain=zono, model=model, n_design=3, n_mc=4, rng=np.random.SeedSequence(seed)
    )
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        c = asub.conditional_samples(
            domain=zono, model=model, n_design=3, n_mc=4, rng=seed, executor=ex, batch_size=5
        )
    assert a == b == c
    assert c.provenance.parameters["batch_size"] == 5
    d = asub.conditional_samples(
        domain=zono, model=model, n_design=3, n_mc=4, rng=_random.replay(a)
    )
    assert d == a


def test_conditional_samples_given_design(zono):
    model = _model(zono.subspace.space)
    design = asub.active_design(domain=zono, n_points=3, rng=1)
    a = asub.conditional_samples(
        domain=zono, model=model, design=design, n_mc=2, include_vertices=False, rng=0
    )
    b = asub.conditional_samples(
        domain=zono,
        model=model,
        design=np.array(design.points),
        n_mc=2,
        include_vertices=False,
        rng=0,
    )
    assert a == b
    assert not a.is_vertex.any()
    np.testing.assert_array_equal(a.Y, design.points)
    # a bare callable is evaluated in domain.subspace.space
    c = asub.conditional_samples(
        domain=zono, model=model.value, design=design, n_mc=2, include_vertices=False, rng=0
    )
    np.testing.assert_array_equal(c.f_values, a.f_values)


def test_conditional_samples_validation(zono, square, interval):
    model = _model(zono.subspace.space)
    with pytest.raises(asub.InputError) as ei:
        asub.conditional_samples(domain=zono, model=model, design=np.array([[100.0, 0.0]]), rng=0)
    assert ei.value.code == "input.out_of_domain"
    with pytest.raises(asub.InputError) as ei:
        asub.conditional_samples(domain=zono, model=model, design=np.zeros((1, 3)), rng=0)
    assert ei.value.code == "input.shape"
    with pytest.raises(asub.InputError) as ei:
        asub.conditional_samples(domain=zono, model=model, design=np.array([[np.nan, 0.0]]), rng=0)
    assert ei.value.code == "input.nonfinite"
    other = asub.active_design(domain=square, n_points=1, rng=0)
    with pytest.raises(asub.InputError) as ei:
        asub.conditional_samples(domain=zono, model=model, design=other, rng=0)
    assert ei.value.details["reason"] == "domain_mismatch"
    with pytest.raises(asub.InputError) as ei:
        asub.conditional_samples(
            domain=zono, model=model, design=np.zeros((1, 2)), n_design=3, rng=0
        )
    assert ei.value.code == "input.parameter"
    for bad in (1, 0, 2.0, True):
        with pytest.raises(asub.InputError) as ei:
            asub.conditional_samples(domain=zono, model=model, n_mc=bad, rng=0)
        assert ei.value.code == "input.parameter"
    with pytest.raises(asub.InputError) as ei:
        asub.conditional_samples(domain=zono, model=_model(_ref(5)), rng=0)
    assert ei.value.details["reason"] == "space_mismatch"
    with pytest.raises(asub.InputError) as ei:
        asub.conditional_samples(domain=zono, model=model, qoi="g", rng=0)
    assert ei.value.details["reason"] == "qoi_mismatch"
    with pytest.raises(asub.InputError):
        asub.conditional_samples(domain=zono, model=model, include_vertices=1, rng=0)
    with pytest.raises(asub.InputError):
        asub.conditional_samples(domain=zono, model=model, batch_size=0, rng=0)
    # 1-D design accepted when n == 1
    cs = asub.conditional_samples(
        domain=interval, model=_model(interval.subspace.space), design=[0.1, 0.2], n_mc=2, rng=0
    )
    assert cs.Y.shape[0] == 2 + 2


def test_conditional_samples_model_error(zono):
    def bad(X):
        raise RuntimeError("boom")

    with pytest.raises(asub.ModelError):
        asub.conditional_samples(domain=zono, model=bad, design=np.zeros((1, 2)), n_mc=2, rng=0)


# ---------------------------------------------------------------------------
# V09.15 round trips and immutability
# ---------------------------------------------------------------------------


def test_round_trip(zono, gauss, assert_roundtrip, assert_readonly):
    d = asub.active_design(domain=zono, n_points=3, rng=0)
    assert_roundtrip(d)
    assert_readonly(d)
    g = asub.active_design(domain=gauss, n_points=5)
    assert_roundtrip(g)
    cs = asub.conditional_samples(
        domain=zono, model=_model(zono.subspace.space), design=d, n_mc=2, rng=0
    )
    assert_roundtrip(cs)
    assert_readonly(cs)
    assert cs.n_samples == cs.X.shape[0]


# ---------------------------------------------------------------------------
# D-170 greedy farthest-point above 2000 pair constraints
# ---------------------------------------------------------------------------


def _zonotope(m, n, seed):
    sp = asub.Uniform(-np.ones(m), 2 * np.ones(m))
    return asub.active_domain(
        asub.subspace_from_basis(np.random.default_rng(seed).normal(size=(m, n)), space=sp)
    )


def test_method_switch(zono, monkeypatch):
    assert asub.active_design(domain=zono, n_points=8, rng=0).method == "maximin_slsqp"
    # 10 facets: 8 points give 28 pairs + 80 facet rows = 108 constraints, 7 points give 91
    monkeypatch.setattr(de, "_SLSQP_MAX_CONSTRAINTS", 107)
    d = asub.active_design(domain=zono, n_points=8, rng=0)
    assert d.method == "maximin_greedy"
    assert asub.active_design(domain=zono, n_points=7, rng=0).method == "maximin_slsqp"


@pytest.mark.parametrize(("m", "n", "N"), [(6, 3, 125), (6, 4, 625)])
def test_greedy_default_sizes(m, n, N, seed):
    import time

    dom = _zonotope(m, n, 5)
    t0 = time.perf_counter()
    d = asub.active_design(domain=dom, n_points=N, rng=seed)
    assert time.perf_counter() - t0 < 10.0
    assert d.method == "maximin_greedy"
    assert d.points.shape == (N, n)
    assert _facet_slack(dom, d.points) <= 0.0
    dpp, dpv = de._sq_dists(d.points, dom.vertices)
    assert d.min_distance == pytest.approx(np.sqrt(min(dpp.min(), dpv.min())), rel=1e-14)
    assert d.min_distance > 0
    again = asub.active_design(domain=dom, n_points=N, rng=seed)
    assert np.array_equal(d.points, again.points)


def test_greedy_versus_slsqp(zono, seed):
    """Greedy farthest-point is a 2-approximation of the maximin; check at least half of
    SLSQP's distance.
    """
    for N in (5, 12, 20):
        s = asub.active_design(domain=zono, n_points=N, rng=seed)
        g, dmin = de._maximin_greedy(zono, N, np.random.default_rng(seed))
        assert dmin >= 0.5 * s.min_distance
        assert _facet_slack(zono, g) <= 0.0


def test_greedy_round_trip(assert_roundtrip, monkeypatch, zono):
    monkeypatch.setattr(de, "_SLSQP_MAX_CONSTRAINTS", 0)
    d = asub.active_design(domain=zono, n_points=4, rng=0)
    assert d.method == "maximin_greedy"
    assert_roundtrip(d)
