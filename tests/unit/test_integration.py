"""Unit tests of active_subspaces.integration (spec 10 §4.2, §5.1-§5.2, §6; spec 17 V10.4,
V10.6, V10.12).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import integration as I
from active_subspaces.errors import AccuracyWarning, InputError, NumericalError


def _sub(m, n, seed=0, space=None):
    rng = np.random.default_rng(seed)
    space = asub.Uniform.reference(m) if space is None else space
    return asub.subspace_from_basis(rng.standard_normal((m, n)), space=space)


def _dom(m, n, seed=0, space=None):
    return asub.active_domain(_sub(m, n, seed, space))


def _square():
    """m = 3, W1 = [e1 e2]: the active domain is [-1, 1]^2."""
    sub = asub.subspace_from_basis(np.eye(3)[:, :2], space=asub.Uniform.reference(3))
    return asub.active_domain(sub)


def _code(excinfo):
    return excinfo.value.code


def _short_chains():
    """Hit-and-run chains with n_mc < 20 warn accuracy.correlated_draws (D-114, amended
    2026-10-29).
    """
    return pytest.warns(AccuracyWarning, match=r"accuracy\.correlated_draws")


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_all_is_sorted_and_reexported():
    assert I.__all__ == tuple(sorted(I.__all__))
    for name in I.__all__:
        assert getattr(asub, name) is getattr(I, name)
    assert "integration" in asub.__all__


# ---------------------------------------------------------------------------
# Rules (§5.1)
# ---------------------------------------------------------------------------


def test_gaussian_rule_size():
    """V10.4: N = 8, n = 3 gives k = 2 (not 1) and N_q = 8."""
    dom = _dom(5, 3, space=asub.Gaussian.reference(5))
    rule = I.active_quadrature_rule(dom, n_nodes=8)
    assert rule.kind == "gauss_hermite"
    assert rule.nodes.shape == (8, 3)
    assert rule.degree == 3
    assert rule.n_requested == (8,)
    assert rule.provenance.rng is None
    assert abs(rule.weights.sum() - 1) <= 1e-14


def test_gaussian_rule_per_dimension():
    dom = _dom(4, 2, space=asub.Gaussian.reference(4))
    rule = I.active_quadrature_rule(dom, n_nodes=[2, 3])
    assert rule.nodes.shape == (6, 2)
    assert rule.degree == 3
    assert rule.n_requested == (2, 3)
    assert np.unique(rule.nodes[:, 1]).size == 3  # C order: last coordinate fastest
    assert np.array_equal(rule.nodes[:3, 0], np.full(3, rule.nodes[0, 0]))


@pytest.mark.parametrize(
    ("N", "n", "k"), [(1, 2, 1), (26, 3, 2), (27, 3, 3), (1000, 3, 10), (999, 3, 9), (64, 6, 2)]
)
def test_integer_root_exact(N, n, k):
    assert I._int_root(N, n) == k


def test_interval_rule():
    dom = _dom(4, 1)
    rule = I.active_quadrature_rule(dom, n_nodes=5)
    assert rule.kind == "box_spline_gauss"
    assert rule.degree == 9
    assert rule.nodes.shape == (5, 1)
    t = rule.nodes[:, 0]
    assert np.all(np.diff(t) > 0)
    assert t[2] == 0.0
    h = float(dom.vertices.max())
    assert np.all(np.abs(t) < h)
    assert np.all(rule.weights > 0)
    assert abs(rule.weights.sum() - 1) <= 1e-14
    assert rule.design is None
    assert rule.n_weight_samples is None
    assert rule.weight_deficit == 0.0


def test_zonotope_rule_basic(seed):
    dom = _dom(5, 2)
    rule = I.active_quadrature_rule(dom, n_nodes=4, n_weight_samples=20_000, rng=seed)
    assert rule.kind == "delaunay_centroid"
    assert rule.degree is None
    assert rule.design.shape == (4, 2)
    assert rule.n_weight_samples == 20_000
    assert rule.weight_deficit == 0.0
    assert np.all(rule.weights > 0)
    assert abs(rule.weights.sum() - 1) <= 1e-12
    assert np.all(asub.in_active_domain(dom, Y=rule.nodes))
    # weights are cell counts / N_X
    counts = rule.weights * 20_000
    assert np.allclose(counts, np.round(counts), atol=1e-8)


def test_zonotope_no_design_points(seed):
    rule = I.active_quadrature_rule(_square(), n_nodes=0, n_weight_samples=1000, rng=seed)
    assert rule.design.shape == (0, 2)
    assert rule.nodes.shape == (2, 2)  # two triangles of the square


def test_weight_deficit(seed):
    """V10.6: one vertex removed gives accuracy.weight_deficit, d > 0, and weights that still
    sum to 1.
    """
    dom = _square()
    cut = dataclasses.replace(
        dom, vertices=dom.vertices[1:], vertex_preimages=dom.vertex_preimages[1:]
    )
    with pytest.warns(AccuracyWarning) as rec:
        rule = I.active_quadrature_rule(cut, n_nodes=0, n_weight_samples=20_000, rng=seed)
    codes = [w.message.code for w in rec]
    assert codes == ["accuracy.weight_deficit"]
    assert (
        0.45 < rule.weight_deficit < 0.55
    )  # the hull is a triangle with half the area of the square
    assert rec[0].message.details["deficit"] == rule.weight_deficit
    assert abs(rule.weights.sum() - 1) <= 1e-12


def test_zonotope_rule_stream_layout(seed):
    """§5.7: s_design, s_weights = gen.spawn(2); the N_X draws come from s_weights in order
    (chunking invisible).
    """
    dom = _dom(4, 2)
    rule = I.active_quadrature_rule(dom, n_nodes=0, n_weight_samples=70_000, rng=seed)
    gen = np.random.default_rng(np.random.SeedSequence(seed))
    _, s_weights = gen.spawn(2)
    Y = s_weights.uniform(-1, 1, size=(70_000, 4)) @ dom.subspace.W1
    import scipy.spatial

    T = scipy.spatial.Delaunay(np.asarray(dom.vertices))
    counts = np.bincount(T.find_simplex(Y), minlength=T.nsimplex)
    keep = counts > 0
    assert np.array_equal(rule.weights, counts[keep] / 70_000)


def test_rule_determinism(seed):
    dom = _dom(5, 2)
    a = I.active_quadrature_rule(dom, n_nodes=3, n_weight_samples=5000, rng=seed)
    b = I.active_quadrature_rule(dom, n_nodes=3, n_weight_samples=5000, rng=seed)
    c = I.active_quadrature_rule(dom, n_nodes=3, n_weight_samples=5000, rng=seed + 1)
    assert a == b
    assert a != c


@pytest.mark.parametrize(
    ("kwargs", "space"),
    [
        ({"n_nodes": [2]}, "uniform1"),  # sequence for a uniform domain
        ({"n_nodes": 0}, "uniform1"),
        ({"n_nodes": -1}, "uniform2"),
        ({"n_nodes": 2.5}, "uniform1"),
        ({"n_nodes": True}, "uniform1"),
        ({"n_nodes": [2, 2, 2]}, "gaussian2"),
        ({"n_nodes": [2, 0]}, "gaussian2"),
        ({"n_nodes": 0}, "gaussian2"),
        ({"n_nodes": 2, "n_weight_samples": 0}, "uniform2"),
    ],
)
def test_rule_bad_arguments(kwargs, space):
    dom = {
        "uniform1": _dom(3, 1),
        "uniform2": _dom(3, 2),
        "gaussian2": _dom(3, 2, space=asub.Gaussian.reference(3)),
    }[space]
    with pytest.raises(InputError) as e:
        I.active_quadrature_rule(dom, **kwargs)
    assert _code(e) == "input.parameter"


def test_rule_bad_domain():
    with pytest.raises(InputError) as e:
        I.active_quadrature_rule(_sub(3, 1), n_nodes=2)
    assert _code(e) == "input.parameter"


def test_rule_roundtrip(assert_roundtrip, assert_readonly, seed):
    for rule in (
        I.active_quadrature_rule(_dom(4, 2), n_nodes=2, n_weight_samples=2000, rng=seed),
        I.active_quadrature_rule(_square(), n_nodes=0, n_weight_samples=500, rng=seed),
        I.active_quadrature_rule(_dom(4, 1), n_nodes=3),
        I.active_quadrature_rule(_dom(4, 2, space=asub.Gaussian.reference(4)), n_nodes=[2, 3]),
    ):
        assert_roundtrip(rule)
        assert_readonly(rule)


# ---------------------------------------------------------------------------
# Designs (§5.2)
# ---------------------------------------------------------------------------


def test_design_shapes_and_weights(seed):
    dom = _dom(4, 1)
    d = I.integration_design(dom, n_nodes=3, n_mc=4, n_replicates=2, rng=seed)
    assert d.nodes.shape == (6, 1)
    assert d.X.shape == (24, 4)
    assert np.array_equal(d.node_replicate, [0, 0, 0, 1, 1, 1])
    assert np.array_equal(d.point_node, np.repeat(np.arange(6), 4))
    assert abs(d.point_weights.sum() - 1) <= 1e-12
    assert d.inactive_sampler == "hit_and_run"
    assert d.burn_in == 30
    assert d.thin == 3
    assert d.n_weight_samples is None
    assert np.allclose(d.X @ dom.subspace.W1, d.nodes[d.point_node], atol=1e-12)
    phys = d.X_physical
    assert phys.shape == (24, 4)
    assert np.all(np.abs(phys) <= 1)


def test_design_rule_only(seed):
    d = I.integration_design(_dom(3, 1), n_nodes=3, n_mc=0, rng=seed)
    assert d.X.shape == (0, 3)
    assert d.inactive_sampler == "none"
    assert d.burn_in is None
    assert d.point_weights.shape == (0,)


def test_design_stream_layout(seed):
    """§5.7: reps = gen.spawn(R); s_rule, s_inactive = reps[r].spawn(2) for every r and every
    domain kind.
    """
    dom = _dom(5, 2)
    R = 2
    d = I.integration_design(
        dom, n_nodes=2, n_mc=3, n_replicates=R, n_weight_samples=3000, rng=seed
    )
    gen = np.random.default_rng(np.random.SeedSequence(seed))
    for r, rep in enumerate(gen.spawn(R)):
        s_rule, s_inactive = rep.spawn(2)
        rule = I.active_quadrature_rule(dom, n_nodes=2, n_weight_samples=3000, rng=s_rule)
        block = d.node_replicate == r
        assert np.array_equal(d.nodes[block], rule.nodes)
        assert np.array_equal(d.node_weights[block], rule.weights)
        inv = asub.inverse_map(dom.subspace, Y=rule.nodes, n_draws=3, rng=s_inactive)
        rows = np.repeat(block, 3)
        assert np.array_equal(d.X[rows], inv.X)


def test_design_gaussian_sampler(seed):
    dom = _dom(3, 1, space=asub.Gaussian.reference(3))
    d = I.integration_design(dom, n_nodes=4, n_mc=2, rng=seed)
    assert d.inactive_sampler == "gaussian"
    assert d.burn_in is None
    assert d.thin is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_mc": -1},
        {"n_replicates": 0},
        {"burn_in": -1},
        {"thin": 0},
        {"backend": "gurobi"},
        {"rng": -3},
        {"n_weight_samples": 0},
    ],
)
def test_design_bad_arguments(kwargs):
    with pytest.raises(InputError) as e:
        I.integration_design(_dom(3, 1), n_nodes=2, **kwargs)
    assert _code(e) == "input.parameter"


def test_design_roundtrip(assert_roundtrip, assert_readonly, seed):
    d = I.integration_design(
        _dom(4, 2), n_nodes=2, n_mc=2, n_replicates=2, n_weight_samples=1000, rng=seed
    )
    assert_roundtrip(d)
    assert_readonly(d)


# ---------------------------------------------------------------------------
# Estimates (§3.2, §5.2)
# ---------------------------------------------------------------------------


def _ridge_model(dom):
    a = np.asarray(dom.subspace.W1)[:, 0]
    return lambda X: (X @ a) ** 2 + 0.1 * X.sum(axis=1)


def test_integrate_equals_design_path(seed):
    """V10.12: integrate == estimate_integral(integration_design(...), f=model(X_physical)),
    bitwise.
    """
    dom = _dom(4, 1)
    model = _ridge_model(dom)
    with _short_chains():
        res = I.integrate(model, domain=dom, n_nodes=3, n_mc=5, n_replicates=2, rng=seed)
    design = I.integration_design(dom, n_nodes=3, n_mc=5, n_replicates=2, rng=seed)
    with _short_chains():
        ref = I.estimate_integral(design, f=model(design.X_physical))
    assert res == ref
    assert res.provenance.function == "active_subspaces.integrate"
    assert res.provenance.parameters["model"] is None
    assert res.provenance.parameters["batch_size"] is None
    assert res.provenance.rng == design.provenance.rng
    assert abs(res.design.point_weights.sum() - 1) <= 1e-12
    assert res.n_evaluations == 30


def test_estimate_fields(seed):
    dom = _dom(2, 1)
    res = I.integrate(_ridge_model(dom), domain=dom, n_nodes=3, n_mc=4, rng=seed)
    d = res.design
    F = res.f.reshape(3, 4)
    assert np.array_equal(res.node_means, F.mean(axis=1))
    assert np.allclose(res.node_variances, F.var(axis=1, ddof=1), rtol=1e-14)
    assert res.estimate == pytest.approx(float(np.sum(d.node_weights * res.node_means)), rel=1e-15)
    assert res.total_std_error is None
    assert d.inactive_sampler == "exact_interval"
    assert res.autocorrelation_time == 1.0
    expected = np.sqrt(
        np.sum(d.node_weights**2 * res.autocorrelation_time * res.node_variances / 4)
    )
    assert res.inactive_std_error == pytest.approx(expected, rel=1e-14)


def test_single_draw_has_no_inactive_error(seed):
    dom = _dom(3, 1, space=asub.Gaussian.reference(3))
    res = I.integrate(_ridge_model(dom), domain=dom, n_nodes=3, n_mc=1, rng=seed)
    assert res.inactive_std_error is None
    assert res.autocorrelation_time is None
    assert np.all(np.isnan(res.node_variances))


def test_short_hit_and_run_chains_through_integrate(seed):
    """§5.2: 2 <= N_MC < 4 hit-and-run draws: tau = 1 and accuracy.correlated_draws."""
    dom = _dom(4, 1)
    with pytest.warns(AccuracyWarning) as rec:
        res = I.integrate(_ridge_model(dom), domain=dom, n_nodes=2, n_mc=3, rng=seed)
    assert [w.message.code for w in rec] == ["accuracy.correlated_draws"]
    assert res.autocorrelation_time == 1.0


@pytest.mark.parametrize("n_mc", [2, 5, 19, 20])
def test_short_hit_and_run_chains_warn(seed, n_mc):
    """V10.26, D-114 amended 2026-10-29: hit-and-run with 2 <= N_MC < 20 warns once with
    details["n_mc"]; 20 does not.
    """
    dom = _dom(4, 1)
    design = I.integration_design(dom, n_nodes=2, n_mc=n_mc, n_weight_samples=500, rng=seed)
    assert design.inactive_sampler == "hit_and_run"
    f = _ridge_model(dom)(design.X_physical)
    if n_mc < 20:
        with pytest.warns(AccuracyWarning) as rec:
            res = I.estimate_integral(design, f=f)
        assert [w.message.code for w in rec] == ["accuracy.correlated_draws"]
        assert rec[0].message.details == {"n_mc": n_mc}
    else:
        res = I.estimate_integral(design, f=f)
    assert (res.autocorrelation_time == 1.0) if n_mc < 4 else (res.autocorrelation_time >= 1.0)


@pytest.mark.parametrize("space", [asub.Gaussian.reference(3), asub.Uniform.reference(2)])
def test_independent_draws_never_warn_correlated(seed, space):
    """§8: a Gaussian or exact-interval design with N_MC = 5 does not warn [D-114]."""
    dom = _dom(space.dim, 1, space=space)
    design = I.integration_design(dom, n_nodes=2, n_mc=5, rng=seed)
    assert design.inactive_sampler in ("gaussian", "exact_interval")
    res = I.estimate_integral(design, f=_ridge_model(dom)(design.X_physical))
    assert res.autocorrelation_time == 1.0


def test_replicate_combination(seed):
    """D-111: mu = mean of mu_r; total = s_R / sqrt(R); inactive = sqrt(sum se_r^2) / R."""
    dom = _dom(3, 1, space=asub.Gaussian.reference(3))
    model = _ridge_model(dom)
    res = I.integrate(model, domain=dom, n_nodes=2, n_mc=3, n_replicates=3, rng=seed)
    mu_r = res.replicate_estimates
    assert res.estimate == pytest.approx(mu_r.mean(), rel=1e-15)
    assert res.total_std_error == pytest.approx(mu_r.std(ddof=1) / np.sqrt(3), rel=1e-14)
    d = res.design
    se2 = [
        np.sum(d.node_weights[b] ** 2 * res.node_variances[b] / 3)
        for b in (d.node_replicate == r for r in range(3))
    ]
    assert res.inactive_std_error == pytest.approx(np.sqrt(np.sum(se2)) / 3, rel=1e-14)


def test_estimate_errors(seed):
    dom = _dom(3, 1)
    d = I.integration_design(dom, n_nodes=2, n_mc=3, rng=seed)
    with pytest.raises(InputError) as e:
        I.estimate_integral(d, f=np.ones(5))
    assert _code(e) == "input.size_mismatch"
    with pytest.raises(InputError) as e:
        I.estimate_integral(d, f=np.full(6, np.nan))
    assert _code(e) == "input.nonfinite"
    d0 = I.integration_design(dom, n_nodes=2, n_mc=0, rng=seed)
    with pytest.raises(InputError) as e:
        I.estimate_integral(d0, f=np.ones(1))
    assert _code(e) == "input.parameter"
    with pytest.raises(InputError) as e:
        I.estimate_integral("design", f=np.ones(1))
    assert _code(e) == "input.parameter"
    with pytest.warns(RuntimeWarning), pytest.raises(NumericalError) as e:
        I.estimate_integral(d, f=np.full(6, 1e308))
    assert _code(e) == "numerical.nonfinite"


def test_integrate_column_values(seed):
    dom = _dom(2, 1)
    model = _ridge_model(dom)
    a = I.integrate(model, domain=dom, n_nodes=2, n_mc=3, rng=seed)
    b = I.integrate(lambda X: model(X)[:, None], domain=dom, n_nodes=2, n_mc=3, rng=seed)
    assert a == b


def test_integrate_model_checks(seed):
    dom = _dom(3, 1)
    other = asub.Model(space=asub.Uniform(-2 * np.ones(3), np.ones(3)), value=lambda X: X[:, 0])
    with pytest.raises(InputError) as e:
        I.integrate(other, domain=dom, n_nodes=2)
    assert _code(e) == "input.parameter"
    assert e.value.details["reason"] == "space_mismatch"
    with pytest.raises(InputError) as e:
        I.integrate(lambda X: X[:, 0], domain=dom, n_nodes=2, n_mc=0)
    assert _code(e) == "input.parameter"
    with pytest.raises(InputError) as e:
        I.integrate(lambda X: X[:, 0], domain=dom, n_nodes=2, batch_size=0)
    assert _code(e) == "input.parameter"
    with pytest.raises(InputError):
        I.integrate(3.0, domain=dom, n_nodes=2)


def test_integrate_named_model(seed):
    dom = _dom(2, 1)
    model = asub.Model(space=dom.subspace.space, value=lambda X: X[:, 0] ** 2, name="square")
    res = I.integrate(model, domain=dom, n_nodes=2, n_mc=2, batch_size=3, rng=seed)
    assert res.provenance.parameters["model"] == "square"
    assert res.provenance.parameters["batch_size"] == 3


def test_integrate_physical_space(uniform_space, seed):
    """The model sees physical points; the estimate equals the normalized-coordinate one."""
    space = uniform_space(3)
    sub = _sub(3, 1, space=space)  # m - n = 2: hit-and-run, n_mc = 4
    dom = asub.active_domain(sub)
    a = np.asarray(sub.W1)[:, 0]

    def model(Xp):
        assert np.all(Xp >= space.lower)
        assert np.all(Xp <= space.upper)
        return (space.to_normalized(Xp) @ a) ** 2

    with _short_chains():
        res = I.integrate(model, domain=dom, n_nodes=2, n_mc=4, rng=seed)
    assert res.estimate == pytest.approx(1 / 3, rel=1e-12)


def test_integrate_determinism(seed):
    dom = _dom(4, 2)
    model = _ridge_model(dom)
    kw = {"domain": dom, "n_nodes": 2, "n_mc": 4, "n_weight_samples": 2000}
    with _short_chains():
        assert I.integrate(model, rng=seed, **kw) == I.integrate(model, rng=seed, **kw)
        assert I.integrate(model, rng=seed, **kw) != I.integrate(model, rng=seed + 1, **kw)


def test_result_roundtrip(assert_roundtrip, assert_readonly, seed):
    dom = _dom(4, 1)
    with _short_chains():
        res = I.integrate(
            _ridge_model(dom), domain=dom, n_nodes=3, n_mc=4, n_replicates=2, rng=seed
        )
    assert_roundtrip(res)
    assert_readonly(res)


# ---------------------------------------------------------------------------
# Geyer autocorrelation time (§3.2, D-114)
# ---------------------------------------------------------------------------


def test_geyer_tau_ar1():
    """AR(1) with phi = 0.5 has integrated autocorrelation time (1 + phi) / (1 - phi) = 3."""
    gen = np.random.default_rng(0)
    C, N, phi = 200, 400, 0.5
    F = np.empty((C, N))
    F[:, 0] = gen.standard_normal(C) / np.sqrt(1 - phi**2)
    for j in range(1, N):
        F[:, j] = phi * F[:, j - 1] + gen.standard_normal(C)
    assert I._geyer_tau(F) == pytest.approx(3.0, rel=0.1)


def test_geyer_tau_limits():
    gen = np.random.default_rng(1)
    assert I._geyer_tau(np.ones((5, 10))) == 1.0
    assert I._geyer_tau(gen.standard_normal((500, 20))) == pytest.approx(1.0, abs=0.15)
    walk = np.cumsum(gen.standard_normal((5, 8)), axis=1)
    assert 1.0 <= I._geyer_tau(walk) <= 8.0
    alternating = np.tile([1.0, -1.0], (3, 4))
    assert I._geyer_tau(alternating) == 1.0


# ---------------------------------------------------------------------------
# Surrogates (§5.2)
# ---------------------------------------------------------------------------


def test_integrate_surrogate_callable(seed):
    dom = _dom(3, 1)
    res = I.integrate_surrogate(lambda Y: Y[:, 0] ** 2, domain=dom, n_nodes=2)
    assert res.estimate == pytest.approx(
        1 / 3, rel=1e-13
    )  # E[(w^T x)^2] = ||w||^2 / 3 with ||w|| = 1
    assert res.source == "surrogate"
    assert res.surrogate is None
    assert res.f.shape == (0,)
    assert res.inactive_std_error is None
    assert res.autocorrelation_time is None
    assert res.total_std_error is None
    assert res.design.n_mc == 0
    assert res.n_evaluations == 0
    assert np.all(np.isnan(res.node_variances))


def test_integrate_surrogate_object(assert_roundtrip, seed):
    dom = _dom(4, 1)
    sub = dom.subspace
    X = np.random.default_rng(seed).uniform(-1, 1, (40, 4))
    w = np.asarray(sub.W1)[:, 0]
    s = asub.fit_surrogate(X=X, f=(X @ w) ** 2, subspace=sub, kind="poly", degree=2)
    res = I.integrate_surrogate(s, domain=dom, n_nodes=3, n_replicates=2)
    assert res.surrogate == s
    assert res.estimate == pytest.approx(1 / 3, rel=1e-10)
    assert res.total_std_error == 0.0
    assert_roundtrip(res)


def test_integrate_surrogate_errors(seed):
    dom = _dom(4, 1)
    X = np.random.default_rng(seed).uniform(-1, 1, (40, 4))
    other = _sub(4, 1, seed=5)
    s = asub.fit_surrogate(X=X, f=X[:, 0] ** 2, subspace=other, kind="poly", degree=2)
    with pytest.raises(InputError) as e:
        I.integrate_surrogate(s, domain=dom, n_nodes=2)
    assert e.value.details["reason"] == "subspace_mismatch"
    full = asub.fit_surrogate(
        X=X, f=X[:, 0] ** 2, space=dom.subspace.space, subspace=None, kind="poly", degree=1
    )
    with pytest.raises(InputError) as e:
        I.integrate_surrogate(full, domain=dom, n_nodes=2)
    assert e.value.details["reason"] == "subspace_mismatch"
    with pytest.raises(InputError) as e:
        I.integrate_surrogate(lambda Y: Y, domain=dom, n_nodes=2)
    assert _code(e) == "input.shape"
    with pytest.raises(NumericalError) as e:
        I.integrate_surrogate(lambda Y: np.full(Y.shape[0], np.inf), domain=dom, n_nodes=2)
    assert _code(e) == "numerical.nonfinite"
    with pytest.raises(InputError) as e:
        I.integrate_surrogate("g", domain=dom, n_nodes=2)
    assert _code(e) == "input.parameter"


def test_no_global_rng_and_no_files(tmp_path, monkeypatch, seed):
    monkeypatch.chdir(tmp_path)
    dom = _dom(4, 2)
    with _short_chains():
        I.integrate(
            _ridge_model(dom), domain=dom, n_nodes=1, n_mc=4, n_weight_samples=500, rng=seed
        )
    assert list(tmp_path.iterdir()) == []
