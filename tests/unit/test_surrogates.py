"""Contracts of spec 09 surrogates (V09.13-V09.15, validation, protocol, serialization)."""

from __future__ import annotations

import concurrent.futures

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import _random


def _ref(m):
    return asub.Uniform(-np.ones(m), np.ones(m))


@pytest.fixture(scope="module")
def problem():
    gen = np.random.default_rng(0)
    m = 4
    lower = gen.uniform(-3, 3, m)
    sp = asub.Uniform(lower, lower + gen.uniform(0.5, 4, m))
    X = sp.sample(30, rng=1)
    x = sp.to_normalized(X)
    G = sp.grads_to_physical(
        np.c_[2 * x[:, 0] + x[:, 1], x[:, 0] + 0.2, 0.1 * np.ones(30), 0.05 * x[:, 3]]
    )
    f = x[:, 0] ** 2 + x[:, 0] * x[:, 1] + 0.2 * x[:, 1] + 0.1 * x[:, 2] + 0.025 * x[:, 3] ** 2
    sub = asub.active_subspace(grads=G, space=sp, n=2)
    return sp, sub, X, f


# ---------------------------------------------------------------------------
# Validation (§5.1 step 1, §6)
# ---------------------------------------------------------------------------


def _code(exc_info):
    return exc_info.value.code, exc_info.value.details


def test_space_rules(problem):
    sp, sub, X, f = problem
    s = asub.fit_surrogate(X=X, f=f, space=sp, subspace=sub, kind="poly")
    assert s.space == sp
    assert s.subspace == sub
    with pytest.raises(asub.InputError) as ei:
        asub.fit_surrogate(X=X, f=f, space=asub.Uniform(sp.lower, 2 * sp.upper + 10), subspace=sub)
    assert _code(ei)[0] == "input.parameter"
    assert ei.value.details["reason"] == "space_mismatch"
    with pytest.raises(asub.InputError) as ei:
        asub.fit_surrogate(X=X, f=f, subspace=None)
    assert ei.value.details["param"] == "space"
    with pytest.raises(TypeError):
        asub.fit_surrogate(X=X, f=f, space=sp)  # subspace is a required keyword
    with pytest.raises(asub.InputError) as ei:
        asub.fit_surrogate(X=X, f=f, subspace="sub")
    assert ei.value.details["param"] == "subspace"


@pytest.mark.filterwarnings(r"ignore:.*\[accuracy\.flat_objective\]")  # plateau of J [D-192]
def test_parameter_errors(problem):
    _sp, sub, X, f = problem
    cases = [
        ({"kind": "gp"}, "kind", None),
        ({"degree": -1}, "degree", None),
        ({"degree": True}, "degree", None),
        ({"degree": 2.0}, "degree", None),
        ({"qoi": "g"}, "qoi", "qoi_mismatch"),
        ({"kind": "rbf", "sample_weights": np.ones(30)}, "sample_weights", "not_for_kind"),
        ({"kind": "poly", "noise_variance": np.ones(30)}, "noise_variance", "not_for_kind"),
        ({"kind": "poly", "log10_eta": 0.0}, "log10_eta", "not_for_kind"),
        ({"kind": "poly", "log10_eta_bounds": (-3.0, 1.0)}, "log10_eta_bounds", "not_for_kind"),
        ({"log10_eta_bounds": (1.0, 1.0)}, "log10_eta_bounds", None),
        ({"log10_eta_bounds": (0.0, np.inf)}, "log10_eta_bounds", None),
        ({"log10_eta": 3.0}, "log10_eta", None),
        ({"noise_variance": -np.ones(30)}, "noise_variance", "negative"),
    ]
    for kw, param, reason in cases:
        with pytest.raises(asub.InputError) as ei:
            asub.fit_surrogate(X=X, f=f, subspace=sub, **kw)
        assert ei.value.code == "input.parameter", kw
        assert ei.value.details["param"] == param, kw
        if reason is not None:
            assert ei.value.details["reason"] == reason, kw
    # poly with the default bounds is fine
    asub.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", log10_eta_bounds=(-10, 1))


def test_array_errors(problem):
    _sp, sub, X, f = problem
    bad = [
        ({"X": X[:, :3]}, "input.shape"),
        ({"X": np.where(np.arange(30)[:, None] == 2, np.nan, X)}, "input.nonfinite"),
        ({"X": X + 100}, "input.out_of_space"),
        ({"f": f[:-1]}, "input.size_mismatch"),
        ({"f": np.c_[f, f]}, "input.vector_qoi"),
        ({"noise_variance": np.ones(29)}, "input.size_mismatch"),
        ({"noise_variance": np.ones((30, 2))}, "input.shape"),
        ({"noise_variance": np.full(30, np.inf)}, "input.nonfinite"),
    ]
    for kw, code in bad:
        args = {"X": X, "f": f, **kw}
        with pytest.raises(asub.InputError) as ei:
            asub.fit_surrogate(subspace=sub, **args)
        assert ei.value.code == code, kw
    for w, code in (
        (-np.ones(30), "input.weights"),
        (np.zeros(30), "input.weights"),
        (np.ones(3), "input.size_mismatch"),
    ):
        with pytest.raises(asub.InputError) as ei:
            asub.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", sample_weights=w)
        assert ei.value.code == code


def test_too_few_and_singular():
    sp = _ref(2)
    X = np.random.default_rng(0).uniform(-1, 1, (5, 2))
    for kind in ("poly", "rbf"):
        with pytest.raises(asub.InputError) as ei:
            asub.fit_surrogate(X=X, f=X[:, 0], space=sp, subspace=None, kind=kind, degree=2)
        assert ei.value.code == "input.too_few_samples"
        assert ei.value.details == {"required": 6, "got": 5}
    w = np.ones(8)
    w[:3] = 0
    X8 = np.random.default_rng(1).uniform(-1, 1, (8, 2))
    with pytest.raises(asub.InputError) as ei:
        asub.fit_surrogate(
            X=X8, f=X8[:, 0], space=sp, subspace=None, kind="poly", degree=2, sample_weights=w
        )
    assert ei.value.details == {"required": 6, "got": 5}
    # n = 1 with fewer than d + 1 distinct y: rank-deficient
    Y = np.array([-0.5, -0.5, 0.5, 0.5, 0.5])[:, None]
    for kind in ("poly", "rbf"):
        with pytest.raises(asub.NumericalError) as ei:
            asub.fit_surrogate(
                X=Y, f=Y[:, 0] + 1.0, space=_ref(1), subspace=None, kind=kind, degree=2
            )
        assert ei.value.code == "numerical.singular"
        assert ei.value.details == {"rank": 2, "p": 3}


def test_evaluate_coordinates(problem):
    _sp, sub, X, f = problem
    s = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="poly")
    with pytest.raises(asub.InputError) as ei:
        s.evaluate(np.zeros((3, 3)))
    assert ei.value.code == "input.shape"
    with pytest.raises(asub.InputError) as ei:
        s.gradient(np.full((2, 2), np.nan))
    assert ei.value.code == "input.nonfinite"
    # 1-D accepted when k == 1 [D-64]; far points are fine (no domain check)
    s1 = asub.fit_surrogate(X=X, f=f, subspace=asub.with_dimension(sub, n=1), kind="rbf")
    assert s1.evaluate(np.array([0.1, 50.0])).shape == (2,)
    np.testing.assert_array_equal(s1.evaluate([0.1, 0.2]), s1.evaluate([[0.1], [0.2]]))
    assert s1.gradient([0.1]).shape == (1, 1)


@pytest.mark.filterwarnings(r"ignore:.*\[accuracy\.flat_objective\]")  # plateau of J [D-192]
def test_quadratic_terms_errors(problem):
    _sp, sub, X, f = problem
    s0 = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=0)
    with pytest.raises(asub.InputError) as ei:
        asub.quadratic_terms(s0)
    assert ei.value.code == "input.parameter"
    r = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="rbf")
    with pytest.raises(asub.InputError):
        asub.quadratic_terms(r)


# ---------------------------------------------------------------------------
# Special cases
# ---------------------------------------------------------------------------


def test_constant_f_rbf(problem):
    _sp, sub, X, _f = problem
    s = asub.fit_surrogate(X=X, f=np.full(30, 3.5), subspace=sub, kind="rbf")
    assert s.objective is None
    assert s.amplitude == 0.0
    assert s.training_variance == 0.0
    assert s.log10_eta == 0.0
    assert s.loo_r_squared is None
    assert np.all(s.loo_residuals == 0)
    assert np.all(s.radial_coefficients == 0)
    assert np.all(s.nugget == 1e-6)
    np.testing.assert_allclose(asub.predict(s, X[:4]), 3.5, rtol=1e-12)
    s2 = asub.fit_surrogate(X=X, f=np.full(30, 3.5), subspace=sub, kind="rbf", log10_eta=-2.0)
    assert s2.log10_eta == -2.0


def test_constant_f_poly_r2_none(problem):
    _sp, sub, X, _f = problem
    s = asub.fit_surrogate(X=X, f=np.full(30, -1.0), subspace=sub, kind="poly", degree=1)
    assert s.loo_r_squared is None
    assert s.loo_residuals is not None


def test_subspace_none_and_full_dimension(problem):
    sp, sub, X, f = problem
    s = asub.fit_surrogate(X=X, f=f, space=sp, subspace=None, kind="poly", degree=2)
    assert s.active_dim == 4
    assert s.Y.shape == (30, 4)
    assert s.qoi == "f"
    np.testing.assert_allclose(s.Y, sp.to_normalized(X), rtol=1e-15)
    np.testing.assert_allclose(asub.predict(s, X), f, atol=1e-12)
    full = asub.with_dimension(sub, n=4)
    s4 = asub.fit_surrogate(X=X, f=f, subspace=full, kind="rbf", degree=1)
    assert s4.active_dim == 4
    assert s4.length_scales.shape == (4,)


@pytest.mark.filterwarnings(r"ignore:.*\[accuracy\.flat_objective\]")  # plateau of J [D-192]
def test_deterministic_and_provenance(problem):
    _sp, sub, X, f = problem
    a = asub.fit_surrogate(X=X, f=f, subspace=sub, noise_variance=np.full(30, 1e-4))
    b = asub.fit_surrogate(X=X, f=f, subspace=sub, noise_variance=np.full(30, 1e-4))
    assert a == b
    assert np.array_equal(a.radial_coefficients, b.radial_coefficients)
    p = a.provenance
    assert p.function == "active_subspaces.fit_surrogate"
    assert p.rng is None
    assert set(p.parameters) >= {
        "X",
        "f",
        "noise_variance",
        "kind",
        "degree",
        "log10_eta_bounds",
        "qoi",
        "subspace",
    }
    assert p.parameters["X"]["sha256"] == asub.results._v.digest(X)["sha256"]


# ---------------------------------------------------------------------------
# V09.14 protocol [D-99], [D-104]
# ---------------------------------------------------------------------------


class _Linear:
    """A minimal user surrogate: g(y) = c^T y."""

    def __init__(self, sub):
        self.space = sub.space
        self.subspace = sub
        self.qoi = sub.qoi
        self.c = np.arange(1.0, sub.n + 1)

    @property
    def active_dim(self):
        return self.subspace.n

    def evaluate(self, Y):
        return np.asarray(Y) @ self.c

    def gradient(self, Y):
        return np.tile(self.c, (np.asarray(Y).shape[0], 1))


@pytest.mark.filterwarnings(r"ignore:.*\[accuracy\.flat_objective\]")  # plateau of J [D-192]
def test_protocol(problem):
    sp, sub, X, f = problem
    for s in (
        asub.fit_surrogate(X=X, f=f, subspace=sub, kind="poly"),
        asub.fit_surrogate(X=X, f=f, subspace=sub),
    ):
        assert isinstance(s, asub.Surrogate)
        assert s.active_dim == 2
        assert s.n_samples == 30
    user = _Linear(sub)
    assert isinstance(user, asub.Surrogate)
    x = sp.to_normalized(X[:5])
    np.testing.assert_allclose(asub.predict(user, X[:5]), x @ sub.W1 @ user.c, rtol=1e-13)
    G = asub.predict_gradient(user, X[:5])
    np.testing.assert_allclose(
        G, np.tile(sp.grads_to_physical((sub.W1 @ user.c)[None, :]), (5, 1)), rtol=1e-13
    )
    with pytest.raises(asub.InputError) as ei:
        asub.predict(object(), X)
    assert ei.value.details["param"] == "surrogate"
    with pytest.raises(asub.InputError) as ei:
        asub.predict(user, X + 100)
    assert ei.value.code == "input.out_of_space"


@pytest.mark.filterwarnings(r"ignore:.*\[accuracy\.flat_objective\]")  # plateau of J [D-192]
def test_protocol_reduced_model(problem):
    rmod = pytest.importorskip("active_subspaces.reduced_model")
    sp, sub, X, f = problem
    rm = rmod.build_reduced_model(X=X, f=f, space=sp, subspace=sub)
    assert isinstance(rm, asub.Surrogate)
    np.testing.assert_array_equal(asub.predict(rm, X), asub.predict(rm.surrogate, X))


# ---------------------------------------------------------------------------
# V09.13 model path [C-18], [D-48]
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def interval_domain():
    sp = asub.Uniform(np.zeros(3), np.array([1.0, 2.0, 3.0]))
    sub = asub.subspace_from_basis(np.array([[1.0], [0.5], [-0.25]]), space=sp)
    return asub.active_domain(sub)


@pytest.mark.filterwarnings(r"ignore:.*\[accuracy\.flat_objective\]")  # plateau of J [D-192]
def test_model_path_noise_variance(interval_domain, seed):
    dom = interval_domain
    sp = dom.subspace.space
    model = asub.Model(space=sp, value=lambda X: X[:, 0] + X[:, 2] ** 2, name="m1")
    s = asub.fit_surrogate_from_model(
        domain=dom, model=model, kind="rbf", n_design=5, n_mc=6, rng=seed
    )
    cs = s.samples
    assert cs.Y.shape == (7, 1)
    assert cs.is_vertex.sum() == 2
    for k in range(7):
        fk = cs.f_values[cs.index == k]
        assert cs.variances[k] == pytest.approx(
            fk.var(ddof=1) if fk.size > 1 else 0.0, rel=1e-14, abs=1e-15
        )
    np.testing.assert_allclose(s.noise_variance, cs.variances / cs.counts, rtol=1e-15)
    np.testing.assert_array_equal(s.f, cs.means)
    np.testing.assert_array_equal(s.Y, cs.Y)
    sf2 = np.mean((cs.means - cs.means.mean()) ** 2)
    np.testing.assert_allclose(s.nugget, np.maximum(s.noise_variance / sf2, 1e-6), rtol=1e-14)
    p = s.provenance
    assert p.function == "active_subspaces.fit_surrogate_from_model"
    assert p.parameters["model"] == "m1"
    assert p.rng is not None


@pytest.mark.parametrize("kind", ["poly", "rbf"])
def test_model_path_function_of_y(interval_domain, seed, kind):
    dom = interval_domain
    sp, W1 = dom.subspace.space, dom.subspace.W1

    def g(X):
        y = (sp.to_normalized(X) @ W1)[:, 0]
        return 1.0 + y - 0.5 * y**2

    s = asub.fit_surrogate_from_model(domain=dom, model=g, kind=kind, n_design=4, n_mc=3, rng=seed)
    scale = np.max(np.abs(s.f))
    assert np.max(np.abs(s.samples.variances)) <= 1e-14 * scale**2
    assert np.max(np.abs(s.residuals)) <= 1e-10 * scale
    yt = np.linspace(dom.vertices.min(), dom.vertices.max(), 7)
    np.testing.assert_allclose(
        s.evaluate(yt), 1.0 + yt - 0.5 * yt**2, rtol=1e-8, atol=1e-8 * scale
    )


@pytest.mark.filterwarnings(r"ignore:.*\[accuracy\.flat_objective\]")  # plateau of J [D-192]
def test_model_path_determinism(interval_domain, seed):
    dom = interval_domain
    model = asub.Model(space=dom.subspace.space, value=lambda X: np.cos(X).sum(axis=1))
    a = asub.fit_surrogate_from_model(domain=dom, model=model, n_design=4, n_mc=3, rng=seed)
    b = asub.fit_surrogate_from_model(domain=dom, model=model, n_design=4, n_mc=3, rng=seed)
    with concurrent.futures.ThreadPoolExecutor(3) as ex:
        c = asub.fit_surrogate_from_model(
            domain=dom, model=model, n_design=4, n_mc=3, rng=seed, executor=ex, batch_size=2
        )
    assert a == b == c
    cs = asub.conditional_samples(domain=dom, model=model, n_design=4, n_mc=3, rng=seed)
    assert cs == a.samples
    d = asub.fit_surrogate_from_model(
        domain=dom, model=model, n_design=4, n_mc=3, rng=_random.replay(a)
    )
    assert d == a


def test_model_path_errors(interval_domain):
    dom = interval_domain
    model = asub.Model(space=dom.subspace.space, value=lambda X: X[:, 0])
    with pytest.raises(asub.InputError) as ei:
        asub.fit_surrogate_from_model(domain=dom.subspace, model=model)
    assert ei.value.details["param"] == "domain"
    with pytest.raises(asub.InputError) as ei:
        asub.fit_surrogate_from_model(domain=dom, model=model, n_mc=1, rng=0)
    assert ei.value.details["param"] == "n_mc"
    with pytest.raises(asub.InputError) as ei:
        asub.fit_surrogate_from_model(
            domain=dom, model=asub.Model(space=_ref(3), value=lambda X: X[:, 0]), rng=0
        )
    assert ei.value.details["reason"] == "space_mismatch"
    with pytest.raises(asub.InputError) as ei:
        asub.fit_surrogate_from_model(domain=dom, model=model, kind="poly", log10_eta=0.0, rng=0)
    assert ei.value.details["reason"] == "not_for_kind"
    with pytest.raises(asub.InputError) as ei:
        asub.fit_surrogate_from_model(
            domain=dom, model=model, n_design=1, n_mc=2, degree=4, kind="poly", rng=0
        )
    assert ei.value.code == "input.too_few_samples"


# ---------------------------------------------------------------------------
# V09.15 round trips and immutability
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings(r"ignore:.*\[accuracy\.flat_objective\]")  # plateau of J [D-192]
def test_round_trip(problem, interval_domain, assert_roundtrip, assert_readonly):
    sp, sub, X, f = problem
    results = [
        asub.fit_surrogate(
            X=X, f=f, subspace=sub, kind="poly", sample_weights=np.linspace(1, 2, 30)
        ),
        asub.fit_surrogate(X=X, f=f, space=sp, subspace=None, kind="poly", degree=1),
        asub.fit_surrogate(X=X, f=f, subspace=sub, noise_variance=np.full(30, 1e-3)),
        asub.fit_surrogate(X=X, f=np.ones(30), subspace=sub),
    ]
    model = asub.Model(space=interval_domain.subspace.space, value=lambda X: X.sum(axis=1))
    results.append(
        asub.fit_surrogate_from_model(
            domain=interval_domain, model=model, n_design=4, n_mc=2, rng=0
        )
    )
    results.append(
        asub.fit_surrogate_from_model(
            domain=interval_domain, model=model, kind="poly", n_design=4, n_mc=2, rng=0
        )
    )
    for r in results:
        assert_roundtrip(r)
        assert_readonly(r)
        back = asub.from_dict(r.to_dict())
        Yt = r.Y[:3]
        np.testing.assert_array_equal(back.evaluate(Yt), r.evaluate(Yt))


def test_f_variance_validation_and_edges():
    """f_variance must be finite and > 0 and is "rbf"-only; a constant f keeps log10_eta = 0;
    a zero-width published interval fixes log10_eta [D-192].
    """
    space = asub.Gaussian.reference(2)
    X = space.sample(30, rng=0)
    G = np.column_stack([np.ones(30), 0.1 * X[:, 1]])
    sub = asub.active_subspace(grads=G, space=space, n=1)
    f = X[:, 0] + 0.1 * X[:, 1] ** 2
    for bad in (0.0, -1.0, np.inf, np.nan, True):
        with pytest.raises(asub.InputError) as e:
            asub.fit_surrogate(X=X, f=f, subspace=sub, f_variance=bad)
        assert e.value.code == "input.parameter"
    with pytest.raises(asub.InputError) as e:
        asub.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", f_variance=1.0)
    assert e.value.details["reason"] == "not_for_kind"
    total = float(np.abs(sub.eigenvalues).sum())
    const = asub.fit_surrogate(X=X, f=np.full(30, 2.0), subspace=sub, f_variance=10 * total)
    assert const.log10_eta == 0.0
    point = asub.fit_surrogate(X=X, f=f, subspace=sub, f_variance=total)  # lo == hi
    assert point.log10_eta == pytest.approx(np.log10(2.0), rel=1e-15)


def test_predict_interval_validation_order():
    """not_polynomial, then the options and the degrees of freedom, then X (09 §5.5)."""
    space = asub.Uniform.reference(1)
    X = np.linspace(-1, 1, 9)[:, None]
    s = asub.fit_surrogate(X=X, f=X[:, 0] ** 2, space=space, subspace=None, kind="poly", degree=1)
    with pytest.raises(asub.InputError) as e:
        asub.predict_interval(s, np.zeros((2, 5)), level=2.0)
    assert e.value.code == "input.parameter"
