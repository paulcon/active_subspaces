"""Unit tests of active_subspaces.optimization (spec 10 §4.3, §5.3-§5.5, §6; spec 17
V10.15-V10.19, V10.23).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import optimization as O
from active_subspaces.errors import (
    AccuracyWarning,
    ConvergenceWarning,
    InputError,
    SolverError,
)
from active_subspaces.solvers import ScipyBackend, SolverOutput


def _sub(m, n, seed=0, space=None):
    rng = np.random.default_rng(seed)
    space = asub.Uniform.reference(m) if space is None else space
    return asub.subspace_from_basis(rng.standard_normal((m, n)), space=space)


def _dom(m, n, seed=0, space=None):
    return asub.active_domain(_sub(m, n, seed, space))


def _axis_dom(m, n, space):
    """W = I: y = x[:n], z = x[n:]."""
    return asub.active_domain(asub.subspace_from_basis(np.eye(m)[:, :n], space=space))


def _data(m, f, seed=1, M=60, gaussian=False):
    gen = np.random.default_rng(seed)
    X = gen.standard_normal((M, m)) if gaussian else gen.uniform(-1, 1, (M, m))
    return X, f(X)


def _code(excinfo):
    return excinfo.value.code


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_all_is_sorted_and_reexported():
    assert O.__all__ == tuple(sorted(O.__all__))
    for name in O.__all__:
        assert getattr(asub, name) is getattr(O, name)
    assert "optimization" in asub.__all__


# ---------------------------------------------------------------------------
# minimize_active (§5.3)
# ---------------------------------------------------------------------------


def _bowl(center):
    c = np.asarray(center, dtype=float)
    return (lambda Y: np.sum((Y - c) ** 2, axis=1)), (lambda Y: 2 * (Y - c))


def test_interval_brent():
    dom = _dom(4, 1)
    g, _ = _bowl([0.3])
    res = O.minimize_active(
        g, domain=dom, rng=-1
    )  # rng is not used (or normalized) for an interval
    assert res.method == "bounded_brent"
    assert res.y_star == pytest.approx([0.3], abs=1e-8)
    assert res.starts.shape == (0, 1)
    assert res.local_minima.shape == (1, 1)
    assert res.run_status == ("converged",)
    assert res.best_run == 0
    assert res.provenance.rng is None


def test_interval_boundary_minimum():
    dom = _dom(3, 1)
    h = float(dom.vertices.max())
    res = O.minimize_active(lambda Y: Y[:, 0], domain=dom)
    assert res.y_star[0] == pytest.approx(
        -h, abs=1e-6
    )  # bounded Brent never evaluates the end point itself


def test_zonotope_slsqp(seed):
    dom = _dom(5, 2)
    g, dg = _bowl([0.2, -0.1])
    res = O.minimize_active(g, domain=dom, grad=dg, n_starts=4, rng=seed)
    assert res.method == "slsqp"
    assert res.starts.shape == (4, 2)
    assert np.allclose(res.y_star, [0.2, -0.1], atol=1e-6)
    assert all(s == "converged" for s in res.run_status)
    assert res.value == pytest.approx(0.0, abs=1e-10)


def test_starts_stream_layout(seed):
    """§5.7: start j is W1^T x0 with x0 drawn from child j of rng (uniform or normal)."""
    for space, draw in (
        (asub.Uniform.reference(5), "uniform"),
        (asub.Gaussian.reference(5), "normal"),
    ):
        dom = _dom(5, 2, space=space)
        g, dg = _bowl([0.1, 0.1])
        res = O.minimize_active(g, domain=dom, grad=dg, n_starts=3, rng=seed)
        kids = np.random.default_rng(np.random.SeedSequence(seed)).spawn(3)
        x0 = [k.uniform(-1, 1, 5) if draw == "uniform" else k.standard_normal(5) for k in kids]
        assert np.array_equal(res.starts, np.array([x @ dom.subspace.W1 for x in x0]))


def test_gaussian_bfgs(seed):
    dom = _dom(4, 2, space=asub.Gaussian.reference(4))
    g, dg = _bowl([0.5, -1.0])
    res = O.minimize_active(g, domain=dom, grad=dg, n_starts=3, rng=seed)
    assert res.method == "bfgs"
    assert np.allclose(res.y_star, [0.5, -1.0], atol=1e-6)


def test_multistart_failures(seed):
    """V10.16: NaN on part of the plane warns failed_starts; NaN everywhere raises
    solver.failed; starts in the domain.
    """
    dom = _dom(5, 2)
    g0, dg0 = _bowl([-0.3, 0.0])

    def g(Y):
        return np.where(Y[:, 0] > 0.0, np.nan, g0(Y))

    with pytest.warns(ConvergenceWarning) as rec:
        res = O.minimize_active(g, domain=dom, grad=dg0, n_starts=10, max_iter=100, rng=seed)
    assert "convergence.failed_starts" in [w.message.code for w in rec]
    failed = np.array([s == "failed" for s in res.run_status])
    assert failed.any()
    assert not failed.all()
    assert np.all(np.isnan(res.local_values[failed]))
    assert np.all(np.isnan(res.local_minima[failed]))
    assert np.allclose(res.y_star, [-0.3, 0.0], atol=1e-6)
    assert np.all(asub.in_active_domain(dom, Y=res.starts))  # [C-13]
    with pytest.raises(SolverError) as e:
        O.minimize_active(
            lambda Y: np.full(Y.shape[0], np.nan), domain=dom, n_starts=3, max_iter=100, rng=seed
        )
    assert _code(e) == "solver.failed"
    assert len(e.value.details["runs"]) == 3


def test_max_iter_interval():
    """V10.17 [C-14]: max_iter = 2 on an interval warns convergence.max_iter and returns the
    best iterate.
    """
    dom = _dom(3, 1)
    g, _ = _bowl([0.123])
    with pytest.warns(ConvergenceWarning) as rec:
        res = O.minimize_active(g, domain=dom, max_iter=2)
    assert [w.message.code for w in rec] == ["convergence.max_iter"]
    assert res.run_status == ("max_iter",)
    assert np.isfinite(res.value)


def test_max_iter_multistart(seed):
    dom = _dom(4, 2, space=asub.Gaussian.reference(4))
    with pytest.warns(ConvergenceWarning) as rec:
        res = O.minimize_active(
            lambda Y: np.sum(Y**4 + Y, axis=1), domain=dom, n_starts=2, max_iter=1, rng=seed
        )
    assert "convergence.max_iter" in [w.message.code for w in rec]
    assert "max_iter" in res.run_status


def test_gaussian_extrapolation(seed):
    """V10.18 [D-112]: g = (y1 - 10)^2 + ||y||^2 has ||y*|| = 5 > r_2 = 3.72."""
    dom = _dom(4, 2, space=asub.Gaussian.reference(4))

    def g(Y):
        return (Y[:, 0] - 10) ** 2 + np.sum(Y**2, axis=1)

    def dg(Y):
        G = 2 * Y
        G[:, 0] += 2 * (Y[:, 0] - 10)
        return G

    with pytest.warns(AccuracyWarning) as rec:
        res = O.minimize_active(g, domain=dom, grad=dg, n_starts=2, rng=seed)
    assert [w.message.code for w in rec] == ["accuracy.extrapolation"]
    assert rec[0].message.details["radius"] == pytest.approx(
        np.sqrt(13.815510557964274), rel=1e-12
    )
    assert np.allclose(res.y_star, [5, 0], atol=1e-6)


def test_user_exception_propagates(seed):
    class Boom(Exception):
        pass

    def g(Y):
        raise Boom("from g")

    with pytest.raises(Boom):
        O.minimize_active(g, domain=_dom(4, 2), n_starts=2, rng=seed)
    with pytest.raises(KeyError):
        O.minimize_active(lambda Y: {}["k"], domain=_dom(3, 1))


def test_surrogate_path(seed):
    sub = _sub(4, 1)
    dom = asub.active_domain(sub)
    w = np.asarray(sub.W1)[:, 0]
    X, f = _data(4, lambda X: (X @ w - 0.2) ** 2)
    s = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)
    res = O.minimize_active(s, domain=dom, grad="ignored")
    assert res.y_star == pytest.approx([0.2], abs=1e-6)
    other = asub.fit_surrogate(X=X, f=f, subspace=_sub(4, 1, seed=9), kind="poly", degree=2)
    with pytest.raises(InputError) as e:
        O.minimize_active(other, domain=dom)
    assert e.value.details["reason"] == "subspace_mismatch"


@pytest.mark.parametrize(
    "kwargs", [{"n_starts": 0}, {"max_iter": 0}, {"n_starts": 1.5}, {"rng": "x"}]
)
def test_minimize_active_bad_arguments(kwargs):
    with pytest.raises(InputError) as e:
        O.minimize_active(lambda Y: Y[:, 0], domain=_dom(4, 2), **kwargs)
    assert _code(e) == "input.parameter"


def test_minimize_active_bad_objects():
    with pytest.raises(InputError):
        O.minimize_active(3, domain=_dom(4, 2))
    with pytest.raises(InputError):
        O.minimize_active(lambda Y: Y[:, 0], domain=_sub(4, 2))
    with pytest.raises(InputError):
        O.minimize_active(lambda Y: Y[:, 0], domain=_dom(4, 2), grad=3)


def test_active_result_roundtrip(assert_roundtrip, assert_readonly, seed):
    dom = _dom(5, 2)
    g0, _ = _bowl([-0.3, 0.0])
    with pytest.warns(ConvergenceWarning):
        res = O.minimize_active(
            lambda Y: np.where(Y[:, 0] > 0, np.nan, g0(Y)), domain=dom, n_starts=6, rng=seed
        )
    assert_roundtrip(res)
    assert_readonly(res)
    assert_roundtrip(O.minimize_active(g0, domain=_dom(3, 1)))


def test_minimize_active_determinism(seed):
    dom = _dom(5, 2)
    g, _ = _bowl([0.2, 0.1])
    a = O.minimize_active(g, domain=dom, n_starts=3, rng=seed)
    b = O.minimize_active(g, domain=dom, n_starts=3, rng=seed)
    assert a == b
    assert not np.array_equal(
        a.starts, O.minimize_active(g, domain=dom, n_starts=3, rng=seed + 1).starts
    )


# ---------------------------------------------------------------------------
# lift (§5.4)
# ---------------------------------------------------------------------------


def test_lift_exact_quadratic_uniform():
    """With p = m the fitted quadratic is exact and x* minimizes f over the fiber."""
    m = 4
    dom = _axis_dom(m, 1, asub.Uniform.reference(m))
    x0 = np.array([0.2, -0.3, 0.4, 0.1])
    A = np.diag([3.0, 2.0, 1.0, 0.5])

    def f(X):
        return np.einsum("ij,jk,ik->i", X - x0, A, X - x0)

    X, fx = _data(m, f)
    res = O.lift([0.2], domain=dom, X=X, f=fx, lift_dim=4)
    assert np.allclose(res.x, x0, atol=1e-8)
    assert res.lift_dim == 4
    assert res.n_samples == 60
    assert res.n_clipped == 0
    assert res.model_value == pytest.approx(0.0, abs=1e-10)
    assert np.allclose(res.hessian, 2 * A, atol=1e-8)
    assert res.curvature_floor == pytest.approx(1e-2 * 4.0, rel=1e-8)
    assert res.projection_distance == 0.0


def test_lift_default_dimension():
    dom5 = _dom(5, 1)
    X, f = _data(5, lambda X: np.sum(X**2, axis=1))
    assert O.lift([0.0], domain=dom5, X=X, f=f).lift_dim == 3  # m - n = 4 > 2: p = n + 2
    dom3 = _dom(3, 1)
    X, f = _data(3, lambda X: np.sum(X**2, axis=1))
    assert O.lift([0.0], domain=dom3, X=X, f=f).lift_dim == 2  # m - n = 2: p = n + 1


def test_lift_gaussian_closed_form():
    """Gaussian: zeta* = -Q_+^{-1} c(y), z_b = 0 [C-16]."""
    m = 4
    dom = _axis_dom(m, 1, asub.Gaussian.reference(m))
    x0 = np.array([0.5, -0.4, 0.3, 0.0])

    def f(X):
        return (
            (X[:, 0] - x0[0]) ** 2
            + 2 * (X[:, 1] - x0[1]) ** 2
            + (X[:, 2] - x0[2]) ** 2
            + X[:, 0] * X[:, 1]
        )

    X, fx = _data(m, f, gaussian=True)
    res = O.lift([0.5], domain=dom, X=X, f=fx, lift_dim=3)
    # minimize over (x1, x2) with x0 fixed at 0.5: d/dx1: 4 (x1 + 0.4) + 0.5 = 0
    assert res.x == pytest.approx([0.5, -0.4 - 0.125, 0.3, 0.0], abs=1e-8)
    assert res.z.shape == (3,)
    assert res.z[-1] == 0.0


def test_indefinite_lift():
    """V10.15: an indefinite Q warns accuracy.curvature_clipped; the QP is convex and solves."""
    m = 3
    dom = _axis_dom(m, 1, asub.Uniform.reference(m))

    def f(X):
        return X[:, 0] + X[:, 1] ** 2 - X[:, 2] ** 2 + 0.3 * X[:, 1]

    X, fx = _data(m, f)
    with pytest.warns(AccuracyWarning) as rec:
        res = O.lift([0.1], domain=dom, X=X, f=fx, lift_dim=3)
    assert [w.message.code for w in rec] == ["accuracy.curvature_clipped"]
    assert rec[0].message.details["tau"] == pytest.approx(0.02, rel=1e-8)
    assert res.n_clipped == 1
    assert np.sort(res.curvature_eigenvalues) == pytest.approx([-2.0, 2.0], abs=1e-8)
    assert res.x[1] == pytest.approx(-0.15, abs=1e-6)  # the convex direction is minimized exactly
    assert np.all(np.abs(res.x) <= 1)


def test_constant_model_lift():
    """An exactly constant fit (f = 0): c = 0, tau = 1, the minimum-norm z (here z = 0)."""
    dom = _axis_dom(3, 1, asub.Uniform.reference(3))
    X, fx = _data(3, lambda X: np.zeros(X.shape[0]))
    with pytest.warns(AccuracyWarning):
        res = O.lift([0.4], domain=dom, X=X, f=fx, lift_dim=3)
    assert res.n_clipped == 2
    assert res.curvature_floor == 1.0
    assert res.x == pytest.approx([0.4, 0.0, 0.0], abs=1e-8)


@pytest.mark.parametrize("gaussian", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_rounding_level_curvature_lift(gaussian, seed):
    """V10.27, D-110 amended: on an exact ridge the fitted max|theta| is round-off; the lift takes
    c = 0, tau = 1
    and returns the minimum-norm z, not a step of many standard deviations (spec 10 §3.6, §8)."""
    m = 6
    gen = np.random.default_rng(0)
    a = gen.standard_normal(m)
    a /= np.linalg.norm(a)
    sp = asub.Gaussian(mean=np.zeros(m), cov=np.ones(m)) if gaussian else asub.Uniform.reference(m)
    dom = asub.active_domain(asub.subspace_from_basis(a[:, None], space=sp))
    X = sp.sample(200, rng=seed)
    f = (X @ a) ** 2
    with pytest.warns(AccuracyWarning) as rec:
        res = O.lift(np.array([0.3]), domain=dom, X=X, f=f)
    assert {w.message.code for w in rec} == {"accuracy.curvature_clipped"}
    assert np.max(np.abs(res.curvature_eigenvalues)) < 1e-12
    assert res.curvature_floor == 1.0
    assert np.linalg.norm(res.z) <= 1e-8


def test_lift_errors():
    dom = _dom(4, 1)
    X, f = _data(4, lambda X: np.sum(X**2, axis=1))
    h = float(dom.vertices.max())
    with pytest.raises(InputError) as e:
        O.lift([h * 1.01], domain=dom, X=X, f=f)
    assert _code(e) == "input.out_of_domain"
    for bad in (1, 5):
        with pytest.raises(InputError) as e:
            O.lift([0.0], domain=dom, X=X, f=f, lift_dim=bad)
        assert _code(e) == "input.parameter"
    with pytest.raises(InputError) as e:
        O.lift([0.0], domain=dom, X=X[:14], f=f[:14], lift_dim=4)
    assert _code(e) == "input.too_few_samples"
    assert e.value.details["minimum"] == 15
    for delta in (0.0, 1.0, -0.1):
        with pytest.raises(InputError) as e:
            O.lift([0.0], domain=dom, X=X, f=f, curvature_floor=delta)
        assert _code(e) == "input.parameter"
    with pytest.raises(InputError) as e:
        O.lift([0.0], domain=dom, X=X, f=f, space=asub.Uniform(-2 * np.ones(4), np.ones(4)))
    assert e.value.details["reason"] == "space_mismatch"
    with pytest.raises(InputError) as e:
        O.lift([0.0, 0.0], domain=dom, X=X, f=f)
    assert _code(e) == "input.shape"
    with pytest.raises(InputError) as e:
        O.lift([np.nan], domain=dom, X=X, f=f)
    assert _code(e) == "input.nonfinite"
    with pytest.raises(InputError) as e:
        O.lift([0.0], domain=dom, X=X, f=f[:-1])
    assert _code(e) == "input.size_mismatch"
    with pytest.raises(InputError) as e:
        O.lift([0.0], domain=dom, X=X, f=f, backend="gurobi")
    assert _code(e) == "input.parameter"


def test_lift_rank_deficient():
    dom = _dom(4, 1)
    X = np.zeros((30, 4))
    X[:, 0] = np.linspace(-1, 1, 30)
    with pytest.raises(asub.NumericalError) as e:
        O.lift([0.0], domain=dom, X=X, f=X[:, 0])
    assert _code(e) == "numerical.singular"


def test_lift_physical_data(uniform_space):
    """X is physical; the fit is in normalized coordinates; x_physical maps back."""
    space = uniform_space(3)
    dom = _axis_dom(3, 1, space)
    gen = np.random.default_rng(2)
    Xn = gen.uniform(-1, 1, (40, 3))
    f = (Xn[:, 1] - 0.3) ** 2 + (Xn[:, 2] + 0.2) ** 2 + Xn[:, 0]
    res = O.lift([0.5], domain=dom, X=space.from_normalized(Xn), f=f, space=space, lift_dim=3)
    assert res.x == pytest.approx([0.5, 0.3, -0.2], abs=1e-8)
    assert res.x_physical == pytest.approx(space.from_normalized(res.x[None, :])[0], rel=1e-14)


class _Recorder:
    """A SolverBackend that delegates to SciPy and records its calls; optionally reports the
    first QP infeasible.
    """

    def __init__(self, fail_first_qp=False):
        self.name = "recorder"
        self.calls = []
        self._inner = ScipyBackend()
        self._fail = fail_first_qp

    def lp(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        self.calls.append("lp")
        return self._inner.lp(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    def qp(self, H, g, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        self.calls.append("qp")
        if self._fail and self.calls.count("qp") == 1:
            return SolverOutput(
                x=None, objective=None, status="infeasible", message="mock", n_iter=0
            )
        return self._inner.qp(H, g, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)


def test_backend_protocol():
    """V10.23: a mock backend satisfying SolverBackend is used by lift (calls recorded); a bad
    string raises.
    """
    dom = _dom(4, 1)
    X, f = _data(4, lambda X: np.sum((X - 0.1) ** 2, axis=1))
    be = _Recorder()
    assert isinstance(be, asub.SolverBackend)
    res = O.lift([0.05], domain=dom, X=X, f=f, backend=be)
    assert be.calls == ["qp"]
    assert res.provenance.parameters["backend"] == "recorder"
    assert res == O.lift([0.05], domain=dom, X=X, f=f)
    gauss = _dom(4, 1, space=asub.Gaussian.reference(4))
    be2 = _Recorder()
    O.lift([0.05], domain=gauss, X=X, f=f, backend=be2)
    assert be2.calls == []  # closed form, no solver
    with pytest.raises(InputError) as e:
        O.lift([0.05], domain=dom, X=X, f=f, backend="cvxopt")
    assert _code(e) == "input.parameter"


def test_lift_projection_path():
    """§5.4 step 5: an infeasible QP triggers the L1 projection LP and a second QP."""
    dom = _dom(4, 2)
    X, f = _data(4, lambda X: np.sum((X - 0.1) ** 2, axis=1))
    v = np.asarray(dom.vertices)[0]
    be = _Recorder(fail_first_qp=True)
    res = O.lift(v, domain=dom, X=X, f=f, backend=be)
    assert be.calls == ["qp", "lp", "qp"]
    assert res.projection_distance <= 1e-6 * (1 + np.linalg.norm(v))
    assert np.allclose(res.x, dom.vertex_preimages[0], atol=1e-7)


def test_lift_projection_too_far(monkeypatch):
    dom = _dom(4, 2)
    X, f = _data(4, lambda X: np.sum((X - 0.1) ** 2, axis=1))
    monkeypatch.setattr(O, "_project_y", lambda y, W1, be: y + 1e-3)
    with pytest.raises(SolverError) as e:
        O.lift([0.0, 0.0], domain=dom, X=X, f=f, backend=_Recorder(fail_first_qp=True))
    assert _code(e) == "solver.infeasible"


def test_lift_roundtrip(assert_roundtrip, assert_readonly):
    dom = _dom(4, 1)
    X, f = _data(4, lambda X: np.sum((X - 0.1) ** 2, axis=1))
    res = O.lift([0.05], domain=dom, X=X, f=f)
    assert_roundtrip(res)
    assert_readonly(res)


# ---------------------------------------------------------------------------
# minimize (§5.5)
# ---------------------------------------------------------------------------


def test_f_model(uniform_space, seed):
    """V10.19: f_model equals the model at x_star_physical; None without a model [D-108],
    [A-10].
    """
    space = uniform_space(4)
    sub = _sub(4, 1, space=space)
    dom = asub.active_domain(sub)
    w = np.asarray(sub.W1)[:, 0]

    def model(Xp):
        x = space.to_normalized(Xp)
        return (x @ w - 0.2) ** 2 + 0.1 * np.sum(x**2, axis=1)

    gen = np.random.default_rng(seed)
    Xp = space.from_normalized(gen.uniform(-1, 1, (50, 4)))

    def g(Y):
        return (Y[:, 0] - 0.2) ** 2

    res = O.minimize(g, domain=dom, X=Xp, f=model(Xp), model=model, rng=seed)
    assert res.f_model == model(res.x_star_physical[None, :])[0]
    assert res.f_star == res.active.value
    assert np.array_equal(res.x_star, res.lift.x)
    assert np.array_equal(res.y_star, res.lift.y)
    assert res.provenance.parameters["model"] is None
    res0 = O.minimize(g, domain=dom, X=Xp, f=model(Xp), rng=seed)
    assert res0.f_model is None
    assert res0.active == res.active
    assert res0.lift == res.lift


def test_minimize_model_checks(seed):
    dom = _dom(4, 1)
    X, f = _data(4, lambda X: np.sum(X**2, axis=1))

    def g(Y):
        return Y[:, 0] ** 2

    other = asub.Model(space=asub.Uniform(-2 * np.ones(4), np.ones(4)), value=lambda X: X[:, 0])
    with pytest.raises(InputError) as e:
        O.minimize(g, domain=dom, X=X, f=f, model=other)
    assert e.value.details["reason"] == "space_mismatch"
    with pytest.raises(InputError) as e:
        O.minimize(g, domain=dom, X=X, f=f, space=asub.Gaussian.reference(4))
    assert e.value.details["reason"] == "space_mismatch"
    named = asub.Model(space=dom.subspace.space, value=lambda X: np.sum(X**2, axis=1), name="bowl")
    res = O.minimize(g, domain=dom, X=X, f=f, model=named, batch_size=1)
    assert res.provenance.parameters["model"] == "bowl"
    assert res.provenance.parameters["batch_size"] == 1
    assert res.f_model == pytest.approx(float(np.sum(res.x_star**2)), rel=1e-14)


def test_minimize_roundtrip_and_determinism(assert_roundtrip, assert_readonly, seed):
    dom = _dom(5, 2)
    X, f = _data(5, lambda X: np.sum((X - 0.1) ** 2, axis=1), M=80)
    g, dg = _bowl([0.1, 0.0])
    a = O.minimize(
        g,
        domain=dom,
        X=X,
        f=f,
        grad=dg,
        model=lambda X: np.sum((X - 0.1) ** 2, axis=1),
        n_starts=3,
        rng=seed,
    )
    b = O.minimize(
        g,
        domain=dom,
        X=X,
        f=f,
        grad=dg,
        model=lambda X: np.sum((X - 0.1) ** 2, axis=1),
        n_starts=3,
        rng=seed,
    )
    assert a == b
    assert a.provenance.rng == a.active.provenance.rng
    assert_roundtrip(a)
    assert_readonly(a)


def test_minimize_result_invariants(make_provenance, seed):
    dom = _dom(4, 1)
    X, f = _data(4, lambda X: np.sum(X**2, axis=1))
    res = O.minimize(lambda Y: Y[:, 0] ** 2, domain=dom, X=X, f=f)
    with pytest.raises(AssertionError):
        dataclasses.replace(res, f_star=res.f_star + 1.0)


def test_no_files(tmp_path, monkeypatch, seed):
    monkeypatch.chdir(tmp_path)
    dom = _dom(4, 2)
    X, f = _data(4, lambda X: np.sum(X**2, axis=1))
    with pytest.raises(SolverError):
        O.minimize_active(
            lambda Y: np.full(Y.shape[0], np.nan), domain=dom, n_starts=2, max_iter=50, rng=seed
        )
    assert O.lift([0.0, 0.0], domain=dom, X=X, f=f).x.shape == (4,)
    assert list(tmp_path.iterdir()) == []
