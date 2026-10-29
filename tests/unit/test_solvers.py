"""Contracts of the solver-backend protocol (spec 10 §4.4, §5.6, §6; spec 17 V10.21-V10.24)."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
import scipy.optimize

import active_subspaces as ac
from active_subspaces import solvers
from active_subspaces.errors import InputError, SolverError


class RecordingBackend:
    """A third-party backend: solves through SciPy and records each call."""

    name = "recording"

    def __init__(self, status: str | None = None):
        self.calls: list[tuple[str, tuple, dict]] = []
        self.status = status
        self._inner = solvers.ScipyBackend()

    def lp(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        kw = {"A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq, "bounds": bounds}
        self.calls.append(("lp", (c,), kw))
        if self.status is not None:
            return solvers.SolverOutput(None, None, self.status, "forced", 7)
        return self._inner.lp(c, **kw)

    def qp(self, H, g, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        kw = {"A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq, "bounds": bounds}
        self.calls.append(("qp", (H, g), kw))
        if self.status is not None:
            return solvers.SolverOutput(None, None, self.status, "forced", 7)
        return self._inner.qp(H, g, **kw)


# ---------------------------------------------------------------------------
# Spec 17 V10.21-V10.24
# ---------------------------------------------------------------------------


def test_bounds_none_free():
    out = solvers.solve_lp([1.0], A_ub=[[-1.0]], b_ub=[3.0])
    assert out.status == "optimal"
    assert out.x[0] == pytest.approx(-3.0, rel=1e-8)
    assert out.objective == pytest.approx(-3.0, rel=1e-8)


def test_infeasible_unbounded_nonpsd():
    with pytest.raises(SolverError) as exc:
        solvers.solve_lp([1.0], A_ub=[[1.0]], b_ub=[-1.0], bounds=[[0.0, np.inf]])
    assert exc.value.code == "solver.infeasible"
    d = exc.value.details
    assert d["problem"] == "lp"
    assert d["backend"] == "scipy"
    assert d["status"] == "infeasible"
    assert d["shape"] == [1, 1, 0]
    assert isinstance(d["message"], str)
    assert "n_iter" in d

    with pytest.raises(SolverError) as exc:
        solvers.solve_lp([1.0, 0.0])
    assert exc.value.code == "solver.unbounded"
    assert exc.value.details["shape"] == [2, 0, 0]

    with pytest.raises(SolverError) as exc:
        solvers.solve_lp(
            [1.0, -1.0], A_eq=[[1.0, 1.0]], b_eq=[0.0], bounds=[[-np.inf, 5.0], [-np.inf, np.inf]]
        )
    assert exc.value.code == "solver.unbounded"

    with pytest.raises(InputError) as exc:
        solvers.solve_qp([[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0])
    assert exc.value.code == "input.parameter"
    assert exc.value.details["reason"] == "not_psd"


def test_qp_infeasible_and_unbounded():
    with pytest.raises(SolverError) as exc:
        solvers.solve_qp([[1.0]], [1.0], A_ub=[[1.0], [-1.0]], b_ub=[-1.0, -1.0])
    assert exc.value.code == "solver.infeasible"
    assert exc.value.details["problem"] == "qp"
    assert exc.value.details["shape"] == [1, 2, 0]

    with pytest.raises(SolverError) as exc:
        solvers.solve_qp(np.diag([1.0, 0.0]), [0.0, 1.0])
    assert exc.value.code == "solver.unbounded"

    with pytest.raises(SolverError) as exc:
        solvers.solve_qp(np.zeros((1, 1)), [1.0])
    assert exc.value.code == "solver.unbounded"


def test_backend_protocol():
    be = RecordingBackend()
    assert isinstance(be, solvers.SolverBackend)
    assert solvers.get_backend(be) is be
    c = [1, 2]  # integers are coerced to float64 before the backend sees them
    out = solvers.solve_lp(c, A_ub=[[-1, 0], [0, -1]], b_ub=[1, 1], backend=be)
    np.testing.assert_allclose(out.x, [-1.0, -1.0])
    kind, args, kw = be.calls[0]
    assert kind == "lp"
    assert args[0].dtype == np.float64
    assert args[0].shape == (2,)
    assert kw["A_ub"].dtype == np.float64
    assert kw["A_eq"] is None
    assert kw["bounds"] is None

    out = solvers.solve_qp(np.eye(2), [1.0, -1.0], bounds=[[0, 1], [0, 1]], backend=be)
    np.testing.assert_allclose(out.x, [0.0, 1.0], atol=1e-8)
    kind, args, kw = be.calls[1]
    assert kind == "qp"
    assert args[0].shape == (2, 2)
    assert kw["bounds"].shape == (2, 2)

    with pytest.raises(InputError) as exc:
        solvers.get_backend("gurobi")
    assert exc.value.code == "input.parameter"
    assert exc.value.details["param"] == "backend"
    with pytest.raises(InputError):
        solvers.solve_lp([1.0], backend="SCIPY")


def test_no_files_no_global_rng(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = np.random.get_state()  # noqa: NPY002 (checks the legacy global state, spec 05 §8)
    for call in (
        lambda: solvers.solve_lp([1.0], A_ub=[[1.0]], b_ub=[-1.0], bounds=[[0.0, 1.0]]),
        lambda: solvers.solve_lp([-1.0]),
        lambda: solvers.solve_qp([[1.0]], [0.0], A_ub=[[1.0], [-1.0]], b_ub=[-2.0, 1.0]),
        lambda: solvers.solve_qp(
            *_hard_qp()[:2],
            A_ub=_hard_qp()[2],
            b_ub=_hard_qp()[3],
            backend=solvers.ScipyBackend(max_iter=1),
        ),
    ):
        with pytest.raises(SolverError):
            call()
    assert os.listdir(tmp_path) == []
    after = np.random.get_state()  # noqa: NPY002 (checks the legacy global state, spec 05 §8)
    assert np.array_equal(state[1], after[1])
    assert state[2:] == after[2:]


# ---------------------------------------------------------------------------
# Status mapping (spec 10 §4.4, §5.6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scipy_status", "status", "code"),
    [
        (1, "max_iter", "solver.failed"),
        (2, "infeasible", "solver.infeasible"),
        (3, "unbounded", "solver.unbounded"),
        (4, "failed", "solver.failed"),
    ],
)
def test_linprog_status_map(monkeypatch, scipy_status, status, code):
    def fake(*args, **kwargs):
        return scipy.optimize.OptimizeResult(
            status=scipy_status, x=None, message="mocked", nit=5, success=False
        )

    monkeypatch.setattr(scipy.optimize, "linprog", fake)
    out = solvers.ScipyBackend().lp(np.ones(2))
    assert out.status == status
    assert out.x is None
    assert out.objective is None
    assert out.n_iter == 5
    with pytest.raises(SolverError) as exc:
        solvers.solve_lp(np.ones(2))
    assert exc.value.code == code
    assert exc.value.details["status"] == status
    assert exc.value.details["message"] == "mocked"
    assert exc.value.details["n_iter"] == 5


def test_linprog_max_iter_keeps_last_iterate(monkeypatch):
    def fake(*args, **kwargs):
        return scipy.optimize.OptimizeResult(
            status=1, x=np.array([0.5, 0.25]), message="limit", nit=9, success=False
        )

    monkeypatch.setattr(scipy.optimize, "linprog", fake)
    out = solvers.ScipyBackend().lp(np.array([1.0, 2.0]))
    assert out.status == "max_iter"
    assert out.objective == 1.0
    np.testing.assert_array_equal(out.x, [0.5, 0.25])
    with pytest.raises(SolverError) as exc:
        solvers.solve_lp(np.array([1.0, 2.0]))
    assert exc.value.code == "solver.failed"
    assert exc.value.details["status"] == "max_iter"


def test_linprog_exception_is_numerical_failure(monkeypatch):
    def fake(*args, **kwargs):
        raise ValueError("HiGHS blew up")

    monkeypatch.setattr(scipy.optimize, "linprog", fake)
    out = solvers.ScipyBackend().lp(np.ones(1))
    assert out.status == "failed"
    assert "HiGHS blew up" in out.message
    with pytest.raises(SolverError) as exc:
        solvers.solve_lp(np.ones(1))
    assert exc.value.code == "solver.failed"


def _hard_qp():
    gen = np.random.default_rng(0)
    B = gen.standard_normal((20, 20))
    return (
        B @ B.T + np.eye(20),
        10 * gen.standard_normal(20),
        gen.standard_normal((30, 20)),
        np.ones(30),
    )


@pytest.mark.parametrize("method", ["slsqp", "trust-constr"])
def test_qp_iteration_limit(method):
    H, g, A, b = _hard_qp()
    be = solvers.ScipyBackend(max_iter=1, qp_method=method)
    out = be.qp(H, g, A_ub=A, b_ub=b)
    assert out.status == "max_iter"
    assert out.x is not None
    assert out.x.shape == (20,)
    assert out.objective == pytest.approx(0.5 * out.x @ H @ out.x + g @ out.x)
    with pytest.raises(SolverError) as exc:
        solvers.solve_qp(H, g, A_ub=A, b_ub=b, backend=be)
    assert exc.value.code == "solver.failed"
    assert exc.value.details["status"] == "max_iter"
    assert exc.value.details["shape"] == [20, 30, 0]
    out = solvers.solve_qp(H, g, A_ub=A, b_ub=b, backend=solvers.ScipyBackend(qp_method=method))
    assert out.status == "optimal"
    assert np.all(A @ out.x <= b + 1e-8)


def test_qp_fallback_to_other_method(monkeypatch):
    real = scipy.optimize.minimize
    methods = []

    def flaky(fun, x0, *args, method=None, **kwargs):
        methods.append(method)
        if method == "SLSQP":
            return scipy.optimize.OptimizeResult(
                x=x0, success=False, status=8, message="mocked failure", nit=1
            )
        return real(fun, x0, *args, method=method, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", flaky)
    out = solvers.solve_qp(np.eye(2), [1.0, 1.0], A_ub=[[-1.0, 0.0]], b_ub=[0.0])
    assert methods == ["SLSQP", "trust-constr"]
    assert out.message.startswith("trust-constr")
    np.testing.assert_allclose(
        out.x, [0.0, -1.0], atol=1e-4
    )  # trust-constr ends near, not on, the bound


def test_qp_both_methods_fail(monkeypatch):
    def broken(fun, x0, *args, method=None, **kwargs):
        if method == "SLSQP":
            raise np.linalg.LinAlgError("singular")
        return scipy.optimize.OptimizeResult(
            x=x0 + 10.0, success=True, status=1, message="drifted", nit=3
        )

    monkeypatch.setattr(scipy.optimize, "minimize", broken)
    out = solvers.ScipyBackend().qp(
        np.eye(2), np.ones(2), A_ub=np.array([[1.0, 0.0]]), b_ub=np.array([0.0])
    )
    assert out.status == "failed"
    assert out.x is None
    assert "violation" in out.message
    with pytest.raises(SolverError) as exc:
        solvers.solve_qp(np.eye(2), np.ones(2), A_ub=[[1.0, 0.0]], b_ub=[0.0])
    assert exc.value.code == "solver.failed"


# ---------------------------------------------------------------------------
# Backends that misbehave
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("infeasible", "solver.infeasible"),
        ("unbounded", "solver.unbounded"),
        ("max_iter", "solver.failed"),
        ("failed", "solver.failed"),
    ],
)
def test_backend_status_to_code(status, code):
    be = RecordingBackend(status=status)
    with pytest.raises(SolverError) as exc:
        solvers.solve_lp([1.0], backend=be)
    assert exc.value.code == code
    assert exc.value.details == {
        "problem": "lp",
        "backend": "recording",
        "status": status,
        "message": "forced",
        "n_iter": 7,
        "shape": [1, 0, 0],
    }
    with pytest.raises(SolverError) as exc:
        solvers.solve_qp([[1.0]], [1.0], backend=be)
    assert exc.value.code == code
    assert exc.value.details["problem"] == "qp"


def test_backend_optimal_without_solution_is_failure():
    be = RecordingBackend(status="optimal")
    with pytest.raises(SolverError) as exc:
        solvers.solve_lp([1.0], backend=be)
    assert exc.value.code == "solver.failed"


def test_backend_wrong_return_type_is_failure():
    class Bad(RecordingBackend):
        def lp(self, c, **kw):
            return {"x": c}

    with pytest.raises(SolverError) as exc:
        solvers.solve_lp([1.0], backend=Bad())
    assert exc.value.code == "solver.failed"


def test_backend_exception_propagates_unchanged():
    class Boom(RecordingBackend):
        def lp(self, c, **kw):
            raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        solvers.solve_lp([1.0], backend=Boom())


def test_get_backend():
    assert solvers.get_backend(None) == solvers.ScipyBackend()
    assert solvers.get_backend("scipy") == solvers.ScipyBackend()
    be = solvers.ScipyBackend(qp_method="trust-constr")
    assert solvers.get_backend(be) is be
    assert isinstance(be, solvers.SolverBackend)

    class NoQP:
        name = "half"

        def lp(self, c, **kw):
            return None

    for bad in (NoQP(), 3, "", object()):
        with pytest.raises(InputError) as exc:
            solvers.get_backend(bad)
        assert exc.value.code == "input.parameter"


# ---------------------------------------------------------------------------
# Validation (spec 10 §5.6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"c": [1.0, 2.0], "A_ub": [[1.0, 0.0]]}, "pair"),
        ({"c": [1.0, 2.0], "b_eq": [1.0]}, "pair"),
        ({"c": [1.0, 2.0], "A_ub": [[1.0, 0.0, 0.0]], "b_ub": [1.0]}, "shape"),
        ({"c": [1.0, 2.0], "A_ub": [[1.0, 0.0]], "b_ub": [1.0, 2.0]}, "shape"),
        ({"c": [[1.0, 2.0]]}, "shape"),
        ({"c": []}, "shape"),
        ({"c": [1.0, np.nan]}, "nonfinite"),
        ({"c": [1.0, 2.0], "A_eq": [[np.inf, 0.0]], "b_eq": [1.0]}, "nonfinite"),
        ({"c": [True, False]}, "dtype"),
        ({"c": [1.0, 2.0], "bounds": [[0.0, 1.0]]}, "shape"),
        ({"c": [1.0, 2.0], "bounds": [[0.0, 1.0], [2.0, 1.0]]}, "bounds"),
        ({"c": [1.0, 2.0], "bounds": [[np.inf, np.inf], [0.0, 1.0]]}, "bounds"),
        ({"c": [1.0, 2.0], "bounds": [[0.0, 1.0], [-np.inf, -np.inf]]}, "bounds"),
        ({"c": [1.0, 2.0], "bounds": [[0.0, np.nan], [0.0, 1.0]]}, "nonfinite"),
    ],
)
def test_lp_validation(kwargs, reason):
    c = kwargs.pop("c")
    with pytest.raises(InputError) as exc:
        solvers.solve_lp(c, **kwargs)
    assert exc.value.code == "input.parameter"
    assert exc.value.details["reason"] == reason


def test_qp_validation():
    with pytest.raises(InputError) as exc:
        solvers.solve_qp([[1.0, 0.5], [0.0, 1.0]], [0.0, 0.0])
    assert exc.value.details["reason"] == "not_symmetric"
    with pytest.raises(InputError) as exc:
        solvers.solve_qp(np.eye(3), [0.0, 0.0])
    assert exc.value.details["reason"] == "shape"
    with pytest.raises(InputError) as exc:
        solvers.solve_qp([[np.nan]], [0.0])
    assert exc.value.details["reason"] == "nonfinite"
    # Asymmetry below 1e-12 ||H||_inf is symmetrized; tiny negative eigenvalues are accepted.
    H = np.array([[2.0, 1.0 + 1e-13], [1.0, 2.0]])
    out = solvers.solve_qp(H, [1.0, 1.0])
    np.testing.assert_allclose(out.x, [-1 / 3, -1 / 3], rtol=1e-12)
    out = solvers.solve_qp(np.diag([1.0, -1e-12]), [1.0, 0.0])
    np.testing.assert_allclose(out.x, [-1.0, 0.0], atol=1e-15)


def test_inputs_not_modified():
    H = np.array([[2.0, 1.0], [1.0, 2.0]])
    g = np.array([1.0, -1.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([0.5])
    B = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    copies = [a.copy() for a in (H, g, A, b, B)]
    solvers.solve_qp(H, g, A_ub=A, b_ub=b, bounds=B)
    solvers.solve_lp(g, A_ub=A, b_ub=b, A_eq=A, b_eq=b, bounds=B)
    for a, c in zip((H, g, A, b, B), copies, strict=False):
        np.testing.assert_array_equal(a, c)


def test_scipy_backend_parameters():
    be = solvers.ScipyBackend()
    assert (be.name, be.qp_method, be.tol, be.max_iter) == ("scipy", "slsqp", 1e-10, 1000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        be.tol = 1.0
    for kwargs in (
        {"qp_method": "cvxopt"},
        {"tol": 0.0},
        {"tol": -1.0},
        {"max_iter": 0},
        {"max_iter": 2.5},
    ):
        with pytest.raises(InputError) as exc:
            solvers.ScipyBackend(**kwargs)
        assert exc.value.code == "input.parameter"


def test_solver_output_value_type():
    out = solvers.SolverOutput([1.0, 2.0], 3.0, "optimal", "ok", 4)
    assert not out.x.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        out.x[0] = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.status = "failed"
    assert out == solvers.SolverOutput(np.array([1.0, 2.0]), 3.0, "optimal", "ok", 4)
    assert out != solvers.SolverOutput(np.array([1.0, 2.5]), 3.0, "optimal", "ok", 4)
    assert solvers.SolverOutput(None, None, "failed", "x", None) == solvers.SolverOutput(
        None, None, "failed", "x", None
    )
    with pytest.raises(InputError):
        solvers.SolverOutput(None, None, "weird", "", None)


def test_deterministic():
    args = (np.array([[2.0, 0.5], [0.5, 1.0]]), np.array([-1.0, 3.0]))
    kw = {
        "A_ub": np.array([[1.0, 1.0]]),
        "b_ub": np.array([0.3]),
        "bounds": np.array([[-2.0, 2.0], [-2.0, 2.0]]),
    }
    a = solvers.solve_qp(*args, **kw)
    b = solvers.solve_qp(*args, **kw)
    assert a == b


def test_exports():
    assert solvers.__all__ == tuple(sorted(solvers.__all__))
    assert set(solvers.__all__) == {
        "SolverBackend",
        "SolverOutput",
        "ScipyBackend",
        "get_backend",
        "solve_lp",
        "solve_qp",
    }
    for name in solvers.__all__:
        assert getattr(ac, name) is getattr(solvers, name)
    assert "solvers" in ac.__all__
