"""Known LP/QP solutions through the solver protocol (spec 10 §8, spec 17 V10.20)."""

from __future__ import annotations

import numpy as np
import pytest

from active_subspaces import solvers

F8 = 1e-8
BOX5 = np.tile([-1.0, 1.0], (5, 1))


@pytest.fixture(params=["slsqp", "trust-constr"])
def backend(request):
    return solvers.ScipyBackend(qp_method=request.param)


def test_2016_qp_problems(backend):
    # 2016 linear_program_ineq: min 1^T x s.t. A x >= b.
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([0.1, 0.1, 0.1])
    out = solvers.solve_lp(np.ones(2), A_ub=-A, b_ub=-b, backend=backend)
    np.testing.assert_allclose(out.x, [0.1, 0.1], rtol=F8, atol=F8)
    assert out.status == "optimal"
    assert abs(out.objective - 0.2) <= F8

    # 2016 linear_program_eq: min 1^T x s.t. A x = b, -1 <= x <= 1.
    A = np.array([[2.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0, 0.0]])
    b = np.array([0.5, 0.5])
    out = solvers.solve_lp(np.ones(5), A_eq=A, b_eq=b, bounds=BOX5, backend=backend)
    np.testing.assert_allclose(out.x, [0.75, -1.0, 0.75, -1.0, -1.0], rtol=F8, atol=F8)

    # 2016 quadratic_program_bnd: min c^T x + x^T Q x (no 1/2), so H = 2 Q in the v1 form.
    out = solvers.solve_qp(2 * np.eye(5), np.ones(5), bounds=BOX5, backend=backend)
    np.testing.assert_allclose(out.objective, -1.25, rtol=F8)
    np.testing.assert_allclose(
        out.x, -0.5 * np.ones(5), atol=F8 if backend.qp_method == "slsqp" else 1e-5
    )

    # 2016 quadratic_program_ineq: min c^T x + x^T Q x s.t. A x >= b.
    A = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    b = np.array([-1.0, -1.0])
    out = solvers.solve_qp(2 * np.eye(5), np.ones(5), A_ub=-A, b_ub=-b, backend=backend)
    np.testing.assert_allclose(out.objective, -1.25, rtol=F8)
    np.testing.assert_allclose(
        out.x, -0.5 * np.ones(5), atol=F8 if backend.qp_method == "slsqp" else 1e-5
    )


def test_qp_simplex_projection(backend):
    # Projection of a onto the probability simplex: min 1/2 ||v - a||^2, sum v = 1, v >= 0.
    a = np.array([0.5, 0.3, -0.2])
    bounds = np.array([[0.0, np.inf]] * 3)
    out = solvers.solve_qp(
        np.eye(3), -a, A_eq=np.ones((1, 3)), b_eq=[1.0], bounds=bounds, backend=backend
    )
    expected = np.array([0.6, 0.4, 0.0])
    # trust-constr (the fallback) is an interior-point method: it stops about 1e-5 inside an active
    # bound.
    tol = F8 if backend.qp_method == "slsqp" else 1e-4
    np.testing.assert_allclose(out.x, expected, atol=tol)
    np.testing.assert_allclose(out.objective, 0.5 * expected @ expected - a @ expected, rtol=tol)


def test_qp_objective_has_one_half():
    # min 1/2 * 4 v^2 - 4 v: v* = 1, objective -2 (the 2016 form without 1/2 would give v* = 1/2).
    out = solvers.solve_qp([[4.0]], [-4.0])
    assert abs(out.x[0] - 1.0) <= 1e-14
    assert abs(out.objective + 2.0) <= 1e-14


def test_unconstrained_qp_singular_hessian():
    out = solvers.solve_qp(np.diag([2.0, 0.0]), [2.0, 0.0])
    np.testing.assert_allclose(out.x, [-1.0, 0.0], atol=1e-15)
    assert out.objective == pytest.approx(-1.0, abs=1e-15)
    assert out.n_iter == 0


def test_lp_with_free_variables_and_mixed_blocks():
    # max v1 + v2 s.t. v1 + 2 v2 <= 4, 3 v1 + v2 <= 6, v1 - v2 = 0  ->  v1 = v2 = 4/3.
    out = solvers.solve_lp(
        [-1.0, -1.0],
        A_ub=[[1.0, 2.0], [3.0, 1.0]],
        b_ub=[4.0, 6.0],
        A_eq=[[1.0, -1.0]],
        b_eq=[0.0],
    )
    np.testing.assert_allclose(out.x, [4 / 3, 4 / 3], rtol=F8)
    assert out.objective == pytest.approx(-8 / 3, rel=F8)


def test_lp_chebyshev_center():
    # The start LP of spec 08: largest ball in the square [-1, 1]^2 on the line x1 = y; r* = 1 -
    # |y|.
    y = 0.25
    c = np.array([0.0, 0.0, -1.0])
    kappa = np.ones((2, 1))
    A_ub = np.block([[np.eye(2), kappa], [-np.eye(2), kappa]])
    b_ub = np.ones(4)
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, np.inf]])
    out = solvers.solve_lp(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=[[1.0, 0.0, 0.0]], b_eq=[y], bounds=bounds
    )
    assert out.x[2] == pytest.approx(0.75, rel=F8)
    assert out.x[0] == pytest.approx(y, abs=F8)
