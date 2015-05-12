from unittest import TestCase
import unittest
import numpy as np
# import active_subspaces.qp_solvers.gurobi_solver as gb

class TestGurobi(TestCase):

    @unittest.skip("Don't think we are using this anymore")
    def test_linear_program_eq(self):
        c = np.ones((5,1))
        A = np.array([[2.0,1.0,0.,0.,0.],[0.,0.,2.0,1.0,0.]])
        b = np.array([[0.5],[0.5]])
        lb, ub = -np.ones((5,1)), np.ones((5,1))
        gs = gb.GurobiSolver()
        x = gs.linear_program_eq(c, A, b, lb, ub)
        xtrue = np.array([0.75,-1.0,0.75,-1.0,-1.0]).reshape((5,1))
        np.testing.assert_almost_equal(x,xtrue)

    @unittest.skip("Don't think we are using this anymore")
    def test_quadratic_program_bnd(self):
        c = np.ones((5,1))
        Q = np.eye(5)
        lb, ub = -np.ones((5,1)), np.ones((5,1))
        gs = gb.GurobiSolver()
        x = gs.quadratic_program_bnd(c, Q, lb, ub)
        xtrue = -0.5*np.ones((5,1))
        np.testing.assert_almost_equal(x,xtrue)

    @unittest.skip("Don't think we are using this anymore")
    def test_quadratic_program_ineq(self):
        c = np.ones((5,1))
        Q = np.eye(5)
        A = np.array([[1.,0.,0.,0.,0.],[0.,1.,0.,0.,0.]])
        b = np.array([[-1.0],[-1.0]])
        gs = gb.GurobiSolver()
        x = gs.quadratic_program_ineq(c, Q, A, b)
        xtrue = -0.5*np.ones((5,1))
        np.testing.assert_almost_equal(x,xtrue)

if __name__ == '__main__':
    unittest.main()

