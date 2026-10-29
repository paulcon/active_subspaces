import numpy as np
from unittest import TestCase
import unittest
import active_subspaces.utils.designs as dn

class TestDesigns(TestCase):

    def test_interval_design(self):
        y = dn.interval_design(-1.0, 1.0, 3)
        ytrue = np.array([[-0.5], [0.], [0.5]])
        np.testing.assert_almost_equal(y, ytrue)

    def test_maximin_design(self):
        vert = np.array([[-1.0,-1.0], [1.0,-1.0], [-1.0,1.0], [1.0,1.0]])
        Y = dn.maximin_design(vert, 1)
        np.testing.assert_almost_equal(Y, np.zeros((1,2)), decimal=4)

    def test_maximin_design_random_state(self):
        vert = np.array([[-1.0,-1.0], [1.0,-1.0], [-1.0,1.0], [1.0,1.0]])
        np.random.seed(99)
        Y1 = dn.maximin_design(vert, 3)
        np.random.seed(999)
        Y2 = dn.maximin_design(vert, 3)
        np.testing.assert_almost_equal(Y1, Y2, decimal=6)

    def test_maximin_design_repeatable(self):
        vert = np.array([[-1.0,-1.0], [1.0,-1.0], [-1.0,1.0], [1.0,1.0]])
        Y1 = dn.maximin_design(vert, 10)
        Y2 = dn.maximin_design(vert, 10)
        np.testing.assert_almost_equal(Y1, Y2)

    def test_gauss_hermite_design_1(self):
        Y = dn.gauss_hermite_design([1])
        Ytrue = np.array([[0.0]])
        np.testing.assert_almost_equal(Y, Ytrue)

    def test_gauss_hermite_design_2(self):
        Y = dn.gauss_hermite_design([1,1])
        Ytrue = np.array([[0.0, 0.0]])
        np.testing.assert_almost_equal(Y, Ytrue)

    def test_gradient_with_finite_difference(self):
        vert = np.array([[-1.0,-1.0], [1.0,-1.0], [-1.0,1.0], [1.0,1.0]])
        N, n, h = 5, 2, 1e-6
        for i in range(5):
            y0 = np.random.normal(size=(N*n, ))
            f0 = dn._maximin_design_obj(y0, vert)
            df0 = dn._maximin_design_grad(y0, vert)
            for j in range(N*n):
                y0p = y0.copy()
                y0p[j] += h
                f0p = dn._maximin_design_obj(y0p, vert)
                df0_fd = (f0p - f0)/h
                np.testing.assert_almost_equal(df0[j], df0_fd, decimal=5)


if __name__ == '__main__':
    unittest.main()
