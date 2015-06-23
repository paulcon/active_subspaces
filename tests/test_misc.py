from unittest import TestCase
import unittest
import active_subspaces.utils.misc as ut
import numpy as np

class TestUtils(TestCase):
    def test_bounded_normalizer(self):
        M, m = 10, 3
        XX = np.hstack((np.random.uniform(-4.0,7.0,size=(M,1)),
                        np.random.uniform(4.0,6.0,size=(M,1)),
                        np.random.uniform(2.0,3.0,size=(M,1))))
        lb = np.array([-4.0,4.0,2.0])
        ub = np.array([7.0,6.0,3.0])
        bn = ut.BoundedNormalizer(lb, ub)
        X0 = bn.normalize(XX)
        X1 = bn.unnormalize(X0)
        np.testing.assert_almost_equal(XX, X1)
        np.testing.assert_array_less(X0, 1.0)
        np.testing.assert_array_less(-X0, 1.0)

    def test_unbounded_normalizer(self):
        M, m = 100, 3
        XX = np.hstack((np.random.normal(-4.0,7.0,size=(M,1)),
                        np.random.normal(4.0,6.0,size=(M,1)),
                        np.random.normal(2.0,3.0,size=(M,1))))

        C = np.diag([7.0**2, 6.0**2, 3.0**2])
        mu = np.array([-4.0, 4.0, 2.0]).reshape((3,1))

        un = ut.UnboundedNormalizer(mu, C)
        X0 = un.normalize(XX)
        X1 = un.unnormalize(X0)
        np.testing.assert_almost_equal(XX, X1)
        np.testing.assert_allclose(np.mean(X0,axis=0).reshape((m,1)), np.zeros((m,1)), atol=1.0)

    def test_process_inputs_bad_inputs(self):
        self.assertRaises(ValueError, ut.process_inputs, np.array([1.0,1.0,-1.0]))

    def test_process_inputs(self):
        X0 = np.random.uniform(-1.0,1.0,size=(10,3))
        X,M,m = ut.process_inputs(X0)
        np.testing.assert_almost_equal(X, X0)
        np.testing.assert_almost_equal(M, 10)
        np.testing.assert_almost_equal(m, 3)

    def test_conditional_expectations(self):
        f = np.array([0.2,0.2,1.0,2.0]).reshape((4,1))
        ind = np.array([0,0,1,1]).reshape((4,1))
        E, V = ut.conditional_expectations(f, ind)
        np.testing.assert_almost_equal(E,np.array([[0.2],[1.5]]))
        np.testing.assert_almost_equal(V,np.array([[0.0],[0.25]]))


if __name__ == '__main__':
    unittest.main()
