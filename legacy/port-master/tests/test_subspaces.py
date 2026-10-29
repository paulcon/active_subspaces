from unittest import TestCase
import unittest
import active_subspaces.subspaces as ss
import numpy as np

class TestSubspaces(TestCase):

    def test_sorted_eigh(self):
        np.random.seed(42)
        X = np.random.normal(size=(3,3))
        C = np.dot(X.transpose(),X)
        e, W = ss.sorted_eigh(C)
        np.testing.assert_array_less(e[1], e[0])
        
    def test_active_subspace(self):
        np.random.seed(42)
        df = np.random.normal(size=(10,3))
        weights = np.ones((10,1)) / 10
        e, W = ss.active_subspace(df, weights)
        np.testing.assert_array_less(e[1], e[0])
        
    def test_ols_subspace(self):
        np.random.seed(42)
        X = np.random.normal(size=(20,3))
        f = np.random.normal(size=(20,1))
        weights = np.ones((20,1)) / 20
        e, W = ss.ols_subspace(X, f, weights)
        np.testing.assert_array_less(e[1], e[0])
        
    def test_qphd_subspace(self):
        np.random.seed(42)
        X = np.random.normal(size=(50,3))
        f = np.random.normal(size=(50,1))
        weights = np.ones((50,1)) / 50
        e, W = ss.qphd_subspace(X, f, weights)
        np.testing.assert_array_less(e[1], e[0])
        
    def test_opg_subspace(self):
        np.random.seed(42)
        X = np.random.normal(size=(50,3))
        f = np.random.normal(size=(50,1))
        weights = np.ones((50,1)) / 50
        e, W = ss.opg_subspace(X, f, weights)
        np.testing.assert_array_less(e[1], e[0])
        
    def test_bootstrap_replicate(self):
        np.random.seed(42)
        X = np.random.normal(size=(10,3))
        f = np.random.normal(size=(10,1))
        df = np.random.normal(size=(10,3))
        weights = np.ones((10,1)) / 10
        X0, f0, df0, w0 = ss._bootstrap_replicate(X, f, df, weights)
        assert(np.any(weights==w0[0]))
        assert(np.any(f==f0[1]))
        
    def test_bootstrap_ranges(self):
        np.random.seed(42)
        X = np.random.normal(size=(50,3))
        f = np.random.normal(size=(50,1))
        df = np.random.normal(size=(50,3))
        weights = np.ones((50,1)) / 50
        
        e, W = ss.active_subspace(df, weights)
        ssmethod = lambda X, f, df, weights: ss.active_subspace(df, weights)
        d = ss._bootstrap_ranges(e, W, None, None, df, weights, ssmethod, nboot=10)
        
        e, W = ss.ols_subspace(X, f, weights)
        ssmethod = lambda X, f, df, weights: ss.ols_subspace(X, f, weights)
        d = ss._bootstrap_ranges(e, W, X, f, None, weights, ssmethod, nboot=10)
        
        e, W = ss.qphd_subspace(X, f, weights)
        ssmethod = lambda X, f, df, weights: ss.qphd_subspace(X, f, weights)
        d = ss._bootstrap_ranges(e, W, X, f, None, weights, ssmethod, nboot=10)
        
        e, W = ss.opg_subspace(X, f, weights)
        ssmethod = lambda X, f, df, weights: ss.opg_subspace(X, f, weights)
        d = ss._bootstrap_ranges(e, W, X, f, None, weights, ssmethod, nboot=10)

    def test_eig_partition(self):
        np.random.seed(42)
        df = np.random.normal(size=(10,3))
        weights = np.ones((10,1)) / 10
        e, W = ss.active_subspace(df, weights)
        d = ss.eig_partition(e)

    def test_errbnd_partition(self):
        np.random.seed(42)
        df = np.random.normal(size=(10,3))
        weights = np.ones((10,1)) / 10
        e, W = ss.active_subspace(df, weights)
        ssmethod = lambda X, f, df, weights: ss.active_subspace(df, weights)
        e_br, sub_br, li_F = ss._bootstrap_ranges(e, W, None, None, df, weights, ssmethod, nboot=10)
        sub_err = sub_br[:,1].reshape((2, 1))
        d = ss.errbnd_partition(e, sub_err)

    def test_ladle_partition(self):
        np.random.seed(42)
        df = np.random.normal(size=(10,3))
        weights = np.ones((10,1)) / 10
        e, W = ss.active_subspace(df, weights)
        ssmethod = lambda X, f, df, weights: ss.active_subspace(df, weights)
        e_br, sub_br, li_F = ss._bootstrap_ranges(e, W, None, None, df, weights, ssmethod, nboot=10)
        d = ss.ladle_partition(e, li_F)
        
    def test_subspace_class(self):
        np.random.seed(42)
        X = np.random.normal(size=(50,3))
        f = np.random.normal(size=(50,1))
        df = np.random.normal(size=(50,3))
        weights = np.ones((50,1)) / 50
        
        sub = ss.Subspaces()
        sub.compute(X, f, df, weights)
        sub.compute(X, f, df, weights, sstype='AS')
        sub.compute(X, f, df, weights, sstype='OLS')
        sub.compute(X, f, df, weights, sstype='QPHD')
        sub.compute(X, f, df, weights, sstype='OPG')
        
        sub.compute(X, f, df, weights, sstype='AS', nboot=10)
        sub.compute(X, f, df, weights, sstype='OLS', nboot=10)
        sub.compute(X, f, df, weights, sstype='QPHD', nboot=10)
        sub.compute(X, f, df, weights, sstype='OPG', nboot=10)
        
        sub.compute(X, f, df, weights, sstype='AS', ptype='EVG', nboot=100)
        sub.compute(X, f, df, weights, sstype='AS', ptype='RS', nboot=100)
        sub.compute(X, f, df, weights, sstype='AS', ptype='LI', nboot=100)
        

if __name__ == '__main__':
    unittest.main()
