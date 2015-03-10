from unittest import TestCase
import unittest
import active_subspaces.subspaces as ss
import helper
import numpy as np

class TestSubspaces(TestCase):

    def test_spectral_decomposition(self):
        data = helper.load_test_npz('test_spec_decomp.npz')
        df = data['df']
        e, W = ss.spectral_decomposition(df)
        np.testing.assert_equal(e, data['e'])
        np.testing.assert_equal(W, data['W'])
        
    def test_compute_partition(self):
        e = np.array([1.0, 1.0, 0.5, 0.5])
        n = ss.compute_partition(e)
        np.testing.assert_equal(n, 2)
        
    def test_bootstrap_ranges(self):
        data = helper.load_test_npz('test_spec_decomp.npz')
        df, e, W = data['df'], data['e'], data['W']
        np.random.seed(1234)
        e_br, sub_br = ss.bootstrap_ranges(df, e, W, n_boot=100)
        data = helper.load_test_npz('test_spec_br.npz')
        np.testing.assert_equal(e_br, data['e_br'])
        np.testing.assert_equal(sub_br, data['sub_br'])

if __name__ == '__main__':
    unittest.main()