from unittest import TestCase
import unittest
import active_subspaces.subspaces as ss
import helper
import numpy as np

class TestSubspaces(TestCase):
    writeData = False

    def test_spectral_decomposition_0(self):
        data = helper.load_test_npz('test_spec_decomp_0.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']
        e, W = ss.spectral_decomposition(df0)
        np.testing.assert_almost_equal(e, e0)
        np.testing.assert_almost_equal(W, W0)

    def test_spectral_decomposition_1(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']
        e, W = ss.spectral_decomposition(df0)
        np.testing.assert_almost_equal(e, e0)
        np.testing.assert_almost_equal(W, W0)

    def test_compute_partition(self):
        e = np.array([1.0, 1.0, 0.5, 0.5]).reshape((4,1))
        n = ss.compute_partition(e)
        np.testing.assert_almost_equal(n, 2)

    def test_bootstrap_ranges_0(self):
        data = helper.load_test_npz('test_spec_decomp_0.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']
        np.random.seed(42)
        e_br, sub_br = ss.bootstrap_ranges(df0, e0, W0, n_boot=100)

        if self.writeData:
            np.savez('data/test_spec_br_0', e_br=e_br, sub_br=sub_br)
        data_br = helper.load_test_npz('test_spec_br_0.npz')
        np.testing.assert_almost_equal(e_br, data_br['e_br'])
        np.testing.assert_almost_equal(sub_br, data_br['sub_br'])

    def test_bootstrap_ranges_1(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']
        np.random.seed(42)
        e_br, sub_br = ss.bootstrap_ranges(df0, e0, W0, n_boot=100)

        if self.writeData:
            np.savez('data/test_spec_br_1', e_br=e_br, sub_br=sub_br)
        data_br = helper.load_test_npz('test_spec_br_1.npz')
        np.testing.assert_almost_equal(e_br, data_br['e_br'])
        np.testing.assert_almost_equal(sub_br, data_br['sub_br'])

    def test_subspaces_0(self):
        data = helper.load_test_npz('test_spec_decomp_0.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        np.random.seed(42)
        sub.compute(df0, n_boot=100)
        np.testing.assert_almost_equal(sub.eigenvalues, e0)
        np.testing.assert_almost_equal(sub.eigenvectors, W0)

        data_br = helper.load_test_npz('test_spec_br_0.npz')
        np.testing.assert_almost_equal(sub.e_br, data_br['e_br'])
        np.testing.assert_almost_equal(sub.sub_br, data_br['sub_br'])

    def test_subspaces_1(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        np.random.seed(42)
        sub.compute(df0, n_boot=100)
        np.testing.assert_almost_equal(sub.eigenvalues, e0)
        np.testing.assert_almost_equal(sub.eigenvectors, W0)

        data_br = helper.load_test_npz('test_spec_br_1.npz')
        np.testing.assert_almost_equal(sub.e_br, data_br['e_br'])
        np.testing.assert_almost_equal(sub.sub_br, data_br['sub_br'])

    def test_subspaces_2(self):
        data = helper.load_test_npz('test_spec_decomp_0.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        np.random.seed(42)
        sub.compute(df0, n_boot=0)
        np.testing.assert_almost_equal(sub.eigenvalues, e0)
        np.testing.assert_almost_equal(sub.eigenvectors, W0)
        np.testing.assert_equal(sub.e_br, None)
        np.testing.assert_equal(sub.sub_br, None)

    def test_subspaces_3(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        np.random.seed(42)
        sub.compute(df0, n_boot=0)
        np.testing.assert_almost_equal(sub.eigenvalues, e0)
        np.testing.assert_almost_equal(sub.eigenvectors, W0)
        np.testing.assert_equal(sub.e_br, None)
        np.testing.assert_equal(sub.sub_br, None)


if __name__ == '__main__':
    unittest.main()
