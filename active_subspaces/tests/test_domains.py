from unittest import TestCase
import unittest
import active_subspaces.subspaces as ss
import active_subspaces.domains as dom
import helper
import numpy as np
import pdb

class TestDomains(TestCase):
    writeData = False

    def test_unbounded_active_variable_domain(self):
        data = helper.load_test_npz('test_spec_decomp_0.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)

        uavd = dom.UnboundedActiveVariableDomain(sub)
        np.testing.assert_equal(uavd.vertY, None)
        np.testing.assert_equal(uavd.vertX, None)
        np.testing.assert_equal(uavd.convhull, None)
        np.testing.assert_equal(uavd.constraints, None)
        np.testing.assert_almost_equal(uavd.n, sub.W1.shape[1])
        np.testing.assert_almost_equal(uavd.m, sub.W1.shape[0])

    def test_bounded_active_variable_domain_0(self):
        data = helper.load_test_npz('test_spec_decomp_0.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)

        data_bavd = helper.load_test_npz('bavd_0.npz')
        bavd = dom.BoundedActiveVariableDomain(sub)

        np.testing.assert_almost_equal(bavd.vertY, np.dot(bavd.vertX, sub.W1))
        np.testing.assert_almost_equal(bavd.vertY, data_bavd['vertY'])
        np.testing.assert_almost_equal(bavd.vertX, data_bavd['vertX'])
        np.testing.assert_almost_equal(bavd.n, sub.W1.shape[1])
        np.testing.assert_almost_equal(bavd.m, sub.W1.shape[0])

    def test_bounded_active_variable_domain_1(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)

        data_bavd = helper.load_test_npz('bavd_1.npz')
        np.random.seed(42)
        bavd = dom.BoundedActiveVariableDomain(sub)

        np.testing.assert_almost_equal(bavd.vertY, np.dot(bavd.vertX, sub.W1))
        np.testing.assert_almost_equal(bavd.vertY, data_bavd['vertY'])
        np.testing.assert_almost_equal(bavd.vertX, data_bavd['vertX'])
        np.testing.assert_almost_equal(bavd.n, sub.W1.shape[1])
        np.testing.assert_almost_equal(bavd.m, sub.W1.shape[0])

    def test_unbounded_active_variable_map_0(self):
        data = helper.load_test_npz('test_spec_decomp_0.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        m, n = sub.W1.shape

        uavd = dom.UnboundedActiveVariableDomain(sub)
        uavm = dom.UnboundedActiveVariableMap(uavd)

        X = np.random.normal(size=(100,m))
        Y,Z = uavm.forward(X)
        X0 = np.dot(Y, sub.W1.T) + np.dot(Z, sub.W2.T)
        np.testing.assert_almost_equal(X0, X)

    def test_unbounded_active_variable_map_1(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        m, n = sub.W1.shape

        uavd = dom.UnboundedActiveVariableDomain(sub)
        uavm = dom.UnboundedActiveVariableMap(uavd)

        X = np.random.normal(size=(100,m))
        Y,Z = uavm.forward(X)
        X0 = np.dot(Y, sub.W1.T) + np.dot(Z, sub.W2.T)
        np.testing.assert_almost_equal(X0, X)

    def test_unbounded_active_variable_map_2(self):
        data = helper.load_test_npz('test_spec_decomp_0.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        m, n = sub.W1.shape

        uavd = dom.UnboundedActiveVariableDomain(sub)
        uavm = dom.UnboundedActiveVariableMap(uavd)

        X = np.random.normal(size=(100,m))
        Y,Z = uavm.forward(X)
        X0 = uavm.inverse(Y, N=10)[0]
        np.testing.assert_almost_equal(np.dot(X0, sub.W1), np.kron(Y, np.ones((10,1))) )

    def test_unbounded_active_variable_map_3(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        m, n = sub.W1.shape

        uavd = dom.UnboundedActiveVariableDomain(sub)
        uavm = dom.UnboundedActiveVariableMap(uavd)

        X = np.random.normal(size=(100,m))
        Y,Z = uavm.forward(X)
        X0 = uavm.inverse(Y, N=10)[0]
        np.testing.assert_almost_equal(np.dot(X0, sub.W1), np.kron(Y, np.ones((10,1))) )

    def test_bounded_active_variable_map_0(self):
        data = helper.load_test_npz('test_spec_decomp_0.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        m, n = sub.W1.shape

        bavd = dom.BoundedActiveVariableDomain(sub)
        bavm = dom.BoundedActiveVariableMap(bavd)

        X = np.random.uniform(-1.0,1.0,size=(100,m))
        Y,Z = bavm.forward(X)
        X0 = np.dot(Y, sub.W1.T) + np.dot(Z, sub.W2.T)
        np.testing.assert_almost_equal(X0, X)

    def test_bounded_active_variable_map_1(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        m, n = sub.W1.shape

        bavd = dom.BoundedActiveVariableDomain(sub)
        bavm = dom.BoundedActiveVariableMap(bavd)

        X = np.random.uniform(-1.0,1.0,size=(100,m))
        Y,Z = bavm.forward(X)
        X0 = np.dot(Y, sub.W1.T) + np.dot(Z, sub.W2.T)
        np.testing.assert_almost_equal(X0, X)

    def test_bounded_active_variable_map_2(self):
        data = helper.load_test_npz('test_spec_decomp_0.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        m, n = sub.W1.shape

        bavd = dom.BoundedActiveVariableDomain(sub)
        bavm = dom.BoundedActiveVariableMap(bavd)

        X = np.random.uniform(-1.0,1.0,size=(10,m))
        Y,Z = bavm.forward(X)
        X0 = bavm.inverse(Y, N=10)[0]
        np.testing.assert_almost_equal(np.dot(X0, sub.W1), np.kron(Y, np.ones((10,1))) )

    def test_bounded_active_variable_map_3(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        m, n = sub.W1.shape

        bavd = dom.BoundedActiveVariableDomain(sub)
        bavm = dom.BoundedActiveVariableMap(bavd)

        X = np.random.uniform(-1.0,1.0,size=(10,m))
        Y,Z = bavm.forward(X)
        X0 = bavm.inverse(Y, N=10)[0]
        np.testing.assert_almost_equal(np.dot(X0, sub.W1), np.kron(Y, np.ones((10,1))) )

    def test_rejection_sample_z(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        W1, W2 = sub.W1, sub.W2
        m, n = W1.shape

        np.random.seed(43)
        x = np.random.uniform(-1.0,1.0,size=(1,m))
        y = np.dot(x, W1).reshape((n, ))
        N = 10
        np.random.seed(42)
        Z = dom.rejection_sampling_z(N, y, W1, W2)

        if self.writeData:
            np.savez('data/test_sampling_z_0_0',Z=Z)
        data_test = helper.load_test_npz('test_sampling_z_0_0.npz')
        np.testing.assert_almost_equal(Z, data_test['Z'])

    def test_hit_and_run_z(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        W1, W2 = sub.W1, sub.W2
        m, n = W1.shape

        np.random.seed(43)
        x = np.random.uniform(-1.0,1.0,size=(1,m))
        y = np.dot(x, W1).reshape((n, ))
        N = 10
        np.random.seed(42)
        Z = dom.hit_and_run_z(N, y, W1, W2)

        if self.writeData:
            np.savez('data/test_sampling_z_0_1',Z=Z)
        data_test = helper.load_test_npz('test_sampling_z_0_1.npz')
        np.testing.assert_almost_equal(Z, data_test['Z'])

    def test_random_walk_z(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        W1, W2 = sub.W1, sub.W2
        m, n = W1.shape

        np.random.seed(43)
        x = np.random.uniform(-1.0,1.0,size=(1,m))
        y = np.dot(x, W1).reshape((n, ))
        N = 10
        np.random.seed(42)
        Z = dom.random_walk_z(N, y, W1, W2)

        if self.writeData:
            np.savez('data/test_sampling_z_0_2',Z=Z)
        data_test = helper.load_test_npz('test_sampling_z_0_2.npz')
        np.testing.assert_almost_equal(Z, data_test['Z'])

    def test_sample_z(self):
        data = helper.load_test_npz('test_spec_decomp_1.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        W1, W2 = sub.W1, sub.W2
        m, n = W1.shape

        np.random.seed(43)
        x = np.random.uniform(-1.0,1.0,size=(1,m))
        y = np.dot(x, W1).reshape((n, ))
        N = 10
        np.random.seed(42)
        Z = dom.sample_z(N, y, W1, W2)

        data_test = helper.load_test_npz('test_sampling_z_0_0.npz')
        np.testing.assert_almost_equal(Z, data_test['Z'])


if __name__ == '__main__':
    unittest.main()
