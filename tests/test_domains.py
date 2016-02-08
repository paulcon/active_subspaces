from unittest import TestCase
import unittest
import active_subspaces.subspaces as ss
import active_subspaces.domains as dom
import numpy as np
import pdb

class TestDomains(TestCase):
    
    def test_unbounded_active_variable_domain(self):
        np.random.seed(42)
        df = np.random.normal(size=(10,3))
        
        sub = ss.Subspaces()
        sub.compute(df=df)
        uavd = dom.UnboundedActiveVariableDomain(sub)

    def test_bounded_active_variable_domain_0(self):
        np.random.seed(42)
        df = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df)

        bavd = dom.BoundedActiveVariableDomain(sub)

    def test_bounded_active_variable_domain_1(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)

        bavd = dom.BoundedActiveVariableDomain(sub)

    def test_unbounded_active_variable_map_0(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        m, n = sub.W1.shape

        uavd = dom.UnboundedActiveVariableDomain(sub)
        uavm = dom.UnboundedActiveVariableMap(uavd)

        X = np.random.normal(size=(100,m))
        Y,Z = uavm.forward(X)
        X0 = np.dot(Y, sub.W1.T) + np.dot(Z, sub.W2.T)
        np.testing.assert_almost_equal(X0, X)

    def test_unbounded_active_variable_map_1(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        m, n = sub.W1.shape

        uavd = dom.UnboundedActiveVariableDomain(sub)
        uavm = dom.UnboundedActiveVariableMap(uavd)

        X = np.random.normal(size=(100,m))
        Y,Z = uavm.forward(X)
        X0 = np.dot(Y, sub.W1.T) + np.dot(Z, sub.W2.T)
        np.testing.assert_almost_equal(X0, X)

    def test_unbounded_active_variable_map_2(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        m, n = sub.W1.shape

        uavd = dom.UnboundedActiveVariableDomain(sub)
        uavm = dom.UnboundedActiveVariableMap(uavd)

        X = np.random.normal(size=(100,m))
        Y,Z = uavm.forward(X)
        X0 = uavm.inverse(Y, N=10)[0]
        np.testing.assert_almost_equal(np.dot(X0, sub.W1), np.kron(Y, np.ones((10,1))) )

    def test_unbounded_active_variable_map_3(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        m, n = sub.W1.shape

        uavd = dom.UnboundedActiveVariableDomain(sub)
        uavm = dom.UnboundedActiveVariableMap(uavd)

        X = np.random.normal(size=(100,m))
        Y,Z = uavm.forward(X)
        X0 = uavm.inverse(Y, N=10)[0]
        np.testing.assert_almost_equal(np.dot(X0, sub.W1), np.kron(Y, np.ones((10,1))) )

    def test_bounded_active_variable_map_0(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        m, n = sub.W1.shape

        bavd = dom.BoundedActiveVariableDomain(sub)
        bavm = dom.BoundedActiveVariableMap(bavd)

        X = np.random.uniform(-1.0,1.0,size=(100,m))
        Y,Z = bavm.forward(X)
        X0 = np.dot(Y, sub.W1.T) + np.dot(Z, sub.W2.T)
        np.testing.assert_almost_equal(X0, X)

    def test_bounded_active_variable_map_1(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        m, n = sub.W1.shape

        bavd = dom.BoundedActiveVariableDomain(sub)
        bavm = dom.BoundedActiveVariableMap(bavd)

        X = np.random.uniform(-1.0,1.0,size=(100,m))
        Y,Z = bavm.forward(X)
        X0 = np.dot(Y, sub.W1.T) + np.dot(Z, sub.W2.T)
        np.testing.assert_almost_equal(X0, X)
    
    def test_bounded_active_variable_map_2(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        m, n = sub.W1.shape

        bavd = dom.BoundedActiveVariableDomain(sub)
        bavm = dom.BoundedActiveVariableMap(bavd)

        X = np.random.uniform(-1.0,1.0,size=(10,m))
        Y,Z = bavm.forward(X)
        X0 = bavm.inverse(Y, N=10)[0]
        np.testing.assert_almost_equal(np.dot(X0, sub.W1), np.kron(Y, np.ones((10,1))) )
        np.testing.assert_equal(np.floor(np.abs(X0)), np.zeros(X0.shape))
    
    def test_bounded_active_variable_map_3(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        m, n = sub.W1.shape
        
        bavd = dom.BoundedActiveVariableDomain(sub)
        bavm = dom.BoundedActiveVariableMap(bavd)

        X = np.random.uniform(-1.0,1.0,size=(10,m))
        Y,Z = bavm.forward(X)
        X0 = bavm.inverse(Y, N=10)[0]
        np.testing.assert_almost_equal(np.dot(X0, sub.W1), np.kron(Y, np.ones((10,1))) )
        np.testing.assert_equal(np.floor(np.abs(X0)), np.zeros(X0.shape))
    
    def test_rejection_sample_z(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        W1, W2 = sub.W1, sub.W2
        m, n = W1.shape

        np.random.seed(43)
        x = np.random.uniform(-1.0,1.0,size=(1,m))
        y = np.dot(x, W1).reshape((n, ))
        N = 10
        np.random.seed(42)
        Z = dom.rejection_sampling_z(N, y, W1, W2)

        

    def test_hit_and_run_z(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        W1, W2 = sub.W1, sub.W2
        m, n = W1.shape

        np.random.seed(43)
        x = np.random.uniform(-1.0,1.0,size=(1,m))
        y = np.dot(x, W1).reshape((n, ))
        N = 10
        np.random.seed(42)
        Z = dom.hit_and_run_z(N, y, W1, W2)

    def test_random_walk_z(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        W1, W2 = sub.W1, sub.W2
        m, n = W1.shape

        np.random.seed(43)
        x = np.random.uniform(-1.0,1.0,size=(1,m))
        y = np.dot(x, W1).reshape((n, ))
        N = 10
        np.random.seed(42)
        Z = dom.random_walk_z(N, y, W1, W2)


    def test_sample_z(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        W1, W2 = sub.W1, sub.W2
        m, n = W1.shape

        np.random.seed(43)
        x = np.random.uniform(-1.0,1.0,size=(1,m))
        y = np.dot(x, W1).reshape((n, ))
        N = 10
        np.random.seed(42)
        Z = dom.sample_z(N, y, W1, W2)

    

if __name__ == '__main__':
    unittest.main()
