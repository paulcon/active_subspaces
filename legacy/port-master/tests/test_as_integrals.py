from unittest import TestCase
import unittest
import active_subspaces.response_surfaces as asm
import active_subspaces.integrals as asi
import active_subspaces.subspaces as ss
import active_subspaces.domains as dom
import numpy as np

class TestASIntegrals(TestCase):


    def quad_fun(self, x):
        A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
                    [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
                    [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
        x = x.reshape((3,1))
        return 0.5*np.dot(x.T,np.dot(A,x))

    def quad_dfun(self, x):
        A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
                    [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
                    [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
        return np.dot(A,x.reshape((3,1)))

    def test_rs_ubnd_int(self):
        np.random.seed(42)
        X0 = np.random.normal(size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(1)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X0, f0)

        I = asi.av_integrate(asrs, avm, 10)

    def test_rs_bnd_int(self):
        np.random.seed(42)
        X0 = np.random.uniform(-1.0,1.0,size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(1)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X0, f0)

        I = asi.av_integrate(asrs, avm, 10)

    def test_rs_ubnd_2d_int(self):
        np.random.seed(42)
        X0 = np.random.normal(size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(2)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X0, f0)

        I = asi.av_integrate(asrs, avm, 10)

    def test_rs_bnd_2d_int(self):
        np.random.seed(42)
        X0 = np.random.uniform(-1.0,1.0,size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(2)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X0, f0)

        I = asi.av_integrate(asrs, avm, 10)

    def test_fun_ubnd_int(self):
        np.random.seed(42)
        X0 = np.random.normal(size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(1)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)

        mu, lb, ub = asi.integrate(self.quad_fun, avm, 10)

    def test_fun_bnd_int(self):
        np.random.seed(42)
        X0 = np.random.uniform(-1.0,1.0,size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(1)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)

        mu, lb, ub = asi.integrate(self.quad_fun, avm, 10)

    def test_fun_ubnd_2d_int(self):
        np.random.seed(42)
        X0 = np.random.normal(size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(2)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)

        mu, lb, ub = asi.integrate(self.quad_fun, avm, 10)

    def test_fun_bnd_2d_int(self):
        np.random.seed(42)
        X0 = np.random.uniform(-1.0,1.0,size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(2)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)

        mu, lb, ub = asi.integrate(self.quad_fun, avm, 10)

if __name__ == '__main__':
    unittest.main()
