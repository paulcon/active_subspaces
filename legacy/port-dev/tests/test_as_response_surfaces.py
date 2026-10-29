from unittest import TestCase
import unittest
import active_subspaces.response_surfaces as asm
import active_subspaces.subspaces as ss
import active_subspaces.domains as dom
import active_subspaces.utils.simrunners as srun
import active_subspaces.utils.response_surfaces as rs
import numpy as np

class TestASResponseSurfaces(TestCase):

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

    def test_rs_0(self):
        np.random.seed(42)
        df0 = np.random.normal(size=(10,2))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asm.ActiveSubspaceResponseSurface(avm)

    def test_rs_data_train_gp_ubnd(self):
        np.random.seed(42)
        X0 = np.random.normal(size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X0, f0)

        np.random.seed(43)
        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_data_train_pr_ubnd(self):
        np.random.seed(42)
        X0 = np.random.normal(size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, respsurf=pr)
        asrs.train_with_data(X0, f0)

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_data_train_gp_bnd(self):
        np.random.seed(42)
        X0 = np.random.uniform(-1.0,1.0,size=(10,3))
        f0 = np.random.normal(size=(10,1))
        df0 = np.random.normal(size=(10,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X0, f0)

        np.random.seed(43)
        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_data_train_pr_bnd(self):
        np.random.seed(42)
        X0 = np.random.uniform(-1.0,1.0,size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)
        asrs.train_with_data(X0, f0)

        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_data_train_gp_ubnd_2d(self):
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

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_data_train_pr_ubnd_2d(self):
        np.random.seed(42)
        X0 = np.random.normal(size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(2)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)
        asrs.train_with_data(X0, f0)

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_data_train_gp_bnd_2d(self):
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

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_data_train_pr_bnd_2d(self):
        np.random.seed(42)
        X0 = np.random.uniform(-1.0,1.0,size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(2)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)
        asrs.train_with_data(X0, f0)

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_fun_train_gp_ubnd(self):
        np.random.seed(42)
        X0 = np.random.normal(size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)

        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_fun_train_pr_ubnd(self):
        np.random.seed(42)
        X0 = np.random.normal(size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)

        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_fun_train_gp_bnd(self):
        np.random.seed(42)
        X0 = np.random.uniform(-1.0,1.0,size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)

        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_fun_train_pr_bnd(self):
        np.random.seed(42)
        X0 = np.random.uniform(-1.0,1.0,size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)

        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_fun_train_gp_ubnd_2d(self):
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

        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_fun_train_pr_ubnd_2d(self):
        np.random.seed(42)
        X0 = np.random.normal(size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(2)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)

        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_fun_train_gp_bnd_2d(self):
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

        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

    def test_rs_fun_train_pr_bnd_2d(self):
        np.random.seed(42)
        X0 = np.random.uniform(-1.0,1.0,size=(50,3))
        f0 = np.random.normal(size=(50,1))
        df0 = np.random.normal(size=(50,3))

        sub = ss.Subspaces()
        sub.compute(df=df0)
        sub.partition(2)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)

        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)


if __name__ == '__main__':
    unittest.main()
