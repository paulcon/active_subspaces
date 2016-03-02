from unittest import TestCase
import unittest
import active_subspaces.response_surfaces as asm
import active_subspaces.optimizers as aso
import active_subspaces.subspaces as ss
import active_subspaces.domains as dom
import active_subspaces.base as base
import numpy as np

class TestBase(TestCase):

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
        X = np.random.normal(size=(50,3))
        f = np.zeros((50,1))
        df = np.zeros((50,3))
        for i in range(50):
            x = X[i,:]
            f[i,0] = self.quad_fun(x)
            df[i,:] = self.quad_dfun(x).reshape((3, ))

        model = base.ActiveSubspaceReducedModel(3, False)
        model.build_from_data(X, f, df=df)

        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()

    def test_rs_bnd_int(self):
        np.random.seed(42)
        X = np.random.uniform(-1.,1.,size=(50,3))
        f = np.zeros((50,1))
        df = np.zeros((50,3))
        for i in range(50):
            x = X[i,:]
            f[i,0] = self.quad_fun(x)
            df[i,:] = self.quad_dfun(x).reshape((3, ))

        model = base.ActiveSubspaceReducedModel(3, True)
        model.build_from_data(X, f, df=df)

        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()

    def test_rs_ubnd_2d_int(self):
        np.random.seed(42)
        X = np.random.normal(size=(50,3))
        f = np.zeros((50,1))
        df = np.zeros((50,3))
        for i in range(50):
            x = X[i,:]
            f[i,0] = self.quad_fun(x)
            df[i,:] = self.quad_dfun(x).reshape((3, ))

        model = base.ActiveSubspaceReducedModel(3, False)
        model.build_from_data(X, f, df=df, avdim=2)

        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()

    def test_rs_bnd_2d_int(self):
        np.random.seed(42)
        X = np.random.uniform(-1.,1.,size=(50,3))
        f = np.zeros((50,1))
        df = np.zeros((50,3))
        for i in range(50):
            x = X[i,:]
            f[i,0] = self.quad_fun(x)
            df[i,:] = self.quad_dfun(x).reshape((3, ))

        model = base.ActiveSubspaceReducedModel(3, True)
        model.build_from_data(X, f, df=df, avdim=2)

        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()


    def test_rs_diag(self):
        np.random.seed(42)
        X = np.random.normal(size=(50,3))
        f = np.zeros((50,1))
        df = np.zeros((50,3))
        for i in range(50):
            x = X[i,:]
            f[i,0] = self.quad_fun(x)
            df[i,:] = self.quad_dfun(x).reshape((3, ))

        model = base.ActiveSubspaceReducedModel(3, False)
        model.build_from_data(X, f, df=df)

        model.diagnostics()

    def test_rs_predict(self):
        np.random.seed(42)
        X = np.random.normal(size=(50,3))
        f = np.zeros((50,1))
        df = np.zeros((50,3))
        for i in range(50):
            x = X[i,:]
            f[i,0] = self.quad_fun(x)
            df[i,:] = self.quad_dfun(x).reshape((3, ))

        modelN = base.ActiveSubspaceReducedModel(3, False)
        modelN.build_from_data(X, f, df=df)

        XN = np.random.normal(size=X.shape)
        modelN.predict(XN)

        modelU = base.ActiveSubspaceReducedModel(3, True)
        modelU.build_from_data(X, f, df=df)

        XU = np.random.uniform(-1.0, 1.0, size=X.shape)
        modelU.predict(XU)

    def test_fun_rs_ubnd_int(self):
        np.random.seed(42)
        X = np.random.normal(size=(50,3))
        f = np.zeros((50,1))
        df = np.zeros((50,3))
        for i in range(50):
            x = X[i,:]
            f[i,0] = self.quad_fun(x)
            df[i,:] = self.quad_dfun(x).reshape((3, ))

        model = base.ActiveSubspaceReducedModel(3, False)
        model.build_from_interface(self.quad_fun, avdim=1)

        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()

    def test_fun_rs_bnd_int(self):
        np.random.seed(42)
        X = np.random.uniform(-1.,1.,size=(100,3))
        f = np.zeros((100,1))
        df = np.zeros((100,3))
        for i in range(100):
            x = X[i,:]
            f[i,0] = self.quad_fun(x)
            df[i,:] = self.quad_dfun(x).reshape((3, ))

        model = base.ActiveSubspaceReducedModel(3, True)
        model.build_from_interface(self.quad_fun, avdim=1)

        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()


    def test_fun_rs_ubnd_2d_int(self):
        np.random.seed(42)
        X = np.random.normal(size=(100,3))
        f = np.zeros((100,1))
        df = np.zeros((100,3))
        for i in range(100):
            x = X[i,:]
            f[i,0] = self.quad_fun(x)
            df[i,:] = self.quad_dfun(x).reshape((3, ))

        model = base.ActiveSubspaceReducedModel(3, False)
        model.build_from_interface(self.quad_fun, avdim=2)

        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()


    def test_fun_rs_bnd_2d_int(self):
        np.random.seed(42)
        X = np.random.uniform(-1.,1.,size=(50,3))
        f = np.zeros((50,1))
        df = np.zeros((50,3))
        for i in range(50):
            x = X[i,:]
            f[i,0] = self.quad_fun(x)
            df[i,:] = self.quad_dfun(x).reshape((3, ))

        model = base.ActiveSubspaceReducedModel(3, True)
        model.build_from_interface(self.quad_fun, avdim=2)

        avg = model.average(20)[0]
        prob, pl, pu = model.probability(0.0, 1.0)
        fstar, xstar = model.minimum()


if __name__ == '__main__':
    unittest.main()
