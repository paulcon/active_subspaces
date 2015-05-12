from unittest import TestCase
import unittest
import active_subspaces.response_surfaces as asm
import active_subspaces.subspaces as ss
import active_subspaces.domains as dom
import active_subspaces.utils.simrunners as srun
import active_subspaces.utils.response_surfaces as rs
import helper
import numpy as np

class TestASResponseSurfaces(TestCase):
    writeData = False

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
        data = helper.load_test_npz('test_spec_decomp_0.npz')
        df0, e0, W0 = data['df'], data['e'], data['W']

        sub = ss.Subspaces()
        sub.compute(df0)
        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asm.ActiveSubspaceResponseSurface(avm)

    def test_rs_data_train_gp_ubnd(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X, f)

        np.random.seed(43)
        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)

        if self.writeData:
            np.savez('data/test_rs_0_1',ff=ff,dff=dff)
        data_test = helper.load_test_npz('test_rs_0_1.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'data gp ubnd'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_data_train_pr_ubnd(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, respsurf=pr)
        asrs.train_with_data(X, f)

        np.random.seed(43)
        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_0_2',ff=ff,dff=dff)

        data_test = helper.load_test_npz('test_rs_0_2.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'data pr ubnd'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_data_train_gp_bnd(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X, f)

        np.random.seed(43)
        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_1_0',ff=ff,dff=dff)
        data_test = helper.load_test_npz('test_rs_1_0.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'data gp bnd'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_data_train_pr_bnd(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)
        asrs.train_with_data(X, f)

        np.random.seed(43)
        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_1_1',ff=ff,dff=dff)

        data_test = helper.load_test_npz('test_rs_1_1.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'data pr bnd'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_data_train_gp_ubnd_2d(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)
        sub.partition(2)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X, f)

        np.random.seed(43)
        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_2_0',ff=ff,dff=dff)

        data_test = helper.load_test_npz('test_rs_2_0.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'data gp ubnd 2d'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_data_train_pr_ubnd_2d(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)
        sub.partition(2)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)
        asrs.train_with_data(X, f)

        np.random.seed(43)
        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_2_1',ff=ff,dff=dff)

        data_test = helper.load_test_npz('test_rs_2_1.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'data pr ubnd 2d'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_data_train_gp_bnd_2d(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)
        sub.partition(2)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X, f)

        np.random.seed(43)
        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_3_0',ff=ff,dff=dff)

        data_test = helper.load_test_npz('test_rs_3_0.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'data gp bnd 2d'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_data_train_pr_bnd_2d(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)
        sub.partition(2)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)
        asrs.train_with_data(X, f)

        np.random.seed(43)
        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_3_1',ff=ff,dff=dff)

        data_test = helper.load_test_npz('test_rs_3_1.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'data pr bnd 2d'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_fun_train_gp_ubnd(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)

        np.random.seed(43)
        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_4_0',ff=ff,dff=dff)
        data_test = helper.load_test_npz('test_rs_4_0.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'fun gp ubnd'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_fun_train_pr_ubnd(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)

        np.random.seed(43)
        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_4_1',ff=ff,dff=dff)
        data_test = helper.load_test_npz('test_rs_4_1.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'fun pr ubnd'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_fun_train_gp_bnd(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)

        np.random.seed(43)
        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_5_0',ff=ff,dff=dff)
        data_test = helper.load_test_npz('test_rs_5_0.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'fun gp bnd'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_fun_train_pr_bnd(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)

        np.random.seed(43)
        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_5_1',ff=ff,dff=dff)
        data_test = helper.load_test_npz('test_rs_5_1.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'fun pr bnd'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_fun_train_gp_ubnd_2d(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)
        sub.partition(2)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)

        np.random.seed(43)
        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_6_0',ff=ff,dff=dff)
        data_test = helper.load_test_npz('test_rs_6_0.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'fun gp ubnd 2d'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_fun_train_pr_ubnd_2d(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)
        sub.partition(2)

        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)

        np.random.seed(43)
        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.normal(size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_6_1',ff=ff,dff=dff)
        data_test = helper.load_test_npz('test_rs_6_1.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'fun pr ubnd 2d'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_fun_train_gp_bnd_2d(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)
        sub.partition(2)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)

        np.random.seed(43)
        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_7_0',ff=ff,dff=dff)
        data_test = helper.load_test_npz('test_rs_7_0.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'fun gp bnd 2d'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))

    def test_rs_fun_train_pr_bnd_2d(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']

        sub = ss.Subspaces()
        sub.compute(df)
        sub.partition(2)

        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        pr = rs.PolynomialApproximation()
        asrs = asm.ActiveSubspaceResponseSurface(avm, pr)

        np.random.seed(43)
        asrs.train_with_interface(self.quad_fun, 10)

        XX = np.random.uniform(-1.0,1.0,size=(10, 3))
        ff, dff = asrs.predict(XX, compgrad=True)
        if self.writeData:
            np.savez('data/test_rs_7_1',ff=ff,dff=dff)
        data_test = helper.load_test_npz('test_rs_7_1.npz')
        np.testing.assert_almost_equal(ff, data_test['ff'])
        np.testing.assert_almost_equal(asrs(XX), data_test['ff'])
        np.testing.assert_almost_equal(dff, data_test['dff'])
        np.testing.assert_almost_equal(asrs.gradient(XX), data_test['dff'])


        sr = srun.SimulationRunner(self.quad_fun)
        f_true = sr.run(XX)
        dsr = srun.SimulationGradientRunner(self.quad_dfun)
        df_true = dsr.run(XX)

        print '\n'
        print 'fun pr bnd 2d'
        print 'f error: {:6.4e}'.format(np.linalg.norm(ff-f_true)/np.linalg.norm(f_true))
        print 'df error: {:6.4e}'.format(np.linalg.norm(dff-df_true)/np.linalg.norm(df_true))


if __name__ == '__main__':
    unittest.main()
