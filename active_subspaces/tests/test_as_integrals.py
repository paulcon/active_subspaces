from unittest import TestCase
import unittest
import active_subspaces.response_surfaces as asm
import active_subspaces.integrals as asi
import active_subspaces.subspaces as ss
import active_subspaces.domains as dom
import helper
import numpy as np

class TestASIntegrals(TestCase):
    
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
        
    def test_rs_ubnd_int(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        sub = ss.Subspaces()
        sub.compute(df)
        
        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X, f)
        
        np.random.seed(43)
        I = asi.av_integrate(asrs.respsurf, avm, 10)
        if self.writeData:
            np.savez('data/test_int_0_1',I=I)
        data_test = helper.load_test_npz('test_int_0_1.npz')
        np.testing.assert_almost_equal(I, data_test['I'])
        
        print '\n'
        print 'rs ubnd: {:6.4f}'.format(I)

    def test_rs_bnd_int(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        sub = ss.Subspaces()
        sub.compute(df)
        
        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        asrs = asm.ActiveSubspaceResponseSurface(avm)
        asrs.train_with_data(X, f)
        
        np.random.seed(43)
        I = asi.av_integrate(asrs.respsurf, avm, 10)
        if self.writeData:
            np.savez('data/test_int_0_2',I=I)
        data_test = helper.load_test_npz('test_int_0_2.npz')
        np.testing.assert_almost_equal(I, data_test['I'])
        
        print '\n'
        print 'rs bnd: {:6.4f}'.format(I)
                
    def test_rs_ubnd_2d_int(self):
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
        I = asi.av_integrate(asrs.respsurf, avm, 10)
        if self.writeData:
            np.savez('data/test_int_0_3',I=I)
        data_test = helper.load_test_npz('test_int_0_3.npz')
        np.testing.assert_almost_equal(I, data_test['I'])
        
        print '\n'
        print 'rs ubnd 2d: {:6.4f}'.format(I)
        
    def test_rs_bnd_2d_int(self):
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
        I = asi.av_integrate(asrs.respsurf, avm, 10)
        if self.writeData:
            np.savez('data/test_int_0_4',I=I)
        data_test = helper.load_test_npz('test_int_0_4.npz')
        np.testing.assert_almost_equal(I, data_test['I'])
        
        print '\n'
        print 'rs bnd 2d: {:6.4f}'.format(I)
        
    def test_fun_ubnd_int(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        sub = ss.Subspaces()
        sub.compute(df)
        
        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        
        np.random.seed(43)
        mu, lb, ub = asi.integrate(self.quad_fun, avm, 10)
        if self.writeData:
            np.savez('data/test_int_1_1',mu=mu,lb=lb,ub=ub)
        data_test = helper.load_test_npz('test_int_1_1.npz')
        np.testing.assert_almost_equal(mu, data_test['mu'])
        np.testing.assert_almost_equal(lb, data_test['lb'])
        np.testing.assert_almost_equal(ub, data_test['ub'])
        
        print '\n'
        print 'fun ubnd: {:6.4f}, {:6.4f}, {:6.4f}'.format(lb,mu,ub)
        
    def test_fun_bnd_int(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        sub = ss.Subspaces()
        sub.compute(df)
        
        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        
        np.random.seed(43)
        mu, lb, ub = asi.integrate(self.quad_fun, avm, 10)
        if self.writeData:
            np.savez('data/test_int_1_2',mu=mu,lb=lb,ub=ub)
        data_test = helper.load_test_npz('test_int_1_2.npz')
        np.testing.assert_almost_equal(mu, data_test['mu'])
        np.testing.assert_almost_equal(lb, data_test['lb'])
        np.testing.assert_almost_equal(ub, data_test['ub'])
        
        print '\n'
        print 'fun bnd: {:6.4f}, {:6.4f}, {:6.4f}'.format(lb,mu,ub)
        
    def test_fun_ubnd_2d_int(self):
        data = helper.load_test_npz('test_rs_0.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        sub = ss.Subspaces()
        sub.compute(df)
        sub.partition(2)
        
        avd = dom.UnboundedActiveVariableDomain(sub)
        avm = dom.UnboundedActiveVariableMap(avd)
        
        np.random.seed(43)
        mu, lb, ub = asi.integrate(self.quad_fun, avm, 10)
        if self.writeData:
            np.savez('data/test_int_1_3',mu=mu,lb=lb,ub=ub)
        data_test = helper.load_test_npz('test_int_1_3.npz')
        np.testing.assert_almost_equal(mu, data_test['mu'])
        np.testing.assert_almost_equal(lb, data_test['lb'])
        np.testing.assert_almost_equal(ub, data_test['ub'])
        
        print '\n'
        print 'fun 2d ubnd: {:6.4f}, {:6.4f}, {:6.4f}'.format(lb,mu,ub)
        
    def test_fun_bnd_2d_int(self):
        data = helper.load_test_npz('test_rs_1.npz')
        X, f, df = data['X'], data['f'], data['df']
        
        sub = ss.Subspaces()
        sub.compute(df)
        sub.partition(2)
        
        avd = dom.BoundedActiveVariableDomain(sub)
        avm = dom.BoundedActiveVariableMap(avd)
        
        np.random.seed(43)
        mu, lb, ub = asi.integrate(self.quad_fun, avm, 10)
        if self.writeData:
            np.savez('data/test_int_1_4',mu=mu,lb=lb,ub=ub)
        data_test = helper.load_test_npz('test_int_1_4.npz')
        np.testing.assert_almost_equal(mu, data_test['mu'])
        np.testing.assert_almost_equal(lb, data_test['lb'])
        np.testing.assert_almost_equal(ub, data_test['ub'])
        
        print '\n'
        print 'fun bnd 2d: {:6.4f}, {:6.4f}, {:6.4f}'.format(lb,mu,ub)
  
if __name__ == '__main__':
    unittest.main()