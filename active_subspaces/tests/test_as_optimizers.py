from unittest import TestCase
import unittest
import active_subspaces.as_response_surfaces as asm
import active_subspaces.as_optimizers as aso
import active_subspaces.subspaces as ss
import active_subspaces.domains as dom
import helper
import numpy as np

class TestASOptimizers(TestCase):
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
        xstar, fstar = aso.minimize(asrs, X, f)
        #np.savez('data/test_opt_0_1',xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_opt_0_1.npz')
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'ubnd min: {:6.4f}'.format(fstar)
        print 'ubnd xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
    
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
        xstar, fstar = aso.minimize(asrs, X, f)
        #np.savez('data/test_opt_0_2',xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_opt_0_2.npz')
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'bnd min: {:6.4f}'.format(fstar)
        print 'bnd xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
    
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
        xstar, fstar = aso.minimize(asrs, X, f)
        #np.savez('data/test_opt_0_3',xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_opt_0_3.npz')
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'ubnd 2d min: {:6.4f}'.format(fstar)
        print 'ubnd 2d xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
        
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
        xstar, fstar = aso.minimize(asrs, X, f)
        #np.savez('data/test_opt_0_4',xstar=xstar, fstar=fstar)
        data_test = helper.load_test_npz('test_opt_0_4.npz')
        np.testing.assert_equal(xstar, data_test['xstar'])
        np.testing.assert_equal(fstar, data_test['fstar'])
                
        print '\n'
        print 'bnd 2d min: {:6.4f}'.format(fstar)
        print 'bnd 2d xmin: {:6.4f}, {:6.4f}, {:6.4f}'.format(xstar[0,0],xstar[0,1],xstar[0,2])
        
if __name__ == '__main__':
    unittest.main()