from unittest import TestCase
import unittest
import active_subspaces.sdr as sdr
import active_subspaces.utils.simrunners as sr
import helper
import numpy as np

class TestSDR(TestCase):
    def quad_fun(self, x):
        A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
                    [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
                    [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
        x = x.reshape((3,1))
        return 0.5*np.dot(x.T,np.dot(A,x))
        
    def test_lingrad(self):
        
        np.random.seed(42)
        X = np.random.normal(size=(100,3))
        f = sr.SimulationRunner(self.quad_fun).run(X)
        w = sdr.lingrad(X, f)
        
        #np.savez('data/test_sdr_0_0',w=w)
        data_test = helper.load_test_npz('test_sdr_0_0.npz')
        np.testing.assert_equal(w, data_test['w'])
    
    def test_quick_check(self):
        
        np.random.seed(42)
        X = np.random.normal(size=(100,3))
        f = sr.SimulationRunner(self.quad_fun).run(X)
        w = sdr.quick_check(X, f)
        
        #np.savez('data/test_sdr_0_1',w=w)
        data_test = helper.load_test_npz('test_sdr_0_1.npz')
        np.testing.assert_equal(w, data_test['w'])
        
    def test_quadratic_model_check(self):
        
        np.random.seed(42)
        X = np.random.normal(size=(100,3))
        f = sr.SimulationRunner(self.quad_fun).run(X)
        gamma = 0.33*np.ones((3,1))
        e, W = sdr.quadratic_model_check(X, f, gamma)
        
        #np.savez('data/test_sdr_0_2',e=e, W=W)
        data_test = helper.load_test_npz('test_sdr_0_2.npz')
        np.testing.assert_equal(e, data_test['e']) 
        np.testing.assert_equal(W, data_test['W']) 

if __name__ == '__main__':
    unittest.main()