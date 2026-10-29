from unittest import TestCase
import unittest
import active_subspaces.gradfree as gf
import active_subspaces.utils.simrunners as simrun
import numpy as np

class TestGradFree(TestCase):


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

    def test_measurement_mats(self):
        m, k, M = 3, 2, 10
        E = gf._gauss_measurements(m, k, M)
        E = gf._bernoulli_measurements(m, k, M)
        E = gf._orthogonal_measurements(m, k, M)
        
    def test_gradient_measurements(self):
        m, k, M = 3, 2, 10
        sr = simrun.SimulationRunner(self.quad_fun)
        
        X = np.random.uniform(-1., 1., size=(M, m))
        E = gf._gauss_measurements(m, k, M)
        dfm = gf.gradient_measurements(sr, X, E)
    
    def test_simple_projection(self):
        m, k, M = 3, 2, 100
        sr = simrun.SimulationRunner(self.quad_fun)
        
        X = np.random.uniform(-1., 1., size=(M, m))
        E = gf._gauss_measurements(m, k, M)
        dfm = gf.gradient_measurements(sr, X, E)
        
        W, V, s = gf.simple_projection(dfm, E)
        
    def test_alternating_minimization(self):
        m, k, M = 3, 2, 100
        sr = simrun.SimulationRunner(self.quad_fun)
        
        X = np.random.uniform(-1., 1., size=(M, m))
        E = gf._gauss_measurements(m, k, M)
        dfm = gf.gradient_measurements(sr, X, E)
        
        W, V, s = gf.alternating_minimization(dfm, E, 2)
    
if __name__ == '__main__':
    unittest.main()
