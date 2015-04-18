import numpy as np
from unittest import TestCase
import unittest
import active_subspaces.utils.designs as dn

class TestDesigns(TestCase):
    
    def test_interval_design(self):
        y = dn.interval_design(-1.0, 1.0, 3)
        ytrue = np.array([[-0.5], [0.], [0.5]])
        np.testing.assert_equal(y, ytrue)
    
    def test_maximin_design(self):
        vert = np.array([[-1.0,-1.0], [1.0,-1.0], [-1.0,1.0], [1.0,1.0]])
        Y = dn.maximin_design(vert, 1)
        np.testing.assert_almost_equal(Y, np.zeros((1,2)), decimal=6)
        
    def test_maximin_design_random_state(self):
        vert = np.array([[-1.0,-1.0], [1.0,-1.0], [-1.0,1.0], [1.0,1.0]])
        np.random.seed(99)
        a = np.random.normal(size=(1,2))
        Y1 = dn.maximin_design(vert, 3)
        np.random.seed(999)
        b = np.random.normal(size=(2,1))
        Y2 = dn.maximin_design(vert, 3)
        np.testing.assert_almost_equal(Y1, Y2, decimal=6)
  
    def test_maximin_design_repeatable(self):
        vert = np.array([[-1.0,-1.0], [1.0,-1.0], [-1.0,1.0], [1.0,1.0]])
        Y1 = dn.maximin_design(vert, 10)
        Y2 = dn.maximin_design(vert, 10)
        np.testing.assert_equal(Y1, Y2)
        
    def test_gauss_hermite_design_1(self):
        Y = dn.gauss_hermite_design([1])
        Ytrue = np.array([[0.0]])
        np.testing.assert_equal(Y, Ytrue)
        
    def test_gauss_hermite_design_2(self):
        Y = dn.gauss_hermite_design([1,1])
        Ytrue = np.array([[0.0, 0.0]])
        np.testing.assert_equal(Y, Ytrue)
    
if __name__ == '__main__':
    unittest.main()