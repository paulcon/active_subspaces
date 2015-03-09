from unittest import TestCase
import unittest
import active_subspaces.utils.response_surfaces as rs
import helper
import numpy as np
import pdb

class TestResponseSurfaces(TestCase):
    
    def test_full_index_set(self):
        I = rs.full_index_set(7,3)
        data = helper.load_test_npz('test_full_index_set_7_3.npz')
        np.testing.assert_equal(I,data['I'])
        
    def test_polynomial_bases(self):
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        data = helper.load_test_npz('test_poly_bases_3.npz')
        B, I = rs.polynomial_bases(X, 3)
        np.testing.assert_equal(B, data['B'])
        np.testing.assert_equal(I, data['I'])

    def test_grad_polynomial_bases(self):
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        data = helper.load_test_npz('test_grad_poly_bases_3.npz')
        dB = rs.grad_polynomial_bases(X, 3)
        np.testing.assert_equal(dB,data['dB'])    
        
    def test_grad_polynomial_bases_fd(self):
        data = helper.load_test_npz('test_points_10_2.npz')
        X0 = data['X']
        data = helper.load_test_npz('test_grad_poly_bases_3.npz')
        
        dB = rs.grad_polynomial_bases(X0, 3)
        e = 1e-6
        B0 = rs.polynomial_bases(X0, 3)[0]
        
        X1 = X0.copy()
        X1[:,0] += e
        B1 = rs.polynomial_bases(X1, 3)[0]
        dB1 = (B1 - B0)/e
        np.testing.assert_array_almost_equal(dB[:,:,0], dB1, decimal=5)
        
        X2 = X0.copy()
        X2[:,1] += e
        B2 = rs.polynomial_bases(X2, 3)[0]
        dB2 = (B2 - B0)/e
        np.testing.assert_array_almost_equal(dB[:,:,1], dB2, decimal=5)
    
    def test_exponential_squared_covariance(self):
        
        data = helper.load_test_npz('test_points_10_2.npz')
        X1 = data['X']
        X2 = X1.copy()
        C = rs.exponential_squared_covariance(X1, X2, 1.0, np.array([1.0,1.0]))
        
        data = helper.load_test_npz('test_exp_cov.npz')
        np.testing.assert_equal(C, data['C'])
        
    def test_grad_exponential_squared_covariance_fd(self):
        
        data = helper.load_test_npz('test_points_10_2.npz')
        X1 = data['X']
        X2 = X1.copy()
        C0 = rs.exponential_squared_covariance(X1, X2, 1.0, np.array([1.0,1.0]))
        
        dC = rs.grad_exponential_squared_covariance(X1, X2, 1.0, np.array([1.0,1.0]))
        e = 1e-6
        
        X2p1 = X2.copy()
        X2p1[:,0] += e
        C1 = rs.exponential_squared_covariance(X1, X2p1, 1.0, np.array([1.0,1.0]))
        dC1 = (C1 - C0)/e
        np.testing.assert_array_almost_equal(dC[:,:,0], dC1, decimal=5)
        
        X2p2 = X2.copy()
        X2p2[:,1] += e
        C2 = rs.exponential_squared_covariance(X1, X2p2, 1.0, np.array([1.0,1.0]))
        dC2 = (C2 - C0)/e
        np.testing.assert_array_almost_equal(dC[:,:,1], dC2, decimal=5)
        
    def test_grad_exponential_squared_covariance(self):
    
        data = helper.load_test_npz('test_points_10_2.npz')
        X1 = data['X']
        X2 = X1.copy()
        dC = rs.grad_exponential_squared_covariance(X1, X2, 1.0, np.array([1.0,1.0]))
        
        data = helper.load_test_npz('test_grad_exp_cov.npz')
        np.testing.assert_equal(dC, data['dC'])



if __name__ == '__main__':
    unittest.main()