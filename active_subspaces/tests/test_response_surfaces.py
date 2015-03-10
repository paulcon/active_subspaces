from unittest import TestCase
import unittest
import active_subspaces.utils.response_surfaces as rs
import helper
import numpy as np
import matplotlib.pyplot as plt
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
    
     
    def test_exact_polynomial_approximation_1d(self):
        data = helper.load_test_npz('train_points_10_2.npz')
        X = data['X']
        X_1d = X[:,0].copy()
        f_1d = 2 + 5*X_1d
        
        pr = rs.PolynomialRegression(N=1)
        pr.train(X_1d, f_1d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_1d_test = X[:,0].copy()
        f, df, v = pr.predict(X_1d_test, compgrad=True, compvar=True)
        np.testing.assert_almost_equal(f, 2+5*X_1d_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df, 5*np.ones((10,1)), decimal=10)
        
        f_1d = 2 - 3*X_1d + 5*X_1d*X_1d
        pr = rs.PolynomialRegression(N=2)
        pr.train(X_1d, f_1d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_1d_test = X[:,0].copy()
        f, df, v = pr.predict(X_1d_test, compgrad=True, compvar=True)
        f_test = 2 - 3*X_1d_test + 5*X_1d_test*X_1d_test
        df_test = -3 + 10*X_1d_test
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df, df_test.reshape((10,1)), decimal=10)
    
 
    def test_exact_polynomial_approximation_2d(self):
        data = helper.load_test_npz('train_points_10_2.npz')
        X = data['X']
        X_train = X.copy()
        f_2d = 2 + 5*X_train[:,0] - 4*X_train[:,1]
        
        pr = rs.PolynomialRegression(N=1)
        pr.train(X_train, f_2d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_test = X.copy()
        f, df, v = pr.predict(X_test, compgrad=True, compvar=True)
        f_test = 2 + 5*X_test[:,0] - 4*X_test[:,1]
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,0].reshape((10,1)), 5*np.ones((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,1].reshape((10,1)), -4*np.ones((10,1)), decimal=10)
        
        f_2d = 2 - 3*X_train[:,1] + 5*X_train[:,0]*X_train[:,1]
        pr = rs.PolynomialRegression(N=2)
        pr.train(X_train, f_2d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_test = X.copy()
        f, df, v = pr.predict(X_test, compgrad=True, compvar=True)
        f_test = 2 - 3*X_test[:,1] + 5*X_test[:,0]*X_test[:,1]
        df1_test = 5*X_test[:,1]
        df2_test = -3 + 5*X_test[:,0]
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,0].reshape((10,1)), df1_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,1].reshape((10,1)), df2_test.reshape((10,1)), decimal=10)
    
 
    def test_exact_gp_approximation_1d(self):
        data = helper.load_test_npz('train_points_10_2.npz')
        X = data['X']
        X_1d = X[:,0].copy()
        f_1d = 2 + 5*X_1d
        
        gp = rs.GaussianProcess(N=1)
        gp.train(X_1d, f_1d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_1d_test = X[:,0].copy()
        f, df, v = gp.predict(X_1d_test, compgrad=True, compvar=True)
        np.testing.assert_almost_equal(f, 2+5*X_1d_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df, 5*np.ones((10,1)), decimal=10)
        
        f_1d = 2 - 3*X_1d + 5*X_1d*X_1d
        gp = rs.GaussianProcess(N=2)
        gp.train(X_1d, f_1d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_1d_test = X[:,0].copy()
        f, df, v = gp.predict(X_1d_test, compgrad=True, compvar=True)
        f_test = 2 - 3*X_1d_test + 5*X_1d_test*X_1d_test
        df_test = -3 + 10*X_1d_test
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df, df_test.reshape((10,1)), decimal=10)
    
 
    def test_exact_gp_approximation_2d(self):
        data = helper.load_test_npz('train_points_10_2.npz')
        X = data['X']
        X_train = X.copy()
        f_2d = 2 + 5*X_train[:,0] - 4*X_train[:,1]
        
        gp = rs.GaussianProcess(N=1)
        gp.train(X_train, f_2d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_test = X.copy()
        f, df, v = gp.predict(X_test, compgrad=True, compvar=True)
        f_test = 2 + 5*X_test[:,0] - 4*X_test[:,1]
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,0].reshape((10,1)), 5*np.ones((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,1].reshape((10,1)), -4*np.ones((10,1)), decimal=10)
        
        f_2d = 2 - 3*X_train[:,1] + 5*X_train[:,0]*X_train[:,1]
        gp = rs.GaussianProcess(N=2)
        gp.train(X_train, f_2d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_test = X.copy()
        f, df, v = gp.predict(X_test, compgrad=True, compvar=True)
        f_test = 2 - 3*X_test[:,1] + 5*X_test[:,0]*X_test[:,1]
        df1_test = 5*X_test[:,1]
        df2_test = -3 + 5*X_test[:,0]
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,0].reshape((10,1)), df1_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,1].reshape((10,1)), df2_test.reshape((10,1)), decimal=10)
    
 
    def test_polynomial_grad_1d(self):
        data = helper.load_test_npz('train_points_10_2.npz')
        X = data['X']
        X_1d = X[:,0].copy()
        f_1d = np.cos(X_1d)
        
        pr = rs.PolynomialRegression(N=7)
        pr.train(X_1d, f_1d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_1d_test = X[:,0].copy()
        f0, df0, v0 = pr.predict(X_1d_test, compgrad=True, compvar=True)

        e = 1e-6
        X_1d_testp = X_1d_test.copy() + e
        f1 = pr.predict(X_1d_testp)[0]
        df0_fd = (f1 - f0)/e
        np.testing.assert_almost_equal(df0, df0_fd, decimal=5)
    
 
    def test_polynomial_grad_2d(self):
        data = helper.load_test_npz('train_points_200_2.npz')
        X = data['X']
        X_train = X.copy()
        ff0 = np.cos(X_train[:,0]).reshape((200,1))
        ff1 = np.sin(X_train[:,1]).reshape((200,1))
        f_2d = ff0*ff1
        
        pr = rs.PolynomialRegression(N=5)
        pr.train(X_train, f_2d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_test = X.copy()
        f0, df0, v0 = pr.predict(X_test, compgrad=True, compvar=True)

        e = 1e-6
        X_testp = X_test.copy()
        X_testp[:,0] += e
        f1 = pr.predict(X_testp)[0]
        df1_fd = (f1 - f0)/e
        np.testing.assert_almost_equal(df0[:,0].reshape((10,1)), df1_fd, decimal=5)
        X_testp = X_test.copy()
        X_testp[:,1] += e
        f1 = pr.predict(X_testp)[0]
        df2_fd = (f1 - f0)/e
        np.testing.assert_almost_equal(df0[:,1].reshape((10,1)), df2_fd, decimal=5)
    
     
    def test_gp_grad_1d(self):
        data = helper.load_test_npz('train_points_10_2.npz')
        X = data['X']
        X_1d = X[:,0].copy()
        f_1d = np.cos(X_1d)
        
        gp = rs.GaussianProcess(N=0)
        gp.train(X_1d, f_1d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_1d_test = X[:,0].copy()
        f0, df0, v0 = gp.predict(X_1d_test, compgrad=True, compvar=True)

        e = 1e-6
        X_1d_testp = X_1d_test.copy() + e
        f1 = gp.predict(X_1d_testp)[0]
        df0_fd = (f1 - f0)/e
        np.testing.assert_almost_equal(df0, df0_fd, decimal=5)
    
 
    def test_gp_grad_2d(self):
        data = helper.load_test_npz('train_points_200_2.npz')
        X = data['X']
        X_train = X.copy()
        ff0 = np.cos(X_train[:,0]).reshape((200,1))
        ff1 = np.sin(X_train[:,1]).reshape((200,1))
        f_2d = ff0*ff1
        
        gp = rs.GaussianProcess(N=5)
        gp.train(X_train, f_2d)
        data = helper.load_test_npz('test_points_10_2.npz')
        X = data['X']
        X_test = X.copy()
        f0, df0, v0 = gp.predict(X_test, compgrad=True, compvar=True)

        e = 1e-6
        X_testp = X_test.copy()
        X_testp[:,0] += e
        f1 = gp.predict(X_testp)[0]
        df1_fd = (f1 - f0)/e
        np.testing.assert_almost_equal(df0[:,0].reshape((10,1)), df1_fd, decimal=5)
        X_testp = X_test.copy()
        X_testp[:,1] += e
        f1 = gp.predict(X_testp)[0]
        df2_fd = (f1 - f0)/e
        np.testing.assert_almost_equal(df0[:,1].reshape((10,1)), df2_fd, decimal=5)
    
 
    def test_poly_order_1d(self):
        data = helper.load_test_npz('test_points_uniform_50_2.npz')
        X = data['X']
        X_1d_test = X[:,0].copy().reshape((50,1))
        
        X_train = np.linspace(-1.0, 1.0, 201).reshape((201,1))
        f_train = np.sin(np.pi*X_train)
        
        
        print '\nPOLY 1D ORDER CONVERGENCE\n'
        for N in range(3,10):
            pr = rs.PolynomialRegression(N=N)
            pr.train(X_train, f_train)
        
            f, df, v = pr.predict(X_1d_test, compgrad=True, compvar=True)
            f_true = np.sin(np.pi*X_1d_test)
            err_f = np.linalg.norm(f - f_true)/np.linalg.norm(f_true)
            df_true = np.pi*np.cos(np.pi*X_1d_test)
            err_df = np.linalg.norm(df - df_true)/np.linalg.norm(df_true)
            print 'Order: %d, Error in f: %6.4e, Error in df: %6.4e' % (N, err_f, err_df)
    
 
    def test_poly_order_2d(self):
        data = helper.load_test_npz('test_points_uniform_50_2.npz')
        X = data['X']
        X_test = X.copy()
        
        xx = np.linspace(-1.0, 1.0, 21)
        X1, X2 = np.meshgrid(xx, xx)
        X_train = np.hstack((X1.reshape((441,1)), X2.reshape((441,1))))
        f_train = np.sin(np.pi*X1.reshape((441,1)))*np.cos(np.pi*X2.reshape((441,1)))
        
        print '\nPOLY 2D ORDER CONVERGENCE\n'
        for N in range(3,10):
            pr = rs.PolynomialRegression(N=N)
            pr.train(X_train, f_train)
        
            f, df, v = pr.predict(X_test, compgrad=True, compvar=True)
            f_true = np.sin(np.pi*X[:,0].reshape((50,1)))*np.cos(np.pi*X[:,1].reshape((50,1)))
            df1_true = np.cos(np.pi*X[:,1].reshape((50,1)))*np.pi*np.cos(np.pi*X[:,0].reshape((50,1)))
            df2_true = -np.sin(np.pi*X[:,0].reshape((50,1)))*np.pi*np.sin(np.pi*X[:,1].reshape((50,1)))
            err_f = np.linalg.norm(f - f_true)/np.linalg.norm(f_true)
            err_df1 = np.linalg.norm(df[:,0].reshape((50,1)) - df1_true)/np.linalg.norm(df1_true)
            err_df2 = np.linalg.norm(df[:,1].reshape((50,1)) - df2_true)/np.linalg.norm(df2_true)
            print 'Order: %d, Error in f: %6.4e, Error in df1: %6.4e, Error in df2: %6.4e' % (N, err_f, err_df1, err_df2)
    
 
    def test_gp_points_1d(self):
        data = helper.load_test_npz('test_points_uniform_50_2.npz')
        X = data['X']
        X_1d_test = X[:,0].copy().reshape((50,1))
        
        print '\nGP 1D POINT CONVERGENCE\n'
        for N in range(1,9):
            X_train = np.linspace(-1.0, 1.0, 2**N+1).reshape((2**N+1,1))
            f_train = np.sin(np.pi*X_train)
            gp = rs.GaussianProcess(N=0)
            #pdb.set_trace()
            gp.train(X_train, f_train)
        
            f, df, v = gp.predict(X_1d_test, compgrad=True, compvar=True)
            
            f_true = np.sin(np.pi*X_1d_test)
            err_f = np.linalg.norm(f - f_true)/np.linalg.norm(f_true)
            df_true = np.pi*np.cos(np.pi*X_1d_test)
            err_df = np.linalg.norm(df - df_true)/np.linalg.norm(df_true)
            print 'Points: %d, Error in f: %6.4e, Error in df: %6.4e' % (2**N+1, err_f, err_df)
    
     
    def test_gp_points_2d(self):
        data = helper.load_test_npz('test_points_uniform_50_2.npz')
        X = data['X']
        X_test = X.copy()
        
        print '\nGP 2D POINT CONVERGENCE\n'
        for N in range(1,5):
            xx = np.linspace(-1.0, 1.0, 2**N+1)
            X1, X2 = np.meshgrid(xx, xx)
            X_train = np.hstack((X1.reshape(((2**N+1)**2,1)), X2.reshape(((2**N+1)**2,1))))
            f_train = np.sin(np.pi*X1.reshape(((2**N+1)**2,1)))*np.cos(np.pi*X2.reshape(((2**N+1)**2,1)))
            
            gp = rs.GaussianProcess(N=2)
            gp.train(X_train, f_train)
        
            f, df, v = gp.predict(X_test, compgrad=True, compvar=True)
            f_true = np.sin(np.pi*X[:,0].reshape((50,1)))*np.cos(np.pi*X[:,1].reshape((50,1)))
            df1_true = np.cos(np.pi*X[:,1].reshape((50,1)))*np.pi*np.cos(np.pi*X[:,0].reshape((50,1)))
            df2_true = -np.sin(np.pi*X[:,0].reshape((50,1)))*np.pi*np.sin(np.pi*X[:,1].reshape((50,1)))
            err_f = np.linalg.norm(f - f_true)/np.linalg.norm(f_true)
            err_df1 = np.linalg.norm(df[:,0].reshape((50,1)) - df1_true)/np.linalg.norm(df1_true)
            err_df2 = np.linalg.norm(df[:,1].reshape((50,1)) - df2_true)/np.linalg.norm(df2_true)
            print 'Points: %d, Error in f: %6.4e, Error in df1: %6.4e, Error in df2: %6.4e' % ((2**N+1)**2, err_f, err_df1, err_df2)

    def test_gp_as(self):
        data = helper.load_test_npz('test_points_uniform_50_2.npz')
        X_test = data['X'].copy()
        data = helper.load_test_npz('train_points_200_2.npz')
        X_train = data['X'].copy()
        f_train = 2 + 5*X_train[:,0] - 4*X_train[:,1] +2*X_train[:,0]*X_train[:,1]

        gp = rs.GaussianProcess(N=1)
        e = np.array([1.0, 0.5, 0.1, 0.05, 0.01])
        gp.train(X_train, f_train, e=e)
        f, df, vf = gp.predict(X_test, compgrad=True, compvar=True)
        data = helper.load_test_npz('test_gp_0.npz')
        np.testing.assert_equal(f, data['f'])
        np.testing.assert_equal(df, data['df'])
        np.testing.assert_equal(vf, data['vf'])
        
        v = 0.0001*np.ones(f_train.shape)
        gp.train(X_train, f_train, e=e, v=v)
        f, df, vf = gp.predict(X_test, compgrad=True, compvar=True)
        data = helper.load_test_npz('test_gp_1.npz')
        np.testing.assert_equal(f, data['f'])
        np.testing.assert_equal(df, data['df'])
        np.testing.assert_equal(vf, data['vf'])
                
        
        
if __name__ == '__main__':
    unittest.main()