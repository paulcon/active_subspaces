from unittest import TestCase
import unittest
import active_subspaces.utils.response_surfaces as rs
import numpy as np

class TestResponseSurfaces(TestCase):

    def test_index_set(self):
        I = rs.index_set(7,3)

    def test_polynomial_bases(self):
        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        B, I = rs.polynomial_bases(X, 3)

    def test_grad_polynomial_bases(self):
        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        dB = rs.grad_polynomial_bases(X, 3)

    def test_grad_polynomial_bases_fd(self):
        np.random.seed(42)
        X0 = np.random.normal(size=(10,2))

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

    def test_exponential_squared(self):

        np.random.seed(42)
        X1 = np.random.normal(size=(10,2))
        X2 = X1.copy()
        C = rs.exponential_squared(X1, X2, 1.0, np.array([1.0,1.0]))

    def test_grad_exponential_squared_fd(self):

        np.random.seed(42)
        X1 = np.random.normal(size=(10,2))
        X2 = X1.copy()
        C0 = rs.exponential_squared(X1, X2, 1.0, np.array([1.0,1.0]))

        dC = rs.grad_exponential_squared(X1, X2, 1.0, np.array([1.0,1.0]))
        e = 1e-6

        X2p1 = X2.copy()
        X2p1[:,0] += e
        C1 = rs.exponential_squared(X1, X2p1, 1.0, np.array([1.0,1.0]))
        dC1 = (C1 - C0)/e
        np.testing.assert_array_almost_equal(dC[:,:,0], dC1, decimal=5)

        X2p2 = X2.copy()
        X2p2[:,1] += e
        C2 = rs.exponential_squared(X1, X2p2, 1.0, np.array([1.0,1.0]))
        dC2 = (C2 - C0)/e
        np.testing.assert_array_almost_equal(dC[:,:,1], dC2, decimal=5)

    def test_grad_exponential_squared(self):

        np.random.seed(42)
        X1 = np.random.normal(size=(10,2))
        X2 = X1.copy()
        dC = rs.grad_exponential_squared(X1, X2, 1.0, np.array([1.0,1.0]))

    def test_exact_polynomial_approximation_1d(self):
        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        M = X.shape[0]
        X_1d = X[:,0].copy().reshape((M,1))
        f_1d = 2 + 5*X_1d

        pr = rs.PolynomialApproximation(N=1)
        pr.train(X_1d, f_1d)
        print 'Rsqr: {:6.4f}'.format(pr.Rsqr)

        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        M = X.shape[0]
        X_1d_test = X[:,0].copy().reshape((M,1))
        f, df = pr.predict(X_1d_test, compgrad=True)
        np.testing.assert_almost_equal(f, 2+5*X_1d_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df, 5*np.ones((10,1)), decimal=10)

        f_1d = 2 - 3*X_1d + 5*X_1d*X_1d
        pr = rs.PolynomialApproximation(N=2)
        pr.train(X_1d, f_1d)
        print 'Rsqr: {:6.4f}'.format(pr.Rsqr)
        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        M = X.shape[0]
        X_1d_test = X[:,0].copy().reshape((M,1))
        f, df = pr.predict(X_1d_test, compgrad=True)
        f_test = 2 - 3*X_1d_test + 5*X_1d_test*X_1d_test
        df_test = -3 + 10*X_1d_test
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df, df_test.reshape((10,1)), decimal=10)


    def test_exact_polynomial_approximation_2d(self):
        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        X_train = X.copy()
        f_2d = 2 + 5*X_train[:,0] - 4*X_train[:,1]

        pr = rs.PolynomialApproximation(N=1)
        pr.train(X_train, f_2d.reshape((f_2d.size,1)))
        print 'Rsqr: {:6.4f}'.format(pr.Rsqr)

        X = np.random.normal(size=(10,2))
        X_test = X.copy()
        f, df = pr.predict(X_test, compgrad=True)
        f_test = 2 + 5*X_test[:,0] - 4*X_test[:,1]
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,0].reshape((10,1)), 5*np.ones((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,1].reshape((10,1)), -4*np.ones((10,1)), decimal=10)

        f_2d = 2 - 3*X_train[:,1] + 5*X_train[:,0]*X_train[:,1]
        pr = rs.PolynomialApproximation(N=2)
        pr.train(X_train, f_2d.reshape((f_2d.size,1)))
        print 'Rsqr: {:6.4f}'.format(pr.Rsqr)
        
        X = np.random.normal(size=(10,2))
        X_test = X.copy()
        f, df = pr.predict(X_test, compgrad=True)
        f_test = 2 - 3*X_test[:,1] + 5*X_test[:,0]*X_test[:,1]
        df1_test = 5*X_test[:,1]
        df2_test = -3 + 5*X_test[:,0]
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,0].reshape((10,1)), df1_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,1].reshape((10,1)), df2_test.reshape((10,1)), decimal=10)


    def test_exact_polynomial_approximation_3d(self):
        np.random.seed(42)
        X = np.random.normal(size=(20,3))
        X_train = X.copy()
        f_3d = 2 + 5*X_train[:,0] - 4*X_train[:,1] + 2*X_train[:,2]

        pr = rs.PolynomialApproximation(N=3)
        pr.train(X_train, f_3d.reshape((f_3d.size,1)))


    def test_exact_rbf_approximation_1d(self):
        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        M = X.shape[0]
        X_1d = X[:,0].copy().reshape((M,1))
        f_1d = 2 + 5*X_1d

        gp = rs.RadialBasisApproximation(N=1)
        gp.train(X_1d, f_1d)
        print 'Rsqr: {:6.4f}'.format(gp.Rsqr)
        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        M = X.shape[0]
        X_1d_test = X[:,0].copy().reshape((M,1))
        f, df = gp.predict(X_1d_test, compgrad=True)
        np.testing.assert_almost_equal(f, 2+5*X_1d_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df, 5*np.ones((10,1)), decimal=10)

        f_1d = 2 - 3*X_1d + 5*X_1d*X_1d
        gp = rs.RadialBasisApproximation(N=2)
        gp.train(X_1d, f_1d)
        print 'Rsqr: {:6.4f}'.format(gp.Rsqr)
        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        M = X.shape[0]
        X_1d_test = X[:,0].copy().reshape((M,1))
        f, df = gp.predict(X_1d_test, compgrad=True)
        f_test = 2 - 3*X_1d_test + 5*X_1d_test*X_1d_test
        df_test = -3 + 10*X_1d_test
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df, df_test.reshape((10,1)), decimal=10)


    def test_exact_rbf_approximation_2d(self):
        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        X_train = X.copy()
        f_2d = 2 + 5*X_train[:,0] - 4*X_train[:,1]

        gp = rs.RadialBasisApproximation(N=1)
        gp.train(X_train, f_2d.reshape((f_2d.size,1)))
        print 'Rsqr: {:6.4f}'.format(gp.Rsqr)

        X = np.random.normal(size=(10,2))
        X_test = X.copy()
        f, df = gp.predict(X_test, compgrad=True)
        f_test = 2 + 5*X_test[:,0] - 4*X_test[:,1]
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,0].reshape((10,1)), 5*np.ones((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,1].reshape((10,1)), -4*np.ones((10,1)), decimal=10)

        f_2d = 2 - 3*X_train[:,1] + 5*X_train[:,0]*X_train[:,1]
        gp = rs.RadialBasisApproximation(N=2)
        gp.train(X_train, f_2d.reshape((f_2d.size,1)))
        print 'Rsqr: {:6.4f}'.format(gp.Rsqr)

        X = np.random.normal(size=(10,2))
        X_test = X.copy()
        f, df = gp.predict(X_test, compgrad=True)
        f_test = 2 - 3*X_test[:,1] + 5*X_test[:,0]*X_test[:,1]
        df1_test = 5*X_test[:,1]
        df2_test = -3 + 5*X_test[:,0]
        np.testing.assert_almost_equal(f, f_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,0].reshape((10,1)), df1_test.reshape((10,1)), decimal=10)
        np.testing.assert_almost_equal(df[:,1].reshape((10,1)), df2_test.reshape((10,1)), decimal=10)

    def test_polynomial_grad_1d(self):
        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        M = X.shape[0]
        X_1d = X[:,0].copy().reshape((M,1))
        f_1d = np.cos(X_1d)

        pr = rs.PolynomialApproximation(N=7)
        pr.train(X_1d, f_1d)

        X = np.random.normal(size=(10,2))
        M = X.shape[0]
        X_1d_test = X[:,0].copy().reshape((M,1))
        f0, df0 = pr.predict(X_1d_test, compgrad=True)

        e = 1e-6
        X_1d_testp = X_1d_test.copy() + e
        f1 = pr.predict(X_1d_testp)[0]
        df0_fd = (f1 - f0)/e
        np.testing.assert_almost_equal(df0, df0_fd, decimal=5)


    def test_polynomial_grad_2d(self):
        np.random.seed(42)
        X = np.random.normal(size=(200,2))
        X_train = X.copy()
        ff0 = np.cos(X_train[:,0]).reshape((200,1))
        ff1 = np.sin(X_train[:,1]).reshape((200,1))
        f_2d = ff0*ff1

        pr = rs.PolynomialApproximation(N=5)
        pr.train(X_train, f_2d)

        X = np.random.normal(size=(10,2))
        X_test = X.copy()
        f0, df0 = pr.predict(X_test, compgrad=True)

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


    def test_rbf_grad_1d(self):
        np.random.seed(42)
        X = np.random.normal(size=(10,2))
        M = X.shape[0]
        X_1d = X[:,0].copy().reshape((M,1))
        f_1d = np.cos(X_1d)

        gp = rs.RadialBasisApproximation(N=0)
        gp.train(X_1d, f_1d)
        
        X = np.random.normal(size=(10,2))
        M = X.shape[0]
        X_1d_test = X[:,0].copy().reshape((M,1))
        f0, df0 = gp.predict(X_1d_test, compgrad=True)

        e = 1e-6
        X_1d_testp = X_1d_test.copy() + e
        f1 = gp.predict(X_1d_testp)[0]
        df0_fd = (f1 - f0)/e
        np.testing.assert_almost_equal(df0, df0_fd, decimal=5)


    def test_rbf_grad_2d(self):
        np.random.seed(42)
        X = np.random.normal(size=(200,2))
        X_train = X.copy()
        ff0 = np.cos(X_train[:,0]).reshape((200,1))
        ff1 = np.sin(X_train[:,1]).reshape((200,1))
        f_2d = ff0*ff1

        gp = rs.RadialBasisApproximation(N=5)
        gp.train(X_train, f_2d)

        X = np.random.normal(size=(10,2))
        X_test = X.copy()
        f0, df0 = gp.predict(X_test, compgrad=True)

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
        np.random.seed(42)
        X = np.random.uniform(-1.0,1.0,size=(50,2))
        X_1d_test = X[:,0].copy().reshape((50,1))

        X_train = np.linspace(-1.0, 1.0, 201).reshape((201,1))
        f_train = np.sin(np.pi*X_train)


        print '\nPOLY 1D ORDER CONVERGENCE\n'
        for N in range(3,10):
            pr = rs.PolynomialApproximation(N=N)
            pr.train(X_train, f_train)

            f, df = pr.predict(X_1d_test, compgrad=True)
            f_true = np.sin(np.pi*X_1d_test)
            err_f = np.linalg.norm(f - f_true)/np.linalg.norm(f_true)
            df_true = np.pi*np.cos(np.pi*X_1d_test)
            err_df = np.linalg.norm(df - df_true)/np.linalg.norm(df_true)
            print 'Order: %d, Error in f: %6.4e, Error in df: %6.4e' % (N, err_f, err_df)


    def test_poly_order_2d(self):
        np.random.seed(42)
        X = np.random.uniform(-1.0,1.0,size=(50,2))
        X_test = X.copy()

        xx = np.linspace(-1.0, 1.0, 21)
        X1, X2 = np.meshgrid(xx, xx)
        X_train = np.hstack((X1.reshape((441,1)), X2.reshape((441,1))))
        f_train = np.sin(np.pi*X1.reshape((441,1)))*np.cos(np.pi*X2.reshape((441,1)))

        print '\nPOLY 2D ORDER CONVERGENCE\n'
        for N in range(3,10):
            pr = rs.PolynomialApproximation(N=N)
            pr.train(X_train, f_train)

            f, df = pr.predict(X_test, compgrad=True)
            f_true = np.sin(np.pi*X[:,0].reshape((50,1)))*np.cos(np.pi*X[:,1].reshape((50,1)))
            df1_true = np.cos(np.pi*X[:,1].reshape((50,1)))*np.pi*np.cos(np.pi*X[:,0].reshape((50,1)))
            df2_true = -np.sin(np.pi*X[:,0].reshape((50,1)))*np.pi*np.sin(np.pi*X[:,1].reshape((50,1)))
            err_f = np.linalg.norm(f - f_true)/np.linalg.norm(f_true)
            err_df1 = np.linalg.norm(df[:,0].reshape((50,1)) - df1_true)/np.linalg.norm(df1_true)
            err_df2 = np.linalg.norm(df[:,1].reshape((50,1)) - df2_true)/np.linalg.norm(df2_true)
            print 'Order: %d, Error in f: %6.4e, Error in df1: %6.4e, Error in df2: %6.4e' % (N, err_f, err_df1, err_df2)


    def test_rbf_points_1d(self):
        np.random.seed(42)
        X = np.random.uniform(-1.0,1.0,size=(50,2))
        X_1d_test = X[:,0].copy().reshape((50,1))

        print '\nRBF 1D POINT CONVERGENCE\n'
        for N in range(1,9):
            X_train = np.linspace(-1.0, 1.0, 2**N+1).reshape((2**N+1,1))
            f_train = np.sin(np.pi*X_train)
            gp = rs.RadialBasisApproximation(N=0)
            #pdb.set_trace()
            gp.train(X_train, f_train)

            f, df = gp.predict(X_1d_test, compgrad=True)

            f_true = np.sin(np.pi*X_1d_test)
            err_f = np.linalg.norm(f - f_true)/np.linalg.norm(f_true)
            df_true = np.pi*np.cos(np.pi*X_1d_test)
            err_df = np.linalg.norm(df - df_true)/np.linalg.norm(df_true)
            print 'Points: %d, Error in f: %6.4e, Error in df: %6.4e' % (2**N+1, err_f, err_df)


    def test_rbf_points_2d(self):
        np.random.seed(42)
        X = np.random.uniform(-1.0,1.0,size=(50,2))
        X_test = X.copy()

        print '\nRBF 2D POINT CONVERGENCE\n'
        for N in range(1,5):
            xx = np.linspace(-1.0, 1.0, 2**N+1)
            X1, X2 = np.meshgrid(xx, xx)
            X_train = np.hstack((X1.reshape(((2**N+1)**2,1)), X2.reshape(((2**N+1)**2,1))))
            f_train = np.sin(np.pi*X1.reshape(((2**N+1)**2,1)))*np.cos(np.pi*X2.reshape(((2**N+1)**2,1)))

            gp = rs.RadialBasisApproximation(N=2)
            gp.train(X_train, f_train)

            f, df = gp.predict(X_test, compgrad=True)
            f_true = np.sin(np.pi*X[:,0].reshape((50,1)))*np.cos(np.pi*X[:,1].reshape((50,1)))
            df1_true = np.cos(np.pi*X[:,1].reshape((50,1)))*np.pi*np.cos(np.pi*X[:,0].reshape((50,1)))
            df2_true = -np.sin(np.pi*X[:,0].reshape((50,1)))*np.pi*np.sin(np.pi*X[:,1].reshape((50,1)))
            err_f = np.linalg.norm(f - f_true)/np.linalg.norm(f_true)
            err_df1 = np.linalg.norm(df[:,0].reshape((50,1)) - df1_true)/np.linalg.norm(df1_true)
            err_df2 = np.linalg.norm(df[:,1].reshape((50,1)) - df2_true)/np.linalg.norm(df2_true)
            print 'Points: %d, Error in f: %6.4e, Error in df1: %6.4e, Error in df2: %6.4e' % ((2**N+1)**2, err_f, err_df1, err_df2)

    def test_rbf_as(self):
        np.random.seed(42)
        X_test = np.random.uniform(-1.0,1.0,size=(50,2))
        np.random.seed(42)
        X_train = np.random.uniform(-1.0,1.0,size=(200,2))
        f_train = 2 + 5*X_train[:,0] - 4*X_train[:,1] +2*X_train[:,0]*X_train[:,1]

        gp = rs.RadialBasisApproximation(N=1)
        e = np.array([1.0, 0.5, 0.1, 0.05, 0.01])
        gp.train(X_train, f_train.reshape((f_train.size,1)), e=e)
        f, df = gp.predict(X_test, compgrad=True)

        v = 0.0001*np.ones(f_train.shape)
        gp.train(X_train, f_train.reshape((f_train.size,1)), e=e, v=v)
        f, df = gp.predict(X_test, compgrad=True)

if __name__ == '__main__':
    unittest.main()
