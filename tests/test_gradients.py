from unittest import TestCase
import unittest
import numpy as np
import active_subspaces.gradients as gr

class TestGradients(TestCase):
    writeData = False    
    
    def test_local_linear_gradients(self):

        np.random.seed(42)
        X = np.random.uniform(-1.0,1.0,size=(200,2))
        f = 2 - 5*X[:,0] + 4*X[:,1]
        
        df = gr.local_linear_gradients(X, f)
        M = df.shape[0]
        np.testing.assert_array_almost_equal(df, np.tile(np.array([-5.0, 4.0]), (M,1)), decimal=9)

        df = gr.local_linear_gradients(X, f, p=8)
        M = df.shape[0]
        np.testing.assert_array_almost_equal(df, np.tile(np.array([-5.0, 4.0]), (M,1)), decimal=9)
        
        f = 2 - np.sin(X[:,0]) + np.cos(X[:,1])        
        np.random.seed(1234)
        df = gr.local_linear_gradients(X, f)
        
    def test_finite_difference_gradients(self):
        def myfun(x):
            return 2 - 5*x[0,0] + 4*x[0,1]
            
        np.random.seed(42)
        X = np.random.uniform(-1.0,1.0,size=(10,2))

        df = gr.finite_difference_gradients(X, myfun)
        M = df.shape[0]
        df_test = np.tile(np.array([-5.0, 4.0]), (M,1))
        np.testing.assert_array_almost_equal(df, df_test, decimal=6)

if __name__ == '__main__':
    unittest.main()