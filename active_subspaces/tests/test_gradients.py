from unittest import TestCase
import unittest
import helper
import numpy as np
import active_subspaces.gradients as gr

class TestGradients(TestCase):

    def test_local_linear_gradients(self):
        data = helper.load_test_npz('train_points_200_2.npz')
        X = data['X'].copy()
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
        #np.savez('data/test_llm_gradients',df=df)
        data = helper.load_test_npz('test_llm_gradients.npz')
        np.testing.assert_equal(df, data['df'])        

    @unittest.skip('not implemented yet')
    def test_finite_difference_gradients(self):
        self.fail('not implemented yet')

if __name__ == '__main__':
    unittest.main()