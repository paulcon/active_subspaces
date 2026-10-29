from unittest import TestCase
import unittest
import active_subspaces.utils.quadrature as gq
import numpy as np

class TestQuadrature(TestCase):
    
    def test_r_hermite_type_error(self):
        self.assertRaises(TypeError, gq.r_hermite, 'string')

    def test_r_hermite_0(self):
        self.assertRaises(ValueError, gq.r_hermite, 0)

    def test_r_hermite_1(self):
        v = gq.r_hermite(1)
        self.assertIsInstance(v, np.ndarray)
        np.testing.assert_almost_equal(v, np.array([[0.0, 1.0]]))

    def test_r_hermite_2(self):
        v = gq.r_hermite(2)
        self.assertIsInstance(v, np.ndarray)
        np.testing.assert_almost_equal(v, np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0]]))

    def test_r_hermite_5(self):
        v = gq.r_hermite(5)
        self.assertIsInstance(v, np.ndarray)
        np.testing.assert_almost_equal(v, np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0]]))

    def test_jacobi_matrix_multi_dimension_1(self):
        self.assertRaises(ValueError, gq.jacobi_matrix, np.array([0.0]))

    def test_jacobi_matrix_multi_dimension_num_columns_1(self):
        self.assertRaises(ValueError, gq.jacobi_matrix, np.array([[0.0], [0.1]]))

    def test_jacobi_matrix_multi_dimension_num_columns_2(self):
        self.assertRaises(ValueError, gq.jacobi_matrix, np.array([[0.0, 2.4, 5.2], [0.1, 21.5, 0.3]]))

    def test_jacobi_matrix_one_row_1(self):
        a = np.array([[0.0, 1.0]])
        np.testing.assert_almost_equal(gq.jacobi_matrix(a), 0.0)

    def test_jacobi_matrix_one_row_2(self):
        a = np.array([[2.0, 1.0]])
        np.testing.assert_almost_equal(gq.jacobi_matrix(a), 2.0)

    def test_jacobi_matrix_5(self):
        gq.jacobi_matrix(gq.r_hermite(5))

    def test_gh1d_7pts(self):
        p,w = gq.gh1d(7)

    def test_gauss_hermite_1d_array_arg(self):
        p,w = gq.gauss_hermite([7])

    def test_gauss_hermite_1d_int_arg(self):
        p,w = gq.gauss_hermite(7)

    def test_gauss_hermite_2d(self):
        p,w = gq.gauss_hermite([3,3])

    def test_gauss_hermite_3d(self):
        p,w = gq.gauss_hermite([3,3,4])

    def test_gauss_hermite_type_error(self):
        self.assertRaises(TypeError, gq.gauss_hermite, 'sting')

if __name__ == '__main__':
    unittest.main()
