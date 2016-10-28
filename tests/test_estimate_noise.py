from unittest import TestCase
import unittest
import active_subspaces
from active_subspaces import SimulationRunner, estimate_noise
import numpy as np

def isclose_order_of_magnitude(a, b, **kwargs):
	return np.isclose(np.log10(a), np.log10(b), **kwargs)



class TestEstimateNoise(TestCase):

	def test_normal(self):
		true_noise = 1e-3
		def f(x):
			return x + true_noise*np.random.randn(*x.shape)

		sr = SimulationRunner(f)
		x = np.array([10])
		est_noise = estimate_noise(sr, x)
		print true_noise, est_noise
		assert isclose_order_of_magnitude(true_noise, est_noise, atol = 1)
	
	def test_normal_large_h(self):
		true_noise = 1e-3
		def f(x):
			return x + true_noise*np.random.randn(*x.shape)

		sr = SimulationRunner(f)
		x = np.array([10])
		est_noise = estimate_noise(sr, x, h = 1e4)
		print true_noise, est_noise
		assert isclose_order_of_magnitude(true_noise, est_noise, atol = 1)

	def test_normal_large_nf(self):
		true_noise = 1e-3
		def f(x):
			return x + true_noise*np.random.randn(*x.shape)

		sr = SimulationRunner(f)
		x = np.array([10])
		est_noise = estimate_noise(sr, x, nf = 51)
		print true_noise, est_noise
		assert isclose_order_of_magnitude(true_noise, est_noise, atol = 0.2)
	
	# This test currently fails
	def test_normal_small_h(self):
		def f(x):
			return np.round(x, 2)
	
		sr = SimulationRunner(f)
		x = np.array([0])
		est_noise = estimate_noise(sr, x, h = 1e-2)
		true_noise = 1e-2
		print true_noise, est_noise
		assert isclose_order_of_magnitude(true_noise, est_noise, atol = 1)


if __name__ == '__main__':
    unittest.main()

