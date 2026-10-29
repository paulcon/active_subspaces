import numpy as np
from unittest import TestCase
import unittest
import active_subspaces.utils.simrunners as sruns

def fun(x):
    return 0.5*np.dot(x, x.T)

def dfun1(x):
    m = x.size
    return x.reshape((m,1))

def dfun2(x):
    m = x.size
    return x.reshape((1,m))

def dfun3(x):
    m = x.size
    return x.reshape(m)

class TestSimrunners(TestCase):

    def test_simulation_runner(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        sr = sruns.SimulationRunner(fun)
        ftrue = np.array([[0.5], [0.5], [0.0]])
        f = sr.run(X)
        np.testing.assert_almost_equal(f, ftrue)

    def test_simulation_grad_runner1(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        sgr = sruns.SimulationGradientRunner(dfun1)
        df = sgr.run(X)
        np.testing.assert_almost_equal(df, X)

    def test_simulation_grad_runner2(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        sgr = sruns.SimulationGradientRunner(dfun2)
        df = sgr.run(X)
        np.testing.assert_almost_equal(df, X)

    def test_simulation_grad_runner3(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        sgr = sruns.SimulationGradientRunner(dfun3)
        df = sgr.run(X)
        np.testing.assert_almost_equal(df, X)

if __name__ == '__main__':
    unittest.main()
