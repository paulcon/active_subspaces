"""Runners for simulations"""

import numpy as np
from utils import process_inputs

class SimulationRunner():
    """
    Description for SimulationRunner
    """

    def __init__(self, fun):
        """
        Arguments:
            fun:
        """

        self.fun = fun

    def run(self, X):
        """
        Description of run.

        Arguments:
            X:
        """

        # right now this just wraps a sequential for-loop.
        # should be parallelized

        X, M, m = process_inputs(X)
        F = np.zeros((M, 1))
        for i in range(M):
            F[i] = self.fun(X[i,:].reshape((1,m)))
        return F

class SimulationGradientRunner():
    """
    Description for SimulationGradientRunner
    """

    def __init__(self, dfun):
        """
        Arguments:
            dfun:
        """

        self.dfun = dfun

    def run(self, X):
        """
        Description of run.

        Arguments:
            X:
        """

        # right now this just wraps a sequential for-loop.
        # should be parallelized

        X, M, m = process_inputs(X)
        dF = np.zeros((M, m))
        for i in range(M):
            df = self.dfun(X[i,:].reshape((1,m)))
            dF[i,:] = df.reshape(m)
        return dF
