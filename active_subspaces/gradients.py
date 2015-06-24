"""Utilities for approximating gradients."""
import numpy as np
import logging
from utils.misc import process_inputs
from utils.simrunners import SimulationRunner

def local_linear_gradients(X, f, p=None):
    """
    Estimate a collection of gradients from input/output pairs.

    :param ndarray X: M-by-m matrix that contains the m-dimensional inputs.
    :param ndarray f: M-by-1 matrix that contains scalar outputs.
    :param int p: How many nearest neighbors to use when constructing the
        local linear model.

    :return df: M-by-m matrix that contains estimated partial derivatives
        approximated by the local linear models.
    :rtype: ndarray

    **Notes**

    If `p` is not specified, the default value is floor(1.7*m).
    """

    X, M, m = process_inputs(X)
    if M<=m: raise Exception('Not enough samples for local linear models.')

    if p is None:
        p = int(np.minimum(np.floor(1.7*m), M))
    elif not isinstance(p, int):
        raise TypeError('p must be an integer.')

    if p < m+1 or p > M:
        raise Exception('p must be between m+1 and M')

    MM = np.minimum(int(np.ceil(10*m*np.log(m))), M-1)
    logging.getLogger(__name__).debug('Computing {:d} local linear approximations with {:d} points in {:d} dims.'.format(MM, M, m))
    df = np.zeros((MM, m))
    for i in range(MM):
        ii = np.random.randint(M)
        x = X[ii,:]
        ind = np.argsort(np.sum((X - x)**2, axis=1))
        A = np.hstack((np.ones((p,1)), X[ind[1:p+1],:]))
        u = np.linalg.lstsq(A, f[ind[1:p+1]])[0]
        df[i,:] = u[1:].T
    return df

def finite_difference_gradients(X, fun, h=1e-6):
    """
    Compute finite difference gradients with a given interface.

    :param ndarray X: M-by-m matrix that contains the points to estimate the
        gradients with finite differences.
    :param function fun: Function that returns the simulation's quantity of
        interest given inputs.
    :param float h: The finite difference step size.

    :return: df, M-by-m matrix that contains estimated partial derivatives
        approximated by finite differences
    :rtype: ndarray
    """
    X, M, m = process_inputs(X)
    logging.getLogger(__name__).debug('Computing finite diff grads at {:d} points in {:d} dims.'.format(M, m))

    # points to run simulations including the perturbed inputs
    XX = np.kron(np.ones((m+1, 1)),X) + \
        h*np.kron(np.vstack((np.zeros((1, m)), np.eye(m))), np.ones((M, 1)))

    # run the simulation
    if isinstance(fun, SimulationRunner):
        F = fun.run(XX)
    else:
        F = SimulationRunner(fun).run(XX)

    df = (F[M:].reshape((m, M)).transpose() - F[:M]) / h
    return df.reshape((M,m))
