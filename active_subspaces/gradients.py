"""Utilities for approximating gradients."""
import numpy as np
from .utils.misc import process_inputs
from .utils.simrunners import SimulationRunner

def local_linear_gradients(X, f, p=None, weights=None):
    """Estimate a collection of gradients from input/output pairs.
    
    Given a set of input/output pairs, choose subsets of neighboring points and
    build a local linear model for each subset. The gradients of these local
    linear models comprise estimates of sampled gradients.

    Parameters
    ----------
    X : ndarray 
        M-by-m matrix that contains the m-dimensional inputs
    f : ndarray 
        M-by-1 matrix that contains scalar outputs
    p : int, optional
        how many nearest neighbors to use when constructing the local linear 
        model (default 1)
    weights : ndarray, optional
        M-by-1 matrix that contains the weights for each observation (default 
        None)

    Returns
    -------
    df : ndarray
        M-by-m matrix that contains estimated partial derivatives approximated 
        by the local linear models

    Notes
    -----
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
        
    if weights is None:
        weights = np.ones((M, 1)) / M

    MM = np.minimum(int(np.ceil(10*m*np.log(m))), M-1)
    df = np.zeros((MM, m))
    for i in range(MM):
        ii = np.random.randint(M)
        x = X[ii,:]
        D2 = np.sum((X - x)**2, axis=1)
        ind = np.argsort(D2)
        ind = ind[D2 != 0]
        A = np.hstack((np.ones((p,1)), X[ind[:p],:])) * np.sqrt(weights[ii])
        b = f[ind[:p]] * np.sqrt(weights[ii])
        u = np.linalg.lstsq(A, b)[0]
        df[i,:] = u[1:].T
    return df

def finite_difference_gradients(X, fun, h=1e-6):
    """Compute finite difference gradients with a given interface.

    Parameters
    ----------
    X : ndarray 
        M-by-m matrix that contains the points to estimate the gradients with 
        finite differences
    fun : function
        function that returns the simulation's quantity of interest given inputs
    h : float, optional 
        the finite difference step size (default 1e-6)

    Returns
    -------
    df : ndarray 
        M-by-m matrix that contains estimated partial derivatives approximated 
        by finite differences
    """
    X, M, m = process_inputs(X)

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
