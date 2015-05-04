"""Utilities for approximating gradients."""
import numpy as np
from utils.utils import process_inputs
from utils.simrunners import SimulationRunner

def local_linear_gradients(X, f, p=None):
    """
    Estimate a collection of gradients from input/output pairs.
    
    Parameters
    ----------
    X : ndarray
        `X` is an ndarray of size M-by-m that contains the m-dimensional inputs.
    f : ndarray
        `f` is an ndarray of size M-by-1 that contains scalar outputs.
    p : int, optional
        `p` determines how many nearest neighbors to use when constructing the
        local linear model. (Default is None)
                    
    Returns
    -------
    df : ndarray
        An ndarray of size M-by-m that contains estimated partial derivatives
        approximated by the local linear models. 
    
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

    MM = np.minimum(int(np.ceil(10*m*np.log(m))), M-1)
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
    
    Parameters
    ----------
    X : ndarray
        `X` is an ndarray of size M-by-m that contains the points to estimate
        the gradients with finite differences.
    fun : function
        `fun` is the function that returns the simulation's quantity of interest
        given inputs.
    h : float, optional
        `h` is the finite difference step size. (Default is 1e-6)
                    
    Returns
    -------
    df : ndarray
        An ndarray of size M-by-m that contains estimated partial derivatives
        approximated by finite differences 
    
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