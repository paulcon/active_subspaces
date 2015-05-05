""" Some implementations of sufficient dimension reduction."""
import numpy as np
from utils.plotters import sufficient_summary, eigenvectors
from utils.response_surfaces import PolynomialApproximation

def linear_gradient_check(X, f, n_boot=1000, in_labels=None, out_label=None):
    """
    Use the normalized gradient of a global linear model to define the active
    subspace.
    
    Parameters
    ----------
    X : ndarray
        `X` is an ndarray of shape M-by-m containing points in the simulation 
        input space.
    f : ndarray
        `f` is an ndarray of shape M-by-1 containing the corresponding 
        simulation outputs.
    n_boot : int, optional
        `n_boot` is the number of bootstrap replicates. (Default is 1000)
    in_labels : list of str, optional
        `in_labels` contains strings that label the input parameters (Default
        is None)
    out_label : str, optional
        `out_label` is a string that labels the simulation output.
                    
    Returns
    -------
    w : ndarray
        `w` is an ndarray of shape m-by-1 that is the normalized gradient of 
        the global linear model.
        
    See Also
    --------
    sdr.quadratic_model_check
            
    Notes
    -----
    This is usually my first step when analyzing a new data set. It can be used
    to identify a one-dimensional active subspace under two conditions: (i) the
    simulation output is roughly a monotonic function of the inputs and (ii)
    the simulation output is well represented by f(x) \approx g(w^T x). 
    
    The function produces the summary plot, which can verify these assumptions.
    
    It also plots the components of `w`, which often provide insight into the
    important parameters of the model. 
    """

    M, m = X.shape
    w = _lingrad(X, f)

    # bootstrap
    ind = np.random.randint(M, size=(M, n_boot))
    w_lb, w_ub = np.ones((m, 1)), -np.ones((m, 1))
    for i in range(n_boot):
        w_boot = _lingrad(X[ind[:,i],:], f[ind[:,i]]).reshape((m, 1))
        for j in range(m):
            if w_boot[j,0] < w_lb[j,0]:
                w_lb[j,0] = w_boot[j,0]
            if w_boot[j,0] > w_ub[j,0]:
                w_ub[j,0] = w_boot[j,0]
    w_br = np.hstack((w_lb, w_ub))

    # make sufficient summary plot
    y = np.dot(X, w)
    sufficient_summary(y, f, out_label=out_label)

    # plot weights
    eigenvectors(w, W_br=w_br, in_labels=in_labels, out_label=out_label)

    return w

def quadratic_model_check(X, f, gamma):
    """
    Use the Hessian of a least-squares-fit quadratic model to identify active 
    and inactive subspaces
    
    Parameters
    ----------
    X : ndarray
        `X` is an ndarray of shape M-by-m containing points in the simulation 
        input space.
    f : ndarray
        `f` is an ndarray of shape M-by-1 containing the corresponding 
        simulation outputs.
    gamma : float
        `gamma` is the variance of the simulation inputs. If the inputs are 
        bounded by a hypercube, then `gamma` is 1/3.
                    
    Returns
    -------
    e : ndarray
        `e` is an ndarray of shape m-by-1 that contains the eigenvalues of the
        quadratic model's Hessian.
    W : ndarray
        `W` is an ndarray of shape m-by-m that contains the eigenvectors of the 
        quadratic model's Hessian.
        
    See Also
    --------
    sdr.linear_gradient_check
            
    Notes
    -----
    This approach is very similar to Ker Chau Li's principal Hessian directions.
    """

    M, m = X.shape
    gamma = gamma.reshape((1, m))

    pr = PolynomialApproximation(2)
    pr.train(X, f)

    # get regression coefficients
    b, A = pr.g, pr.H
    
    # compute eigenpairs
    e, W = np.linalg.eig(np.outer(b, b.T) + \
        np.dot(A, np.dot(np.diagflat(gamma), A)))
    ind = np.argsort(e)[::-1]
    e, W = e[ind], W[:,ind]*np.sign(W[0,ind])
    
    return e.reshape((m,1)), W.reshape((m,m))
    
def _lingrad(X, f):
    """
    A private function that builds the linear model with least-squares and 
    returns the normalized gradient of the linear model. 
    """

    M, m = X.shape
    A = np.hstack((np.ones((M, 1)), X))
    u = np.linalg.lstsq(A, f)[0]
    w = u[1:] / np.linalg.norm(u[1:])
    return w