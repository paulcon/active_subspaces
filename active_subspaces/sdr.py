""" Some implementations of sufficient dimension reduction."""
import numpy as np
from utils.plotters import sufficient_summary, eigenvectors
from utils.response_surfaces import PolynomialApproximation

def linear_gradient_check(X, f, n_boot=1000, in_labels=None, out_label=None):
    """
    Use the normalized gradient of a global linear model to define the active
    subspace.

    :param ndarray X: M-by-m matrix containing points in the simulation input
        space.
    :param ndarray f: M-by-1 matrix containing the corresponding simulation
        outputs.
    :param int n_boot: The number of bootstrap replicates.
    :param str[] in_labels: Contains strings that label the input parameters.
    :param str out_label: String that labels the simulation output.

    :return: w, m-by-1 matrix that is the normalized gradient of
        the global linear model.
    :rtype: ndarray

    **See Also**

    sdr.quadratic_model_check

    **Notes**

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

def quadratic_model_check(X, f, gamma, weights=None):
    """
    Use the Hessian of a least-squares-fit quadratic model to identify active
    and inactive subspaces

    :param ndarray X: M-by-m matrix containing points in the simulation input
        space.
    :param ndarray f: M-by-1 containing the corresponding simulation outputs.
    :param ndarray gamma: The variance of the simulation inputs. If the inputs
        are bounded by a hypercube, then `gamma` is 1/3.
    :param ndarray weights: M-by-1 containing weights for the least-squares.

    :return: e, m-by-1 that contains the eigenvalues of the quadratic model's
        Hessian.
    :rtype: ndarray

    :return: W, m-by-m matrix that contains the eigenvectors of the quadratic
        model's Hessian.
    :rtype: ndarray

    **See Also**

    sdr.linear_gradient_check

    **Notes**

    This approach is very similar to Ker Chau Li's principal Hessian directions.
    """

    M, m = X.shape
    gamma = gamma.reshape((1, m))

    pr = PolynomialApproximation(2)
    pr.train(X, f, weights)

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
