import numpy as np
from plotters import sufficient_summary, eigenvectors
from response_surfaces import PolynomialRegression
import abc

class Normalizer():
    """
    Abstract Base Class for Normalizers
    """
    __metaclass__  = abc.ABCMeta

    @abc.abstractmethod
    def normalize(self, X):
        """
        Description of normalize

        Arguments:
            X:
        Output:

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def unnormalize(self, X):
        """
        Description of unnormalize

        Arguments:
            X:
        Output:

        """
        raise NotImplementedError()

class BoundedNormalizer(Normalizer):
    """
    Description of BoundedNormalizer
    """

    def __init__(self, lb, ub):
        """
        Arguments:
            lb:
            ub:
        """
        m = lb.size
        self.lb = lb.reshape((1, m))
        self.ub = ub.reshape((1, m))

    def normalize(self, X):
        """See Normalizer#normalize"""
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    def unnormalize(self, X):
        """See Normalizer#unnormalize"""
        return (self.ub - self.lb) * (X + 1.0) / 2.0 + self.lb

class UnboundedNormalizer(Normalizer):
    """
    Description of UnboundedNormalizer
    """

    def __init__(self, mu, C):
        """
        Arguments:
            mu:
            C:
        """
        m = mu.size
        self.mu = mu.reshape((1, m))
        self.L = np.linalg.cholesky(C)

    def normalize(self, X):
        """See Normalizer#normalize"""
        X0 = X - self.mu
        return np.linalg.solve(self.L,X0.T).T

    def unnormalize(self, X):
        """See Normalizer#unnormalize"""
        X0 = np.dot(X,self.L.T)
        return X0 + self.mu

def process_inputs(X):
    """
    Description of process_inputs.

    Arguments:
        X:
    Outputs:
        X:
        M:
        m:
    """

    if len(X.shape) == 2:
        M, m = X.shape
    elif len(X.shape) == 1:
        M = X.shape[0]
        m = 1
        X = X.reshape((M, m))
    else:
        raise ValueError('Bad inputs.')
    return X, M, m

def lingrad(X, f):
    """
    Description of lingrad.

    Arguments:
        X:
        f:
    Outputs:
        w:
    """

    M, m = X.shape
    A = np.hstack((np.ones((M, 1)), X))
    u = np.linalg.lstsq(A, f)[0]
    w = u[1:] / np.linalg.norm(u[1:])
    return w

def quick_check(X, f, n_boot=1000, in_labels=None, out_label=None):
    """
    Description of quick_check.

    Arguments:
        X:
        f:
        n_boot: (deafult=1000)
        in_labels: (deafult=None)
        out_label: (deafult=None)
    Outputs:
        w:
    """

    M, m = X.shape
    w = lingrad(X, f)

    # bootstrap
    ind = np.random.randint(M, size=(M, n_boot))
    w_boot = np.zeros((m, n_boot))
    for i in range(n_boot):
        w_boot[:,i] = lingrad(X[ind[:,i],:], f[ind[:,i]])

    # make sufficient summary plot
    y = np.dot(X, w)
    sufficient_summary(y, f, out_label=out_label)

    # plot weights
    eigenvectors(w, W_boot=w_boot, in_labels=in_labels, out_label=out_label)

    return w

def quadratic_model_check(X, f, gamma, k):
    """
    Description of quadratic_model_check.

    Arguments:
        X:
        f:
        gamma:
        k:
    Outputs:
        w:
    """

    M, m = X.shape
    gamma = gamma.reshape((1, m))

    pr = PolynomialRegression(2)
    pr.train(X, f)

    # get regression coefficients
    b, A = pr.g, pr.H
    
    # compute eigenpairs
    e, W = np.linalg.eig(np.outer(b, b.T) + \
        np.dot(A, np.dot(np.diagflat(gamma), A)))
    ind = np.argsort(e)[::-1]
    e, W = e[ind], W[:,ind]*np.sign(W[0,ind])
    
    return e[:k], W


def conditional_expectations(f, ind):
    """
    Description of conditional_expectations.

    Arguments:
        f:
        ind:
    Outputs:
        Ef:
        Vf:
    """
    n = int(np.amax(ind)) + 1
    Ef, Vf = np.zeros((n, 1)), np.zeros((n, 1))
    for i in range(n):
        fi = f[ind == i]
        Ef[i] = np.mean(fi)
        Vf[i] = np.var(fi)
    return Ef, Vf








