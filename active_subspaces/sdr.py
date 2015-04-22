import numpy as np
from utils.plotters import sufficient_summary, eigenvectors
from utils.response_surfaces import PolynomialRegression

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
    w_lb, w_ub = np.ones((m, 1)), -np.ones((m, 1))
    for i in range(n_boot):
        w_boot = lingrad(X[ind[:,i],:], f[ind[:,i]]).reshape((m, 1))
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
    Description of quadratic_model_check.

    Arguments:
        X:
        f:
        gamma:
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
    
    return e.reshape((m,1)), W.reshape((m,m))