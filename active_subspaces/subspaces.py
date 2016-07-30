"""Utilities for computing active and inactive subspaces."""
from __future__ import division
import numpy as np
from utils.misc import process_inputs, process_inputs_outputs
from utils.response_surfaces import PolynomialApproximation
from gradients import local_linear_gradients

SQRTEPS = np.sqrt(np.finfo(float).eps)

class Subspaces():
    """
    A class for computing active and inactive subspaces.

    :cvar ndarray eigenvalues: m-by-1 matrix where m is the dimension of the input space.
    :cvar ndarray eigenvectors: m-by-m matrix that contains the eigenvectors oriented column-wise.
    :cvar ndarray W1: m-by-n matrix that contains the basis for the active subspace.
    :cvar ndarray W2: m-by-(m-n) matix that contains the basis for the inactive subspaces.
    :cvar ndarray e_br: m-by-2 matrix that contains the bootstrap ranges for the eigenvalues.
    :cvar ndarray sub_br: m-by-3 matrix that contains the bootstrap ranges (first and third column) 
    and the mean (second column) of the error in the estimated subspaces approximated by bootstrap

    **Notes**

    The attributes `W1` and `W2` are convenience variables. They are identical
    to the first n and last (m-n) columns of `eigenvectors`, respectively.
    """

    eigenvalues, eigenvectors = None, None
    W1, W2 = None, None
    e_br, sub_br = None, None

    def compute(self, X=None, f=None, df=None, weights=None, sstype='AS', ptype='EVG', nboot=0):
        """
        TODO: docs
        
        Subspace types (sstype):
            AS, active subspace
            OLS, ols sdr
            QPHD, qphd, sdr
            OPG, opg, sdr
            
        Partition types (ptype):
            EVG, eigenvalue gaps
            RS, response surface error bound
            LI, Li's ladle plot
        """
        
        # Check inputs
        if X is not None:
            X, M, m = process_inputs(X)
        elif df is not None:
            df, M, m = process_inputs(df)
        else:
            raise Exception('One of input/output pairs (X,f) or gradients (df) must not be None')
            
        if weights is None:
            # default weights is for Monte Carlo
            weights = np.ones((M, 1)) / M
        
        # Compute the subspace
        if sstype == 'AS':
            if df is None:
                raise Exception('df is None')
            e, W = active_subspace(df, weights)
            ssmethod = lambda X, f, df, weights: active_subspace(df, weights)
        elif sstype == 'OLS':
            if X is None or f is None:
                raise Exception('X or f is None')
            e, W = ols_subspace(X, f, weights)
            ssmethod = lambda X, f, df, weights: ols_subspace(X, f, weights)
        elif sstype == 'QPHD':
            if X is None or f is None:
                raise Exception('X or f is None')
            e, W = qphd_subspace(X, f, weights)
            ssmethod = lambda X, f, df, weights: qphd_subspace(X, f, weights)
        elif sstype == 'OPG':
            if X is None or f is None:
                raise Exception('X or f is None')
            e, W = opg_subspace(X, f, weights)
            ssmethod = lambda X, f, df, weights: opg_subspace(X, f, weights)
        else:
            e, W = None, None
            ssmethod = None
            raise Exception('Unrecognized subspace type: {}'.format(sstype))
        
        self.eigenvalues, self.eigenvectors = e, W    
        
        # Compute bootstrap ranges and partition
        if nboot > 0:
            e_br, sub_br, li_F = bootstrap_ranges(e, W, X, f, df, weights, ssmethod, nboot)
        else:
            if ptype == 1 or ptype == 2:
                raise Exception('Need to run bootstrap for partition type {}'.format(ptype))
            
            e_br, sub_br = None, None
            
        self.e_br, self.sub_br = e_br, sub_br
        
        # Compute the partition
        if ptype == 'EVG':
            n = eig_partition(e)[0]
        elif ptype == 'RS':
            sub_err = sub_br[:,1].reshape((m-1, 1))
            n = errbnd_partition(e, sub_err)[0]
        elif ptype == 'LI':
            n = ladle_partition(e, li_F)[0]
        else:
            raise Exception('Unrecognized partition type: {}'.format(ptype))
        
        self.partition(n)


    def partition(self, n):
        """
        TODO: docs
        """
        if not isinstance(n, int):
            raise TypeError('n should be an integer')

        m = self.eigenvectors.shape[0]
        if n<1 or n>m:
            raise ValueError('n must be positive and less than the dimension of the eigenvectors.')

        self.W1, self.W2 = self.eigenvectors[:,:n], self.eigenvectors[:,n:]

def active_subspace(df, weights):
    """
    TODO: docs
    """
    df, M, m = process_inputs(df)
        
    # compute the matrix
    C = np.dot(df.transpose(), df * weights)
    
    return sorted_eigh(C)

def ols_subspace(X, f, weights):
    """
    TODO: docs
    """
    X, f, M, m = process_inputs_outputs(X, f)
    
    # solve weighted least squares
    A = np.hstack((np.ones((M, 1)), X)) * np.sqrt(weights)
    b = f * np.sqrt(weights)
    u = np.linalg.lstsq(A, b)[0]
    w = u[1:].reshape((m, 1))
    
    # compute rank-1 C
    C = np.dot(w, w.transpose())
    
    return sorted_eigh(C)

def qphd_subspace(X, f, weights):
    """
    TODO: docs
    """
    X, f, M, m = process_inputs_outputs(X, f)
    
    # check if the points are uniform or Gaussian, set 2nd moment
    if np.amax(X) > 1.0 or np.amin < -1.0:
        gamma = 1.0
    else:
        gamma = 1.0 / 3.0
    
    # compute a quadratic approximation
    pr = PolynomialApproximation(2)
    pr.train(X, f, weights)

    # get regression coefficients
    b, A = pr.g, pr.H

    # compute C
    C = np.outer(b, b.transpose()) + gamma*np.dot(A, A.transpose())

    return sorted_eigh(C)
    
def opg_subspace(X, f, weights):
    """
    TODO: docs
    """
    X, f, M, m = process_inputs_outputs(X, f)
    
    # Obtain gradient approximations using local linear regressions
    df = local_linear_gradients(X, f, weights=weights)
    
    # Use gradient approximations to compute active subspace
    opg_weights = np.ones((df.shape[0], 1)) / df.shape[0]
    e, W = active_subspace(df, opg_weights)
    
    return e, W
    
def eig_partition(e):
    """
    TODO: docs
    """
    # dealing with zeros for the log
    ediff = np.fabs(np.diff(e.reshape((e.size,))))

    # crappy threshold for choosing active subspace dimension
    n = np.argmax(ediff) + 1
    return n, ediff

def errbnd_partition(e, sub_err):
    """
    TODO: docs
    """
    m = e.shape[0]
    
    errbnd = np.zeros((m-1, 1))
    for i in range(m-1):
        errbnd[i] = np.sqrt(np.sum(e[:i+1]))*sub_err[i] + np.sqrt(np.sum(e[i+1:]))
    
    n = np.argmin(errbnd) + 1
    return n, errbnd

def ladle_partition(e, li_F):
    """
    TODO: docs
    """
    G = li_F + e.reshape((e.size, 1)) / np.sum(e)
    n = np.argmin(G) + 1
    return n, G
    
def bootstrap_ranges(e, W, X, f, df, weights, ssmethod, nboot=100):
    """
    TODO: docs
    """
    if df is not None:
        df, M, m = process_inputs(df)
    else:
        X, M, m = process_inputs(X)
    
    e_boot = np.zeros((m, nboot))
    sub_dist = np.zeros((m-1, nboot))
    sub_det = np.zeros((m-1, nboot))
    
    # TODO: should be able to parallelize this
    for i in range(nboot):
        X0, f0, df0, weights0 = bootstrap_replicate(X, f, df, weights)
        e0, W0 = ssmethod(X0, f0, df0, weights0)
        e_boot[:,i] = e0.reshape((m,))
        for j in range(m-1):
            sub_dist[j,i] = np.linalg.norm(np.dot(W[:,:j+1].T, W0[:,j+1:]), ord=2)
            sub_det[j,i] = np.linalg.det(np.dot(W[:,:j+1].T, W0[:,:j+1]))
    
    # bootstrap ranges for the eigenvalues
    e_br = np.hstack(( np.amin(e_boot, axis=1).reshape((m, 1)), \
                        np.amax(e_boot, axis=1).reshape((m, 1)) ))
    
    # bootstrap ranges and mean for subspace distance
    sub_br = np.hstack(( np.amin(sub_dist, axis=1).reshape((m-1, 1)), \
                        np.mean(sub_dist, axis=1).reshape((m-1, 1)), \
                        np.amax(sub_dist, axis=1).reshape((m-1, 1)) ))
    
    # metric from Li's ladle plot paper
    li_F = np.vstack(( np.zeros((1,1)), np.sum(1.0 - np.fabs(sub_det), axis=1).reshape((m-1, 1)) / nboot ))
    li_F = li_F / np.sum(li_F)

    return e_br, sub_br, li_F
    
def sorted_eigh(C):
    """
    TODO: docs
    """
    e, W = np.linalg.eigh(C)
    e = abs(e)
    ind = np.argsort(e)
    e = e[ind[::-1]]
    W = W[:,ind[::-1]]
    s = np.sign(W[0,:])
    s[s==0] = 1
    W = W*s
    return e.reshape((e.size,1)), W
    
def bootstrap_replicate(X, f, df, weights):
    """
    TODO: docs
    """
    M = weights.shape[0]
    ind = np.random.randint(M, size=(M, ))
    
    if X is not None:
        X0 = X[ind,:].copy()
    else:
        X0 = None
        
    if f is not None:
        f0 = f[ind,:].copy()
    else:
        f0 = None
        
    if df is not None:
        df0 = df[ind,:].copy()
    else:
        df0 = None
        
    weights0 = weights[ind,:].copy()
    
    return X0, f0, df0, weights0
    
