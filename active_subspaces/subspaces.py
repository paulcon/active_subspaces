"""Utilities for computing active and inactive subspaces."""
from __future__ import division
import numpy as np
import logging
from scipy.spatial import distance_matrix
from utils.misc import process_inputs, process_inputs_outputs
import utils.quadrature as quad

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

    def compute(self, X=None, f=None, df=None, weights=None, sstype=0, ptype=0, nboot=0):
        """
        TODO: docs
        
        Subspace types (sstype):
            0, active subspace
            1, normalized active subspace
            2, active subspace x
            3, normalized active subspace x
            4, swarm subspace
            
        Partition types (ptype):
            0, eigenvalue gaps
            1, response surface error bound
            2, Li's ladle plot
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
        if sstype == 0:
            if df is None:
                raise Exception('df is None')
            e, W = active_subspace(df=df, weights=weights)
            ssmethod = active_subspace
        elif sstype == 1:
            if df is None:
                raise Exception('df is None')
            e, W = normalized_active_subspace(df=df, weights=weights)
            ssmethod = normalized_active_subspace
        elif sstype == 2:
            if X is None or df is None:
                raise Exception('X or df is None')
            e, W = active_subspace_x(X=X, df=df, weights=weights)
            ssmethod = active_subspace_x
        elif sstype == 3:
            if X is None or df is None:
                raise Exception('X or df is None')            
            e, W = normalized_active_subspace_x(X=X, df=df, weights=weights)
            ssmethod = normalized_active_subspace_x
        elif sstype == 4:
            if X is None or f is None:
                raise Exception('X or f is None')
            e, W = swarm_subspace(X=X, f=f, weights=weights)
            ssmethod = swarm_subspace
        else:
            e, W = None, None
            ssmethod = None
            raise Exception('Unrecognized subspace type: {:d}'.format(sstype))
        
        self.eigenvalues, self.eigenvectors = e, W    
        
        # Compute bootstrap ranges and partition
        if nboot > 0:
            e_br, sub_br, li_F = bootstrap_ranges(e, W, X, f, df, weights, ssmethod, nboot)
        else:
            if ptype == 1 or ptype == 2:
                raise Exception('Need to run bootstrap for partition type {:d}'.format(ptype))
            
            e_br, sub_br = None, None
            
        self.e_br, self.sub_br = e_br, sub_br
        
        # Compute the partition
        if ptype == 0:
            n = eig_partition(e)
        elif ptype == 1:
            sub_err = sub_br[:,1].reshape((m-1, 1))
            n = errbnd_partition(e, sub_err)
        elif ptype == 2:
            n = ladle_partition(e, li_F)
        else:
            raise Exception('Unrecognized partition type: {:d}'.format(ptype))
        
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

def active_subspace(X=None, f=None, df, weights):
    """
    TODO: docs
    """
    df, M, m = process_inputs(df)
        
    # compute the matrix
    C = np.dot(df.transpose(), df * weights)
    
    return sorted_eigh(C)
    
def normalized_active_subspace(X=None, f=None, df, weights):
    """
    TODO: docs
    """
    df, M, m = process_inputs(df)
        
    # get row norms
    ndf = np.sqrt(np.sum(df*df, axis=1))
    
    # find rows with norm too close to zero and set elements to exactly zero
    ind = ndf < SQRTEPS
    df[ind,:], ndf[ind] = 0.0, 1.0
    
    # normalize rows
    df = df / ndf.reshape((M, 1))
    
    # compute the matrix
    C = np.dot(df.transpose(), df * weights)
    
    return sorted_eigh(C)
    
def active_subspace_x(X, f=None, df, weights):
    """
    TODO: docs
    """
    df, M, m = process_inputs(df)
    
    # compute the matrix
    A = np.dot(df.transpose(), X * weights)
    C = 0.5*(A + A.transpose())
    
    return sorted_eigh(C)
    
def normalized_active_subspace_x(X, f=None, df, weights):
    """
    TODO: docs
    """
    df, M, m = process_inputs(df)
    
    # get row norms
    ndf = np.sqrt(np.sum(df*df, axis=1))
    nX = np.sqrt(np.sum(X*X, axis=1))
    
    # find rows with norm too close to zero and set elements to exactly zero
    ind = ndf < SQRTEPS
    df[ind,:], ndf[ind] = 0.0, 1.0
    
    ind = nX < SQRTEPS
    X[ind,:], nX[ind] = 0.0, 1.0
    
    # normalize rows
    df = df / ndf.reshape((M, 1))
    X = X / nX.reshape((M, 1))
    
    # compute the matrix
    A = np.dot(df.transpose(), X * weights)
    C = 0.5*(A + A.transpose())
    
    return sorted_eigh(C)           

def swarm_subspace(X, f, df=None, weights):
    """
    TODO: docs
    """
    X, f, M, m = process_inputs_outputs(X, f)
    
    # integration weights
    W = np.dot(weights,weights.transpose())
    
    # distance matrix, getting rid of zeros
    D2 = np.power(distance_matrix(X,X),2)
    ind = D2 < SQRTEPS
    W[ind], D2[ind] = 0.0, 1.0
    
    # all weights
    A = (np.power(f-f.transpose(), 2) * W) / D2
    
    C = np.zeros((m, m))
    for i in range(M):
        P = X - X[i,:]
        C = C + np.dot(P.transpose(), P*A[:,i])
    
    return sorted_eigh(C)

def eig_partition(evals):
    """
    TODO: docs
    """
    # dealing with zeros for the log
    e = evals.copy()
    ind = e==0.0
    e[ind] = 1e-100
    ediff = np.fabs(np.diff(np.log10(e.reshape((e.size,)))))

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
        err[i] = np.sqrt(np.sum(e[:i+1,:]))*sub_err[i] + np.sqrt(np.sum(e[i+1:,:]))
        
    n = np.argmin(err) + 1
    return n, errbnd

def ladle_partition(e, li_F):
    """
    TODO: docs
    """
    Phi = e / np.sum(e)
    G = li_F + Phi
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
    e_br = np.hstack(( np.amin(e_boot, axis=1), np.amax(e_boot, axis=1) ))
    
    # bootstrap ranges and mean for subspace distance
    sub_br = np.hstack(( np.amin(sub_dist, axis=1), np.mean(sub_dist, axis=1), np.amax(sub_dist, axis=1) ))
    
    # metric from Li's ladle plot paper
    li_F = np.sum(1.0 - np.fabs(sub_det), axis=1) / nboot
    li_F = li_F / np.sum(li_F)

    return e_br, sub_br, li_F
    
def sorted_eigh(C):
    """
    TODO: docs
    """
    e, W = np.linalg.eigh(C)
    ind = np.argsort(e)
    e = e[ind[::-1]]
    W = W[:,ind[::-1]]
    W = W*np.sign(W[0,:])
    return e, W
    
def bootstrap_replicate(X, f, df, weights):
    """
    TODO: docs
    """
    ind = np.random.randint(M, size=(M, 1))
    
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
    
"""
bootstrap_ranges comments
Use a nonparametric bootstrap to estimate variability in the computed
eigenvalues and subspaces.

:param ndarray df: M-by-m matrix of evaluations of the gradient.
:param ndarray e: m-by-1 vector of eigenvalues.
:param ndarray W: eigenvectors.
:param ndarray f: M-by-1 vector of function evaluations.
:param ndarray X: M-by-m array for c_index = 0,1,2,3 *******OR******** M-by-2m matrix for c_index = 4.
:param int c_index: an integer specifying which C matrix to compute, the default matrix is 0
:param int n_boot: index number for alternative subspaces.

:return: [e_br, sub_br], e_br: m-by-2 matrix that contains the bootstrap ranges for the eigenvalues, 
sub_br: m-by-3 matrix that contains the bootstrap ranges (first and third column) and the mean (second column) 
of the error in the estimated subspaces approximated by bootstrap
:rtype: [ndarray, ndarray]

**Notes**

The mean of the subspace distance bootstrap replicates is an interesting
quantity. Still trying to figure out what its precise relation is to
the subspace error. They seem to behave similarly in test problems. And an
initial "coverage" study suggested that they're unbiased for a quadratic
test problem. Quadratics, though, may be special cases.
"""


"""
spectral_decomposition comments
Use the SVD to compute the eigenvectors and eigenvalues for the
active subspace analysis.

:param ndarray df: an ndarray of size M-by-m that contains evaluations of the gradient.
:param ndarray f: an ndarray of size M that contains evaluations of the function.
:param ndarray X: an ndarray of size M-by-m that contains data points in the input space.
:param function: a specified function that outputs f(x), and df(x) the gradient vector for a data point x
:param int c_index: an integer specifying which C matrix to compute, the default matrix is 0.
:param int comp_flag: an integer specifying computation method: 0 for monte carlo, 1 for LG quadrature.
:param int N: number of quadrature points per dimension.

:return: [e, W], [ eigenvalues, eigenvectors ]
:rtype: [ndarray, ndarray]

**Notes**

If the number M of gradient samples is less than the dimension m of the
inputs, then the method builds an arbitrary basis for the nullspace, which
corresponds to the inactive subspace.
"""