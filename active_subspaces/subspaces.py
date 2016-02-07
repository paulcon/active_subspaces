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

    def compute(self, df ,f = 0, X = 0,function=0, c_index = 0, comp_flag =0,N=5, n_boot=200):
        """
        Compute the active and inactive subspaces from a collection of
        sampled gradients.

        :param ndarray df: an ndarray of size M-by-m that contains evaluations of the gradient.
        :param ndarray f: an ndarray of size M that contains evaluations of the function.
        :param ndarray X: an ndarray of size M-by-m that contains data points in the input space.
        :param function: a specified function that outputs f(x), and df(x) the gradient vector for a data point x
        :param int c_index: an integer specifying which C matrix to compute, the default matrix is 0.
        :param int comp_flag: an integer specifying computation method: 0 for monte carlo, 1 for LG quadrature.
        :param int N: number of quadrature points per dimension.
        :param int n_boot: number of bootstrap replicates to use when computing bootstrap ranges.

        **Notes**

        This method sets the class's attributes `W1`, `W2`, `eigenvalues`, and
        `eigenvectors`. If `n_boot` is greater than zero, then this method
        also runs a bootstrap to compute and set `e_br` and `sub_br`.
        """
        
        if c_index != 4:
            df, M, m = process_inputs(df)
        else:
            M = np.shape(X)[0]
            m = np.shape(X)[1]/2
        if not isinstance(n_boot, int):
            raise TypeError('n_boot must be an integer.')
        evecs = np.zeros((m,m))
        evals = np.zeros(m)
        e_br = np.zeros((m,2))
        sub_br = np.zeros((m-1,3))
        # compute eigenvalues and eigenvecs
        if c_index == 0:
            logging.getLogger('PAUL').info('Computing spectral decomp with {:d} samples in {:d} dims.'.format(M, m))
            evals, evecs = spectral_decomposition(df=df)
            if comp_flag == 0:
                # compute bootstrap ranges for eigenvalues and subspace distances
                if n_boot > 0:
                    logging.getLogger('PAUL').info('Bootstrapping {:d} spectral decomps of size {:d} by {:d}.'.format(n_boot, M, m))
                    e_br, sub_br = bootstrap_ranges(df, evals, evecs, n_boot=n_boot)
        elif c_index == 1:  
            if comp_flag == 0:
                evals, evecs = spectral_decomposition(df,f,X,c_index=c_index,comp_flag=comp_flag)
                # compute bootstrap ranges for eigenvalues and subspace distances
                if n_boot > 0:
                    logging.getLogger('PAUL').info('Bootstrapping {:d} spectral decomps of size {:d} by {:d}.'.format(n_boot, M, m))
                    e_br, sub_br = bootstrap_ranges(df, evals, evecs,f, X, c_index,n_boot)
            elif comp_flag == 1:
                evals, evecs = spectral_decomposition(df,f,X,function,c_index,N,comp_flag)        
        elif c_index == 2:
            if comp_flag == 0:
                evals, evecs = spectral_decomposition(df,f,X,c_index=c_index,comp_flag=comp_flag)
                # compute bootstrap ranges for eigenvalues and subspace distances
                if n_boot > 0:
                    logging.getLogger('PAUL').info('Bootstrapping {:d} spectral decomps of size {:d} by {:d}.'.format(n_boot, M, m))
                    e_br, sub_br = bootstrap_ranges(df, evals, evecs,f, X, c_index,n_boot)
            elif comp_flag == 1:
                evals, evecs = spectral_decomposition(df,f,X,function,c_index,N,comp_flag)
                
        elif c_index == 3:
            if comp_flag == 0:
                evals, evecs = spectral_decomposition(df,f,X,c_index=c_index,comp_flag=comp_flag)
                # compute bootstrap ranges for eigenvalues and subspace distances
                if n_boot > 0:
                    logging.getLogger('PAUL').info('Bootstrapping {:d} spectral decomps of size {:d} by {:d}.'.format(n_boot, M, m))
                    e_br, sub_br = bootstrap_ranges(df, evals, evecs,f, X, c_index,n_boot)
            elif comp_flag == 1:
                evals, evecs = spectral_decomposition(df,f,X,function,c_index,N,comp_flag)
        elif c_index == 4:
            if comp_flag == 0:
                evals, evecs = spectral_decomposition(df,f,X,c_index=c_index,comp_flag=comp_flag)
                # compute bootstrap ranges for eigenvalues and subspace distances
                if n_boot > 0:
                   # logging.getLogger('PAUL').info('Bootstrapping {:d} spectral decomps of size {:d} by {:d}.'.format(n_boot, M, 2*m))
                    e_br, sub_br = bootstrap_ranges(df,evals, evecs,f, X, c_index,n_boot)
            elif comp_flag == 1:
                evals, evecs = spectral_decomposition(df,f,X,function,c_index,N,comp_flag)   
        self.e_br, self.sub_br = e_br, sub_br    
        self.e_br, self.sub_br = e_br, sub_br    
        self.eigenvalues, self.eigenvectors = evals, evecs

        

        # partition the subspaces with a crappy heuristic
        n = compute_partition(evals)
        self.partition(n)


    def partition(self, n):
        """
        Set the partition between active and inactive subspaces.

        :param int n: dimension of the active subspace.

        **Notes**

        This method sets the class's attributes `W1` and `W2` according to the
        given `n`. It is mostly a convenience method in case one wants to
        manually set the dimension of the active subspace after the basis is
        computed.
        """
        if not isinstance(n, int):
            raise TypeError('n should be an integer')

        m = self.eigenvectors.shape[0]
        if n<1 or n>m:
            raise ValueError('n must be positive and less than the dimension of the eigenvectors.')

        logging.getLogger('PAUL').info('Active subspace dimension is {:d} out of {:d}.'.format(n, m))

        self.W1, self.W2 = self.eigenvectors[:,:n], self.eigenvectors[:,n:]

def compute_partition(evals):
    """
    A heuristic based on eigenvalue gaps for deciding the dimension of the
    active subspace.

    :param ndarray evals: the eigenvalues.

    :return: dimension of the active subspace
    :rtype: int
    """
    # dealing with zeros for the log
    e = evals.copy()
    ind = e==0.0
    e[ind] = 1e-100

    # crappy threshold for choosing active subspace dimension
    n = np.argmax(np.fabs(np.diff(np.log(e.reshape((e.size,)))))) + 1
    return n


def active_subspaces(X=None, f=None, df, weights):
    """
    
    """
    df, M, m = process_inputs(df)
        
    # multiply each row by the weights
    df = df * weights
    
    # compute the matrix
    C = np.dot(df.transpose(), df)
    
    return sorted_eigh(C)
    
def normalized_active_subspaces(X=None, f=None, df, weights):
    """
    
    """
    df, M, m = process_inputs(df)
        
    # get row norms
    ndf = np.sqrt(np.sum(df*df, axis=1))
    
    # find rows with norm too close to zero and set elements to exactly zero
    ind = ndf < SQRTEPS
    df[ind,:], ndf[ind] = 0.0, 1.0
    
    # normalize rows and multiply by weights
    df = df * (weights / ndf.reshape((M, 1)))
    
    # compute the matrix
    C = np.dot(df.transpose(), df)
    
    return sorted_eigh(C)
    
def active_subspaces_x(X, f=None, df, weights):
    """
    
    """
    df, M, m = process_inputs(df)
    
    # multiply by weights
    df, X = df * weights, X * weights
    
    # compute the matrix
    A = np.dot(df.transpose(), X)
    C = 0.5*(A + A.transpose())
    
    return sorted_eigh(C)
    
def normalized_active_subspaces_x(X, f=None, df, weights):
    """
    
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
    
    # normalize rows and multiply by weights
    df = df * (weights / ndf.reshape((M, 1)))
    X = X * (weights / nX.reshape((M, 1)))
    
    # compute the matrix
    A = np.dot(df.transpose(), X)
    C = 0.5*(A + A.transpose())
    
    return sorted_eigh(C)           

def swarm_subspaces(X, f, df=None, weights):
    """
    
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

def rs_partition(e, sub_err):
    """
    
    """
    m = e.shape[0]
    
    err = np.zeros((m-1, 1))
    for i in range(m-1):
        err[i] = np.sqrt(np.sum(e[:i+1,:]))*sub_err[i] + np.sqrt(np.sum(e[i+1:,:]))
        
    n = np.argmin(err) + 1
    return n, err

def ladle(e, li_F):
    """
    
    """
    Phi = e / np.sum(e)
    G = li_F + Phi
    n = np.argmin(G) + 1
    return n, G
    
def bootstrap_ranges(e, W, X, f, df, weights, ssmethod, nboot=100):
    """
    
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
            
    e_br = np.zeros((m, 2))
    sub_br = np.zeros((m-1, 3))
    for i in range(m):
        e_br[i,0] = np.amin(e_boot[i,:])
        e_br[i,1] = np.amax(e_boot[i,:])
    for i in range(m-1):
        sub_br[i,0] = np.amin(sub_dist[i,:])
        sub_br[i,1] = np.mean(sub_dist[i,:])
        sub_br[i,2] = np.amax(sub_dist[i,:])
        
    li_F = np.sum(1.0 - np.fabs(sub_det), axis=1) / M
    li_F = li_F / np.sum(li_F)

    return e_br, sub_br, li_F
    
def sorted_eigh(C):
    """
    
    """
    e, W = np.linalg.eigh(C)
    ind = np.argsort(e)
    e = e[ind[::-1]]
    W = W[:,ind[::-1]]
    W = W*np.sign(W[0,:])
    return e, W
    
def bootstrap_replicate(X, f, df, weights):
    """
    
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