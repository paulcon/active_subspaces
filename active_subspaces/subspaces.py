"""Utilities for computing active and inactive subspaces."""

import numpy as np
from .utils.misc import process_inputs, process_inputs_outputs
from .utils.response_surfaces import PolynomialApproximation
from .gradients import local_linear_gradients

SQRTEPS = np.sqrt(np.finfo(float).eps)

class Subspaces():
    """A class for computing active and inactive subspaces.
    
    Attributes
    ----------
        eigenvals : ndarray 
            m-by-1 matrix of eigenvalues
        eigenvecs : ndarray
            m-by-m matrix, eigenvectors oriented column-wise
        W1 : ndarray
            m-by-n matrix, basis for the active subspace
        W2 : ndarray
            m-by-(m-n) matrix, basis for the inactive subspace
        e_br : ndarray
            m-by-2 matrix, bootstrap ranges for the eigenvalues
        sub_br : ndarray
            m-by-3 matrix, bootstrap ranges (first and third column) and the 
            mean (second column) of the error in the estimated active subspace 
            approximated by bootstrap
            
    Notes
    -----
    The attributes `W1` and `W2` are convenience variables. They are identical
    to the first n and last (m-n) columns of `eigenvecs`, respectively.
    """

    eigenvals, eigenvecs = None, None
    W1, W2 = None, None
    e_br, sub_br = None, None

    def compute(self, X=None, f=None, df=None, weights=None, sstype='AS', ptype='EVG', nboot=0):
        """Compute the active and inactive subspaces.
        
        Given input points and corresponding outputs, or given samples of the 
        gradients, estimate an active subspace. This method has four different
        algorithms for estimating the active subspace: 'AS' is the standard
        active subspace that requires gradients, 'OLS' uses a global linear
        model to estimate a one-dimensional active subspace, 'QPHD' uses a 
        global quadratic model to estimate subspaces, and 'OPG' uses a set of
        local linear models computed from subsets of give input/output pairs.
        
        The function also sets the dimension of the active subspace (and, 
        consequently, the dimenison of the inactive subspace). There are three
        heuristic choices for the dimension of the active subspace. The default
        is the largest gap in the eigenvalue spectrum, which is 'EVG'. The other
        two choices are 'RS', which estimates the error in a low-dimensional 
        response surface using the eigenvalues and the estimated subspace 
        errors, and 'LI' which is a heuristic from Bing Li on order 
        determination. 
        
        Note that either `df` or `X` and `f` must be given, although formally
        all are optional.
        
        Parameters
        ----------
        X : ndarray, optional
            M-by-m matrix of samples of inputs points, arranged as rows (default
            None)
        f : ndarray, optional
            M-by-1 matrix of outputs corresponding to rows of `X` (default None)
        df : ndarray, optional
            M-by-m matrix of samples of gradients, arranged as rows (default
            None)
        weights : ndarray, optional
           M-by-1 matrix of weights associated with rows of `X`
        sstype : str, optional
           defines subspace type to compute. Default is 'AS' for active 
           subspace, which requires `df`. Other  options are `OLS` for a global 
           linear model, `QPHD` for a global quadratic model, and `OPG` for 
           local linear models. The latter three require `X` and `f`.
        ptype : str, optional
            defines the partition type. Default is 'EVG' for largest 
            eigenvalue gap. Other options are 'RS', which is an estimate of the
            response surface error, and 'LI', which is a heuristic proposed by
            Bing Li based on subspace errors and eigenvalue decay.
        nboot : int, optional
            number of bootstrap samples used to estimate the error in the 
            estimated subspace (default 0 means no bootstrap estimates)
            
        Notes
        -----
        Partition type 'RS' and 'LI' require nboot to be greater than 0 (and
        probably something more like 100) to get bootstrap estimates of the 
        subspace error. 
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
        
        self.eigenvals, self.eigenvecs = e, W    
        
        # Compute bootstrap ranges and partition
        if nboot > 0:
            e_br, sub_br, li_F = _bootstrap_ranges(e, W, X, f, df, weights, ssmethod, nboot)
        else:
            if ptype == 'RS' or ptype == 'LI':
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
        """Partition the eigenvectors to define the active subspace.
        
        A convenience function for partitioning the full set of eigenvectors to
        separate the active from inactive subspaces.
        
        Parameters
        ----------
        n : int
            the dimension of the active subspace
        
        """
        if not isinstance(n, int):
            #raise TypeError('n should be an integer')
            n = int(n)
            print(Warning("n should be an integer. Performing conversion."))

        m = self.eigenvecs.shape[0]
        if n<1 or n>m:
            raise ValueError('n must be positive and less than the dimension of the eigenvectors.')

        self.W1, self.W2 = self.eigenvecs[:,:n], self.eigenvecs[:,n:]

def active_subspace(df, weights):
    """Compute the active subspace.
    
    Parameters
    ----------
    df : ndarray
        M-by-m matrix containing the gradient samples oriented as rows
    weights : ndarray
        M-by-1 weight vector, corresponds to numerical quadrature rule used to
        estimate matrix whose eigenspaces define the active subspace
        
    Returns
    -------
    e : ndarray
        m-by-1 vector of eigenvalues
    W : ndarray
        m-by-m orthogonal matrix of eigenvectors
    """
    df, M, m = process_inputs(df)
        
    # compute the matrix
    C = np.dot(df.transpose(), df * weights)
    
    return sorted_eigh(C)

def ols_subspace(X, f, weights):
    """Estimate one-dimensional subspace with global linear model.
    
    Parameters
    ----------
    X : ndarray
        M-by-m matrix of input samples, oriented as rows
    f : ndarray
        M-by-1 vector of output samples corresponding to the rows of `X`
    weights : ndarray
        M-by-1 weight vector, corresponds to numerical quadrature rule used to
        estimate matrix whose eigenspaces define the active subspace
        
    Returns
    -------
    e : ndarray
        m-by-1 vector of eigenvalues
    W : ndarray
        m-by-m orthogonal matrix of eigenvectors
        
    Notes
    -----
    Although the method returns a full set of eigenpairs (to be consistent with
    the other subspace functions), only the first eigenvalue will be nonzero,
    and only the first eigenvector will have any relationship to the input 
    parameters. The remaining m-1 eigenvectors are only orthogonal to the first.
    """
    X, f, M, m = process_inputs_outputs(X, f)
    
    # solve weighted least squares
    A = np.hstack((np.ones((M, 1)), X)) * np.sqrt(weights)
    b = f * np.sqrt(weights)
    u = np.linalg.lstsq(A, b, rcond=None)[0]
    w = u[1:].reshape((m, 1))
    
    # compute rank-1 C
    C = np.dot(w, w.transpose())
    
    return sorted_eigh(C)

def qphd_subspace(X, f, weights):
    """Estimate active subspace with global quadratic model.
    
    This approach is similar to Ker-Chau Li's approach for principal Hessian
    directions based on a global quadratic model of the data. In contrast to
    Li's approach, this method uses the average outer product of the gradient
    of the quadratic model, as opposed to just its Hessian.
    
    Parameters
    ----------
    X : ndarray
        M-by-m matrix of input samples, oriented as rows
    f : ndarray
        M-by-1 vector of output samples corresponding to the rows of `X`
    weights : ndarray
        M-by-1 weight vector, corresponds to numerical quadrature rule used to
        estimate matrix whose eigenspaces define the active subspace
        
    Returns
    -------
    e : ndarray
        m-by-1 vector of eigenvalues
    W : ndarray
        m-by-m orthogonal matrix of eigenvectors

    """
    X, f, M, m = process_inputs_outputs(X, f)
    
    # check if the points are uniform or Gaussian, set 2nd moment
    if np.amax(X) > 1.0 or np.amin(X) < -1.0:
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
    """Estimate active subspace with local linear models.
    
    This approach is related to the sufficient dimension reduction method known 
    sometimes as the outer product of gradient method. See the 2001 paper 
    'Structure adaptive approach for dimension reduction' from Hristache, et al.
    
    Parameters
    ----------
    X : ndarray
        M-by-m matrix of input samples, oriented as rows
    f : ndarray
        M-by-1 vector of output samples corresponding to the rows of `X`
    weights : ndarray
        M-by-1 weight vector, corresponds to numerical quadrature rule used to
        estimate matrix whose eigenspaces define the active subspace
        
    Returns
    -------
    e : ndarray
        m-by-1 vector of eigenvalues
    W : ndarray
        m-by-m orthogonal matrix of eigenvectors
    """
    X, f, M, m = process_inputs_outputs(X, f)
    
    # Obtain gradient approximations using local linear regressions
    df = local_linear_gradients(X, f, weights=weights)
    
    # Use gradient approximations to compute active subspace
    opg_weights = np.ones((df.shape[0], 1)) / df.shape[0]
    e, W = active_subspace(df, opg_weights)
    
    return e, W
    
def eig_partition(e):
    """Partition the active subspace according to largest eigenvalue gap.
    
    Parameters
    ----------
    e : ndarray
        m-by-1 vector of eigenvalues
        
    Returns
    -------
    n : int
        dimension of active subspace
    ediff : float
        largest eigenvalue gap
    """
    # dealing with zeros for the log
    ediff = np.fabs(np.diff(e.reshape((e.size,))))

    # crappy threshold for choosing active subspace dimension
    n = np.argmax(ediff) + 1
    return n, ediff

def errbnd_partition(e, sub_err):
    """Partition the active subspace according to response surface error.
    
    Uses an a priori estimate of the response surface error based on the 
    eigenvalues and subspace error to determine the active subspace dimension. 
    
    Parameters
    ----------
    e : ndarray
        m-by-1 vector of eigenvalues
    sub_err : ndarray
        m-by-1 vector of estimates of subspace error
        
        
    Returns
    -------
    n : int
        dimension of active subspace
    errbnd : float
        estimate of error bound
        
    Notes
    -----
    The error bound should not be used as an estimated error. The bound is only
    used to estimate the subspace dimension.
    
    """
    m = e.shape[0]
    
    errbnd = np.zeros((m-1, 1))
    for i in range(m-1):
        errbnd[i] = np.sqrt(np.sum(e[:i+1]))*sub_err[i] + np.sqrt(np.sum(e[i+1:]))
    
    n = np.argmin(errbnd) + 1
    return n, errbnd

def ladle_partition(e, li_F):
    """Partition the active subspace according to Li's criterion.
    
    Uses a criterion proposed by Bing Li that combines estimates of the subspace
    with estimates of the eigenvalues. 
    
    Parameters
    ----------
    e : ndarray
        m-by-1 vector of eigenvalues
    li_F : float
        measures error in the subspace
        
    Returns
    -------
    n : int
        dimension of active subspace
    G : ndarray
        metrics used to determine active subspace dimension
        
    """
    G = li_F + e.reshape((e.size, 1)) / np.sum(e)
    n = np.argmin(G) + 1
    return n, G
    
def _bootstrap_ranges(e, W, X, f, df, weights, ssmethod, nboot=100):
    """Compute bootstrap ranges for eigenvalues and subspaces.
    
    An implementation of the nonparametric bootstrap that we use in 
    conjunction with the subspace estimation methods to estimate the errors in 
    the eigenvalues and subspaces.
    
    Parameters
    ----------
    e : ndarray
        m-by-1 vector of eigenvalues
    W : ndarray
        m-by-m orthogonal matrix of eigenvectors
    X : ndarray
        M-by-m matrix of input samples, oriented as rows
    f : ndarray
        M-by-1 vector of outputs corresponding to rows of `X`
    df : ndarray
        M-by-m matrix of gradient samples
    weights : ndarray
        M-by-1 vector of weights corresponding to samples
    ssmethod : function
        a function that returns eigenpairs given input/output or gradient
        samples
    nboot : int, optional
        number of bootstrap samples (default 100)
        
    Returns
    -------
    e_br : ndarray
        m-by-2 matrix, first column contains bootstrap lower bound on 
        eigenvalues, second column contains bootstrap upper bound on 
        eigenvalues
    sub_br : ndarray
        (m-1)-by-3 matrix, first column contains bootstrap lower bound on 
        estimated subspace error, second column contains estimated mean of
        subspace error (a reasonable subspace error estimate), third column
        contains estimated upper bound on subspace error
    li_F : float
        Bing Li's metric for order determination based on determinants
    
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
        X0, f0, df0, weights0 = _bootstrap_replicate(X, f, df, weights)
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
    """Compute eigenpairs and sort.
    
    Parameters
    ----------
    C : ndarray
        matrix whose eigenpairs you want
        
    Returns
    -------
    e : ndarray
        vector of sorted eigenvalues
    W : ndarray
        orthogonal matrix of corresponding eigenvectors
    
    Notes
    -----
    Eigenvectors are unique up to a sign. We make the choice to normalize the
    eigenvectors so that the first component of each eigenvector is positive.
    This normalization is very helpful for the bootstrapping. 
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
    
def _bootstrap_replicate(X, f, df, weights):
    """Return a bootstrap replicate.
    
    A bootstrap replicate is a sampling-with-replacement strategy from a given
    data set. 
    
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
    
