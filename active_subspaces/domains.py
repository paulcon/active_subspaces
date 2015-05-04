"""Utilities for building the domains and maps for active variables."""
import numpy as np
import warnings
from utils.utils import process_inputs, BoundedNormalizer
from scipy.spatial import ConvexHull
from utils.qp_solver import QPSolver
from subspaces import Subspaces

class ActiveVariableDomain():
    """
    A base class for the domain of functions of active variables.
    
    Attributes
    ----------
    subspaces : Subspaces
        `subspaces` is a Subspaces object.
    m : int
        `m` is the dimension of the simulation inputs.
    n : int
        `n` is the dimension of the active subspace. 
    vertY : ndarray
        `vertY` is an ndarray that contains n-dimensional vertices that define
        the boundary of the domain when the m-dimensional space is bounded by
        a hypercube.
    vertX : ndarray
        `vertX` is an ndarray of the corners of the m-dimensional hypercube
        that map to the points `vertY`.
    convhull : scipy.spatial.ConvexHull
        `convhull` is the the ConvexHull object defined by the vertices `vertY`.
    constraints : dict
        `constraints` is a dictionary of linear inequality constraints 
        conforming to the specifications used in the scipy.optimizer library.
    
    Notes
    -----
    Attributes `vertY`, `vertX`, `convhull`, and `constraints` are None when the
    m-dimensional parameter space is unbounded. 
    """
    subspaces = None
    m, n = None, None
    vertY, vertX = None, None
    convhull, constraints = None, None
    
class UnboundedActiveVariableDomain(ActiveVariableDomain):
    """
    An class for the domain of functions of active variables when the space
    of simulation parameters is unbounded.
    
    Notes
    -----
    Using this class assumes that the space of simulation inputs is equipped 
    with a Gaussian weight function.
    """
    def __init__(self, subspaces):
        """
        Initialize the UnboundedActiveVariableDomain.
        
        Parameters
        ----------
        subspaces : Subspaces
            `subspaces` is a Subspaces object with the `compute` method already
            called.
        """
        if not isinstance(subspaces, Subspaces):
            raise TypeError('subspaces should be a Subspaces object.')
            
        if subspaces.W1 is None:
            raise ValueError('The given subspaces has not been computed.')
        
        self.subspaces = subspaces
        self.m, self.n = subspaces.W1.shape

class BoundedActiveVariableDomain(ActiveVariableDomain):
    """
    An class for the domain of functions of active variables when the space
    of simulation parameters is bounded.
    
    Notes
    -----
    Using this class assumes that the space of simulation inputs is equipped 
    with a uniform weight function. And the space itself is a hypercube.
    """
    
    def __init__(self, subspaces):
        """
        Initialize the BoundedActiveVariableDomain.
        
        Parameters
        ----------
        subspaces : Subspaces
            `subspaces` is a Subspaces object with the `compute` method already
            called.
        """
        if not isinstance(subspaces, Subspaces):
            raise TypeError('subspaces should be a Subspaces object.')
            
        if subspaces.W1 is None:
            raise ValueError('The given subspaces has not been computed.')

        self.subspaces = subspaces
        self.m, self.n = subspaces.W1.shape
        self.compute_boundary()
        
    def compute_boundary(self):
        """
        Compute and set the boundary of the domain.
        
        Notes
        -----
        This function computes the boundary of the active variable range, i.e.,
        the domain of a function of the active variables, and it sets the 
        attributes to the computed components. It is called when the 
        BoundedActiveVariableDomain is initialized. If the dimension of the 
        active subspaces is manually changed, then this function must be called 
        again to recompute the boundary of the domain.
        """
        W1 = self.subspaces.W1
        m, n = W1.shape
        
        if n == 1:
            Y, X = interval_endpoints(W1)
            convhull = None
            constraints = None
        else:
	    Y, X = zonotope_vertices(W1)
	    numverts = nzv(m,n)[0]
	    if Y.shape[0] != numverts:
	        warnings.warn('Number of zonotope vertices should be {:d} but is {:d}'.format((numverts,Y.shape[0])))
	    
            convhull = ConvexHull(Y)
            A = convhull.equations[:,:n]
            b = convhull.equations[:,n]
            constraints = ({'type' : 'ineq',
                        'fun' : lambda x: np.dot(A, x) - b,
                        'jac' : lambda x: A})

        # store variables
        self.vertY, self.vertX = Y, X
        self.convhull, self.constraints = convhull, constraints

class ActiveVariableMap():
    """
    A base class for the map between active/inactive and original variables.
    
    Attributes
    ----------
    domain : ActiveVariableDomain
        `domain` is an ActiveVariableDomain object.
    
    See Also
    --------
    domains.UnboundedActiveVariableMap
    domains.BoundedActiveVariableMap
    """
    domain = None
    
    def __init__(self, domain):
        """
        Initialize the ActiveVariableMap.
        
        Parameters
        ----------
        domain : ActiveVariableDomain
            `domain` is an ActiveVariableDomain object.
        """
        self.domain = domain

    def forward(self, X):
        """
        Map the points in the original input space to the active and inactive
        variables.
        
        Parameters
        ----------
        X : ndarray
            `X` is an ndarray of size M-by-m. Each row of `X` is a point in the
            original parameter space
            
        Returns
        -------
        Y : ndarray
            `Y` is an ndarray of size M-by-n that contains points in the space
            of active variables. Each row of `Y` corresponds to a row of `X`.
        Z : ndarray
            `Z` is an ndarray of size M-by-(m-n) that contains points in the 
            space of inactive variables. Each row of `Z` corresponds to a row of
            `X`.
            
        """
        X = process_inputs(X)[0]
        W1, W2 = self.domain.subspaces.W1, self.domain.subspaces.W2
        Y, Z = np.dot(X, W1), np.dot(X, W2)
        return Y, Z

    def inverse(self, Y, N=1):
        """
        Map the points in the active variable space to the original parameter
        space.
        
        Parameters
        ----------
        Y : ndarray
            `Y` is an ndarray of size M-by-n that contains points in the space
            of active variables.
        N : int, optional
            `N` is the number of points in the original parameter space that are
            returned that map to the given active variables. (Default is 1)
            
        Returns
        -------
        X : ndarray
            `X` is an ndarray of shape (M*N)-by-m that contains points in the 
            original parameter space.
        ind : ndarray
            `ind` is an ndarray of shape (M*N)-by-1 that contains integer
            indices. These indices identify which rows of `X` map to which
            rows of `Y`. 
            
        Notes
        -----
        The inverse map depends critically on the `regularize_z` function.
        """
        # check inputs
        Y = process_inputs(Y)[0]
        if not isinstance(N, int):
            raise TypeError('N must be an int') 
        
        Z = self.regularize_z(Y, N)
        W = self.domain.subspaces.eigenvectors
        X, ind = _rotate_x(Y, Z, W)
        return X, ind

    def regularize_z(self, Y, N):
        """
        Find points in the space of inactive variables to complete the inverse
        map.
        
        Parameters
        ----------
        Y : ndarray
            `Y` is an ndarray of size M-by-n that contains points in the space
            of active variables.
        N : int
            `N` is the number of points in the original parameter space that are
            returned that map to the given active variables. 
            
        Returns
        -------
        Z : ndarray
            `Z` is an ndarray of shape (M)-by-(m-n)-by-N that contains values of
            the inactive variables.
            
        Notes
        -----
        The base class does not implement `regularize_z`. Specific 
        implementations depend on whether the original variables are bounded or 
        unbounded. They also depend on what the weight function is on the 
        original parameter space. 
        """
        raise NotImplementedError()

class BoundedActiveVariableMap(ActiveVariableMap):
    """
    A class for the map between active/inactive and original variables when the
    original variables are bounded by a hypercube with a uniform density.
    
    See Also
    --------
    domains.UnboundedActiveVariableMap
    """
    def regularize_z(self, Y, N):
        """
        Find points in the space of inactive variables to complete the inverse
        map.
        
        Parameters
        ----------
        Y : ndarray
            `Y` is an ndarray of size M-by-n that contains points in the space
            of active variables.
        N : int
            `N` is the number of points in the original parameter space that are
            returned that map to the given active variables. 
            
        Returns
        -------
        Z : ndarray
            `Z` is an ndarray of shape (M)-by-(m-n)-by-N that contains values of
            the inactive variables.
            
        Notes
        -----
        This implementation of `regularize_z` uses the function `sample_z` to 
        randomly sample values of the inactive variables to complement the
        given values of the active variables. 
        """
        W1, W2 = self.domain.subspaces.W1, self.domain.subspaces.W2
        m, n = W1.shape

        # sample the z's
        # TODO: preallocate and organize properly
        NY = Y.shape[0]
        Zlist = []
        for y in Y:
            Zlist.append(sample_z(N, y, W1, W2))
        Z = np.array(Zlist).reshape((NY, m-n, N))
        return Z

class UnboundedActiveVariableMap(ActiveVariableMap):
    """
    A class for the map between active/inactive and original variables when the
    original variables are ubbounded and the space is equipped with a standard
    Gaussian density.
    
    See Also
    --------
    domains.BoundedActiveVariableMap
    """
    def regularize_z(self, Y, N):
        """
        Find points in the space of inactive variables to complete the inverse
        map.
        
        Parameters
        ----------
        Y : ndarray
            `Y` is an ndarray of size M-by-n that contains points in the space
            of active variables.
        N : int
            `N` is the number of points in the original parameter space that are
            returned that map to the given active variables. 
            
        Returns
        -------
        Z : ndarray
            `Z` is an ndarray of shape (M)-by-(m-n)-by-N that contains values of
            the inactive variables.
            
        Notes
        -----
        This implementation of `regularize_z` samples the inactive variables 
        from a standard (m-n)-variate Gaussian distribution.
        """
        m, n = self.domain.subspaces.W1.shape

        # sample z's
        NY = Y.shape[0]
        Z = np.random.normal(size=(NY, m-n, N))
        return Z

def nzv(m, n, M=None):
    """
    Compute the number of zonotope vertices for a linear map from R^m to R^n.
    
    Parameters
    ----------
    m : int
        `m` is the dimension of the hypercube.
    n : int
        `n` is the dimension of the low-dimesional subspace.
        
    Returns
    -------
    k : int
        `k` is number of vertices defining the zonotope.
    M : ndarray
        `M` is used as a temporary variable for the recursive computation. It
        can be discarded.
    """
    if not isinstance(m, int):
        raise TypeError('m should be an integer.')
        
    if not isinstance(n, int):
        raise TypeError('n should be an integer.')
    
    # number of zonotope vertices
    if M is None:
        M = np.zeros((m, n))
    if m==1 or n==1:
        M[m-1, n-1] = 2
    elif M[m-1, n-1]==0:
        k1, M = nzv(m-1, n-1, M)
        k2, M = nzv(m-1, n, M)
        M[m-1, n-1] = k1 + k2
        for i in range(n-1):
            M = nzv(m, i+1, M)[1]
    k = M[m-1, n-1]
    return k, M

def interval_endpoints(W1):
    """
    Compute the range of a 1d active variable.
    
    Parameters
    ----------
    W1 : ndarray
        `W1` is an ndarray of shape m-by-1 that contains the eigenvector that
        defines the first active variable.
        
    Returns
    -------
    Y : ndarray
        `Y` is an ndarray of shape 2-by-1 that contains the endpoints of the
        interval defining the range of the 1d active variable.
    X : ndarray
        `X` is an ndarray of shape 2-by-m that contains the corners of the 
        m-dimensional hypercube that map to the active variable endpoints.
    """
    
    m = W1.shape[0]
    y0 = np.dot(W1.T, np.sign(W1))[0]
    if y0 < -y0:
        yl, yu = y0, -y0
        xl, xu = np.sign(W1), -np.sign(W1)
    else:
        yl, yu = -y0, y0
        xl, xu = -np.sign(W1), np.sign(W1)
    Y = np.array([yl, yu]).reshape((2,1))
    X = np.vstack((xl.reshape((1, m)), xu.reshape((1, m))))
    return Y, X

def zonotope_vertices(W1, NY=10000):
    """
    Compute the vertices of the zonotope.
    
    Parameters
    ----------
    W1 : ndarray
        `W1` is an ndarray of shape m-by-n that contains the eigenvector bases
        of the n-dimensional active subspace.
    NY : int
        `NY` is the number of samples used to compute the zonotope vertices.
        
    Returns
    -------
    Y : ndarray
        `Y` is an ndarray of shape nzv-by-n that contains the zonotope
        vertices.
    X : ndarray
        `X` is an ndarray of shape nzv-by-m that contains the corners of the 
        m-dimensional hypercube that map to the zonotope vertices.
    """
    if not isinstance(NY, int):
        raise TypeError('NY should be an integer.')
    
    m, n = W1.shape
    
    Xlist = []
    nzv = 0
    for i in range(NY):
        y = np.random.normal(size=(n))
        x = np.sign(np.dot(y, W1.transpose()))
        addx = True
        for xx in Xlist:
            if all(x==xx):
                addx = False
                break
        if addx:
            Xlist.append(x)
            nzv += 1
    X = np.array(Xlist).reshape((nzv, m))
    Y = np.dot(X, W1).reshape((nzv, n))
    return Y, X

def sample_z(N, y, W1, W2):
    """
    Sample values of the inactive variables for a fixed value of the active
    variables when the original variables are bounded by a hypercube.
    
    Parameters
    ----------
    N : int
        `N` is the number of inactive variable samples.
    y : ndarray
        `y` contains the value of the active variables. 
    W1 : ndarray
        `W1` is an ndarray of shape m-by-n that contains the eigenvector bases
        of the n-dimensional active subspace.
    W2 : ndarray
        `W2` is an ndarray of shape m-by-(m-n) that contains the eigenvector 
        bases of the (m-n)-dimensional inactive subspace.
        
    Returns
    -------
    Z : ndarray
        `Z` is an ndarray of shape N-by-(m-n) that contains values of the 
        active variable that correspond to the given `y`.
        
    Notes
    -----
    The trick here is to sample the inactive variables z so that 
    -1 <= W1*y + W2*z <= 1,
    where y is the given value of the active variables. In other words, we need
    to sample z such that it respects the linear equalities 
    W2*z <= 1 - W1*y, -W2*z <= 1 + W1*y.
    These inequalities define a polytope in R^(m-n). We want to sample `N` 
    points uniformly from the polytope. 
    
    This function first tries a simple rejection sampling scheme, which (i) 
    finds a bounding hyperbox for the polytope, (ii) draws points uniformly from
    the bounding hyperbox, and (iii) rejects points outside the polytope. 
    
    If that method does not return enough samples, the method tries a "hit and
    run" method for sampling from the polytope. 
    
    If that doesn't work, it returns an array with `N` copies of a feasible
    point computed with a linear program solver. 
    
    """
    if not isinstance(N, int):
        raise TypeError('N should be an integer.')
    
    Z = rejection_sampling_z(N, y, W1, W2)
    if Z is None:
        warnings.warn('Rejection sampling has failed miserably. Will try hit and run sampling.')
        Z = hit_and_run_z(N, y, W1, W2)
    return Z

def hit_and_run_z(N, y, W1, W2):
    """
    A hit and run method for sampling the inactive variables from a polytope.
    
    See Also
    --------
    domains.sample_z
    
    """
    m, n = W1.shape
    
    # get an initial feasible point
    qps = QPSolver()
    lb = -np.ones((m,1))
    ub = np.ones((m,1))
    c = np.zeros((m,1))
    x0 = qps.linear_program_eq(c, W1.T, y.reshape((n,1)), lb, ub)
    z0 = np.dot(W2.T, x0).reshape((m-n, 1))
    
    # define the polytope A >= b
    s = np.dot(W1, y).reshape((m, 1))
    A = np.vstack((W2, -W2))
    b = np.vstack((-1-s, -1+s)).reshape((2*m, 1))
    
    # tolerance
    ztol = 1e-6
    eps0 = ztol/4.0
    
    Z = np.zeros((N, m-n))
    for i in range(N):
        
        # random direction
        bad_dir = True
        count, maxcount = 0, 50
        while bad_dir:
            d = np.random.normal(size=(m-n,1))
            bad_dir = np.any(np.dot(A, z0 + eps0*d) <= b)
            count += 1
            if count >= maxcount:
                warnings.warn('There are no more directions worth pursuing in hit and run. Got {:d} samples.'.format(i))
                Z[i:,:] = np.tile(z0, (1,N-i)).transpose()
                return Z
                
        # find constraints that impose lower and upper bounds on eps
        f, g = b - np.dot(A,z0), np.dot(A, d)
        
        # find an upper bound on the step
        min_ind = np.logical_and(g<=0, f < -np.sqrt(np.finfo(np.float).eps))
        eps_max = np.amin(f[min_ind]/g[min_ind])
        
        # find a lower bound on the step
        max_ind = np.logical_and(g>0, f < -np.sqrt(np.finfo(np.float).eps))
        eps_min = np.amax(f[max_ind]/g[max_ind])
        
        # randomly sample eps
        eps1 = np.random.uniform(eps_min, eps_max)
        
        # take a step along d
        z1 = z0 + eps1*d
        Z[i,:] = z1.reshape((m-n, ))
        
        # update temp vars
        z0, eps0 = z1, eps1
        
    return Z

def rejection_sampling_z(N, y, W1, W2):
    """
    A rejetion sampling method for sampling the inactive variables from a 
    polytope.
    
    See Also
    --------
    domains.sample_z
    
    """
    m, n = W1.shape
    s = np.dot(W1, y).reshape((m, 1))
    
    # Build a box around z for uniform sampling
    qps = QPSolver()
    A = np.vstack((W2, -W2))
    b = np.vstack((-1-s, -1+s)).reshape((2*m, 1))
    lbox, ubox = np.zeros((1,m-n)), np.zeros((1,m-n))
    for i in range(m-n):
        clb = np.zeros((m-n,1))
        clb[i,0] = 1.0
        lbox[0,i] = qps.linear_program_ineq(clb, A, b)[i,0]
        cub = np.zeros((m-n,1))
        cub[i,0] = -1.0
        ubox[0,i] = qps.linear_program_ineq(cub, A, b)[i,0]
    bn = BoundedNormalizer(lbox, ubox)
    Zbox = bn.unnormalize(np.random.uniform(-1.0,1.0,size=(50*N,m-n)))
    ind = np.all(np.dot(A, Zbox.T) >= b, axis=0)
    
    if np.sum(ind) >= N:
        Z = Zbox[ind,:]
        return Z[:N,:].reshape((N,m-n))
    else:
        return None
    
def random_walk_z(N, y, W1, W2):
    """
    A random walk method for sampling the inactive variables from a polytope.
    
    See Also
    --------
    domains.sample_z
    
    """
    m, n = W1.shape
    s = np.dot(W1, y).reshape((m, 1))

    # linear program to get starting z0
    if np.all(np.zeros((m, 1)) <= 1-s) and np.all(np.zeros((m, 1)) >= -1-s):
        z0 = np.zeros((m-n, 1))
    else:
        qps = QPSolver()
        lb = -np.ones((m,1))
        ub = np.ones((m,1))
        c = np.zeros((m,1))
        x0 = qps.linear_program_eq(c, W1.T, y.reshape((n,1)), lb, ub)
        z0 = np.dot(W2.T, x0).reshape((m-n, 1))
    
    # get MCMC step size
    sig = 0.1*np.minimum(
            np.linalg.norm(np.dot(W2, z0) + s - 1),
            np.linalg.norm(np.dot(W2, z0) + s + 1))
    
    # burn in
    for i in range(10*N):
        zc = z0 + sig*np.random.normal(size=z0.shape)
        if np.all(np.dot(W2, zc) <= 1-s) and np.all(np.dot(W2, zc) >= -1-s):
            z0 = zc

    # sample
    Z = np.zeros((m-n, N))
    for i in range(N):
        zc = z0 + sig*np.random.normal(size=z0.shape)
        if np.all(np.dot(W2, zc) <= 1-s) and np.all(np.dot(W2, zc) >= -1-s):
            z0 = zc
        Z[:,i] = z0.reshape((z0.shape[0], ))

    return Z.reshape((N, m-n))

def _rotate_x(Y, Z, W):
    NY, n = Y.shape
    N = Z.shape[2]
    m = n + Z.shape[1]

    YY = np.tile(Y.reshape((NY, n, 1)), (1, 1, N))
    YZ = np.concatenate((YY, Z), axis=1).transpose((1, 0, 2)).reshape((m, N*NY)).transpose((1, 0))
    X = np.dot(YZ, W.T).reshape((N*NY,m))
    ind = np.kron(np.arange(NY), np.ones(N)).reshape((N*NY,1))
    return X, ind
