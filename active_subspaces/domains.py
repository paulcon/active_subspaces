"""Utilities for building the domains and maps for active variables."""
import numpy as np
from utils.misc import process_inputs, BoundedNormalizer
from scipy.spatial import ConvexHull
from scipy.misc import comb
from utils.qp_solver import QPSolver
from subspaces import Subspaces

class ActiveVariableDomain():
    """A base class for the domain of functions of active variables.
    
    Attributes
    ----------
    subspaces : Subspaces
        subspaces that define the domain
    m : int 
        the dimension of the simulation inputs
    n : int 
        the dimension of the active subspace
    vertY : ndarray 
        n-dimensional vertices that define the boundary of the domain when the 
        m-dimensional space is a hypercube
    vertX : ndarray 
        corners of the m-dimensional hypercube that map to the points `vertY`
    convhull : scipy.spatial.ConvexHull
        the ConvexHull object defined by the vertices `vertY`
    constraints : dict
        a dictionary of linear inequality constraints conforming to the 
        specifications used in the scipy.optimizer library

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
    """Domain of functions with unbounded domains (Gaussian weight).
    
    An class for the domain of functions of active variables when the space
    of simulation parameters is unbounded.

    Notes
    -----
    Using this class assumes that the space of simulation inputs is equipped
    with a Gaussian weight function.
    """
    def __init__(self, subspaces):
        """Initialize the UnboundedActiveVariableDomain.

        Parameters
        ----------
        subspaces : Subspaces 
            a Subspaces object with the `compute` method already called
        """
        if not isinstance(subspaces, Subspaces):
            raise TypeError('subspaces should be a Subspaces object.')

        if subspaces.W1 is None:
            raise ValueError('The given subspaces has not been computed.')

        self.subspaces = subspaces
        self.m, self.n = subspaces.W1.shape

class BoundedActiveVariableDomain(ActiveVariableDomain):
    """Domain of functions with bounded domains (uniform on hypercube).
    
    An class for the domain of functions of active variables when the space
    of simulation parameters is bounded.

    Notes
    -----
    Using this class assumes that the space of simulation inputs is equipped
    with a uniform weight function. And the space itself is a hypercube.
    """

    def __init__(self, subspaces):
        """Initialize the BoundedActiveVariableDomain
        
        Parameters
        ----------
        subspaces : Subspaces 
            a Subspaces object with the `compute` method already called
        """
        if not isinstance(subspaces, Subspaces):
            raise TypeError('subspaces should be a Subspaces object.')

        if subspaces.W1 is None:
            raise ValueError('The given subspaces has not been computed.')

        self.subspaces = subspaces
        self.m, self.n = subspaces.W1.shape
        self.compute_boundary()

    def compute_boundary(self):
        """Compute and set the boundary of the domain.

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
    """A base class for the map between active/inactive and original variables.

    Attributes
    ----------
    domain : ActiveVariableDomain
        an ActiveVariableDomain object

    See Also
    --------
    domains.UnboundedActiveVariableMap
    domains.BoundedActiveVariableMap
    """
    domain = None

    def __init__(self, domain):
        """Initialize the ActiveVariableMap.
        
        Parameters
        ----------
        domain : ActiveVariableDomain
            an ActiveVariableDomain object
        """
        self.domain = domain

    def forward(self, X):
        """Map full variables to active variables.
        
        Map the points in the original input space to the active and inactive
        variables.

        Parameters
        ----------
        X : ndarray
            an M-by-m matrix, each row of `X` is a point in the original 
            parameter space

        Returns
        -------
        Y : ndarray 
            M-by-n matrix that contains points in the space of active variables.
            Each row of `Y` corresponds to a row of `X`.
        Z : ndarray 
            M-by-(m-n) matrix that contains points in the space of inactive 
            variables. Each row of `Z` corresponds to a row of `X`.

        """
        X = process_inputs(X)[0]
        W1, W2 = self.domain.subspaces.W1, self.domain.subspaces.W2
        Y, Z = np.dot(X, W1), np.dot(X, W2)
        return Y, Z

    def inverse(self, Y, N=1):
        """Find points in full space that map to active variable points.
        
        Map the points in the active variable space to the original parameter
        space.
        
        Parameters
        ----------
        Y : ndarray
            M-by-n matrix that contains points in the space of active variables
        N : int, optional
            the number of points in the original parameter space that are 
            returned that map to the given active variables (default 1)

        Returns
        -------
        X : ndarray
            (M*N)-by-m matrix that contains points in the original parameter 
            space
        ind : ndarray
            (M*N)-by-1 matrix that contains integer indices. These indices 
            identify which rows of `X` map to which rows of `Y`.

        Notes
        -----
        The inverse map depends critically on the `regularize_z` function.
        """
        # check inputs
        Y, NY, n = process_inputs(Y)

        if not isinstance(N, int):
            raise TypeError('N must be an int')

        Z = self.regularize_z(Y, N)
        W = self.domain.subspaces.eigenvecs
        X, ind = _rotate_x(Y, Z, W)
        return X, ind

    def regularize_z(self, Y, N):
        """Pick inactive variables associated active variables.
        
        Find points in the space of inactive variables to complete the inverse
        map.
        
        Parameters
        ----------
        Y : ndarray
            M-by-n matrix that contains points in the space of active variables
        N : int
            The number of points in the original parameter space that are 
            returned that map to the given active variables

        Returns
        -------
        Z : ndarray 
            (M)-by-(m-n)-by-N matrix that contains values of the inactive 
            variables
            
        Notes
        -----
        The base class does not implement `regularize_z`. Specific
        implementations depend on whether the original variables are bounded or
        unbounded. They also depend on what the weight function is on the
        original parameter space.
        """
        raise NotImplementedError()

class BoundedActiveVariableMap(ActiveVariableMap):
    """Class for mapping between active and bounded full variables.
    
    A class for the map between active/inactive and original variables when the
    original variables are bounded by a hypercube with a uniform density.

    See Also
    --------
    domains.UnboundedActiveVariableMap
    """
    def regularize_z(self, Y, N):
        """Pick inactive variables associated active variables.
        
        Find points in the space of inactive variables to complete the inverse
        map.
        
        Parameters
        ----------
        Y : ndarray
            M-by-n matrix that contains points in the space of active variables
        N : int
            The number of points in the original parameter space that are 
            returned that map to the given active variables

        Returns
        -------
        Z : ndarray 
            (M)-by-(m-n)-by-N matrix that contains values of the inactive 
            variables
            
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
        
        Zlist = []
        for y in Y:
            Zlist.append(sample_z(N, y, W1, W2))
            
        Z = np.swapaxes(np.array(Zlist),1,2)
        return Z

class UnboundedActiveVariableMap(ActiveVariableMap):
    """Class for mapping between active and unbounded full variables.
    
    A class for the map between active/inactive and original variables when the
    original variables are ubbounded and the space is equipped with a standard
    Gaussian density.

    See Also
    --------
    domains.BoundedActiveVariableMap
    """
    def regularize_z(self, Y, N):
        """Pick inactive variables associated active variables.
        
        Find points in the space of inactive variables to complete the inverse
        map.
        
        Parameters
        ----------
        Y : ndarray
            M-by-n matrix that contains points in the space of active variables
        N : int
            The number of points in the original parameter space that are 
            returned that map to the given active variables

        Returns
        -------
        Z : ndarray 
            (M)-by-(m-n)-by-N matrix that contains values of the inactive 
            variables
            
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

def nzv(m, n):
    """Number of zonotope vertices.
    
    Compute the number of zonotope vertices for a linear map from R^m to R^n.
    
    Parameters
    ----------
    m : int
        the dimension of the hypercube
    n : int
        the dimension of the low-dimesional subspace
        
    Returns
    -------
    N : int 
        the number of vertices defining the zonotope
    """
    if not isinstance(m, int):
        raise TypeError('m should be an integer.')

    if not isinstance(n, int):
        raise TypeError('n should be an integer.')

    # number of zonotope vertices
    N = 0
    for i in range(n):
        N = N + comb(m-1,i)
    N = 2*N
    return int(N)

def interval_endpoints(W1):
    """Compute the range of a 1d active variable.
    
    Parameters
    ----------
    W1 : ndarray
        m-by-1 matrix that contains the eigenvector that defines the first 
        active variable

    Returns
    -------
    Y : ndarray
        2-by-1 matrix that contains the endpoints of the interval defining the 
        range of the 1d active variable
    X : ndarray
        2-by-m matrix that contains the corners of the m-dimensional hypercube 
        that map to the active variable endpoints
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

def unique_rows(S):
    """Return the unique rows from ndarray
    
    Parameters
    ----------
    S : ndarray
        array with rows to reduces
        
    Returns
    -------
    T : ndarray
        version of `S` with unique rows
        
    Notes
    -----
    http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    """
    T = S.view(np.dtype((np.void, S.dtype.itemsize * S.shape[1])))
    return np.unique(T).view(S.dtype).reshape(-1, S.shape[1])

def zonotope_vertices(W1, Nsamples=10000, maxcount=100000):
    """Compute the vertices of the zonotope.

    Parameters
    ----------
    W1 : ndarray 
        m-by-n matrix that contains the eigenvector bases of the n-dimensional 
        active subspace
    Nsamples : int, optional
        number of samples per iteration to check (default 1e4)
    maxcount : int, optional
        maximum number of iterations (default 1e5)

    Returns
    -------
    Y : ndarray 
        nzv-by-n matrix that contains the zonotope vertices
    X : ndarray 
        nzv-by-m matrix that contains the corners of the m-dimensional hypercube
        that map to the zonotope vertices
    """

    m, n = W1.shape
    totalverts = nzv(m,n)
    
    # make sure they're ints
    Nsamples = int(Nsamples)
    maxcount = int(maxcount)

    # initialize
    Z = np.random.normal(size=(Nsamples, n))
    X = unique_rows(np.sign(np.dot(Z, W1.transpose())))
    X = unique_rows(np.vstack((X, -X)))
    N = X.shape[0]
    
    count = 0
    while N < totalverts:
        Z = np.random.normal(size=(Nsamples, n))
        X0 = unique_rows(np.sign(np.dot(Z, W1.transpose())))
        X0 = unique_rows(np.vstack((X0, -X0)))
        X = unique_rows(np.vstack((X, X0)))
        N = X.shape[0]
        count += 1
        if count > maxcount:
            break
    
    numverts = X.shape[0]
    if totalverts > numverts:
        print 'Warning: {} of {} vertices found.'.format(numverts, totalverts)
    
    Y = np.dot(X, W1)
    return Y.reshape((numverts, n)), X.reshape((numverts, m))
    

def sample_z(N, y, W1, W2):
    """Sample inactive variables.
    
    Sample values of the inactive variables for a fixed value of the active
    variables when the original variables are bounded by a hypercube.

    Parameters
    ----------
    N : int 
        the number of inactive variable samples
    y : ndarray 
        the value of the active variables
    W1 : ndarray 
        m-by-n matrix that contains the eigenvector bases of the n-dimensional 
        active subspace
    W2 : ndarray 
        m-by-(m-n) matrix that contains the eigenvector bases of the 
        (m-n)-dimensional inactive subspace

    Returns
    -------
    Z : ndarray
        N-by-(m-n) matrix that contains values of the inactive variable that 
        correspond to the given `y`

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
    point computed as the Chebyshev center of the polytope. Thanks to David
    Gleich for showing me Chebyshev centers.

    """
    if not isinstance(N, int):
        raise TypeError('N should be an integer.')

    Z = rejection_sampling_z(N, y, W1, W2)
    if Z is None:
        Z = hit_and_run_z(N, y, W1, W2)
    return Z

def hit_and_run_z(N, y, W1, W2):
    """A hit and run method for sampling the inactive variables from a polytope.

    Parameters
    ----------
    N : int 
        the number of inactive variable samples
    y : ndarray 
        the value of the active variables
    W1 : ndarray 
        m-by-n matrix that contains the eigenvector bases of the n-dimensional 
        active subspace
    W2 : ndarray 
        m-by-(m-n) matrix that contains the eigenvector bases of the 
        (m-n)-dimensional inactive subspace

    Returns
    -------
    Z : ndarray
        N-by-(m-n) matrix that contains values of the inactive variable that 
        correspond to the given `y`    
    
    See Also
    --------
    domains.sample_z
    
    Notes
    -----
    The interface for this implementation is written specifically for 
    `domains.sample_z`.
    """
    m, n = W1.shape

    # get an initial feasible point using the Chebyshev center. huge props to
    # David Gleich for showing me the Chebyshev center.
    s = np.dot(W1, y).reshape((m, 1))
    normW2 = np.sqrt( np.sum( np.power(W2, 2), axis=1 ) ).reshape((m,1))
    A = np.hstack(( np.vstack((W2, -W2.copy())), np.vstack((normW2, normW2.copy())) ))
    b = np.vstack((1-s, 1+s)).reshape((2*m, 1))
    c = np.zeros((m-n+1,1))
    c[-1] = -1.0

    qps = QPSolver()
    zc = qps.linear_program_ineq(c, -A, -b)
    z0 = zc[:-1].reshape((m-n, 1))

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

        # update temp var
        z0 = z1.copy()

    return Z

def rejection_sampling_z(N, y, W1, W2):
    """A rejection sampling method for sampling the from a polytope.

    Parameters
    ----------
    N : int 
        the number of inactive variable samples
    y : ndarray 
        the value of the active variables
    W1 : ndarray 
        m-by-n matrix that contains the eigenvector bases of the n-dimensional 
        active subspace
    W2 : ndarray 
        m-by-(m-n) matrix that contains the eigenvector bases of the 
        (m-n)-dimensional inactive subspace

    Returns
    -------
    Z : ndarray
        N-by-(m-n) matrix that contains values of the inactive variable that 
        correspond to the given `y`    
    
    See Also
    --------
    domains.sample_z
    
    Notes
    -----
    The interface for this implementation is written specifically for 
    `domains.sample_z`.
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
    """A random walk method for sampling from a polytope.

    Parameters
    ----------
    N : int 
        the number of inactive variable samples
    y : ndarray 
        the value of the active variables
    W1 : ndarray 
        m-by-n matrix that contains the eigenvector bases of the n-dimensional 
        active subspace
    W2 : ndarray 
        m-by-(m-n) matrix that contains the eigenvector bases of the 
        (m-n)-dimensional inactive subspace

    Returns
    -------
    Z : ndarray
        N-by-(m-n) matrix that contains values of the inactive variable that 
        correspond to the given `y`    
    
    See Also
    --------
    domains.sample_z
    
    Notes
    -----
    The interface for this implementation is written specifically for 
    `domains.sample_z`.
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
    """A convenience function for rotating subspace coordinates to x space.
    
    """
    NY, n = Y.shape
    N = Z.shape[2]
    m = n + Z.shape[1]
    
    YY = np.tile(Y.reshape((NY, n, 1)), (1, 1, N))
    YZ = np.concatenate((YY, Z), axis=1).transpose((1, 0, 2)).reshape((m, N*NY)).transpose((1, 0))
    X = np.dot(YZ, W.T).reshape((N*NY,m))
    ind = np.kron(np.arange(NY), np.ones(N)).reshape((N*NY,1))
    return X, ind
