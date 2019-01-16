"""Utilities for exploiting active subspaces when estimating integrals."""

import numpy as np
from active_subspaces.utils import quadrature as gq
from active_subspaces.utils.misc import conditional_expectations
from active_subspaces.utils.designs import maximin_design
from active_subspaces.utils.simrunners import SimulationRunner
from active_subspaces.domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain, \
                    ActiveVariableMap
from active_subspaces.response_surfaces import ActiveSubspaceResponseSurface
from scipy.spatial import Delaunay

def integrate(fun, avmap, N, NMC=10):
    """Approximate the integral of a function of m variables.
    
    Parameters
    ----------
    fun : function 
        an interface to the simulation that returns the quantity of interest 
        given inputs as an 1-by-m ndarray
    avmap : ActiveVariableMap 
        a domains.ActiveVariableMap
    N : int 
        the number of points in the quadrature rule
    NMC : int, optional
        the number of points in the Monte Carlo estimates of the conditional 
        expectation and conditional variance (default 10)

    Returns
    -------
    mu : float 
        an estimate of the integral of the function computed against the weight 
        function on the simulation inputs
    lb : float 
        a central-limit-theorem 95% lower confidence from the Monte Carlo part 
        of the integration
    ub : float 
        a central-limit-theorem 95% upper confidence from the Monte Carlo part 
        of the integration

    See Also
    --------
    integrals.quadrature_rule

    Notes
    -----
    The CLT-based bounds `lb` and `ub` are likely poor estimators of the error.
    They only account for the variance from the Monte Carlo portion. They do
    not include any error from the integration rule on the active variables.
    """
    if not isinstance(avmap, ActiveVariableMap):
        raise TypeError('avmap should be an ActiveVariableMap.')

    if not isinstance(N, int):
        raise TypeError('N should be an integer')

    # get the quadrature rule
    Xp, Xw, ind = quadrature_rule(avmap, N, NMC=NMC)

    # compute the simulation output at each quadrature node
    if isinstance(fun, SimulationRunner):
        f = fun.run(Xp)
    else:
        f = SimulationRunner(fun).run(Xp)

    # estimate conditional expectations and variances
    Ef, Vf = conditional_expectations(f, ind)

    # get weights for the conditional expectations
    w = conditional_expectations(Xw*NMC, ind)[0]

    # estimate the average
    mu = np.dot(Ef.T, w)

    # estimate the variance due to Monte Carlo
    sig2 = np.dot(Vf.T, w*w) / NMC

    # compute 95% confidence bounds from the Monte Carlo
    lb, ub = mu - 1.96*np.sqrt(sig2), mu + 1.96*np.sqrt(sig2)
    return mu[0,0], lb[0,0], ub[0,0]

def av_integrate(avfun, avmap, N):
    """Approximate the integral of a function of active variables.

    Parameters
    ----------
    avfun : function 
        a function of the active variables
    avmap : ActiveVariableMap 
        a domains.ActiveVariableMap
    N : int 
        the number of points in the quadrature rule

    Returns
    -------
    mu : float 
        an estimate of the integral

    Notes
    -----
    This function is usually used when one has already constructed a response
    surface on the active variables and wants to estimate its integral.
    """
    if not isinstance(avmap, ActiveVariableMap):
        raise TypeError('avmap should be an ActiveVariableMap.')

    if not isinstance(N, int):
        raise TypeError('N should be an integer.')

    Yp, Yw = av_quadrature_rule(avmap, N)
    if isinstance(avfun, ActiveSubspaceResponseSurface):
        avf = avfun.predict_av(Yp)[0]
    else:
        avf = SimulationRunner(avfun).run(Yp)
    mu = np.dot(Yw.T, avf)[0,0]
    return mu

def quadrature_rule(avmap, N, NMC=10):
    """Get a quadrature rule on the space of simulation inputs.
    
    Parameters
    ----------
    avmap : ActiveVariableMap 
        a domains.ActiveVariableMap
    N : int 
        the number of quadrature nodes in the active variables
    NMC : int, optional 
        the number of samples in the simple Monte Carlo over the inactive 
        variables (default 10)

    Returns
    -------
    Xp : ndarray 
        (N*NMC)-by-m matrix containing the quadrature nodes on the simulation 
        input space
    Xw : ndarray 
        (N*NMC)-by-1 matrix containing the quadrature weights on the simulation 
        input space
    ind : ndarray 
        array of indices identifies which rows of `Xp` correspond to the same 
        fixed value of the active variables

    See Also
    --------
    integrals.av_quadrature_rule

    Notes
    -----
    This quadrature rule uses an integration rule on the active variables and
    simple Monte Carlo on the inactive variables.

    If the simulation inputs are bounded, then the quadrature nodes on the
    active variables is constructed with a Delaunay triangulation of a
    maximin design. The weights are computed by sampling the original variables,
    mapping them to the active variables, and determining which triangle the
    active variables fall in. These samples are used to estimate quadrature
    weights. Note that when the dimension of the active subspace is
    one-dimensional, this reduces to operations on an interval.

    If the simulation inputs are unbounded, the quadrature rule on the active
    variables is given by a tensor product Gauss-Hermite quadrature rule.
    """
    if not isinstance(avmap, ActiveVariableMap):
        raise TypeError('avmap should be an ActiveVariableMap.')

    if not isinstance(N, int):
        raise TypeError('N should be an integer.')

    if not isinstance(NMC, int):
        raise TypeError('NMC should be an integer.')

    # get quadrature rule on active variables
    Yp, Yw = av_quadrature_rule(avmap, N)

    # get points on x space with MC
    Xp, ind = avmap.inverse(Yp, NMC)
    Xw = np.kron(Yw, np.ones((NMC,1)))/float(NMC)
    return Xp, Xw, ind

def av_quadrature_rule(avmap, N):
    """Get a quadrature rule on the space of active variables.

    Parameters
    ----------
    avmap : ActiveVariableMap 
        a domains.ActiveVariableMap
    N : int 
        the number of quadrature nodes in the active variables
        
    Returns
    -------
    Yp : ndarray 
        quadrature nodes on the active variables
    Yw : ndarray
        quadrature weights on the active variables

    See Also
    --------
    integrals.quadrature_rule
    """
    m, n = avmap.domain.subspaces.W1.shape

    if isinstance(avmap.domain, UnboundedActiveVariableDomain):
        NN = [int(np.floor(np.power(N, 1.0/n))) for i in range(n)]
        Yp, Yw = gq.gauss_hermite(NN)

    elif isinstance(avmap.domain, BoundedActiveVariableDomain):
        if n == 1:
            Yp, Yw = interval_quadrature_rule(avmap, N)
        else:
            Yp, Yw = zonotope_quadrature_rule(avmap, N)
    else:
        raise Exception('There is a problem with the avmap.domain.')
    return Yp, Yw

def interval_quadrature_rule(avmap, N, NX=10000):
    """Quadrature rule on a one-dimensional interval.
    
    Quadrature when the dimension of the active subspace is 1 and the
    simulation parameter space is bounded.

    Parameters
    ----------
    avmap : ActiveVariableMap 
        a domains.ActiveVariableMap
    N : int 
        the number of quadrature nodes in the active variables
    NX : int, optional 
        the number of samples to use to estimate the quadrature weights (default
        10000)

    Returns
    -------
    Yp : ndarray 
        quadrature nodes on the active variables
    Yw : ndarray 
        quadrature weights on the active variables

    See Also
    --------
    integrals.quadrature_rule
    """
    W1 = avmap.domain.subspaces.W1
    a, b = avmap.domain.vertY[0,0], avmap.domain.vertY[1,0]

    # number of dimensions
    m = W1.shape[0]

    # points
    y = np.linspace(a, b, N+1).reshape((N+1, 1))
    points = 0.5*(y[1:] + y[:-1])

    # weights
    Y_samples = np.dot(np.random.uniform(-1.0, 1.0, size=(NX, m)), W1)
    weights = np.histogram(Y_samples.reshape((NX, )), bins=y.reshape((N+1, )), \
        range=(np.amin(y), np.amax(y)))[0]
    weights = weights / float(NX)

    Yp, Yw = points.reshape((N, 1)), weights.reshape((N, 1))
    return Yp, Yw

def zonotope_quadrature_rule(avmap, N, NX=10000):
    """Quadrature rule on a zonotope.
    
    Quadrature when the dimension of the active subspace is greater than 1 and
    the simulation parameter space is bounded.

    Parameters
    ----------
    avmap : ActiveVariableMap 
        a domains.ActiveVariableMap
    N : int 
        the number of quadrature nodes in the active variables
    NX : int, optional 
        the number of samples to use to estimate the quadrature weights (default
        10000)

    Returns
    -------
    Yp : ndarray
        quadrature nodes on the active variables
    Yw : ndarray 
        quadrature weights on the active variables

    See Also
    --------
    integrals.quadrature_rule
    """

    vert = avmap.domain.vertY
    W1 = avmap.domain.subspaces.W1

    # number of dimensions
    m, n = W1.shape

    # points
    y = np.vstack((vert, maximin_design(vert, N)))
    T = Delaunay(y)
    c = []
    for t in T.simplices:
        c.append(np.mean(T.points[t], axis=0))
    points = np.array(c)

    # approximate weights
    Y_samples = np.dot(np.random.uniform(-1.0, 1.0, size=(NX,m)), W1)
    I = T.find_simplex(Y_samples)
    weights = np.zeros((T.nsimplex, 1))
    for i in range(T.nsimplex):
        weights[i] = np.sum(I==i) / float(NX)

    Yp, Yw = points.reshape((T.nsimplex,n)), weights.reshape((T.nsimplex,1))
    return Yp, Yw


