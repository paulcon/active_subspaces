"""Utilities for constructing design-of-experiments."""
import numpy as np
from . import misc as mi
from .quadrature import gauss_hermite
from scipy.spatial import ConvexHull, distance_matrix
from scipy.optimize import minimize

def interval_design(a, b, N):
    """Equally spaced points on an interval.

    Parameters
    ----------
    a : float
        the left endpoint of the interval
    b : float
        the right endpoint of the interval
    N : int
        the number of points in the design

    Returns
    -------
    design, ndarray
        N-by-1 matrix that contains the design points in the interval. It does 
        not contain the endpoints.
    """
    y = np.linspace(a, b, N+2)
    design = mi.atleast_2d_col(y[1:-1])
    return design

def maximin_design(vert, N):
    """Multivariate maximin design constrained by a polytope.

    Parameters
    ----------
    vert : ndarray
        the vertices that define the m-dimensional polytope. The shape of `vert`
        is M-by-m, where M is the number of vertices.
    N : int 
        the number of points in the design

    Returns
    -------
    design : ndarray
        N-by-m matrix that contains the design points in the polytope. It does 
        not contain the vertices.
        
    Notes
    -----
    The objective function used to find the design is the negative of the
    minimum distance between points in the design and the given vertices. The
    routine uses the scipy.minimize function with the SLSQP method to minimize
    the function. The constraints are given by the polytope defined by the
    vertices. The scipy.spatial packages turns the vertices into a set of
    linear inequality constraints.

    The optimization is nonlinear and nonconvex with many local minima. Any
    reasonable local minima is likely to give a good design. However, to
    increase robustness, we use three random starting points in the
    minimization and use the design with the lowest objective value.
    """
    n = vert.shape[1]
    C = ConvexHull(vert)
    A = np.kron(np.eye(N), C.equations[:,:n])
    b = np.kron(np.ones(N), C.equations[:,n])
    cons = ({'type':'ineq',
                'fun' : lambda x: np.dot(A, x) - b,
                'jac' : lambda x: A})

    # some tricks for the globalization
    curr_state = np.random.get_state()

    np.random.seed(42)
    minf = 1e10
    minres = []

    for i in range(3):
        y0 = np.random.normal(size=(N, n))
        res = minimize(_maximin_design_obj, y0, args=(vert, ), jac=_maximin_design_grad, constraints=cons,
                        method='SLSQP', options={'disp':False, 'maxiter':1e2, 'ftol':1e-4})
        if not res.success:
            raise Exception('SLSQP failed with message: {}.'.format(res.message))
        if res.fun < minf:
            minf = res.fun
            minres = res

    np.random.set_state(curr_state)
    design = minres.x.reshape((N, n))
    return design

def gauss_hermite_design(N):
    """Tensor product Gauss-Hermite quadrature points.

    Parameters
    ----------
    N : int[]
        contains the number of points per dimension in the tensor product design

    Returns
    -------
    design : ndarray
        N-by-m matrix that contains the design points
    """
    design = gauss_hermite(N)[0]
    return design

def _maximin_design_obj(y, vert=None):
    """Objective function for the maximin design optimization.

    Parameters
    ----------
    y : ndarray
        contains the coordinates of the points in the design. If there are N 
        points in n dimensions then `y` is shape ((Nn, )).
    vert : ndarray, optional
        contains the fixed vertices defining the zonotope

    Notes
    -----
    This function returns the minimum squared distance between all points in
    the design and between points and vertices.
    """
    Ny, n = vert.shape
    N = y.size / n
    Y = y.reshape((N, n))

    # get minimum distance among points
    D0 = distance_matrix(Y, Y) + 1e5*np.eye(N)
    d0 = np.power(D0.flatten(), 2)
    d0star = np.amin(d0)

    # get minimum distance between points and vertices
    D1 = distance_matrix(Y, vert)
    d1 = np.power(D1.flatten(), 2)
    d1star = np.amin(d1)
    dstar = np.amin([d0star, d1star])
    return -dstar

def _maximin_design_grad(y, vert=None):
    """Gradient of objective function for the maximin design optimization.

    Parameters
    ----------
    y : ndarray
        contains the coordinates of the points in the design. If there are N 
        points in n dimensions then `y` is shape ((Nn, )).
    vert : ndarray
        contains the fixed vertices defining the zonotope
    """
    Ny, n = vert.shape
    v = vert.reshape((Ny*n, ))

    N = y.size / n
    Y = y.reshape((N, n))

    # get minimum distance among points
    D0 = distance_matrix(Y, Y) + 1e5*np.eye(N)
    d0 = np.power(D0.flatten(), 2)
    d0star, k0star = np.amin(d0), np.argmin(d0)

    # get minimum distance between points and vertices
    D1 = distance_matrix(Y, vert)
    d1 = np.power(D1.flatten(), 2)
    d1star, k1star = np.amin(d1), np.argmin(d1)

    g = np.zeros((N*n, ))
    if d0star < d1star:
        dstar, kstar = d0star, k0star
        istar = kstar/N
        jstar = np.mod(kstar, N)

        for k in range(n):
            g[istar*n + k] = 2*(y[istar*n + k] - y[jstar*n + k])
            g[jstar*n + k] = 2*(y[jstar*n + k] - y[istar*n + k])

    else:
        dstar, kstar = d1star, k1star
        istar = kstar/Ny
        jstar = np.mod(kstar, Ny)

        for k in range(n):
            g[istar*n + k] = 2*(y[istar*n + k] - v[jstar*n + k])

    return -g
