"""Utilities for constructing design-of-experiments."""
import numpy as np
import utils as ut
from quadrature import gauss_hermite
from scipy.spatial import ConvexHull, distance_matrix
from scipy.optimize import minimize

def interval_design(a, b, N):
    """
    Equally spaced points on an interval.
    
    Parameters
    ----------
    a : float
        `a` is the left endpoint of the interval.
    b : float
        `b` is the right endpoint of the interval.
    N : int
        `N` is the number of points in the design.
        
    Returns
    -------
    design : ndarray
        `design` is an ndarray of shape N-by-1 that contains the design points 
        in the interval. It does not contain the endpoints. 
        
    """
    y = np.linspace(a, b, N+2)
    design = ut.atleast_2d_col(y[1:-1])
    return design
    
def maximin_design(vert, N):
    """
    Multivariate maximin design constrained by a polytope.
    
    Parameters
    ----------
    vert : ndarray
        `vert` contains the vertices that define the m-dimensional polytope. 
        The shape of `vert` is M-by-m, where M is the number of vertices.
    N : int
        `N` is the number of points in the design.
        
    Returns
    -------
    design : ndarray
        `design` is an ndarray of shape N-by-m that contains the design points 
        in the polytope. It does not contain the vertices.
        
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
    # objective function for maximin design
    def maximin_design_obj(y, vert=None):
        Ny, n = vert.shape
        N = y.size / n
        Y = y.reshape((N, n))
        D0 = distance_matrix(Y, Y) + 1e4*np.eye(N)
        D1 = distance_matrix(Y, vert)
        return -np.amin(np.hstack((D0.flatten(), D1.flatten())))
    
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
        res = minimize(maximin_design_obj, y0, args=(vert, ), constraints=cons,
                        method='SLSQP', options={'disp':False, 'maxiter':1e9, 'ftol':1e-12})
        if res.fun < minf:
            minf = res.fun
            minres = res
    
    np.random.set_state(curr_state)
    design = minres.x.reshape((N, n))
    return design

def gauss_hermite_design(N):
    """
    Tensor product Gauss-Hermite quadrature points. 
    
    Parameters
    ----------
    N : list of int
        `N` is a list that contains the number of points per dimension in the 
        tensor product design.
        
    Returns
    -------
    design : ndarray
        `design` is an ndarray of shape N-by-m that contains the design points.
        
    """
    design = gauss_hermite(N)[0]
    return design