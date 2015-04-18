import numpy as np
import utils as ut
from quadrature import gauss_hermite
from scipy.spatial import ConvexHull, distance_matrix
from scipy.optimize import minimize

def interval_design(a, b, N):
    y = np.linspace(a, b, N+2)
    return ut.atleast_2d_col(y[1:-1])
    
def maximin_design(vert, N):
    
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
    
    # some tricks for the global optimization
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
    return minres.x.reshape((N, n))

def gauss_hermite_design(N):
    return gauss_hermite(N)[0]