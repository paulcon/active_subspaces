import numpy as np
from quadrature import gauss_hermite
from scipy.spatial import ConvexHull, distance_matrix
from scipy.optimize import minimize

def interval_design(a, b, N):
    y = np.linspace(a, b, N+2).reshape((N+2, 1))
    return y[1:-1]
    
def maximin_design(vert, N):
    
    # objective function for maximin design
    def maximin_design_obj(y, vert=None):
        Ny,n = vert.shape
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
    y0 = np.random.normal(size=(N, n))
    res = minimize(maximin_design_obj, y0, args=(vert, ), constraints=cons,
                    method='SLSQP', options={'disp':False, 'maxiter':1e9, 'ftol':1e-12})
    return res.x.reshape(y0.shape)

def unbounded_design(N):
    return gauss_hermite(N)[0]