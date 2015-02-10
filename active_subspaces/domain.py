import numpy as np
import gaussquad as gq
from scipy.spatial import Delaunay,ConvexHull,distance_matrix
from scipy.optimize import minimize

class ActiveVariableDomain():
    def __init__(self, W1):
        
        # TODO: error checking on W1
        self.W1 = W1
        self.n = W1.shape[1]

    def design(self, N):
        raise NotImplementedError()

    def integration_rule(self, N):
        raise NotImplementedError()

class UnboundedActiveVariableDomain(ActiveVariableDomain):
    def design(self, N):
        return gq.gauss_hermite(N)[0]

    def integration_rule(self, N):
        return gq.gauss_hermite(N)

class BoundedActiveVariableDomain(ActiveVariableDomain):
    
    def __init__(self, W1):
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
            constraints = ({'type':'ineq',
                        'fun' : lambda x: np.dot(A, x) - b,
                        'jac' : lambda x: A})

        # store variables
        self.W1 = W1
        self.m, self.n = m, n
        self.vertY, self.vertX = Y, X
        self.convhull, self.constraints = convhull, constraints

    def design(self, N):
        n = self.n
        if n == 1:
            a, b = self.vertY[0], self.vertY[1]
            points = interval_design(a, b, N)
        else:
            points = maximin_design(self.vertY, N)
        return points

    def integration_rule(self, N):
        n = self.n
        W1 = self.W1

        if n == 1:
            a, b = self.vertY[0], self.vertY[1]
            points, weights = interval_quadrature_rule(a, b, W1, N)
        else:
            vert = self.vertY
            points, weights = zonotope_quadrature_rule(vert, W1, N)
        return points, weights

def nzv(m, n, M=None):
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
    m = W1.shape[0]
    y0 = np.dot(W1.T, np.sign(W1))[0]
    if y0 < -y0:
        yl, yu = y0, -y0
        xl, xu = np.sign(W1), -np.sign(W1)
    else:
        yl, yu = -y0, y0
        xl, xu = -np.sign(W1), np.sign(W1)
    Y = np.array([[yl], [yu]])
    X = np.vstack((xl.reshape((1, m)), xu.reshape((1, m))))
    return Y, X

def zonotope_vertices(W1, NY=10000):
    m, n = W1.shape
    
    Xlist = []
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
    X = np.array(Xlist)
    Y = np.dot(X, W1)
    return Y, X

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

def interval_quadrature_rule(a, b, W1, N, NX=10000):
    
    # number of dimensions
    m = W1.shape[0]
    
    # points
    y = np.linspace(a, b, N+1).reshape((N+1, 1))
    points = 0.5*(y[1:] + y[:-1])

    # weights
    Ysamp = np.dot(np.random.uniform(-1.0, 1.0, size=(NX, m)), W1)
    weights = np.histogram(Ysamp.reshape((NX, )), bins=y.reshape((N+1, )), \
        range=(np.amin(y),np.amax(y)))[0]
    weights = weights / float(NX)
    
    return points.reshape((N, 1)), weights.reshape((N, 1))
    
def zonotope_quadrature_rule(vert, W1, N, NX=10000):
    
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
    Ysamp = np.dot(np.random.uniform(-1.0, 1.0, size=(NX,m)), W1)
    I = T.find_simplex(Ysamp)
    weights = np.zeros((T.nsimplex, 1))
    for i in range(T.nsimplex):
        weights[i] = np.sum(I==i) / float(NX)

    return points.reshape((N, n)), weights.reshape((N, 1))
