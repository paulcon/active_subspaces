"""Description of as_integrals"""

import numpy as np
import utils.quadrature as gq
from utils.utils import conditional_expectations
from utils.designs import maximin_design
from utils.simrunners import SimulationRunner
from domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain, \
                    UnboundedActiveVariableMap, BoundedActiveVariableMap
from scipy.spatial import Delaunay

def as_integrate(fun, domain, subspace, N, NMC=10):
    """
    Description of as_integrate

    Arguments:
        fun:
        domain:
        subspace:
        N:
        NMC: (default=10)
    Outputs:
        mu:
        lb:
        ub:
    """
    w, x, ind = as_quadrature_rule(domain, subspace, N, NMC)
    f = SimulationRunner(fun).run(x)
    return compute_integral(w, f, ind)

def as_quadrature_rule(domain, subspace, N, NMC=10):
    W1 = subspace.W1    
    m, n = W1.shape

    if isinstance(domain,UnboundedActiveVariableDomain):
        avmap = UnboundedActiveVariableMap(subspace)
        order = []
        for i in range(domain.n):
            order.append(N)
        y, w = gq.gauss_hermite(order)
                
    elif isinstance(domain,BoundedActiveVariableDomain):
        avmap = BoundedActiveVariableMap(subspace)
        if domain.n == 1:
            a, b = domain.vertY[0], domain.vertY[1]
            y, w = interval_quadrature_rule(a, b, W1, N)
        else:
            vert = domain.vertY
            y, w = zonotope_quadrature_rule(vert, W1, N)
    else:
        raise Exception('Shit, yo!')
    x, ind = avmap.inverse(y,NMC)
    return w, x, ind

def interval_quadrature_rule(a, b, W1, N, NX=10000):
    """
    Description of interval_quadrature_rule

    Arguments:
        a:
        b:
        W1:
        N:
        NX: (default=10000)
    Outputs:
        points:
        weights:
    """

    # number of dimensions
    m = W1.shape[0]

    # points
    y = np.linspace(a, b, N+1).reshape((N+1, 1))
    points = 0.5*(y[1:] + y[:-1])

    # weights
    Ysamp = np.dot(np.random.uniform(-1.0, 1.0, size=(NX, m)), W1)
    weights = np.histogram(Ysamp.reshape((NX, )), bins=y.reshape((N+1, )), \
        range=(np.amin(y), np.amax(y)))[0]
    weights = weights / float(NX)

    return points.reshape((N, 1)), weights.reshape((N, 1))

def zonotope_quadrature_rule(vert, W1, N, NX=10000):
    """
    Description of zonotope_quadrature_rule

    Arguments:
        vert:
        W1:
        N:
        NX: (default=10000)
    Outputs:
        points:
        weights:
    """

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
    return points.reshape((T.nsimplex,n)), weights.reshape((T.nsimplex,1))
    
def compute_integral(w, f, ind):
    NMC = np.sum(ind==0)
    Ef, Vf = conditional_expectations(f, ind)
    
    mu = np.dot(Ef.T, w)
    sig2 = np.dot(Vf.T, w*w) / NMC
    lb, ub = mu - 1.96*np.sqrt(sig2), mu + 1.96*np.sqrt(sig2)
    return mu, lb, ub

