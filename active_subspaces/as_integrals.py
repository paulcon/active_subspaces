"""Description of as_integrals"""

import numpy as np
import utils.quadrature as gq
from utils.utils import conditional_expectations
from utils.designs import maximin_design
from utils.simrunners import SimulationRunner
from domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain
from scipy.spatial import Delaunay

def integrate(fun, avmap, N, NMC=10):
    Yp, Yw = av_quadrature_rule(avmap, N)
    Xp, Xw, ind = quadrature_rule(Yp, Yw, avmap, NMC=NMC)
    f = SimulationRunner(fun).run(Xp)
    return compute_integral(f, Yw, ind)

def av_integrate(avfun, avmap, N):
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
    Yp, Yw = av_quadrature_rule(avmap, N)
    avf = SimulationRunner(avfun).run(Yp)
    return np.dot(Yw.T, avf)[0,0]

def quadrature_rule(Yp, Yw, avmap, NMC=10):
    
    # get points on x space with MC
    Xp, ind = avmap.inverse(Yp, NMC)
    Xw = np.kron(Yw, np.ones((NMC,1)))/float(NMC)
    return Xp, Xw, ind

def av_quadrature_rule(avmap, N):
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

    return points.reshape((N, 1)), weights.reshape((N, 1))

def zonotope_quadrature_rule(avmap, N, NX=10000):
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
    return points.reshape((T.nsimplex,n)), weights.reshape((T.nsimplex,1))
    
def compute_integral(f, w, ind):
    NMC = np.sum(ind==0)
    Ef, Vf = conditional_expectations(f, ind)
    
    mu = np.dot(Ef.T, w)
    sig2 = np.dot(Vf.T, w*w) / NMC
    lb, ub = mu - 1.96*np.sqrt(sig2), mu + 1.96*np.sqrt(sig2)
    return mu[0,0], lb[0,0], ub[0,0]

