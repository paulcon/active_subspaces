import numpy as np
import designs as dn
from domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain, \
                UnboundedActiveVariableMap, BoundedActiveVariableMap
from simrunners import SimulationRunner
from utils import conditional_expectations

class ActiveSubspaceResponseSurface():
    def __init__(self,rs,subspace):
        self.rs = rs
        self.subspace = subspace
    
    def train(self, X, f, Vf=None):
        W1 = self.subspace.W1
        Y = np.dot(X, W1)
        self.rs.train(Y, f)

    def predict(self, X, compgrad=False, compvar=False):
        W1 = self.subspace.W1
        Y = np.dot(X, W1)
        return self.rs.predict(Y, compgrad, compvar)
        
    def gradient(self, X):
        return self.predict(X, compgrad=True)[1]
        
    def __call__(self, X):
        return self.predict(X)[0]
        
def compute_training_data(fun, domain, subspace, N, NMC=10):
    x, ind = as_design(domain, subspace, N, NMC)
    f = SimulationRunner(fun).run(x)
    return get_training_data(x, f, ind)
    
def as_design(domain, subspace, N, NMC=10):
    W1 = subspace.W1    
    m, n = W1.shape
    
    if isinstance(domain, UnboundedActiveVariableDomain):
        avmap = UnboundedActiveVariableMap(subspace)
        y = dn.unbounded_design(N)
        x, ind = avmap.inverse(y, NMC)
        
    elif isinstance(domain, BoundedActiveVariableDomain):
        avmap = BoundedActiveVariableMap(subspace)
        if n==1:
            a, b = domain.vertY[0], domain.vertY[1]
            y = dn.interval_design(a, b, N)
        else:
            vertices = domain.vertY
            y = dn.maximin_design(vertices, N)
        x, ind = avmap.inverse(y, NMC)
        x = np.vstack((x, domain.vertX))
    else:
        raise Exception('Shit, yo')
    
    return x, ind
    
def get_training_data(x, f, ind):
    
    k = len(ind)
    Ef, Vf = conditional_expectations(f[:k], ind)
    
    # get subset of x's
    NMC = np.sum(ind==0)
    indx = np.arange(0, k, NMC)
    X = x[indx,:]
    
    if len(f) > k:
        # vertex values
        X = np.vstack((X, x[k:,:]))
        Ef = np.vstack((Ef, f[k:]))
        Vf = np.vstack((Vf, np.zeros((len(f)-k, 1))))

    return X, Ef, Vf
        
    