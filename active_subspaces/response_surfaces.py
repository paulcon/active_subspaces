import numpy as np
import utils.designs as dn
from utils.simrunners import SimulationRunner
from utils.utils import conditional_expectations
from utils.response_surfaces import RadialBasisApproximation
from domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain

class ActiveSubspaceResponseSurface():
    def __init__(self, avmap, respsurf=None):
        if respsurf == None:
            self.respsurf = RadialBasisApproximation()
        else:
            self.respsurf = respsurf
        self.avmap = avmap
    
    def train(self, Y, f, v=None):
        if isinstance(self.respsurf, RadialBasisApproximation):
            evals = self.avmap.domain.subspaces.eigenvalues
            self.respsurf.train(Y, f, v=v, e=evals)
        else:
            self.respsurf.train(Y, f)
    
    def train_with_data(self, X, f, v=None):
        Y = self.avmap.forward(X)[0]
        self.train(Y, f, v=v)
        
    def train_with_interface(self, fun, N, NMC=10):
        Y, X, ind = as_design(self.avmap, N, NMC=NMC)
        
        if isinstance(self.avmap.domain, BoundedActiveVariableDomain):
            X = np.vstack((X, self.avmap.domain.vertX))
            Y = np.vstack((Y, self.avmap.domain.vertY))
            il = np.amax(ind) + 1
            iu = np.amax(ind) + self.avmap.domain.vertX.shape[0] + 1
            iind = np.arange(il, iu)
            ind = np.vstack(( ind, iind.reshape((iind.size,1)) ))
            
        # run simulation interface at all design points
        if isinstance(fun, SimulationRunner):
            f = fun.run(X)
        else:
            f = SimulationRunner(fun).run(X)
        
        Ef, Vf = conditional_expectations(f, ind)
        self.train(Y, Ef, v=Vf)
        
    def predict_av(self, Y, compgrad=False):
        return self.respsurf.predict(Y, compgrad)
        
    def gradient_av(self, Y):
        return self.respsurf.predict(Y, compgrad=True)[1]
        
    def predict(self, X, compgrad=False):
        Y = self.avmap.forward(X)[0]
        f, dfdy = self.predict_av(Y, compgrad)
        if compgrad:
            W1 = self.avmap.domain.subspaces.W1
            dfdx = np.dot(dfdy, W1.T)
        else:
            dfdx = None
        return f, dfdx
        
    def gradient(self, X):
        return self.predict(X, compgrad=True)[1]
        
    def __call__(self, X):
        return self.predict(X)[0]
        
def as_design(avmap, N, NMC=10):
    # interpret N as total number of points in the design
    if not isinstance(N, int):
        raise Exception('N should be an integer.')
    
    m, n = avmap.domain.subspaces.W1.shape
    
    if isinstance(avmap.domain, UnboundedActiveVariableDomain):
        NN = [int(np.floor(np.power(N, 1.0/n))) for i in range(n)]
        Y = dn.gauss_hermite_design(NN)
        
    elif isinstance(avmap.domain, BoundedActiveVariableDomain):
        
        if n==1:
            a, b = avmap.domain.vertY[0,0], avmap.domain.vertY[1,0]
            Y = dn.interval_design(a, b, N)
        else:
            vertices = avmap.domain.vertY
            Y = dn.maximin_design(vertices, N)
    else:
        raise Exception('There is a problem with the avmap.domain.')
    
    X, ind = avmap.inverse(Y, NMC)
    return Y, X, ind
    