import numpy as np
from utils import process_inputs

class Normalizer():
    def normalize(self,X):
        raise NotImplementedError()
        
    def unnormalize(self,X):
        raise NotImplementedError()
        
class BoundedNormalizer(Normalizer):
    def __init__(self,lb,ub):
        m = lb.size
        self.lb = lb.reshape((1,m))
        self.ub = ub.reshape((1,m))
        
    def normalize(self,X):
        lb,ub = self.lb,self.ub
        return 2.0*(X - lb)/(ub-lb)-1.0
        
    def unnormalize(self,X):
        lb,ub = self.lb,self.ub
        return (ub-lb)*(X + 1.0)/2.0 + lb
        
class UnboundedNormalizer(Normalizer):
    def __init__(self,mu,C):
        m = mu.size
        self.mu = mu.reshape((1,m))
        self.L = np.linalg.cholesky(C)
        
    def normalize(self,X):
        X0 = X - self.mu
        return np.linalg.solve(self.L,X0.T).T
        
    def unnormalize(self,X):
        X0 = np.dot(X,self.L.T)
        return X0 + self.mu

class SimulationRunner():
    def __init__(self,fun):
        self.fun = fun
    
    def run(self,X):
        # right now this just wraps a sequential for-loop. 
        # should be parallelized
        
        X,M,m = process_inputs(X)
        F = np.zeros((M,1))
        for i in range(M):
            F[i] = self.fun(X[i,:])
        return F
        
class SimulationGradientRunner():
    def __init__(self,dfun):
        self.dfun = dfun
    
    def run(self,X):
        # right now this just wraps a sequential for-loop. 
        # should be parallelized
        
        X,M,m = process_inputs(X)
        dF = np.zeros((M,m))
        for i in range(M):
            df = self.dfun(X[i,:])
            dF[i,:] = df.reshape(m)
        return dF

def local_linear_gradients(X,f,p=None):
    # modify this to include a different set of points
    X,M,m = process_inputs(X)
    if M<=m: raise Exception('Not enough samples for local linear models.')

    if p is None:
        p = np.minimum(np.floor(1.7*m),M)
    elif p<m+1 or p>M: 
        raise Exception('p must be between m+1 and M')

    dF = np.zeros((M,m))
    for i in range(M):
        x = X[i,:]
        ind = np.argsort(np.sum((X - x)**2,axis=1))
        A = np.hstack((np.ones((p,1)), X[ind[1:p+1],:]))
        u = np.linalg.lstsq(A,f[ind[1:p+1]])[0]
        dF[i,:] = u[1:].T
    return dF
    
def finite_difference_gradients(X,fun,h=1e-6):
    X,M,m = process_inputs(X)
    
    # points to run simulations
    XX = np.kron(np.ones((m+1,1)),X) + \
        h*np.kron(np.vstack((np.zeros((1,m)),np.eye(m))),np.ones((M,1)))
    
    sr = SimulationRunner(fun)
    F = sr.run(XX)
    
    dF = (F[M:].reshape((M,m)) - F[:M])/h
    return dF
