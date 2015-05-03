import numpy as np
from domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain, \
                ActiveVariableMap
import scipy.optimize as scopt
from utils.response_surfaces import PolynomialApproximation
from utils.qp_solver import QPSolver
from utils.utils import process_inputs_outputs

class MinVariableMap(ActiveVariableMap):
    def train(self, X, f):
        X, f, M, m = process_inputs_outputs(X, f)
        
        W1, W2 = self.domain.subspaces.W1, self.domain.subspaces.W2
        m, n = W1.shape
        W = self.domain.subspaces.eigenvectors

        # train quadratic surface on p>n active vars
        if m-n>2:
            p = n+2
        else:
            p = n+1
        Yp = np.dot(X, W[:,:p])
        pr = PolynomialApproximation(N=2)
        pr.train(Yp, f)
        br, Ar = pr.g, pr.H

        # get coefficients
        b = np.dot(W[:,:p], br)
        A = np.dot(W[:,:p], np.dot(Ar, W[:,:p].T))

        self.bz = np.dot(W2.T, b)
        self.zAy = np.dot(W2.T, np.dot(A, W1))
        self.zAz = np.dot(W2.T, np.dot(A, W2)) + 0.01*np.eye(m-n)

class BoundedMinVariableMap(MinVariableMap):

    def regularize_z(self, Y, N=1):
        if N != 1:
            raise Exception('MinVariableMap needs N=1.')
        
        W1, W2 = self.domain.subspaces.W1, self.domain.subspaces.W2
        m, n = W1.shape
        NY = Y.shape[0]
        qps = QPSolver()

        Zlist = []
        A_ineq = np.vstack((W2, -W2))
        for y in Y:
            c = self.bz.reshape((m-n, 1)) + np.dot(self.zAy, y).reshape((m-n, 1))
            b_ineq = np.vstack((
                -1-np.dot(W1, y).reshape((m, 1)),
                -1+np.dot(W1, y).reshape((m, 1))
                ))
            z = qps.quadratic_program_ineq(c, self.zAz, A_ineq, b_ineq)
            Zlist.append(z)
        return np.array(Zlist).reshape((NY, m-n, N))

class UnboundedMinVariableMap(MinVariableMap):

    def regularize_z(self, Y, N=1):
        if N != 1:
            raise Exception('MinVariableMap needs N=1.')
            
        m, n = self.domain.subspaces.W1.shape
        NY = Y.shape[0]

        Zlist = []
        for y in Y:
            c = self.bz.reshape((m-n, 1)) + np.dot(self.zAy, y).reshape((m-n, 1))
            z = np.linalg.solve(self.zAz, c)
            Zlist.append(z)
        return np.array(Zlist).reshape((NY, m-n, N))

def minimize(avrs, X, f):
    X, f, M, m = process_inputs_outputs(X, f)
    
    # wrappers
    def avfun(y):
        f = avrs.predict_av(y.reshape((1,y.size)))[0]
        return f[0,0]
    def avdfun(y):
        df = avrs.gradient_av(y.reshape((1,y.size)))
        return df.reshape((y.size,))
    
    if isinstance(avrs.avmap.domain, UnboundedActiveVariableDomain):
        mvm = UnboundedMinVariableMap(avrs.avmap.domain)
    elif isinstance(avrs.avmap.domain, BoundedActiveVariableDomain):
        mvm = BoundedMinVariableMap(avrs.avmap.domain)        
    else:
        raise Exception('There is a problem with the avmap.domain.')
        
    ystar, fstar = av_minimize(avfun, mvm, avdfun=avdfun)
    mvm.train(X, f)
    xstar = mvm.inverse(ystar)[0]
    return xstar, fstar

def av_minimize(avfun, avmap, avdfun=None):
    
    if isinstance(avmap.domain, UnboundedActiveVariableDomain):
        ystar, fstar = unbounded_minimize(avfun, avmap, avdfun)
        
    elif isinstance(avmap.domain, BoundedActiveVariableDomain):
        n = avmap.domain.subspaces.W1.shape[1]
        if n==1:
            ystar, fstar = interval_minimize(avfun, avmap)
        else:
            ystar, fstar = zonotope_minimize(avfun, avmap, avdfun)
    else:
        raise Exception('There is a problem with the avmap.domain.')
    
    return ystar.reshape((1,ystar.size)), fstar

def interval_minimize(avfun, avmap):
    yl, yu = avmap.domain.vertY[0,0], avmap.domain.vertY[1,0]
    result = scopt.fminbound(avfun, yl, yu, xtol=1e-9, maxfun=1e4, full_output=1)
    return np.array([[result[0]]]), result[1]
    
def zonotope_minimize(avfun, avmap, avdfun):
    n = avmap.domain.subspaces.W1.shape[1]
    opts = {'disp':False, 'maxiter':1e4, 'ftol':1e-9}
    
    # a bit of globalization
    curr_state = np.random.get_state()
    np.random.seed(42)
    minf = 1e100
    minres = []
    for i in range(10):
        y0 = np.random.normal(size=(1, n))
        cons = avmap.domain.constraints
        result = scopt.minimize(avfun, y0, constraints=cons, method='SLSQP', \
                            jac=avdfun, options=opts)
        if result.fun < minf:
            minf = result.fun
            minres = result
            
    np.random.set_state(curr_state)
    return minres.x, minres.fun
    
def unbounded_minimize(avfun, avmap, avdfun):
    n = avmap.domain.subspaces.W1.shape[1]
    opts = {'disp':False, 'maxiter':1e4}
    
    if avdfun == None:
        method = 'SLSQP'
    else:
        method = 'BFGS'
        
    # some tricks for globalization
    curr_state = np.random.get_state()
    np.random.seed(42)
    minf = 1e100
    minres = []
    for i in range(10):
        y0 = np.random.normal(size=(1, n))
        result = scopt.minimize(avfun, y0, method=method, jac=avdfun, options=opts)
        if result.fun < minf:
            minf = result.fun
            minres = result
    np.random.set_state(curr_state)
    return minres.x, minres.fun