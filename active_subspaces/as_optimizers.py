import numpy as np
from domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain, \
                ActiveVariableMap
from scipy.optimize import minimize, fminbound
from utils.response_surfaces import PolynomialRegression
from qp_solvers.qp_solver import QPSolver

class MinVariableMap(ActiveVariableMap):
    def train(self, X, f):
        W1, W2 = self.W1, self.W2
        m, n = W1.shape

        # train quadratic surface on p>n active vars
        W = np.hstack((W1, W2))
        if m-n>2:
            p = n+2
        else:
            p = n+1
        Yp = np.dot(X, W[:,:p])
        pr = PolynomialRegression(N=2)
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
        W1, W2 = self.W1, self.W2
        m, n = W1.shape
        NY = Y.shape[0]

        Zlist = []
        A_ineq = np.vstack((W2, -W2))
        for y in Y:
            c = self.bz.reshape((m-n, 1)) + np.dot(self.zAy, y).reshape((m-n, 1))
            b_ineq = np.vstack((
                -1-np.dot(W1, y).reshape((m, 1)),
                -1+np.dot(W1, y).reshape((m, 1))
                ))
            z = QPSolver.get_qp_solver().quadratic_program_ineq(c, self.zAz, A_ineq, b_ineq)
            Zlist.append(z)
        return np.array(Zlist).reshape((NY, m-n, N))

class UnboundedMinVariableMap(MinVariableMap):

    def regularize_z(self, Y, N=1):
        m, n = self.W1.shape
        NY = Y.shape[0]

        Zlist = []
        for y in Y:
            c = self.bz.reshape((m-n, 1)) + np.dot(self.zAy, y).reshape((m-n, 1))
            z = np.linalg.solve(self.zAz, c)
            Zlist.append(z)
        return np.array(Zlist).reshape((NY, m-n, N))

def as_minimize(av_fun, domain):
    n = domain.n
    opts = {'disp':True, 'maxiter':1e4, 'ftol':1e-9}
    
    if isinstance(domain, UnboundedActiveVariableDomain):
        y0 = np.random.normal(size=(1, n))
        result = minimize(av_fun, y0, method='SLSQP', options=opts)
        ystar, fval = result.x, result.fun
        
    elif isinstance(domain, BoundedActiveVariableDomain):
        if n==1:
            yl, yu = domain.vertY[0], domain.vertY[1]
            result = fminbound(av_fun, yl, yu, xtol=1e-9, maxfun=1e4, full_output=1)
            ystar, fval = result.xopt, result.fval
        else:
            y0 = np.random.normal(size=(1, n))
            cons = domain.constraints
            result = minimize(av_fun, y0, constraints=cons, method='SLSQP', options=opts)
            ystar, fval = result.x, result.fun
    else:
        raise Exception('Shit, yo')
    
    return ystar, fval
