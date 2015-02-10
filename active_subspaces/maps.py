import numpy as np
import gurobi_wrapper as gw
import respsurf as rs

class ActiveVariableMap():
    def __init__(self, W1, W2):
        self.W1, self.W2 = W1, W2
    
    def forward(self, X):
        return np.dot(X, self.W1), np.dot(X, self.W2)
        
    def inverse(self, Y, N=1):
        Z = self.regularize_z(Y, N)
        W = np.hstack((self.W1, self.W2))
        return rotate_x(Y, Z, W)
        
    def regularize_z(self, Y, N):
        raise NotImplementedError()

class BoundedActiveVariableMap(ActiveVariableMap):
        
    def regularize_z(self, Y, N):
        W1, W2 = self.W1, self.W2
        m, n = W1.shape

        # sample the z's 
        # TODO: preallocate and organize properly
        NY = Y.shape[0]
        Zlist = []
        for y in Y:
            Zlist.append(sample_z(N, y, W1, W2))
        return np.array(Zlist).reshape((NY, m-n, N))
        
class UnboundedActiveVariableMap(ActiveVariableMap):
        
    def regularize_z(self, Y, N):
        m, n = self.W1.shape
        
        # sample z's
        NY = Y.shape[0]
        return np.random.normal(size=(NY, m-n, N))

class MinVariableMap(ActiveVariableMap):
    def train(self, X, f, bflag=False):
        self.bflag = bflag
        W1, W2 = self.W1, self.W2
        m, n = W1.shape
        
        # train quadratic surface on p>n active vars
        W = np.hstack((W1, W2))
        if m-n>2:
            p = n+2
        else:
            p = n+1
        Yp = np.dot(X, W[:,:p])
        pr = rs.PolynomialRegression(N=2)
        pr.train(Yp, f)        
        br, Ar = pr.g, pr.H

        # get coefficients 
        b = np.dot(W[:,:p], br)
        A = np.dot(W[:,:p], np.dot(Ar, W[:,:p].T))
        
        self.bz = np.dot(W2.T, b)
        self.zAy = np.dot(W2.T, np.dot(A, W1))
        self.zAz = np.dot(W2.T, np.dot(A, W2)) + 0.01*np.eye(m-n)
        
    def regularize_z(self, Y, N=1):
        W1, W2 = self.W1, self.W2
        m, n = W1.shape
        NY = Y.shape[0]
        
        Zlist = []
        A_ineq = np.vstack((W2, -W2))
        for y in Y:
            c = self.bz.reshape((m-n, 1)) + np.dot(self.zAy, y).reshape((m-n, 1))
            if self.bflag:
                b_ineq = np.vstack((
                    -1-np.dot(W1, y).reshape((m, 1)),
                    -1+np.dot(W1, y).reshape((m, 1))
                    ))
                z = gw.quadratic_program_ineq(c, self.zAz, A_ineq, b_ineq)
            else:
                z = np.linalg.solve(self.zAz, c)
            Zlist.append(z)
        return np.array(Zlist).reshape((NY, m-n, N))

def sample_z(N, y, W1, W2):
    m, n = W1.shape
    s = np.dot(W1, y).reshape((m, 1))
    if np.all(np.zeros((m, 1)) <= 1-s) and np.all(np.zeros((m, 1)) >= -1-s):
        z0 = np.zeros((m-n, 1))
    else:
        lb = -np.ones(m)
        ub = np.ones(m)
        c = np.zeros(m)
        x0 = gw.linear_program_eq(c, W1.T, y, lb, ub)
        z0 = np.dot(W2.T, x0).reshape((m-n, 1))
        
    # get MCMC step size
    sig = 0.1*np.maximum( 
            np.linalg.norm(np.dot(W2, z0) + s - 1),
            np.linalg.norm(np.dot(W2, z0) + s + 1))        
    
    # burn in
    for i in range(N):
        zc = z0 + sig*np.random.normal(size=z0.shape)
        if np.all(np.dot(W2, zc) <= 1-s) and np.all(np.dot(W2, zc) >= -1-s):
            z0 = zc
    
    # sample
    Z = np.zeros((m-n, N))
    for i in range(N):
        zc = z0 + sig*np.random.normal(size=z0.shape)
        if np.all(np.dot(W2, zc) <= 1-s) and np.all(np.dot(W2, zc) >= -1-s):
            z0 = zc
        Z[:,i] = z0.reshape((z0.shape[0], ))
        
    return Z

def rotate_x(Y, Z, W):
    NY, n = Y.shape
    N = Z.shape[2]
    m = n + Z.shape[1]
    
    YY = np.tile(Y.reshape((NY, n, 1)), (1, 1, N))
    YZ = np.concatenate((YY, Z), axis=1).transpose((1, 0, 2)).reshape((m, N*NY)).transpose((1, 0))
    X = np.dot(YZ, W.T)
    ind = np.kron(np.arange(NY), np.ones(N))
    return X, ind