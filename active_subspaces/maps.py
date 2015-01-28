import numpy as np
import gurobi_wrapper as gw
import respsurf as rs

class ActiveVariableMap():
    def __init__(self,W1,W2):
        self.W1,self.W2 = W1,W2
    
    def forward(self,X):
        return np.dot(X,self.W1),np.dot(X,self.W2)
        
    def inverse(self,Y,N):
        raise NotImplementedError()

class BoundedActiveVariableMap(ActiveVariableMap):
        
    def inverse(self,Y,NMC):
        m,n = self.W1.shape
        W = np.hstack((self.W1,self.W2))

        # sample the z's 
        # TODO: preallocate and organize properly
        NY = Y.shape[0]
        Zlist = []
        for y in Y:
            Zlist.append(sample_z(NMC,y,self.W1,self.W2))
        Z = np.array(Zlist).reshape((NY,m-n,NMC))
        
        return rotate_x(Y,Z,W)
        
class UnboundedActiveVariableMap(ActiveVariableMap):
        
    def inverse(self,Y,NMC):
        m,n = self.W1.shape
        W = np.hstack((self.W1,self.W2))

        # sample z's
        NY = Y.shape[0]
        Z = np.random.normal(size=(NY,m-n,NMC))
        
        return rotate_x(Y,Z,W)

class MinVariableMap(ActiveVariableMap):
    def train(self,X,f,bflag=False):
        self.bflag = bflag
        
        m,n = self.W1.shape
        W = np.hstack((self.W1,self.W2))
        
        # train quadratic surface on p>n active vars
        if m-n>2:
            p = n+2
        else:
            p = n+1
        Yp = np.dot(X,W[:,:p])
        pr = rs.PolynomialRegression(N=2)
        pr.train(Yp,f)        
        gp,Hp = pr.g,pr.H

        # get coefficients 
        g = np.dot(W[:,:p],gp)
        H = np.dot(W[:,:p],np.dot(Hp,W[:,:p].T))
        
        self.gz = np.dot(g.T,self.W2)
        self.Hyz = np.dot(self.W1.T,np.dot(H,self.W2))
        self.Hz = np.dot(self.W2.T,np.dot(H,self.W2)) \
            + 0.1*np.eye(self.W1.shape[0]-n)
        
    def inverse(self,Y,N=0):
        H = self.Hz
        Xlist = []
        for y in Y:
            g = self.gz + np.dot(y,self.Hyz)
            if self.bflag:
                lb = -1.0 - np.dot(self.W1,y)
                ub = 1.0 - np.dot(self.W1,y)
                z = gw.quadratic_program_bnd(g,H,lb,ub)
            else:
                z = np.linalg.solve(H,g)
            x = np.dot(self.W1,y) + np.dot(self.W2,z)
            Xlist.append(x)
        return np.array(Xlist)

def sample_z(N,y,W1,W2):
    m,n = W1.shape
    s = np.dot(W1,y).reshape((m,1))
    if np.all(np.zeros((m,1)) <= 1-s) and np.all(np.zeros((m,1)) >= -1-s):
        z0 = np.zeros((m-n,1))
    else:
        lb = -np.ones(m)
        ub = np.ones(m)
        c = np.zeros(m)
        x0 = gw.linear_program_eq(c,W1.T,y,lb,ub)
        z0 = np.dot(W2.T,x0).reshape((m-n,1))
        
    # get MCMC step size
    sig = 0.1*np.maximum( 
            np.linalg.norm(np.dot(W2,z0)+s-1),
            np.linalg.norm(np.dot(W2,z0)+s+1))        
    
    # burn in
    for i in range(N):
        zc = z0 + sig*np.random.normal(size=z0.shape)
        if np.all(np.dot(W2,zc) <= 1-s) and np.all(np.dot(W2,zc) >= -1-s):
            z0 = zc
    
    # sample
    Z = np.zeros((m-n,N))
    for i in range(N):
        zc = z0 + sig*np.random.normal(size=z0.shape)
        if np.all(np.dot(W2,zc) <= 1-s) and np.all(np.dot(W2,zc) >= -1-s):
            z0 = zc
        Z[:,i] = z0.reshape((z0.shape[0],))
        
    return Z

def rotate_x(Y,Z,W):
    NY,n = Y.shape
    N = Z.shape[2]
    m = n + Z.shape[1]
    
    YY = np.tile(Y.reshape((NY,n,1)),(1,1,N))
    YZ = np.concatenate((YY,Z),axis=1).transpose((1,0,2)).reshape((m,N*NY)).transpose((1,0))
    X = np.dot(YZ,W.T)
    ind = np.kron(np.arange(NY),np.ones(N))
    return X,ind