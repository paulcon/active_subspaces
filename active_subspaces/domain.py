import numpy as np
import gaussquad as gq
from scipy.spatial import Delaunay,ConvexHull,distance_matrix
from scipy.optimize import minimize

class ActiveVariableDomain():
    def __init__(self,W1):
        self.W1 = W1
    
    def design(self,N):
        raise NotImplementedError()
        
    def integration_rule(self,N):
        raise NotImplementedError()
        
class UnoundedActiveVariableDomain(ActiveVariableDomain):
    def design(self,N):
        n = self.W1.shape[1]
        if len(N) != n:
            raise Exception('N should be a list of integers of length n.')
        return gq.gauss_hermite(N)[0]
        
    def integration_rule(self,N):
        n = self.W1.shape[1]
        if len(N) != n:
            raise Exception('N should be a list of integers of length n.')
        return gq.gauss_hermite(N)
        
class BoundedActiveVariableDomain(ActiveVariableDomain):
    def __init__(self,W1):
        self.W1 = W1
        m,n = W1.shape
        if n == 1:
            y0 = np.dot(W1.T,np.sign(W1))[0]
            if y0 < -y0:
                yl,yu = y0,-y0
                xl = np.sign(W1).reshape((1,m))
                xu = -np.sign(W1).reshape((1,m))
            else:
                yl,yu = -y0,y0
                xl = -np.sign(W1).reshape((1,m))
                xu = np.sign(W1).reshape((1,m))
            Y = np.array([yl,yu])
            X = np.vstack((xl,xu))
            self.convhull = None
        else:
	    Y,X = zonotope_vertices(W1)
            self.convhull = ConvexHull(Y)
        self.vertY,self.vertX = Y,X
        
    def design(self,N):
        if self.W1.shape[1] == 1:
            y = np.linspace(self.vertY[0],self.vertY[1],N[0]).reshape((N[0],1))
            Y = y[1:-1]
        else:
            Y = maximin_design(self.vertY,N[0])
        return Y
        
    def constraints(self):
        n = self.W1.shape[1]
        A = self.convhull.equations[:,:n]
        b = self.convhull.equations[:,n]
        cons = ({'type':'ineq',
                'fun' : lambda x: np.dot(A,x)-b,
                'jac' : lambda x: A})
        return cons
        
    def integration_rule(self,N):
        NX = 10000 # maybe user should have control of this
        
        if self.W1.shape[1] == 1:
            y = np.linspace(self.vertY[0],self.vertY[1],N[0]).reshape((N[0],1))
            Y = 0.5*(y[1:]+y[:-1])
            
            Ysamp = np.dot(np.random.uniform(-1.0,1.0, \
                size=(NX,self.W1.shape[0])),self.W1)
            Wy = np.histogram(Ysamp.reshape((NX,)),bins=y.reshape((N[0],)), \
                range=(np.amin(y),np.amax(y)))[0]/float(NX)
            Wy.reshape((N[0]-1,1))
        else:
            y = np.vstack((self.vertY,maximin_design(self.vertY,N[0])))
            T = Delaunay(y)
            c = []
            for t in T.simplices:
                c.append(np.mean(T.points[t],axis=0))
            Y = np.array(c)
            
            # approximate weights
            Ysamp = np.dot(np.random.uniform(-1.0,1.0, \
                size=(NX,self.W1.shape[0])),self.W1)
            I = T.find_simplex(Ysamp)
            Wy = np.zeros((T.nsimplex,1))
            for i in range(T.nsimplex):
                Wy[i] = np.sum(I==i)/float(NX)
                
        return Y,Wy

def nzv(m,n,M=None):
    # number of zonotope vertices
    if M is None:
        M = np.zeros((m,n))
    if m==1 or n==1:
        M[m-1,n-1] = 2
    elif M[m-1,n-1]==0:
        k1,M = nzv(m-1,n-1,M)
        k2,M = nzv(m-1,n,M)
        M[m-1,n-1] = k1 + k2
        for i in range(n-1):
            M = nzv(m,i+1,M)[1]
    k = M[m-1,n-1]
    return k,M

def zonotope_vertices(W1):
    m,n = W1.shape
    NMC = 10000
    Xlist = []
    for i in range(NMC):
        y = np.random.normal(size=(n))
        x = np.sign(np.dot(y,W1.transpose()))
        addx = True
        for xx in Xlist:
            if all(x==xx):
                addx = False
                break
        if addx:
            Xlist.append(x)
    X = np.array(Xlist)
    Y = np.dot(X,W1)
    return Y,X
    
def maximin_design_obj(y,yzv=None):
    Ny,n = yzv.shape
    N = y.size/n
    Y = y.reshape((N,n))
    D0 = distance_matrix(Y,Y) + 1e4*np.eye(N)
    D1 = distance_matrix(Y,yzv)
    return -np.amin(np.hstack((D0.flatten(),D1.flatten())))

def maximin_design(yzv,N):
    n = yzv.shape[1]
    C = ConvexHull(yzv)
    A = np.kron(np.eye(N),C.equations[:,:n])
    b = np.kron(np.ones(N),C.equations[:,n])
    cons = ({'type':'ineq',
                'fun' : lambda x: np.dot(A,x)-b,
                'jac' : lambda x: A})
    y0 = np.random.normal(size=(N,n))
    res = minimize(maximin_design_obj,y0,args=(yzv,),constraints=cons,
                    method='SLSQP',options={'disp':False,'maxiter':1e9,'ftol':1e-12})
    return res.x.reshape(y0.shape)