import numpy as np
import gurobi_wrapper as gw
import regression as rg
import zonotopes as zn
import gaussian_quadrature as gq
from scipy.spatial import Delaunay
from analyze_active_subspace import \
    sufficient_summary_plot,plot_eigenvectors
import pdb

class VariableMap():
    def __init__(self,W,n,bflag):
        W1,W2 = W[:,:n],W[:,n:]
        self.W1,self.W2 = W1,W2
        self.bflag = bflag
        
    def forward(self,X):
        return np.dot(X,self.W1),np.dot(X,self.W2)
        
    def inverse(self,Y,NMC):
        m,n = self.W1.shape
        W = np.hstack((self.W1,self.W2))
        
        if self.bflag:
            # sample the z's 
            NY = Y.shape[0]
            Zlist = []
            for y in Y:
                Zlist.append(sample_z(NMC,y,self.W1,self.W2))
            Z = np.array(Zlist).reshape((NY,m-n,NMC))

        else:
            # sample z's
            NY = Y.shape[0]
            Z = np.random.normal(size=(NY,m-n,NMC))
            
        # rotate back to x
        YY = np.tile(Y.reshape((NY,n,1)),(1,1,NMC))
        YZ = np.concatenate((YY,Z),axis=1).transpose((1,0,2)).reshape((m,NMC*NY)).transpose((1,0))
        X = np.dot(YZ,W.T)
        ind = np.kron(np.arange(NY),np.ones(NMC))
        return X,ind

class OptVariableMap():
    def __init__(self,W,n,X,f,bflag):
        W1,W2 = W[:,:n],W[:,n:]
        self.W1,self.W2 = W1,W2
        self.X,self.f = X,f
        self.bflag = bflag
        
        m = n+2
        Wp = W[:,:m]
        Yp = np.dot(X,Wp)
        Yp2,I = rg.polynomial_bases(Yp,2)
        M = I.shape[0]
        up = np.linalg.lstsq(Yp2,f)[0]
        bp = up[1:m+1]
        Ap = np.zeros((m,m))
        for i in range(m+1,M):
            ind = I[i,:]
            loc = np.nonzero(ind!=0)[0]
            if loc.size==1:
                Ap[loc,loc] = 2.0*up[i]
            elif loc.size==2:
                Ap[loc[0],loc[1]] = up[i]
                Ap[loc[1],loc[0]] = up[i]
            else:
                raise Exception('Error!')
        
        #pdb.set_trace()
        b = np.dot(Wp,bp)
        A = np.dot(Wp,np.dot(Ap,Wp.T))
        
        self.bz = np.dot(b.T,W2)
        self.Ayz = np.dot(W1.T,np.dot(A,W2))
        self.Az = np.dot(W2.T,np.dot(A,W2)) + 0.1*np.eye(W1.shape[0]-n)
        
    def forward(self,X):
        return np.dot(X,self.W1),np.dot(X,self.W2)
        
    def inverse(self,Y):
        A = self.Az
        Xlist = []
        for y in Y:
            b = self.bz + np.dot(y,self.Ayz)
            if self.bflag:
                lb = -1.0 - np.dot(self.W1,y)
                ub = 1.0 - np.dot(self.W1,y)
                z = gw.quadratic_program_bnd(b,A,lb,ub)
            else:
                z = np.linalg.solve(A,b)
            x = np.dot(self.W1,y) + np.dot(self.W2,z)
            Xlist.append(x)
        return np.array(Xlist)

class BoundedNormalizer():
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
        
class UnboundedNormalizer():
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

def lingrad(X,f):
    M,m = X.shape
    A = np.hstack((np.ones((M,1)), X))
    u = np.linalg.lstsq(A,f)[0]
    w = u[1:]/np.linalg.norm(u[1:])
    return w

def sample_function(X,fun,dflag=False):
    M,m = X.shape
    F = np.zeros((M,1))
    if dflag:
        dF = np.zeros((M,m))
        for i in range(M):
            x = X[i,:]
            f,df = fun(x.T)
            F[i] = f
            dF[i,:] = df.T
        return F,dF
    else:
        for i in range(M):
            x = X[i,:]
            F[i] = fun(x.T)
        return F
        
def local_linear_gradients(X,f,XX,p=None):
    M,m = X.shape
    MM = XX.shape[0]
    dF = np.zeros((MM,m))
    if p is None:
        p = np.minimum(2*m,M)
    for i in range(MM):
        x = XX[i,:]
        A = np.sum((X - np.tile(x,(M,1)))**2,axis=1)
        ind = np.argsort(A)
        A = np.hstack((np.ones((p,1)), X[ind[:p],:]))
        u = np.linalg.lstsq(A,f[ind[:p]])[0]
        dF[i,:] = u[1:].T.copy()
    return dF
    
def finite_difference_gradients(X,fun,h=1e-6):
    M,m = X.shape
    dF = np.zeros((M,m))
    for i in range(M):
        x0 = X[i,:]
        f0 = fun(x0.T)
        df = np.zeros((m,1))
        for j in range(m):
            xp = x0.copy()
            xp[j] += h
            df[j] = (fun(xp)-f0)/h
        dF[i,:] = df.T
    return dF

def compute_active_subspace(dF,k,n_boot=1000):
    
    # set integers
    M,m = dF.shape
    k_sub = np.minimum(k,m-1)
    
    # compute active subspace
    U,sig,W = np.linalg.svd(dF,full_matrices=False)
    e = (1.0/M)*(sig[:k]**2)
    W = W.T
    W = W*np.tile(np.sign(W[0,:]),(m,1))
    
    # bootstrap
    e_boot = np.zeros((k,n_boot))
    sub_dist = np.zeros((k_sub,n_boot))
    ind = np.random.randint(M,size=(M,n_boot))
    for i in range(n_boot):
        U0,sig0,W0 = np.linalg.svd(dF[ind[:,i],:],full_matrices=False)
        W0 = W0.T
        W0 = W0*np.tile(np.sign(W0[0,:]),(m,1))
        e_boot[:,i] = (1.0/M)*(sig0[:k]**2)
        for j in range(k_sub):
            sub_dist[j,i] = np.linalg.norm(np.dot(W[:,:j+1].T,W0[:,j+1:]),ord=2)
    
    e_br = np.zeros((k,2))
    sub_br = np.zeros((k_sub,3))
    for i in range(k):
        e_br[i,0] = np.amin(e_boot[i,:])
        e_br[i,1] = np.amax(e_boot[i,:])
    for i in range(k_sub):
        sub_br[i,0] = np.amin(sub_dist[i,:])
        sub_br[i,1] = np.mean(sub_dist[i,:])
        sub_br[i,2] = np.amax(sub_dist[i,:])
    return e,W,e_br,sub_br

def quick_check(X,f,n_boot=1000,in_labels=None,out_label=None):
    M,m = X.shape
    w = lingrad(X,f)
    
    # bootstrap
    ind = np.random.randint(M,size=(M,n_boot))
    w_boot = np.zeros((m,n_boot))
    for i in range(n_boot): 
        w_boot[:,i] = lingrad(X[ind[:,i],:],f[ind[:,i]])
    
    # make sufficient summary plot
    y = np.dot(X,w)
    sufficient_summary_plot(y,f,out_label=out_label)
    
    # plot weights
    plot_eigenvectors(w,W_boot=w_boot,in_labels=in_labels,out_label=out_label)
    
    return w

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
    
    # burn in
    for i in range(N):
        zc = z0 + 0.66*np.random.normal(size=z0.shape)
        if np.all(np.dot(W2,zc) <= 1-s) and np.all(np.dot(W2,zc) >= -1-s):
            z0 = zc
    
    # sample
    Z = np.zeros((m-n,N))
    for i in range(N):
        zc = z0 + 0.66*np.random.normal(size=z0.shape)
        if np.all(np.dot(W2,zc) <= 1-s) and np.all(np.dot(W2,zc) >= -1-s):
            z0 = zc
        Z[:,i] = z0.reshape((z0.shape[0],))
        
    return Z

def response_surface_design(W,n,N,NMC,bflag=0):
    
    # check len(N) == n
    m = W.shape[0]
    W1,W2 = W[:,:n],W[:,n:]
    
    if bflag:
        # uniform case
        if n==1:
            y0 = np.dot(W1.T,np.sign(W1))[0]
            if y0 < -y0:
                yl,yu = y0,-y0
                xl = np.sign(W1).reshape((1,m))
                xu = -np.sign(W1).reshape((1,m))
            else:
                yl,yu = -y0,y0
                xl = -np.sign(W1).reshape((1,m))
                xu = np.sign(W1).reshape((1,m))
            y = np.linspace(yl,yu,N[0]).reshape((N[0],1))
            
            Yzv = np.array([yl,yu])
            Xzv = np.vstack((xl,xu))
            Y = y[1:-1]
            
        else:
            Yzv,Xzv = zn.zonotope_vertices(W1)
            Y = zn.maximin_design(Yzv,N[0])
        
    else:
        # Gaussian case
        if len(N) != n:
            raise Exception('N should be a list of integers of length n.')
        Y = gq.gauss_hermite(N)[0]
    
    vm = VariableMap(W,n,bflag)
    X,ind = vm.inverse(Y,NMC)
    if bflag:
        X = np.vstack((X,Xzv))
        ind = np.hstack((ind,np.arange(Y.shape[0],Y.shape[0]+Yzv.shape[0])))
        Y = np.vstack((Y,Yzv))
    
    return X,ind,Y

def integration_rule(W,n,N,NMC,bflag=0):
    m = W.shape[0]
    W1,W2 = W[:,:n],W[:,n:]
    NX = 10000
    
    if bflag:
        # uniform case
        if n==1:
            y0 = np.dot(W1.T,np.sign(W1))[0]
            if y0 < -y0:
                yl,yu = y0,-y0
            else:
                yl,yu = -y0,y0
            y = np.linspace(yl,yu,N[0]).reshape((N[0],1))
            Y = 0.5*(y[1:]+y[:-1])
            
            Ysamp = np.dot(np.random.uniform(-1.0,1.0,size=(NX,m)),W1)
            Wy = np.histogram(Ysamp.reshape((NX,)),bins=y.reshape((N[0],)), \
                range=(np.amin(y),np.amax(y)))[0]/float(NX)
            
        else:
            # get points
            yzv = zn.zonotope_vertices(W1)[0]
            y = np.vstack((yzv,zn.maximin_design(yzv,N[0])))
            T = Delaunay(y)
            c = []
            for t in T.simplices:
                c.append(np.mean(T.points[t],axis=0))
            Y = np.array(c)
            
            # approximate weights
            Ysamp = np.dot(np.random.uniform(-1.0,1.0,size=(NX,m)),W1)
            I = T.find_simplex(Ysamp)
            Wy = np.zeros((T.nsimplex,1))
            for i in range(T.nsimplex):
                Wy[i] = np.sum(I==i)/float(NX)
            
    else:
        # Gaussian case
        if len(N) != n:
            raise Exception('N should be a list of integers of length n.')
        Y,Wy = gq.gauss_hermite(N)
        
    vm = VariableMap(W,n,bflag)
    X = vm.inverse(Y,NMC)
    
    # weights for integration in x space
    Wx = np.kron(Wy,(1.0/NMC)*np.ones((NMC,1)))
    
    return X,Wx,Y,Wy
    
def quadratic_model_check(X,f,gamma,k):
    M,m = X.shape
    gamma = gamma.reshape((1,m))
    
    pr = rg.PolynomialRegression(2)
    pr.train(X,f)

    # get regression coefficients
    b,A = pr.b,pr.A
    
    # compute eigenpairs
    e,W = np.linalg.eig(np.outer(b,b.T) + \
        np.dot(A,np.dot(np.diagflat(gamma),A)))
    ind = np.argsort(e)[::-1]
    e,W = e[ind],W[:,ind]*np.sign(W[0,ind])
    
    return e[:k],W


    
    
    

