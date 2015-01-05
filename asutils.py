import numpy as np
import gurobi_wrapper as gw
import regression as rg
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

def normalize_uniform(X,xl,xu):
    M,m = X.shape
    xl = xl.reshape(1,m,order='F')
    xu = xu.reshape(1,m,order='F')
    XX = 2*(X - np.tile(xl,(M,1)))/(np.tile(xu,(M,1))-np.tile(xl,(M,1)))-1
    return XX.copy()

def normalize_gaussian(X,mu,C):
    M,m = X.shape
    mu = mu.reshape(1,m,order='F')
    L = np.linalg.cholesky(C)
    X0 = X - np.tile(mu,(M,1))
    XX = np.linalg.solve(L,X0.T)
    return XX.T.copy()

def full_index_set(n,d):
    if d == 1:
        I = np.array([[n]])
    else:
        II = full_index_set(n,d-1)
        m = II.shape[0]
        I = np.hstack((np.zeros((m,1)),II))
        for i in range(1,n+1):
            II = full_index_set(n-i,d-1)
            m = II.shape[0]
            T = np.hstack((i*np.ones((m,1)),II))
            I = np.vstack((I,T))
    return I
    
def index_set(n,d):
    I = np.zeros((1,d))
    for i in range(1,n+1):
        II = full_index_set(i,d)
        I = np.vstack((I,II))
    return I[:,::-1]

def lingrad(X,f):
    M,m = X.shape
    A = np.hstack((np.ones((M,1)), X))
    u = np.linalg.lstsq(A,f)[0]
    w = u[1:]/np.linalg.norm(u[1:])
    return w

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

def quadreg(Xsq,indices,f):
    M,m = indices.shape
    u = np.linalg.lstsq(Xsq,f)[0]
    b = u[1:m+1]
    A = np.zeros((m,m))
    for i in range(m+1,M):
        ind = indices[i,:]
        loc = np.nonzero(ind!=0)[0]
        if loc.size==1:
            A[loc,loc] = 2.0*u[i]
        elif loc.size==2:
            A[loc[0],loc[1]] = u[i]
            A[loc[1],loc[0]] = u[i]
        else:
            print 'error!'
    return b,A
    
def get_eigenpairs(b,A,gamma):
    m = b.shape[0]
    e,W = np.linalg.eig(np.outer(b,b.T) + np.dot(A,np.dot(np.diagflat(gamma),A)))
    ind = np.argsort(e)[::-1]
    e = e[ind]
    W = W[:,ind]
    W = W*np.tile(np.sign(W[0,:]),(m,1))
    return e,W
    
def quadratic_model_check(X,f,gamma,k,n_boot=1000):
    M,m = X.shape
    gamma = gamma.reshape((1,m))
    k_sub = np.minimum(k,m-1)
    
    indices = index_set(2,m)
    mI = indices.shape[0]
    Xsq = np.zeros((M,mI))
    for i in range(mI):
        ind = indices[i,:]
        Xsq[:,i] = np.prod(np.power(X,np.tile(ind,(M,1))),axis=1)
    
    # get regression coefficients
    b,A = quadreg(Xsq,indices,f)
    
    # compute eigenpairs
    e,W = get_eigenpairs(b,A,gamma)
    
    # bootstrap
    sub_dist = np.zeros((m-1,n_boot))
    e_boot = np.zeros((m,n_boot))
    ind = np.random.randint(M,size=(M,n_boot))
    for i in range(n_boot):
        b0,A0 = quadreg(Xsq[ind[:,i],:],indices,f[ind[:,i]])
        e0,W0 = get_eigenpairs(b0,A0,gamma)
        e_boot[:,i] = e0
        for j in range(m-1):
            sub_dist[j,i] = np.linalg.norm(np.dot(W[:,:j+1].T,W0[:,j+1:]),ord=2)

    e_br = np.zeros((k,2))
    sub_br = np.zeros((k_sub,3))
    for i in range(k):
        lam_sort = np.sort(e_boot[i,:])
        e_br[i,0] = lam_sort[np.floor(0.025*n_boot)]
        e_br[i,1] = lam_sort[np.ceil(0.925*n_boot)]
    for i in range(k_sub):
        sub_sort = np.sort(sub_dist[i,:])
        sub_br[i,0] = sub_sort[np.floor(0.025*n_boot)]        
        sub_br[i,1] = np.mean(sub_sort)
        sub_br[i,2] = sub_sort[np.ceil(0.925*n_boot)]
    return e[:k],W,e_br,sub_br


    
    
    

