import numpy as np
from scipy.optimize import fminbound
from scipy.misc import comb
import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self,N=2):
        self.N = N
        
    def train(self,X,f):
        try:
            M,m = X.shape
        except:
            raise Exception('X should be a numpy array of size (M,m), where M is the number of points and m is the dimension.')
         
        B,indices = polynomial_bases(X,self.N)
        Q,R = np.linalg.qr(B)
        p_weights = np.linalg.solve(R,np.dot(Q.T,f))
        
        # store data
        self.X,self.f = X,f
        self.p_weights = p_weights
        self.Q,self.R = Q,R
        
        # organize linear and quadratic coefficients
        self.b = p_weights[1:m+1].copy()
        if self.N>1:
            A = np.zeros((m,m))
            for i in range(m+1,comb(m+2,2)):
                ind = indices[i,:]
                loc = np.nonzero(ind!=0)[0]
                if loc.size==1:
                    A[loc,loc] = 2.0*p_weights[i]
                elif loc.size==2:
                    A[loc[0],loc[1]] = p_weights[i]
                    A[loc[1],loc[0]] = p_weights[i]
                else:
                    raise Exception('Error creating quadratic coefficients.')
        
    def predict(self,Xstar,compgrad=False,compvar=False):
        try:
            M,m = Xstar.shape
        except:
            raise Exception('Xstar should be a numpy array of size (M,m), where M is the number of points and m is the dimension.')
        
        Bstar = polynomial_bases(Xstar,self.N)[0]
        fstar = np.dot(Bstar,self.p_weights)
        
        if compgrad:
            dBstar = grad_polynomial_bases(Xstar,self.N)
            dfstar = np.zeros((M,m))
            for i in range(m):
                dfstar[:,i] = np.dot(dBstar[:,:,i],self.p_weights).reshape((M))
        else:
            dfstar = None
        
        if compvar:
            Rstar = np.linalg.solve(self.R.T,Bstar.T)
            vstar = np.var(self.f)*np.diag(np.dot(Rstar.T,Rstar))
        else:
            vstar = None
            
        return fstar,dfstar,vstar

    def __call__(self,Xstar):
        return self.predict(Xstar)

class GaussianProcess():
    def __init__(self,N=2):
        self.N = N
        
    def train(self,X,f,e=None,gl=0.0,gu=10.0,v=None):
        try:
            M,m = X.shape
        except:
            raise Exception('X should be a numpy array of size (M,m), where M is the number of points and m is the dimension.')
            
        if e is None:
            e = np.hstack((np.ones(m),np.array([np.var(f)])))
        g = fminbound(negative_log_likelihood,gl,gu,args=(X,f,e,self.N,v,))
        
        # set parameters
        sig = g*np.sum(e)
        ell = sig/e[:m]
        
        # covariance matrix of observations
        K = exponential_squared_covariance(X,X,sig,ell)
        if v is None:
            K += g*np.sum(e[m:])*np.eye(M)
        else:
            K += np.diag(v)
        f_weights = np.linalg.solve(K,f)
        
        # coefficients of polynomial basis
        B = polynomial_bases(X,self.N)[0]
        A = np.dot(B.T,np.linalg.solve(K,B))
        p_weights = np.linalg.solve(A,np.dot(B.T,f_weights))
        
        # store parameters
        self.X,self.f = X,f
        self.sig,self.ell = sig,ell
        self.f_weights,self.p_weights = f_weights,p_weights
        self.K,self.A,self.B = K,A,B
        
    def predict(self,Xstar,compgrad=False,compvar=False):
        try:
            M,m = Xstar.shape
        except:
            raise Exception('Xstar should be a numpy array of size (M,m), where M is the number of points and m is the dimension.')

        # predict without polys
        Kstar = exponential_squared_covariance(self.X,Xstar,self.sig,self.ell)
        fstar = np.dot(Kstar.T,self.f_weights)
        
        # update with polys
        Pstar = np.linalg.solve(self.K,Kstar)
        Bstar = polynomial_bases(Xstar,self.N)[0]
        Rstar = Bstar - np.dot(Pstar.T,self.B)
        fstar += np.dot(Rstar,self.p_weights)
        
        if compgrad:
            dKstar = grad_exponential_squared_covariance(self.X,Xstar,self.sig,self.ell)
            dBstar = grad_polynomial_bases(Xstar,self.N)
            dfstar = np.zeros((M,m))
            for i in range(m):
                dPstar = np.linalg.solve(self.K,dKstar[:,:,i])
                dRstar = dBstar[:,:,i] - np.dot(dPstar.T,self.B)
                dfstar[:,i] = (np.dot(dKstar.T,self.f_weights) \
                    + np.dot(dRstar,self.p_weights)).reshape((M))
        else:
            dfstar = None
        
        if compvar:
            Vstar = exponential_squared_covariance(Xstar,Xstar,self.sig,self.ell)
            vstar = np.diag(Vstar) - np.sum(Pstar*Pstar,axis=0)
            vstar += np.diag(np.dot(Rstar,np.linalg.solve(self.A,Rstar.T)))
        else:
            vstar = None
            
        return fstar,dfstar,vstar
            
    def __call__(self,Xstar):
        return self.predict(Xstar)

def negative_log_likelihood(g,X,f,e,N,v):
    M,m = X.shape
    sig = g*np.sum(e)
    ell = sig/e[:m]
    
    # covariance matrix
    K = exponential_squared_covariance(X,X,sig,ell)
    if v is None:
        K += g*np.sum(e[m:])*np.eye(M)
    else:
        K += np.diag(v)
    L = np.linalg.cholesky(K)
    
    # polynomial basis
    B = polynomial_bases(X,N)[0]
    A = np.dot(B.T,np.linalg.solve(K,B))
    AL = np.linalg.cholesky(A)
    
    # negative log likelihood
    z = np.linalg.solve(K,f)
    Bz = np.dot(B.T,z)
    r = np.dot(f.T,z) \
        - np.dot(Bz.T,np.linalg.solve(A,Bz)) \
        + np.sum(np.log(np.diag(L))) \
        + np.sum(np.log(np.diag(AL))) \
        + (M-B.shape[1])*np.log(2*np.pi)
    return 0.5*r

def exponential_squared_covariance(X1,X2,sigma,ell):
    m = X1.shape[0]
    n = X2.shape[0]
    c = -1.0/ell.flatten()
    C = np.zeros((m,n))
    for i in range(n):
        x2 = X2[i,:]
        B = X1 - x2
        C[:,i] = sigma*np.exp(np.dot(B*B,c))
    return C

def grad_exponential_squared_covariance(X1,X2,sigma,ell):
    m,d = X1.shape
    n = X2.shape[0]
    c = -1.0/ell.flatten()
    C = np.zeros((m,n,d))
    for k in range(d):
        for i in range(n):
            x2 = X2[i,:]
            B = X1 - x2
            C[:,i,k] = sigma*(-2.0*c[k]*B[:,k])*np.exp(np.dot(B*B,c))
    return C

def polynomial_bases(X,N):
    M,m = X.shape
    I = index_set(N,m)
    n = I.shape[0]
    B = np.zeros((M,n))
    for i in range(n):
        ind = I[i,:]
        B[:,i] = np.prod(np.power(X,ind),axis=1)
    return B,I
    
def grad_polynomial_bases(X,N):
    M,m = X.shape
    I = index_set(N,m)
    n = I.shape[0]
    B = np.zeros((M,n,m))
    for k in range(m):
        for i in range(n):
            ind = I[i,:]
            indk = ind[k]
            if indk==0:
                B[:,i,k] = np.zeros(M)
            else:
                ind[k] -= 1
                B[:,i,k] = indk*np.prod(np.power(X,ind),axis=1)
    return B

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
    
if __name__ == '__main__':
    '''
    X1,X2 = np.meshgrid(np.linspace(-1.0,1.0,21),np.linspace(-1.0,1.0,21))
    X = np.hstack((X1.reshape((X1.size,1)),X2.reshape((X2.size,1))))
    f = np.sin(np.pi*np.sum(X,axis=1))
    e = np.array([1.0,0.5,0.1])
    Xstar = np.random.uniform(-1.0,1.0,size=(100,2))
    fstar,vstar = gaussian_process_regression(X,f,Xstar,e=e,gl=0.0,gu=100.0,N=5)
    '''
    '''
    X = np.linspace(-1.0,1.0,21).reshape((21,1))
    f = np.sin(np.pi*X)
    e = np.array([1.0,0.5,0.1])
    D = np.load('true.npz')
    Xstar = D['Xstar']
    gp = GaussianProcess(5)
    gp.train(X,f,e,gl=0.0,gu=100.0)
    fstar,dfstar,vstar = gp.predict(Xstar,compvar=True)
    plt.plot(X,f,'k-',Xstar,fstar,'bx')
    plt.show()
    print 'Error: %6.4e,%6.4e' % (np.linalg.norm(fstar-D['fstar']),np.linalg.norm(vstar-D['vstar']))
    print 'Error: %6.4e' % np.linalg.norm(gp(Xstar)[0]-D['fstar'])
    '''
    '''
    X = np.linspace(-1.0,1.0,21).reshape((21,1))
    f = np.sin(np.pi*X)
    e = np.array([1.0,0.5,0.1])
    D = np.load('true.npz')
    Xstar = D['Xstar']
    gp = GaussianProcess(5)
    gp.train(X,f,gl=0.0,gu=100.0)
    fstar,dfstar,vstar = gp.predict(Xstar,compvar=True,compgrad=True)
    h = 1e-9
    fdfstar = (gp(Xstar+h)[0] - gp(Xstar)[0])/h
    print 'Grad err: %6.4e' % np.linalg.norm(dfstar-fdfstar)
    '''
    X = np.linspace(-1.0,1.0,51).reshape((51,1))
    f = np.sin(np.pi*X)
    pr = PolynomialRegression(5)
    pr.train(X,f)
    D = np.load('true.npz')
    Xstar = D['Xstar'] 
    fstar,dfstar,vstar = pr.predict(Xstar,compgrad=True,compvar=True)
    plt.close('all')
    plt.figure()
    plt.plot(X,f,'k-',Xstar,fstar,'bx')
    plt.title('Prediction')
    plt.figure()
    plt.plot(Xstar,np.pi*np.cos(np.pi*Xstar),'ro',Xstar,dfstar,'bx')
    plt.title('Derivative')
    plt.figure()
    plt.plot(Xstar,vstar,'bx')
    plt.title('Variance')
    plt.show()
    