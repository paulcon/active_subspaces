import numpy as np
from asutils import index_set
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

def gaussian_process_regression(X,f,Xstar,N=2,gl=0.0,gu=10.0,e=None,v=None):
    M,m = X.shape
    if e is None:
        e = np.ones(m)
        
    # maximum likelihood
    g = fminbound(negative_log_likelihood,gl,gu,args=(X,f,e,N,v,))
    print 'Optimal g: %6.4f' % g
    
    # set parameters
    sig = g*np.sum(e)
    ell = sig/e[:m]
    
    # covariance matrix of observations
    Ky = exponential_squared_covariance(X,X,sig,ell)
    if v is None:
        Ky += g*np.sum(e[m:])*np.eye(M)
    else:
        Ky += np.diag(v)
    z = np.linalg.solve(Ky,f)
    
    # covariance matrix of test sites
    Kstar = exponential_squared_covariance(X,Xstar,sig,ell)
    
    # prediction without polynomials
    fstar = np.dot(Kstar.T,z)
    V = exponential_squared_covariance(Xstar,Xstar,sig,ell)
    P = np.linalg.solve(Ky,Kstar)
    vstar = np.diag(V) - np.sum((P*P).T,axis=1)
    
    # coefficients of polynomial basis
    B = polynomial_bases(X,N)
    A = np.dot(B.T,np.linalg.solve(Ky,B))
    beta = np.linalg.solve(A,np.dot(B.T,z))
    
    # update predictions with polynomials
    Bstar = polynomial_bases(Xstar,N)
    R = Bstar.T - np.dot(B.T,P)
    fstar += np.dot(R.T,beta)
    vstar += np.diag(np.dot(R.T,np.linalg.solve(A,R)))
    return fstar,vstar

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

def negative_log_likelihood(g,X,f,e,N,v):
    M,m = X.shape
    sig = g*np.sum(e)
    ell = sig/e[:m]
    
    # covariance matrix
    Ky = exponential_squared_covariance(X,X,sig,ell)
    if v is None:
        Ky += g*np.sum(e[m:])*np.eye(M)
    else:
        Ky += np.diag(v)
    L = np.linalg.cholesky(Ky)    
    
    # polynomial basis
    B = polynomial_bases(X,N)
    A = np.dot(B.T,np.linalg.solve(Ky,B))
    AL = np.linalg.cholesky(A)
    
    # negative log likelihood
    z = np.linalg.solve(Ky,f)
    Bz = np.dot(B.T,z)
    #pdb.set_trace()
    r = np.dot(f.T,z) \
        - np.dot(Bz.T,np.linalg.solve(A,Bz)) \
        + np.sum(np.log(np.diag(L))) \
        + np.sum(np.log(np.diag(AL))) \
        + (M-B.shape[1])*np.log(2*np.pi)
    
    return 0.5*r

def polynomial_bases(X,N):
    M,m = X.shape
    I = index_set(N,m)
    n = I.shape[0]
    B = np.zeros((M,n))
    for i in range(n):
        ind = I[i,:]
        B[:,i] = np.prod(np.power(X,ind),axis=1)
    return B
    
if __name__ == '__main__':
    '''
    X1,X2 = np.meshgrid(np.linspace(-1.0,1.0,21),np.linspace(-1.0,1.0,21))
    X = np.hstack((X1.reshape((X1.size,1)),X2.reshape((X2.size,1))))
    f = np.sin(np.pi*np.sum(X,axis=1))
    e = np.array([1.0,0.5,0.1])
    Xstar = np.random.uniform(-1.0,1.0,size=(100,2))
    fstar,vstar = gaussian_process_regression(X,f,Xstar,e=e,gl=0.0,gu=100.0,N=5)
    '''
    X = np.linspace(-1.0,1.0,21).reshape((21,1))
    f = np.sin(np.pi*X)
    e = np.array([1.0,0.5,0.1])
    Xstar = np.random.uniform(-1.0,1.0,size=(100,1))
    fstar,vstar = gaussian_process_regression(X,f,Xstar,e=e,gl=0.0,gu=100.0,N=5)
    plt.plot(X,f,'k-',Xstar,fstar,'bx')
    plt.show()