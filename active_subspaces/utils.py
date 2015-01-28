import numpy as np
import respsurf as rg
from discover import ActiveSubspacePlotter

def process_inputs(X):
    if len(X.shape) == 2:
        M,m = X.shape
    elif len(X.shape) == 1:
        M = X.shape
        m = 1
        X = X.reshape((M,m))
    else:
        raise Exception('Bad inputs.')
    return X,M,m

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
    asp = ActiveSubspacePlotter()
    y = np.dot(X,w)
    asp.sufficient_summary(y,f,out_label=out_label)
    
    # plot weights
    asp.eigenvectors(w,W_boot=w_boot,in_labels=in_labels,out_label=out_label)
    
    return w

def conditional_expectations(F,ind):
    n = int(np.amax(ind))+1
    EF,VF = np.zeros((n,1)),np.zeros((n,1))
    for i in range(n):
        f = F[ind==i]
        EF[i] = np.mean(f)
        VF[i] = np.var(f)
    return EF,VF
    
def quadratic_model_check(X,f,gamma,k):
    M,m = X.shape
    gamma = gamma.reshape((1,m))
    
    pr = rg.PolynomialRegression(2)
    pr.train(X,f)

    # get regression coefficients
    b,A = pr.g,pr.H
    
    # compute eigenpairs
    e,W = np.linalg.eig(np.outer(b,b.T) + \
        np.dot(A,np.dot(np.diagflat(gamma),A)))
    ind = np.argsort(e)[::-1]
    e,W = e[ind],W[:,ind]*np.sign(W[0,ind])
    
    return e[:k],W


    
    
    

