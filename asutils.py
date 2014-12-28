import numpy as np
from analyze_active_subspace import \
    sufficient_summary_plot,plot_eigenvectors

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
    lam,W = np.linalg.eig(np.outer(b,b.T) + np.dot(A,np.dot(np.diagflat(gamma),A)))
    ind = np.argsort(lam)[::-1]
    lam = lam[ind]
    W = W[:,ind]
    W = W*np.tile(np.sign(W[0,:]),(m,1))
    return lam,W
    
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
    lam,W = get_eigenpairs(b,A,gamma)
    
    # bootstrap
    sub_dist = np.zeros((m-1,n_boot))
    lam_boot = np.zeros((m,n_boot))
    ind = np.random.randint(M,size=(M,n_boot))
    for i in range(n_boot):
        b0,A0 = quadreg(Xsq[ind[:,i],:],indices,f[ind[:,i]])
        lam0,W0 = get_eigenpairs(b0,A0,gamma)
        lam_boot[:,i] = lam0
        for j in range(m-1):
            sub_dist[j,i] = np.linalg.norm(np.dot(W[:,:j+1].T,W0[:,j+1:]),ord=2)

    lam_br = np.zeros((k,2))
    sub_br = np.zeros((k_sub,3))
    for i in range(k):
        lam_sort = np.sort(lam_boot[i,:])
        lam_br[i,0] = lam_sort[np.floor(0.025*n_boot)]
        lam_br[i,1] = lam_sort[np.ceil(0.925*n_boot)]
    for i in range(k_sub):
        sub_sort = np.sort(sub_dist[i,:])
        sub_br[i,0] = sub_sort[np.floor(0.025*n_boot)]        
        sub_br[i,1] = np.mean(sub_sort)
        sub_br[i,2] = sub_sort[np.ceil(0.925*n_boot)]
    return lam[:k],W,lam_br,sub_br


    
    
    

