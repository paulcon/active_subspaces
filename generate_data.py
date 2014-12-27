import numpy as np

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
    
    