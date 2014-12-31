import numpy as np

def r_hermite(N):
    if N<=0:
        raise Exception('Parameters out of range.')
    if N==1:
        return np.array([[0.0,1.0]])
    else:
         n = np.array(range(1,N+1))
         B = np.vstack((1.0,0.5*n.reshape((N,1))))
         A = np.zeros(B.shape)
         return np.hstack((A,B))
         
def jacobi_matrix(ab):
    n = ab.shape[0]-1
    if n==0:
        return ab[0,0]
    else:
        J = np.zeros((n,n))
        J[0,0] = ab[0,0]
        J[0,1] = np.sqrt(ab[1,1])
        for i in range(1,n-1):
            J[i,i] = ab[i,0]
            J[i,i-1] = np.sqrt(ab[i,1])
            J[i,i+1] = np.sqrt(ab[i+1,1])
        J[n-1,n-1] = ab[n-1,0]
        J[n-1,n-2] = np.sqrt(ab[n-1,1])
        return J
        
def gh1d(N):
    if N>1:
        ab = r_hermite(N)
        J = jacobi_matrix(ab)
        e,V = np.linalg.eig(J)
        ind = np.argsort(e)
        x = e[ind].reshape((N,1))
        x[np.fabs(x)<1e-12] = 0.0
        w = (V[0,ind]*V[0,ind]).reshape((N,1))
    else:
        x,w = np.array([[0.0]]),np.array([[1.0]])
    return x,w
    
def gauss_hermite(N):
    if len(N)==1:
        return gh1d(N[0])
    else:
        x = np.array([[1.0]])
        w = np.array([[1.0]])
        for n in N:
            xi,wi = gh1d(n)
            
            xL = np.kron(x.copy(),np.ones(xi.shape))
            xU = np.kron(np.ones((x.shape[0],1)),xi)
            x = np.hstack((xL,xU))
            w = np.kron(w.copy(),wi)
        return x[:,1:],w