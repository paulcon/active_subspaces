"""Computing subspaces from compressed measurements of the gradients."""
import numpy as np

def simple_projection(dfm, E):
    """
    Qi and Hughes
    
    Assumes M > m > k. Probably want to relax this.
    
    Inputs
    dfm is gradient measurements (M, k)
    E is measurement matrices (m, k, M)
    
    Outputs
    W is left singular vectors
    V is right singular vectors
    s is singular values
    """
    
    # m is full vector size, k is number of measurements, M is number of samples
    m, k, M = E.shape
    
    # projections
    G = np.zeros((m, M))
    
    # build matrix for projection
    # (faster way to do this?)
    for i in range(M):
        
        E0 = E[:,:,i]
        G[:,i] = np.dot(E0, np.linalg.solve(np.dot(E0.transpose(), E0), \
                        dfm[i,:].reshape((k, 1)))).reshape((m, ))

    # get singular values and vectors
    W, sig, VT = np.linalg.svd(G, full_matrices=0)
    W = W*np.sign(W[0,:])
    V = VT.transpose()
    V = V*np.sign(V[0,:])
    s = (sig.reshape((m, 1)))/M
    
    return W, V, s

def alternating_minimization(dfm, E, r, B0=None):
    """
    Constantine, Eftekhari, Wakin
    
    Assumes M > m > k. Probably want to relax this.
    
    Inputs
    dfm is gradient measurements (M, k)
    E is measurement matrices (m, k, M)
    r is rank of approximation
    B0 is initial guess of B factor in A B^T
    
    Not inputs but intermediates
    A is (m, r)
    B is (M, r)
    
    Outputs
    W is left singular vectors
    V is right singular vectors
    s is singular values
    """
    
    # m is full vector size, k is number of measurements, M is number of samples
    m, k, M = E.shape
    
    if B0 is None:
        B0 = simple_projection(dfm, E)[1]

    B0 = B0[:,:r]
    restol = 1e-3
    maxcount = 100
    stalltol = 1e-5
    
    res = 1e9
    res5 = np.random.uniform(-1., 1., size=(5, ))
    count = 0
    while res > restol and count < maxcount and np.std(res5) > stalltol:
        
        A = _solve_A(B0, dfm, E)
        B = _solve_B(A, dfm, E)
        
        res = _residual(dfm, E, A, B)
        print 'Residual: {:6.4e}'.format(res)
        res5[np.mod(count,5)] = res
        count += 1
        B0 = B
        
    G = np.dot(A, B.transpose())
        
    # get singular values and vectors
    W, sig, VT = np.linalg.svd(G, full_matrices=0)
    W = W*np.sign(W[0,:])
    V = VT.transpose()
    V = V*np.sign(V[0,:])
    s = (sig.reshape((m, 1)))/M
    
    return W, V, s
        
    
def _residual(dfm, E, A, B):
    return np.linalg.norm(dfm - _measurement_operator(np.dot(A, B.transpose()), E), ord='fro') / \
        np.linalg.norm(dfm, ord='fro')
    
def _measurement_operator(X, E):
    
    m, k, M = E.shape
    Y = np.zeros((M, k))
    for i in range(M):
        E0 = E[:,:,i]
        Y[i,:] = np.dot(E0.transpose(), X[:,i]).reshape((k, ))
    return Y
    
def _solve_B(A, dfm, E):
    m, k, M = E.shape
    r = A.shape[1]
    
    B = np.zeros((M, r))
    for i in range(M):
        y = dfm[i,:].reshape((k, 1))
        E0 = E[:,:,i]
        P = np.dot(E0.transpose(), A)
        b = np.linalg.lstsq(P, y)[0]
        B[i,:] = b.reshape((r, ))
    
    return B

def _solve_A(B, dfm, E):
    m, k, M = E.shape
    r = B.shape[1]
    
    T = np.zeros((M*k, r*m))
    for i in range(M):
        E0 = E[:,:,i]
        T[i*k:(i+1)*k,:] = np.kron(B[i,:].reshape((1, r)), E0.transpose())
    
    vecA = np.linalg.lstsq(T, _vec(dfm.transpose()))[0]
    return _unvec(vecA, m, r)
    
def _vec(X):
    m, n = X.shape
    return X.reshape((m*n, 1), order='F')
    
def _unvec(x, m, n):
    return x.reshape((m, n), order='F')
    
def _gauss_measurements(m, k, M):
    return np.random.normal(size=(m, k, M))

def _bernoulli_measurements(m, k, M):
    return 2*(np.random.uniform(size=(m, k, M)) > 0.5).astype(float)-1.
    
def _orthogonal_measurements(m, k, M):
    A = np.random.normal(size=(m, k, M))
    for i in range(M):
        A[:,:,i] = np.linalg.qr(A[:,:,i])[0]
    return A
    
def gradient_measurements(sr, X, E):
    m, k, M = E.shape
    
    dfm = np.zeros((M, k))
    
    # get basic runs
    f0 = sr.run(X)
    
    h = 1e-6;
    for i in range(M):
        E0 = E[:,:,i]
        Xp = X[i,:].reshape((1,m)) + h*E0.transpose()
        fp = sr.run(Xp)
        dfm[i,:] = (fp - f0[i,0]).reshape((k, )) / h
        
    return dfm
        
    
    
    
    
    
    
    
    