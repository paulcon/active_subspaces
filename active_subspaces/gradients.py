import numpy as np
from utils.utils import process_inputs


def local_linear_gradients(X, f, p=None):
    # modify this to include a different set of points
    X, M, m = process_inputs(X)
    if M<=m: raise Exception('Not enough samples for local linear models.')

    if p is None:
        p = np.minimum(np.floor(1.7*m), M)
    elif p < m+1 or p > M: 
        raise Exception('p must be between m+1 and M')

    MM = np.minimum(int(np.ceil(6*m*np.log(m))), M-1)
    df = np.zeros((MM, m))
    for i in range(MM):
        ii = np.random.randint(M)
        x = X[ii,:]
        ind = np.argsort(np.sum((X - x)**2, axis=1))
        A = np.hstack((np.ones((p,1)), X[ind[1:p+1],:]))
        u = np.linalg.lstsq(A, f[ind[1:p+1]])[0]
        df[i,:] = u[1:].T
    return df
    
def finite_difference_gradients(X, simrun, h=1e-6):
    X, M, m = process_inputs(X)
    
    # points to run simulations
    XX = np.kron(np.ones((m+1, 1)),X) + \
        h*np.kron(np.vstack((np.zeros((1, m)), np.eye(m))), np.ones((M, 1)))
    
    # run the simulation
    F = simrun.run(XX)
    
    df = (F[M:].reshape((M, m)) - F[:M]) / h
    return df