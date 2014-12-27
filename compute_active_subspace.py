import numpy as np

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