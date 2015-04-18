import numpy as np

class Subspaces():
    eigenvalues = None
    eigenvectors = None
    W1 = None
    W2 = None
    e_br = None
    sub_br = None

    def compute(self, df, n_boot=200):
        
        if len(df.shape)!=2:
            raise Exception('df is not a 2d array.')
        
        if type(n_boot) is not int:
            raise Exception('n_boot must be an integer.')
        
        # compute eigenvalues and eigenvecs
        evals, evecs = spectral_decomposition(df)
        
        # compute bootstrap ranges for eigenvalues and subspace distances
        if n_boot > 0:
            e_br, sub_br = bootstrap_ranges(df, evals, evecs, n_boot=n_boot)
            self.e_br, self.sub_br = e_br, sub_br

        # partition the subspaces with a crappy heuristic
        n = compute_partition(evals)
        
        self.W1, self.W2 = evecs[:,:n], evecs[:,n:]
        self.eigenvalues, self.eigenvectors = evals, evecs
        
    def partition(self, n=0):
        self.W1, self.W2 = self.eigenvectors[:,:n], self.eigenvectors[:,n:]
        


def compute_partition(evals):
    
    # dealing with zeros for the log
    e = evals.copy()
    ind = e==0.0
    e[ind] = 1e-100
    
    # crappy threshold for choosing active subspace dimension
    return np.argmax(np.fabs(np.diff(np.log(e.reshape((e.size,)))))) + 1

def spectral_decomposition(df):

    # set integers
    M, m = df.shape
    
    # compute active subspace
    if M >= m:
        U, sig, W = np.linalg.svd(df, full_matrices=False)
    else:
        U, sig, W = np.linalg.svd(df, full_matrices=True)
        sig = np.hstack((np.array(sig), np.zeros(m-M)))
    e = (sig**2) / M
    W = W.T
    W = W*np.sign(W[0,:])
    return e.reshape((m,1)), W.reshape((m,m))

def bootstrap_ranges(df, e, W, n_boot=1000):
    
    # number of gradient samples and dimension
    M, m = df.shape

    # bootstrap
    e_boot = np.zeros((m, n_boot))
    sub_dist = np.zeros((m-1, n_boot))
    ind = np.random.randint(M, size=(M, n_boot))

    # can i parallelize this?
    for i in range(n_boot):
        e0, W0 = spectral_decomposition(df[ind[:,i],:])
        e_boot[:,i] = e0.reshape((m,))
        for j in range(m-1):
            sub_dist[j,i] = np.linalg.norm(np.dot(W[:,:j+1].T, W0[:,j+1:]), ord=2)

    e_br = np.zeros((m, 2))
    sub_br = np.zeros((m-1, 3))
    for i in range(m):
        e_br[i,0] = np.amin(e_boot[i,:])
        e_br[i,1] = np.amax(e_boot[i,:])
    for i in range(m-1):
        sub_br[i,0] = np.amin(sub_dist[i,:])
        sub_br[i,1] = np.mean(sub_dist[i,:])
        sub_br[i,2] = np.amax(sub_dist[i,:])

    return e_br, sub_br
