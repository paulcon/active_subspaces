import numpy as np

class Subspaces():
    df = None
    eigenvalues = None
    eigenvectors = None
    W1 = None
    W2 = None
    e_br = None
    sub_br = None

    def compute_spectral_decomposition(self, df):
        self.df = df
        self.eigenvalues, self.eigenvectors = spectral_decomposition(df)

    def compute_bootstrap_ranges(self, n_boot=1000):
        self.e_br, self.sub_br = bootstrap_ranges(self.df, self.eigenvalues, \
                                                self.eigenvectors, n_boot=n_boot)

    def partition(self, n=0):
        if n==0:
            n = compute_partition(self.eigenvalues)
        self.W1, self.W2 = self.eigenvectors[:,:n], self.eigenvectors[:,n:]


def compute_partition(e):
    # crappy threshold for choosing active subspace dimension
    return np.argmax(np.fabs(np.diff(np.log(e)))) + 1

def spectral_decomposition(df):

    # set integers
    M, m = df.shape

    # compute active subspace
    U, sig, W = np.linalg.svd(df, full_matrices=False)
    e = (sig**2) / M
    W = W.T
    W = W*np.sign(W[0,:])
    return e, W

def bootstrap_ranges(df, e, W, n_boot=1000):

    # set integers
    M, m = df.shape
    k = e.shape[0]

    # bootstrap
    e_boot = np.zeros((k, n_boot))
    sub_dist = np.zeros((k-1, n_boot))
    ind = np.random.randint(M, size=(M, n_boot))

    # can i parallelize this?
    for i in range(n_boot):
        e0, W0 = spectral_decomposition(df[ind[:,i],:])
        e_boot[:,i] = e0[:k]
        for j in range(k-1):
            sub_dist[j,i] = np.linalg.norm(np.dot(W[:,:j+1].T, W0[:,j+1:]), ord=2)

    e_br = np.zeros((k, 2))
    sub_br = np.zeros((k-1, 3))
    for i in range(k):
        e_br[i,0] = np.amin(e_boot[i,:])
        e_br[i,1] = np.amax(e_boot[i,:])
    for i in range(k-1):
        sub_br[i,0] = np.amin(sub_dist[i,:])
        sub_br[i,1] = np.mean(sub_dist[i,:])
        sub_br[i,2] = np.amax(sub_dist[i,:])

    return e_br, sub_br
