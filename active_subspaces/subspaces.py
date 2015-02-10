import numpy as np

class Subspaces():
    df = None
    k = None
    eigenvalues = None
    eigenvectors = None
    W1 = None
    W2 = None
    e_br = None
    sub_br = None

    def compute_spectral_decomposition(self,df,k=0):
        self.df = df

        if k==0: k=df.shape[1]
        self.k = k

        self.eigenvalues, self.eigenvectors = spectral_decomposition(df,k)

    def compute_bootstrap_ranges(self,n_boot=1000):
        self.e_br,self.sub_br = bootstrap_ranges(self.df, self.k, self.eigenvalues, self.eigenvectors)

    def partition(self,n):
        self.W1,self.W2 = self.eigenvectors[:,:n],self.eigenvectors[:,n:]

    # crappy threshold for choosing active subspace dimension
    def compute_partition(self):
        return np.argmax(np.fabs(np.diff(np.log(self.eigenvalues)))) + 1

def spectral_decomposition(df,k=0):

    # set integers
    M,m = df.shape
    if k==0: k=m

    # compute active subspace
    U,sig,W = np.linalg.svd(df,full_matrices=False)
    e = (sig[:k]**2)/M
    W = W.T
    W = W*np.sign(W[0,:])
    return e,W

def bootstrap_ranges(df, k, e, W, n_boot=1000):

    # set integers
    M,m = df.shape
    k_sub = np.minimum(k,m-1)

    # bootstrap
    e_boot = np.zeros((k,n_boot))
    sub_dist = np.zeros((k_sub,n_boot))
    ind = np.random.randint(M,size=(M,n_boot))

    # can i parallelize this?
    for i in range(n_boot):
        e0,W0 = spectral_decomposition(df[ind[:,i],:],k)
        e_boot[:,i] = e0[:k]
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

    return e_br, sub_br
