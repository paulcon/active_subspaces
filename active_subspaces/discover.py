import numpy as np
import matplotlib.pyplot as plt
import os

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

class ActiveSubspacePlotter():
    def __init__(self,figtype='eps'):
        self.figtype = '.' + figtype

        # make figs directory
        if not os.path.isdir('figs'):
            os.mkdir('figs')

        # set plot fonts
        myfont = {'family' : 'arial',
                'weight' : 'normal',
                'size'   : 14}
        plt.rc('font', **myfont)

    def eigenvalues(self,e,e_br=None,out_label=None):

        k = e.shape[0]
        if out_label is None:
            out_label = 'Output'

        plt.figure()
        plt.semilogy(range(1,k+1),e,'ko-')
        if e_br is not None:
            plt.fill_between(range(1,k+1),e_br[:,0],e_br[:,1],
                facecolor='0.7', interpolate=True)
        plt.xlabel('Index')
        plt.ylabel('Eigenvalues')
        plt.title(out_label)
        plt.grid(True)
        plt.xticks(range(1,k+1))
        if e_br is None:
            plt.axis([1,k,np.amin(e),np.amax(e)])
        else:
            plt.axis([1,k,np.amin(e_br[:,0]),np.amax(e_br[:,1])])
        figname = 'figs/evals_' + out_label + self.figtype
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.show()

    def subspace_errors(self,sub_br,out_label=None):

        kk = sub_br.shape[0]
        if out_label is None:
            out_label = 'Output'

        plt.figure()
        plt.semilogy(range(1,kk+1),sub_br[:,1],'ko-',markersize=12)
        plt.fill_between(range(1,kk+1),sub_br[:,0],sub_br[:,2],
            facecolor='0.7', interpolate=True)
        plt.xlabel('Subspace dimension')
        plt.ylabel('Subspace distance')
        plt.grid(True)
        plt.xticks(range(1,kk+1))
        plt.axis([1,kk,np.amin(sub_br[:,0]),1])
        figname = 'figs/subspace_' + out_label + self.figtype
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.show()

    def eigenvectors(self,W,W_boot=None,in_labels=None,out_label=None):

        n = len(W.shape)
        m = W.shape[0]
        # set labels for plots
        if in_labels is None:
            in_labels = [str(i) for i in range(1,m+1)]
        if out_label is None:
            out_label = 'Output'

        if n==1:
            plt.figure()
            if W_boot is not None:
                plt.plot(range(1,m+1),W_boot,color='0.7')
            plt.plot(range(1,m+1),W,'ko-',markersize=12)
            plt.ylabel('Weights')
            plt.grid(True)
            if m<=10:
                plt.xticks(range(1,m+1),in_labels,rotation='vertical')
                plt.margins(0.2)
                plt.subplots_adjust(bottom=0.15)
            plt.axis([1,m,-1,1])
            figname = 'figs/evecs_' + out_label + self.figtype
            plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
        else:
            plt.figure()
            for k in range(np.minimum(3,W.shape[1])):
                plt.plot(range(1,m+1),W[:,k],'o-',markersize=12,label='%d' % k)
            plt.ylabel('Eigenvectors')
            plt.grid(True)
            if m<=10:
                plt.xticks(range(1,m+1),in_labels,rotation='vertical')
                plt.margins(0.2)
                plt.subplots_adjust(bottom=0.15)
            plt.axis([1,m,-1,1])
            plt.legend(loc='best')
            figname = 'figs/evecs_' + out_label + self.figtype
            plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

        plt.show()

    def sufficient_summary(self,y,f,out_label=None):

        # check sizes of y
        n = len(y.shape)
        if n == 1:
            y1 = y
        else:
            y1 = y[:,0]
            y2 = y[:,1]

        # set labels for plots
        if out_label is None:
            out_label = 'Output'

        plt.figure()
        plt.plot(y1,f,'bo',markersize=12)
        plt.xlabel('Active variable')
        plt.ylabel(out_label)
        plt.grid(True)
        figname = 'figs/ssp1_' + out_label + self.figtype
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

        if n==2:

            plt.figure()
            plt.scatter(y1,y2,c=f,s=150.0,vmin=np.min(f),vmax=np.max(f))
            plt.xlabel('Active variable 1')
            plt.ylabel('Active variable 2')
            plt.title(out_label)
            plt.colorbar()
            figname = 'figs/ssp2_' + out_label + self.figtype
            plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

        plt.show()

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
