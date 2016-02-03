"""Utilities for computing active and inactive subspaces."""
from __future__ import division
import numpy as np
import logging
from utils.misc import process_inputs
import utils.quadrature as quad

class Subspaces():
    """
    A class for computing active and inactive subspaces.

    :cvar ndarray eigenvalues: m-by-1 matrix where m is the dimension of the input space.
    :cvar ndarray eigenvectors: m-by-m matrix that contains the eigenvectors oriented column-wise.
    :cvar ndarray W1: m-by-n matrix that contains the basis for the active subspace.
    :cvar ndarray W2: m-by-(m-n) matix that contains the basis for the inactive subspaces.
    :cvar ndarray e_br: m-by-2 matrix that contains the bootstrap ranges for the eigenvalues.
    :cvar ndarray sub_br: m-by-3 matrix that contains the bootstrap ranges (first and third column) and the mean (second column) of the error in the estimated subspaces approximated by bootstrap

    **Notes**

    The attributes `W1` and `W2` are convenience variables. They are identical
    to the first n and last (m-n) columns of `eigenvectors`, respectively.
    """

    eigenvalues, eigenvectors = None, None
    W1, W2 = None, None
    e_br, sub_br = None, None

    def compute(self, df ,f = 0, X = 0,function=0, c_index = 0, comp_flag =0,N=5, n_boot=200):
        """
        Compute the active and inactive subspaces from a collection of
        sampled gradients.

        :param ndarray df: an ndarray of size M-by-m that contains evaluations of the gradient.
        :param ndarray f: an ndarray of size M that contains evaluations of the function.
        :param ndarray X: an ndarray of size M-by-m that contains data points in the input space.
        :param function: a specified function that outputs f(x), and df(x) the gradient vector for a data point x
        :param int c_index: an integer specifying which C matrix to compute, the default matrix is 0.
        :param int comp_flag: an integer specifying computation method: 0 for monte carlo, 1 for LG quadrature.
        :param int N: number of quadrature points per dimension.
        :param int n_boot: number of bootstrap replicates to use when computing bootstrap ranges.

        **Notes**

        This method sets the class's attributes `W1`, `W2`, `eigenvalues`, and
        `eigenvectors`. If `n_boot` is greater than zero, then this method
        also runs a bootstrap to compute and set `e_br` and `sub_br`.
        """
        
        if c_index != 4:
            df, M, m = process_inputs(df)
        else:
            M = np.shape(X)[0]
            m = np.shape(X)[1]/2
        if not isinstance(n_boot, int):
            raise TypeError('n_boot must be an integer.')
        evecs = np.zeros((m,m))
        evals = np.zeros(m)
        e_br = np.zeros((m,2))
        sub_br = np.zeros((m-1,3))
        # compute eigenvalues and eigenvecs
        if c_index == 0:
            logging.getLogger('PAUL').info('Computing spectral decomp with {:d} samples in {:d} dims.'.format(M, m))
            evals, evecs = spectral_decomposition(df=df)
            if comp_flag == 0:
                # compute bootstrap ranges for eigenvalues and subspace distances
                if n_boot > 0:
                    logging.getLogger('PAUL').info('Bootstrapping {:d} spectral decomps of size {:d} by {:d}.'.format(n_boot, M, m))
                    e_br, sub_br = bootstrap_ranges(df, evals, evecs, n_boot=n_boot)
        elif c_index == 1:  
            if comp_flag == 0:
                evals, evecs = spectral_decomposition(df,f,X,c_index=c_index,comp_flag=comp_flag)
                # compute bootstrap ranges for eigenvalues and subspace distances
                if n_boot > 0:
                    logging.getLogger('PAUL').info('Bootstrapping {:d} spectral decomps of size {:d} by {:d}.'.format(n_boot, M, m))
                    e_br, sub_br = bootstrap_ranges(df, evals, evecs,f, X, c_index,n_boot)
            elif comp_flag == 1:
                evals, evecs = spectral_decomposition(df,f,X,function,c_index,N,comp_flag)        
        elif c_index == 2:
            if comp_flag == 0:
                evals, evecs = spectral_decomposition(df,f,X,c_index=c_index,comp_flag=comp_flag)
                # compute bootstrap ranges for eigenvalues and subspace distances
                if n_boot > 0:
                    logging.getLogger('PAUL').info('Bootstrapping {:d} spectral decomps of size {:d} by {:d}.'.format(n_boot, M, m))
                    e_br, sub_br = bootstrap_ranges(df, evals, evecs,f, X, c_index,n_boot)
            elif comp_flag == 1:
                evals, evecs = spectral_decomposition(df,f,X,function,c_index,N,comp_flag)
                
        elif c_index == 3:
            if comp_flag == 0:
                evals, evecs = spectral_decomposition(df,f,X,c_index=c_index,comp_flag=comp_flag)
                # compute bootstrap ranges for eigenvalues and subspace distances
                if n_boot > 0:
                    logging.getLogger('PAUL').info('Bootstrapping {:d} spectral decomps of size {:d} by {:d}.'.format(n_boot, M, m))
                    e_br, sub_br = bootstrap_ranges(df, evals, evecs,f, X, c_index,n_boot)
            elif comp_flag == 1:
                evals, evecs = spectral_decomposition(df,f,X,function,c_index,N,comp_flag)
        elif c_index == 4:
            if comp_flag == 0:
                evals, evecs = spectral_decomposition(df,f,X,c_index=c_index,comp_flag=comp_flag)
                # compute bootstrap ranges for eigenvalues and subspace distances
                if n_boot > 0:
                   # logging.getLogger('PAUL').info('Bootstrapping {:d} spectral decomps of size {:d} by {:d}.'.format(n_boot, M, 2*m))
                    e_br, sub_br = bootstrap_ranges(df,evals, evecs,f, X, c_index,n_boot)
            elif comp_flag == 1:
                evals, evecs = spectral_decomposition(df,f,X,function,c_index,N,comp_flag)   
        self.e_br, self.sub_br = e_br, sub_br    
        self.e_br, self.sub_br = e_br, sub_br    
        self.eigenvalues, self.eigenvectors = evals, evecs

        

        # partition the subspaces with a crappy heuristic
        n = compute_partition(evals)
        self.partition(n)


    def partition(self, n):
        """
        Set the partition between active and inactive subspaces.

        :param int n: dimension of the active subspace.

        **Notes**

        This method sets the class's attributes `W1` and `W2` according to the
        given `n`. It is mostly a convenience method in case one wants to
        manually set the dimension of the active subspace after the basis is
        computed.
        """
        if not isinstance(n, int):
            raise TypeError('n should be an integer')

        m = self.eigenvectors.shape[0]
        if n<1 or n>m:
            raise ValueError('n must be positive and less than the dimension of the eigenvectors.')

        logging.getLogger('PAUL').info('Active subspace dimension is {:d} out of {:d}.'.format(n, m))

        self.W1, self.W2 = self.eigenvectors[:,:n], self.eigenvectors[:,n:]

def compute_partition(evals):
    """
    A heuristic based on eigenvalue gaps for deciding the dimension of the
    active subspace.

    :param ndarray evals: the eigenvalues.

    :return: dimension of the active subspace
    :rtype: int
    """
    # dealing with zeros for the log
    e = evals.copy()
    ind = e==0.0
    e[ind] = 1e-100

    # crappy threshold for choosing active subspace dimension
    n = np.argmax(np.fabs(np.diff(np.log(e.reshape((e.size,)))))) + 1
    return n

def spectral_decomposition(df,f=0,X=0,function=0,c_index=0,N=5,comp_flag=0):
    """
    Use the SVD to compute the eigenvectors and eigenvalues for the
    active subspace analysis.

    :param ndarray df: an ndarray of size M-by-m that contains evaluations of the gradient.
    :param ndarray f: an ndarray of size M that contains evaluations of the function.
    :param ndarray X: an ndarray of size M-by-m that contains data points in the input space.
    :param function: a specified function that outputs f(x), and df(x) the gradient vector for a data point x
    :param int c_index: an integer specifying which C matrix to compute, the default matrix is 0.
    :param int comp_flag: an integer specifying computation method: 0 for monte carlo, 1 for LG quadrature.
    :param int N: number of quadrature points per dimension.
    
    :return: [e, W], [ eigenvalues, eigenvectors ]
    :rtype: [ndarray, ndarray]

    **Notes**

    If the number M of gradient samples is less than the dimension m of the
    inputs, then the method builds an arbitrary basis for the nullspace, which
    corresponds to the inactive subspace.
    """
    # set integers
    if c_index != 4:
        df, M, m = process_inputs(df)
    else:
        M = int(np.shape(X)[0])
        m = int(np.shape(X)[1]/2)
    W = np.zeros((m,m))
    e = np.zeros((m,1))
    C = np.zeros((m,m))
    norm_tol = np.finfo(5.0).eps**(0.5)
    # compute active subspace
    if c_index == 0 and comp_flag == 0:
        if M >= m:
            U, sig, W = np.linalg.svd(np.dot(df.T,df), full_matrices=False)
        else:
            U, sig, W = np.linalg.svd(np.dot(df.T,df), full_matrices=True)
            sig = np.hstack((np.array(sig), np.zeros(m-M)))
        e = (sig**2) / M
    elif c_index == 0 and comp_flag == 1:
        xx = (np.ones(m)*N).astype(np.int64).tolist()  
        x,w = quad.gauss_legendre(xx)
        C = np.zeros((m,m))
        N = np.size(w)
        for i in range(0,N):
            [f,DF] = function(x[i,:])
            DF = DF.reshape((m,1))
            C = C + (np.dot(DF,DF.T))*w[i]
        U, sig, W = np.linalg.svd(C, full_matrices=True)
        e = (sig**2)
    elif c_index == 1 and comp_flag == 0:
        C =  (np.dot(X.T,df) + np.dot(df.T,X))/M
        U, sig, W = np.linalg.svd(C, full_matrices=True)
        e = (sig**2)
    elif c_index == 1 and comp_flag == 1:
        xx = (np.ones(m)*N).astype(np.int64).tolist()  
        x,w = quad.gauss_legendre(xx)
        C = np.zeros((m,m))
        N = np.size(w)
        for i in range(0,N):
            [f,DF] = function(x[i,:])
            xxx = x[i,:].reshape((m,1))
            DF = DF.reshape((m,1))
            C = C + (np.dot(xxx,DF.T) + np.dot(DF,xxx.T))*w[i]
        U, sig, W = np.linalg.svd(C, full_matrices=True)
        e = (sig**2)
    elif c_index == 2 and comp_flag == 0:
        row_count = np.shape(df)[0]
        i=0
        while (i< row_count):
            norm = np.linalg.norm(df[i,:])
            if( norm < norm_tol):
                df = np.delete(df,(i),axis=0)
                row_count = row_count-1
            else:
                df[i,:] = df[i,:]/norm
                i = i+1
        C =  np.dot(df.T,df)/M
        U, sig, W = np.linalg.svd(C, full_matrices=True)
        e = (sig**2)
    elif c_index == 2 and comp_flag == 1:
        xx = (np.ones(m)*N).astype(np.int64).tolist()  
        x,w = quad.gauss_legendre(xx)
        C = np.zeros((m,m))
        N = np.size(w)
        for i in range(0,N):
            [f,DF] = function(x[i,:])
            DF = DF.reshape((1,m))
            norm = np.linalg.norm(DF)
            if(norm > norm_tol):
                DF = DF/np.linalg.norm(DF)
                C = C + np.dot(DF.T,DF)*w[i]
        U, sig, W = np.linalg.svd(C, full_matrices=True)
        e = (sig**2)
    elif c_index == 3 and comp_flag == 0: 
        for i in range(0,M):
            xxx = X[i,:]
            DF = df[i,:]
            if(np.linalg.norm(xxx) < norm_tol):
                xxx = np.zeros(m)
            else:
                xxx = xxx/np.linalg.norm(xxx)
            if(np.linalg.norm(DF) < norm_tol):
                DF = np.zeros(m)
            else: 
                DF = DF/np.linalg.norm(DF)
            df[i,:] = DF
            X[i,:] = xxx
        C = C + (np.dot(df.T,X) + np.dot(X.T,df))/M
        U, sig, W = np.linalg.svd(C, full_matrices=True)
        e = (sig**2)
    elif c_index == 3 and comp_flag == 1:
        xx = (np.ones(m)*N).astype(np.int64).tolist()  
        x,w = quad.gauss_legendre(xx)
        C = np.zeros((m,m))
        N = np.size(w)
        for i in range(0,N):
            [f,DF] = function(x[i,:])
            xxx = x[i,:].reshape((m,1))
            if(np.linalg.norm(xxx) < norm_tol):
                xxx = np.zeros((m,1))
            else:
                xxx = xxx/np.linalg.norm(xxx)
            DF = DF.reshape((m,1))
            if(np.linalg.norm(DF) < norm_tol):
                DF = np.zeros(m)
            else:
               # print('shape xxx= ',np.shape(xxx),'shape DF = ', np.shape(DF))
                DF = DF/np.linalg.norm(DF)
            C = C + (np.dot(xxx,DF.T) + np.dot(DF,xxx.T))*w[i]
        U, sig, W = np.linalg.svd(C, full_matrices=True)
        e = (sig**2)
    elif c_index == 4 and comp_flag == 0:
        x = X[:,:m]
        y = X[:,m:]
        A = (f[:M]-f[M:])**2
        for i in range(0,M):
            vec = (x[i,:]-y[i,:]).reshape((m,1))
            if np.linalg.norm(vec) > norm_tol:
                vec =  (vec)/np.linalg.norm(vec)
                C = C + A[i]*np.dot(vec,vec.T)/M
        U, sig, W = np.linalg.svd(C, full_matrices=True)
        e = (sig**2)
    elif c_index == 4 and comp_flag == 1:
        xx = (np.ones(2*m)*N).astype(np.int64).tolist()  
        x,w = quad.gauss_legendre(xx)
        N = np.size(w)
        for i in range(0,N):
            norm = np.linalg.norm(x[i,:m]-x[i,m:]) 
            if( norm > norm_tol):
                xxx = x[i,:m].reshape((m,1))
                yyy = x[i,m:].reshape((m,1))
                [f_x,DF] = function(xxx)
                [f_y,DF] = function(yyy)
                C = C + w[i]*np.dot((xxx-yyy),(xxx-yyy).T)*(f_x-f_y)**2/norm**2
        [evals,WW] = np.linalg.eig(C)
        order = np.argsort(evals)
        order = np.flipud(order)
        e = evals[order]
        W = np.zeros((m,m))
        for jj in range(0,m):
            W[:,jj] = WW[:,order[jj]]
    W = W.T
    W = W*np.sign(W[0,:])
    return e.reshape((m,1)), W.reshape((m,m))

def bootstrap_ranges(df, e, W,f=0,X=0,c_index=0,n_boot=200):
    """
    Use a nonparametric bootstrap to estimate variability in the computed
    eigenvalues and subspaces.

    :param ndarray df: M-by-m matrix of evaluations of the gradient.
    :param ndarray e: m-by-1 vector of eigenvalues.
    :param ndarray W: eigenvectors.
    :param ndarray f: M-by-1 vector of function evaluations.
    :param ndarray X: M-by-m array for c_index = 0,1,2,3 *******OR******** M-by-2m matrix for c_index = 4.
    :param int c_index: an integer specifying which C matrix to compute, the default matrix is 0
    :param int n_boot: index number for alternative subspaces.
    
    :return: [e_br, sub_br], e_br: m-by-2 matrix that contains the bootstrap ranges for the eigenvalues, sub_br: m-by-3 matrix that contains the bootstrap ranges (first and third column) and the mean (second column) of the error in the estimated subspaces approximated by bootstrap
    :rtype: [ndarray, ndarray]

    **Notes**

    The mean of the subspace distance bootstrap replicates is an interesting
    quantity. Still trying to figure out what its precise relation is to
    the subspace error. They seem to behave similarly in test problems. And an
    initial "coverage" study suggested that they're unbiased for a quadratic
    test problem. Quadratics, though, may be special cases.
    """
    # number of gradient samples and dimension
    if c_index != 4:
        df, M, m = process_inputs(df)
    else:
        M = int(np.shape(X)[0])
        m = int((np.shape(X)[1]/2))
        
    # bootstrap
    e_boot = np.zeros((m, n_boot))
    sub_dist = np.zeros((m-1, n_boot))
    ind = np.random.randint(M, size=(M, n_boot))

    # can i parallelize this?
    for i in range(n_boot):
        if c_index == 0:
            e0, W0 = spectral_decomposition(df[ind[:,i],:])
        elif c_index == 1:
            e0, W0 = spectral_decomposition(df=df[ind[:,i],:],f=f,X=X[ind[:,i],:],c_index=c_index)
        elif c_index == 2:
            e0, W0 = spectral_decomposition(df[ind[:,i],:],c_index=c_index)
        elif c_index == 3:
            e0, W0 = spectral_decomposition(df[ind[:,i],:],f,X=X[ind[:,i],:],c_index=c_index)
        elif c_index == 4:
            f_x = f[ind[:,i]]
            f_y = f[ind[:,i]+M]
            f = np.append([[f_x]],[[f_y]],axis=0)
            f = f.reshape(2*np.size(f_x))
            e0, W0 = spectral_decomposition(df=0,f=f,X=X[ind[:,i],:],c_index=c_index)
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
