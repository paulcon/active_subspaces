"""Utilities for computing active and inactive subspaces."""
import numpy as np
import logging
from utils.misc import process_inputs

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

    def compute(self, df, n_boot=200):
        """
        Compute the active and inactive subspaces from a collection of
        sampled gradients.

        :param ndarray df: an ndarray of size M-by-m that contains evaluations of the gradient.
        :param int n_boot: number of bootstrap replicates to use when computing bootstrap ranges.

        **Notes**

        This method sets the class's attributes `W1`, `W2`, `eigenvalues`, and
        `eigenvectors`. If `n_boot` is greater than zero, then this method
        also runs a bootstrap to compute and set `e_br` and `sub_br`.
        """
        df, M, m = process_inputs(df)

        if not isinstance(n_boot, int):
            raise TypeError('n_boot must be an integer.')

        # compute eigenvalues and eigenvecs
        logging.getLogger('PAUL').info('Computing spectral decomp with {:d} samples in {:d} dims.'.format(M, m))
        evals, evecs = spectral_decomposition(df)
        self.eigenvalues, self.eigenvectors = evals, evecs

        # compute bootstrap ranges for eigenvalues and subspace distances
        if n_boot > 0:
            logging.getLogger('PAUL').info('Bootstrapping {:d} spectral decomps of size {:d} by {:d}.'.format(n_boot, M, m))
            e_br, sub_br = bootstrap_ranges(df, evals, evecs, n_boot=n_boot)
            self.e_br, self.sub_br = e_br, sub_br

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

def spectral_decomposition(df):
    """
    Use the SVD to compute the eigenvectors and eigenvalues for the
    active subspace analysis.

    :param ndarray df: ndarray of size M-by-m that contains evaluations of the gradient.

    :return: [e, W], [ eigenvalues, eigenvectors ]
    :rtype: [ndarray, ndarray]

    **Notes**

    If the number M of gradient samples is less than the dimension m of the
    inputs, then the method builds an arbitrary basis for the nullspace, which
    corresponds to the inactive subspace.
    """
    # set integers
    df, M, m = process_inputs(df)

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

def bootstrap_ranges(df, e, W, n_boot=200):
    """
    Use a nonparametric bootstrap to estimate variability in the computed
    eigenvalues and subspaces.

    :param ndarray df: M-by-m matrix of evaluations of the gradient.
    :param ndarray e: m-by-1 vector of eigenvalues
    :param ndarray W: eigenvectors
    :param int n_boot: number of bootstrap replicates to use when computing bootstrap ranges.

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
