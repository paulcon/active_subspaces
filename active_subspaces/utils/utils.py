import numpy as np

class Normalizer():
    """
    Abstract Base Class for Normalizers
    """

    def normalize(self, X):
        """
        Description of normalize

        Arguments:
            X:
        Output:

        """
        raise NotImplementedError()

    def unnormalize(self, X):
        """
        Description of unnormalize

        Arguments:
            X:
        Output:

        """
        raise NotImplementedError()

class BoundedNormalizer(Normalizer):
    """
    Description of BoundedNormalizer
    """

    def __init__(self, lb, ub):
        """
        Arguments:
            lb:
            ub:
        """
        m = lb.size
        self.lb = lb.reshape((1, m))
        self.ub = ub.reshape((1, m))

    def normalize(self, X):
        """See Normalizer#normalize"""
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    def unnormalize(self, X):
        """See Normalizer#unnormalize"""
        return (self.ub - self.lb) * (X + 1.0) / 2.0 + self.lb

class UnboundedNormalizer(Normalizer):
    """
    Description of UnboundedNormalizer
    """

    def __init__(self, mu, C):
        """
        Arguments:
            mu:
            C:
        """
        m = mu.size
        self.mu = mu.reshape((1, m))
        self.L = np.linalg.cholesky(C)

    def normalize(self, X):
        """See Normalizer#normalize"""
        X0 = X - self.mu
        return np.linalg.solve(self.L,X0.T).T

    def unnormalize(self, X):
        """See Normalizer#unnormalize"""
        X0 = np.dot(X,self.L.T)
        return X0 + self.mu

def process_inputs(X):
    """
    Description of process_inputs.

    Arguments:
        X:
    Outputs:
        X:
        M:
        m:
    """

    if len(X.shape) == 2:
        M, m = X.shape
    else:
        raise ValueError('The inputs X should be a two-d numpy array.')
    return X, M, m

def conditional_expectations(f, ind):
    """
    Description of conditional_expectations.

    Arguments:
        f:
        ind:
    Outputs:
        Ef:
        Vf:
    """
    n = int(np.amax(ind)) + 1
    Ef, Vf = np.zeros((n, 1)), np.zeros((n, 1))
    for i in range(n):
        fi = f[ind == i]
        Ef[i] = np.mean(fi)
        Vf[i] = np.var(fi)
    return Ef, Vf

# thanks to Trent for these functions!!!
def atleast_2d_col(A):
    return atleast_2d(A,'col')

def atleast_2d_row(A):
    return atleast_2d(A,'row')

def atleast_2d(A,oned_as='row'):
    ''' ensures A is an array and at least of rank 2
    '''
    
    # not an array yet
    if not isinstance(A,(np.ndarray,np.matrixlib.defmatrix.matrix)):
        if not isinstance(A,(list,tuple)):
            A = [A]
        A = np.array(A)
        
    # check rank
    if np.ndim(A) < 2:
        # expand row or col
        if oned_as == 'row':
            A = A[None,:]
        elif oned_as == 'col':
            A = A[:,None]
        else:
            raise Exception , "oned_as must be 'row' or 'col' "
            
    return A





