"""Miscellaneous utilities."""
import numpy as np

class Normalizer():
    """An abstract class for normalizing inputs.
    
    """
    def normalize(self, X):
        """Return corresponding points in normalized domain.

        Parameters
        ----------
        X : ndarray
            contains all input points one wishes to normalize

        Returns
        -------
        X_norm : ndarray
            contains the normalized inputs corresponding to `X`
            
        Notes
        -----
        Points in `X` should be oriented as an m-by-n ndarray, where each row
        corresponds to an m-dimensional point in the problem domain.
        """
        raise NotImplementedError()

    def unnormalize(self, X):
        """Return corresponding points shifted and scaled to [-1,1]^m.

        Parameters
        ----------
        X : ndarray 
            contains all input points one wishes to unnormalize

        Returns
        -------
        X_unnorm : ndarray 
            contains the unnormalized inputs corresponding to `X`
            
        Notes
        -----
        Points in `X` should be oriented as an m-by-n ndarray, where each row
        corresponds to an m-dimensional point in the normalized domain.
        """
        raise NotImplementedError()

class BoundedNormalizer(Normalizer):
    """A class for normalizing bounded inputs. 
    
    Attributes
    ----------
    lb : ndarray
        a matrix of size m-by-1 that contains lower bounds on the simulation 
        inputs
    ub : ndarray
        a matrix of size m-by-1 that contains upper bounds on the simulation 
        inputs

    See Also
    --------
    utils.misc.UnboundedNormalizer
    """
    lb, ub = None, None

    def __init__(self, lb, ub):
        """Initialize a BoundedNormalizer.

        Parameters
        ----------
        lb : ndarray
            a matrix of size m-by-1 that contains lower bounds on the simulation
            inputs
        ub : ndarray
            a matrix of size m-by-1 that contains upper bounds on the simulation
            inputs
        """
        m = lb.size
        self.lb = lb.reshape((1, m))
        self.ub = ub.reshape((1, m))

    def normalize(self, X):
        """Return corresponding points shifted and scaled to [-1,1]^m.

        Parameters
        ----------
        X : ndarray
            contains all input points one wishes to normalize. The shape of `X` 
            is M-by-m. The components of each row of `X` should be between `lb` 
            and `ub`.

        Returns
        -------
        X_norm : ndarray
            contains the normalized inputs corresponding to `X`. The components 
            of each row of `X_norm` should be between -1 and 1.
        """
        X, M, m = process_inputs(X)
        X_norm = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return X_norm

    def unnormalize(self, X):
        """Return corresponding points shifted and scaled to `[lb, ub]`.

        Parameters
        ----------
        X : ndarray 
            contains all input points one wishes to unnormalize. The shape of 
            `X` is M-by-m. The components of each row of `X` should be between 
            -1 and 1.

        Returns
        -------
        X_unnorm : ndarray 
            contains the unnormalized inputs corresponding to `X`. The 
            components of each row of `X_unnorm` should be between `lb` and 
            `ub`.
        """
        X, M, m = process_inputs(X)
        X_unnorm = (self.ub - self.lb) * (X + 1.0) / 2.0 + self.lb
        return X_unnorm

class UnboundedNormalizer(Normalizer):
    """A class for normalizing unbounded, Gaussian inputs to standard normals.
    
    Attributes
    ----------
    mu : ndarray 
        a matrix of size m-by-1 that contains the mean of the Gaussian 
        simulation inputs
    L : ndarray 
        a matrix size m-by-m that contains the Cholesky factor of the covariance
        matrix of the Gaussian simulation inputs.

    See Also
    --------
    utils.misc.BoundedNormalizer

    Notes
    -----
    A simulation with unbounded inputs is assumed to have a Gaussian weight
    function associated with the inputs. The covariance of the Gaussian weight
    function should be full rank.
    """
    mu, L = None, None

    def __init__(self, mu, C):
        """Initialize an UnboundedNormalizer.

        Parameters
        ----------
        mu : ndarray 
            a matrix of size m-by-1 that contains the mean of the Gaussian 
            simulation inputs
        C : ndarray 
            a matrix of size m-by-m that contains the covariance matrix of the 
            Gaussian simulation inputs
        """
        self.mu = mu.reshape((1, mu.size))
        self.L = np.linalg.cholesky(C)

    def normalize(self, X):
        """Return points transformed to a standard normal distribution.

        Parameters
        ----------
        X : ndarray 
            contains all input points one wishes to normalize. The shape of `X` 
            is M-by-m. The components of each row of `X` should be a draw from a
            Gaussian with mean `mu` and covariance `C`.
            
        Returns
        -------
        X_norm : ndarray 
            contains the normalized inputs corresponding to `X`. The components 
            of each row of `X_norm` should be draws from a standard multivariate
            normal distribution.
        """
        X, M, m = process_inputs(X)
        X0 = X - self.mu
        X_norm = np.linalg.solve(self.L,X0.T).T
        return X_norm

    def unnormalize(self, X):
        """Transform points to original Gaussian.
        
        Return corresponding points transformed to draws from a Gaussian
        distribution with mean `mu` and covariance `C`.

        Parameters
        ----------
        X : ndarray 
            contains all input points one wishes to unnormalize. The shape of 
            `X` is M-by-m. The components of each row of `X` should be draws 
            from a standard multivariate normal.
            
        Returns
        -------
        X_unnorm : ndarray
            contains the unnormalized inputs corresponding to `X`. The 
            components of each row of `X_unnorm` should represent draws from a 
            multivariate normal with mean `mu` and covariance `C`.
        """
        X, M, m = process_inputs(X)
        X0 = np.dot(X,self.L.T)
        X_unnorm = X0 + self.mu
        return X_unnorm

def process_inputs(X):
    """Check a matrix of input values for the right shape.

    Parameters
    ----------
    X : ndarray 
        contains input points. The shape of `X` should be M-by-m.

    Returns
    -------
    X : ndarray
        the same as the input
    M : int
        number of rows in `X`
    m : int 
        number of columns in `X`
    """
    if len(X.shape) == 2:
        M, m = X.shape
    else:
        raise ValueError('The inputs X should be a two-d numpy array.')

    X = X.reshape((M, m))
    return X, M, m

def process_inputs_outputs(X, f):
    """Check matrix of input values and a vector of outputs for correct shapes.

    Parameters
    ----------
    X : ndarray 
        contains input points. The shape of `X` should be M-by-m.
    f : ndarray
        M-by-1 matrix

    Returns
    -------
    X : ndarray
        the same as the input
    f : ndarray
        the same as the output
    M : int
        number of rows in `X`
    m : int 
        number of columns in `X`
    """
    X, M, m = process_inputs(X)

    if len(f.shape) == 2:
        Mf, mf = f.shape
    else:
        raise ValueError('The outputs f should be a two-d numpy array.')

    if Mf != M:
        raise Exception('Different number of inputs and outputs.')

    if mf != 1:
        raise Exception('Only scalar-valued functions.')

    f = f.reshape((M, 1))

    return X, f, M, m

def conditional_expectations(f, ind):
    """Compute conditional expectations and variances for given function values.
    
    Parameters
    ----------
    f : ndarray
        an ndarry of function evaluations
    ind : ndarray[int]
        index array that tells which values of `f` correspond to the same value 
        for the active variable.

    Returns
    -------
    Ef : ndarray
        an ndarray containing the conditional expectations
    Vf : ndarray
        an ndarray containing the conditional variances

    Notes
    -----
    This function computes the mean and variance for all values in the ndarray
    `f` that have the same index in `ind`. The indices in `ind` correspond to
    values of the active variables.
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
    """Wrapper for `atleast_2d(A, 'col')`
    
    Notes
    -----
    Thanks to Trent Lukaczyk for these functions!
    """
    return atleast_2d(A,'col')

def atleast_2d_row(A):
    """Wrapper for `atleast_2d(A, 'row')`
    
    Notes
    -----
    Thanks to Trent Lukaczyk for these functions!
    """
    return atleast_2d(A,'row')

def atleast_2d(A, oned_as='row'):
    """Ensures the array `A` is at least two dimensions.

    Parameters
    ----------
    A : ndarray
        matrix
    oned_as : str, optional
        should be either 'row' or 'col'. It determines whether the array `A` 
        should be expanded as a 2d row or 2d column (default 'row')
    """

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
            raise Exception("oned_as must be 'row' or 'col' ")

    return A





