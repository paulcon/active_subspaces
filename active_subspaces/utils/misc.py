"""Miscellaneous utilities."""
import numpy as np
import logging

class Normalizer():
    """
    An abstract class for normalizing inputs.
    """
    def normalize(self, X):
        raise NotImplementedError()

    def unnormalize(self, X):
        raise NotImplementedError()

class BoundedNormalizer(Normalizer):
    """
    A class for normalizing bounded inputs. Extends the abstract Normalizer
    class.

    :cvar ndarray lb: A matrix of size m-by-1 that contains lower bounds on the
        simulation inputs.
    :cvar ndarray ub: A matrix of size m-by-1 that contains upper bounds on the
        simulation inputs.

    **See Also**

    utils.misc.UnboundedNormalizer
    """
    lb, ub = None, None

    def __init__(self, lb, ub):
        """
        Initialize a BoundedNormalizer.

        :param ndarray lb: A matrix of size m-by-1 that contains lower bounds on
            the simulation inputs.
        :param ndarray ub: A matrix of size m-by-1 that contains upper bounds on
            the simulation inputs.
        """
        m = lb.size
        self.lb = lb.reshape((1, m))
        self.ub = ub.reshape((1, m))

    def normalize(self, X):
        """
        Return corresponding points shifted and scaled to [-1,1]^m.

        :param ndarray X: Contains all input points one wishes to normalize. The
            shape of `X` is M-by-m. The components of each row of `X` should be
            between `lb` and `ub`.

        :return: X_norm, contains the normalized inputs corresponding to `X`.
            The components of each row of `X_norm` should be between -1 and 1.
        :rtype: ndarray
        """
        X, M, m = process_inputs(X)
        X_norm = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return X_norm

    def unnormalize(self, X):
        """
        Return corresponding points shifted and scaled to [-1,1]^m.

        :param ndarray X: Contains all input points one wishes to unnormalize.
            The shape of `X` is M-by-m. The components of each row of `X` should
            be between -1 and 1.

        :return: X_unnorm, Contains the unnormalized inputs corresponding to
            `X`. The components of each row of `X_unnorm` should be between `lb`
            and `ub`.
        :rtype: ndarray
        """
        X, M, m = process_inputs(X)
        X_unnorm = (self.ub - self.lb) * (X + 1.0) / 2.0 + self.lb
        return X_unnorm

class UnboundedNormalizer(Normalizer):
    """
    A class for normalizing unbounded, Gaussian inputs to standard normals.
    Extends the abstract Normalizer class.

    :cvar ndarray mu: A matrix of size m-by-1 that contains the mean of the
        Gaussian simulation inputs.
    :cvar ndarray L: A matrix size m-by-m that contains the Cholesky factor of
        the covariance matrix of the Gaussian simulation inputs.

    **See Also**

    utils.misc.BoundedNormalizer

    **Notes**

    A simulation with unbounded inputs is assumed to have a Gaussian weight
    function associated with the inputs. The covariance of the Gaussian weight
    function should be full rank.
    """
    mu, L = None, None

    def __init__(self, mu, C):
        """
        Initialize an UnboundedNormalizer.

        :param ndarray mu: A matrix of size m-by-1 that contains the mean of the
            Gaussian simulation inputs.
        :param ndarray C: A matrix of size m-by-m that contains the covariance
            matrix of the Gaussian simulation inputs.
        """
        self.mu = mu.reshape((1, mu.size))
        self.L = np.linalg.cholesky(C)

    def normalize(self, X):
        """
        Return corresponding points transformed to a standard normal
        distribution.

        :param ndarray X: Contains all input points one wishes to normalize. The
            shape of `X` is M-by-m. The components of each row of `X` should be
            a draw from a Gaussian with mean `mu` and covariance `C`.

        :return: X_norm, Contains the normalized inputs corresponding to `X`.
            The components of each row of `X_norm` should be draws from a
            standard multivariate normal distribution.
        :rtype: ndarray
        """
        X, M, m = process_inputs(X)
        X0 = X - self.mu
        X_norm = np.linalg.solve(self.L,X0.T).T
        return X_norm

    def unnormalize(self, X):
        """
        Return corresponding points transformed to draws from a Gaussian
        distribution with mean `mu` and covariance `C`.

        :param ndarray X: Contains all input points one wishes to unnormalize.
            The shape of `X` is M-by-m. The components of each row of `X` should
            be draws from a standard multivariate normal.

        :return: X_unnorm, Contains the unnormalized inputs corresponding to
            `X`. The components of each row of `X_unnorm` should represent draws
            from a multivariate normal with mean `mu` and covariance `C`.
        :rtype: ndarray
        """
        X, M, m = process_inputs(X)
        X0 = np.dot(X,self.L.T)
        X_unnorm = X0 + self.mu
        return X_unnorm

def process_inputs(X):
    """
    Check a matrix of input values for the right shape.

    :param ndarray X: Contains input points. The shape of `X` should be M-by-m.

    :return: X, The same as the input.
    :rtype: ndarray
    :return: M, Number of rows in `X`.
    :rtype: int
    :return: m, Number of columns in `X`.
    :rtype: int
    """
    if len(X.shape) == 2:
        M, m = X.shape
    else:
        raise ValueError('The inputs X should be a two-d numpy array.')

    X = X.reshape((M, m))
    return X, M, m

def process_inputs_outputs(X, f):
    """
    Check a matrix of input values and a vector of outputs for the right shapes.

    :param ndarray X: Contains input points. The shape of `X` should be M-by-m.
    :param ndarray f: M-by-1 matrix.

    :return: X, The same as the input.
    :rtype: ndarray
    :return: f, The same as the input.
    :rtype: ndarray
    :return: M, Number of rows in `X`.
    :rtype: int
    :return: m, Number of columns in `X`.
    :rtype: int
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
    """
    Compute conditional expectations and variances for a set of function values.

    :param ndarray f: An ndarry of function evaluations.
    :param ndarray[int] ind: Index array that tells which values of `f`
        correspond to the same value for the active variable.

    :return: Ef, An ndarray containing the conditional expectations.
    :rtype: ndarray
    :return: Vf, An ndarray containing the conditional variances.
    :rtype: ndarray

    **Notes**

    This function computes the mean and variance for all values in the ndarray
    `f` that have the same index in `ind`. The indices in `ind` correspond to
    values of the active variables.
    """

    n = int(np.amax(ind)) + 1
    NMC = np.sum(ind==0)
    logging.getLogger(__name__).debug('Computing {:d} conditional averages with {:d} MC samples.'.format(n, NMC))

    Ef, Vf = np.zeros((n, 1)), np.zeros((n, 1))
    for i in range(n):
        fi = f[ind == i]
        Ef[i] = np.mean(fi)
        Vf[i] = np.var(fi)
    return Ef, Vf

# thanks to Trent for these functions!!!
def atleast_2d_col(A):
    """
    Return the input `A` as a 2d column array.
    """
    return atleast_2d(A,'col')

def atleast_2d_row(A):
    """
    Return the input `A` as a 2d row array.
    """
    return atleast_2d(A,'row')

def atleast_2d(A, oned_as='row'):
    """
    Ensures the array `A` is at least two dimensions.

    :param ndarray A: matrix
    :param str oned_as: Should be either 'row' or 'col'. It determines whether
        the array `A` should be expanded as a 2d row or 2d column.
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
            raise Exception , "oned_as must be 'row' or 'col' "

    return A





