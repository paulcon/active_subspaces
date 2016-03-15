"""Utilities for building response surface approximations."""
import numpy as np
import logging
from scipy.optimize import fminbound
from scipy.misc import comb
from misc import process_inputs, process_inputs_outputs
import pdb

class ResponseSurface():
    """
    An abstract class for response surfaces.

    :cvar int N: Maximum degree of global polynomial in the response surface.
    :cvar float Rsqr: The R-squared coefficient for the response surface.
    :cvar ndarray X: An ndarray of training points for the response surface. The
        shape is M-by-m, where m is the number of dimensions.
    :cvar ndarray f: An ndarray of function values used to train the response
        surface. The shape of `f` is M-by-1.

    **See Also**

    utils.response_surfaces.PolynomialApproximation
    utils.response_surfaces.RadialBasisApproximation
    """
    N = None
    Rsqr = None
    X, f = None, None

    def __init__(self, N=2):
        self.N = N

    def train(self, X, f):
        raise NotImplementedError()

    def predict(self, X, compgrad=False):
        raise NotImplementedError()

    def gradient(self, X):
        return self.predict(X, compgrad=True)[1]

    def __call__(self, X):
        return self.predict(X)[0]

class PolynomialApproximation(ResponseSurface):
    """
    A class for least-squares-fit, global, multivariate polynomial
    approximation.

    :cvar ndarray poly_weights: An ndarray of coefficients for the polynomial
        approximation in the monomial basis.
    :cvar ndarray g: Contains the m coefficients corresponding to the degree 1
        monomials in the polynomial approximation
    :cvar ndarray H: An ndarray of shape m-by-m that contains the coefficients
        of the degree 2 monomials in the approximation.

    **See Also**

    utils.response_surfaces.RadialBasisApproximation

    **Notes**

    All attributes besides the degree `N` are set when the class's `train`
    method is called.
    """
    poly_weights = None
    g, H = None, None

    def train(self, X, f, weights=None):
        """
        Train the least-squares-fit polynomial approximation.

        :param ndarray X: An ndarray of training points for the polynomial
            approximation. The shape is M-by-m, where m is the number of
            dimensions.
        :param ndarray f: An ndarray of function values used to train the
            polynomial approximation. The shape of `f` is M-by-1.
        :param ndarray f: An ndarray of weights for the least-squares.

        **Notes**

        This method sets all the attributes of the class for use in the
        `predict` method.
        """
        X, f, M, m = process_inputs_outputs(X, f)

        # check that there are enough points to train the polynomial
        if M < comb(self.N + m, m):
            raise Exception('Not enough points to fit response surface of order {:d}'.format(self.N))

        logging.getLogger(__name__).debug('Training a degree {:d} polynomial in {:d} dims with {:d} points.'.format(self.N, m, M))

        B, indices = polynomial_bases(X,  self.N)
        p = B.shape[1]
        if weights is not None:
            B, f = weights*B, weights*f

        Q, R = np.linalg.qr(B)
        Qf = np.dot(Q.T, f)
        poly_weights = np.linalg.solve(R, Qf)
        Rsqr = 1.0 - ( np.linalg.norm(np.dot(R, poly_weights) - Qf)**2 / np.var(f) )

        # store data
        self.X, self.f = X, f
        self.poly_weights = poly_weights.reshape((p,1))
        self.Rsqr = Rsqr

        # organize linear and quadratic coefficients
        self.g = poly_weights[1:m+1].copy().reshape((m,1))
        if self.N > 1:
            H = np.zeros((m, m))
            for i in range(m+1, int(m+1+comb(m+1,2))):
                ind = indices[i,:]
                loc = np.nonzero(ind!=0)[0]
                if loc.size==1:
                    H[loc,loc] = 2.0*poly_weights[i]
                elif loc.size==2:
                    H[loc[0],loc[1]] = poly_weights[i]
                    H[loc[1],loc[0]] = poly_weights[i]
                else:
                    raise Exception('Error creating quadratic coefficients.')
            self.H = H

    def predict(self, X, compgrad=False):
        """
        Evaluate the least-squares-fit polynomial approximation at new points.

        :param ndarray X: An ndarray of points to evaluate the polynomial
            approximation. The shape is M-by-m, where m is the number of
            dimensions.
        :param bool compgrad: A flag to decide whether or not to compute the
            gradient of the polynomial approximation at the points `X`.

        :return: f, An ndarray of predictions from the polynomial approximation.
            The shape of `f` is M-by-1.
        :rtype: ndarray
        :return: df, An ndarray of gradient predictions from the polynomial
            approximation. The shape of `df` is M-by-m.
        :rtype: ndarray
        """
        X, M, m = process_inputs(X)

        B = polynomial_bases(X, self.N)[0]
        f = np.dot(B, self.poly_weights).reshape((M, 1))

        if compgrad:
            dB = grad_polynomial_bases(X, self.N)
            df = np.zeros((M, m))
            for i in range(m):
                df[:,i] = np.dot(dB[:,:,i], self.poly_weights).reshape((M))
            df = df.reshape((M, m))
        else:
            df = None

        return f, df

class RadialBasisApproximation(ResponseSurface):
    """
    A class for global, multivariate radial basis approximation with anisotropic
    squared-exponential radial basis and a weighted-least-squares-fit monomial
    basis.

    :cvar ndarray radial_weights: An ndarray of coefficients radial basis
        functions in the model.
    :cvar ndarray poly_weights: An ndarray of coefficients for the polynomial
        approximation in the monomial basis.
    :cvar ndarray K: An ndarray of shape M-by-M that contains the matrix of
        radial basis functions evaluated at the training points.
    :cvar ndarray ell: An ndarray of shape m-by-1 that contains the
        characteristic length scales along each of the inputs.

    **See Also**

    utils.response_surfaces.PolynomialApproximation

    **Notes**

    All attributes besides the degree `N` are set when the class's `train`
    method is called.
    """
    K, ell = None, None
    radial_weights, poly_weights = None, None

    def train(self, X, f, v=None, e=None):
        """
        Train the radial basis approximation.

        :param ndarray X: An ndarray of training points for the polynomial
            approximation. The shape is M-by-m, where m is the number of
            dimensions.
        :parma ndarray f: An ndarray of function values used to train the
            polynomial approximation. The shape of `f` is M-by-1.
        :param ndarray v: Contains the regularization parameters that model
            error in the function values.
        :param ndarray e: An ndarray containing the eigenvalues from the active
            subspace analysis. If present, the radial basis uses it to
            determine the appropriate anisotropy in the length scales.

        **Notes**

        The approximation uses an multivariate, squared exponential radial
        basis. If `e` is not None, then the radial basis is anisotropic with
        length scales determined by `e`. Otherwise, the basis is isotropic.
        The length scale parameters (i.e., the rbf shape parameters) are
        determined with a maximum likelihood heuristic inspired by
        techniques for fitting a Gaussian process model.

        The approximation also includes a monomial basis with monomials of
        total degree up to order `N`. These are fit with weighted least-squares,
        where the weight matrix is the inverse of the matrix of radial basis
        functions evaluated at the training points.

        This method sets all the attributes of the class for use in the
        `predict` method.
        """
        X, f, M, m = process_inputs_outputs(X, f)

        # check that there are enough points to train the polynomial
        if M < comb(self.N + m, m):
            raise Exception('Not enough points to fit response surface of order {:d}'.format(self.N))

        logging.getLogger(__name__).debug('Training an RBF surface with degree {:d} polynomial in {:d} dims with {:d} points.'.format(self.N, m, M))

        # use maximum likelihood to tune parameters
        log10g = fminbound(rbf_objective, -10.0, 1.0, args=(X, f, v, self.N, e, ))
        g = 10**(log10g)

        if e is None:
            ell = g*np.ones((m,1))
            if v is None:
                v = 1e-6*np.ones(f.shape)
        else:
            ell = g*np.sum(e)/e[:m]
            if v is None:
                v = g*np.sum(e[m:])*np.ones(f.shape)

        # covariance matrix of observations
        K = exponential_squared(X, X, 1.0, ell)
        K += np.diag(v.reshape((M,)))
        B = polynomial_bases(X, self.N)[0]
        p = B.shape[1]

        C = np.hstack(( np.vstack(( K, B.T )), np.vstack(( B, np.zeros((p, p)) )) ))
        weights = np.linalg.solve(C, np.vstack(( f, np.zeros((p, 1)) )) )

        radial_weights, poly_weights = weights[:M], weights[M:]

        res = f - np.dot(B, poly_weights)
        Rsqr = 1.0 - (np.dot( res.T, np.linalg.solve(K, res)) / np.dot( f.T, np.linalg.solve(K, f) ))

        # store parameters
        self.X, self.f = X, f
        self.ell, self.K = ell, K
        self.Rsqr = Rsqr[0,0]
        self.radial_weights, self.poly_weights = radial_weights, poly_weights

    def predict(self, X, compgrad=False):
        """
        Evaluate the radial basis approximation at new points.

        :param ndarray X: An ndarray of points to evaluate the polynomial
            approximation. The shape is M-by-m, where m is the number of
            dimensions.
        :param bool compgrad: A flag to decide whether or not to compute the
            gradient of the polynomial approximation at the points `X`.

        :return: f, An ndarray of predictions from the polynomial approximation.
            The shape of `f` is M-by-1.
        :rtype: ndarray
        :return: df, An ndarray of gradient predictions from the polynomial
            approximation. The shape of `df` is M-by-m.
        :rtype: ndarray

        **Notes**

        I'll tell you what. I just refactored this code to use terminology from
        radial basis functions instead of Gaussian processes, and I feel so
        much better about it. Now I don't have to compute that silly
        prediction variance and try to pretend that it has anything to do with
        the actual error in the approximation. Also, computing that variance
        requires another system solve, which might be expensive. So it's both
        expensive and of dubious value. So I got rid of it. Sorry, Gaussian
        process people.
        """
        X, M, m = process_inputs(X)

        #
        K = exponential_squared(X, self.X, 1.0, self.ell)
        B = polynomial_bases(X, self.N)[0]
        f = np.dot(K, self.radial_weights) + np.dot(B, self.poly_weights)
        f = f.reshape((M, 1))

        if compgrad:
            dK = grad_exponential_squared(self.X, X, 1.0, self.ell)
            dB = grad_polynomial_bases(X, self.N)
            df = np.zeros((M, m))
            for i in range(m):
                df[:,i] = (np.dot(dK[:,:,i].T, self.radial_weights) + \
                    np.dot(dB[:,:,i], self.poly_weights)).reshape((M, ))
            df = df.reshape((M, m))
        else:
            df = None

        return f, df

def rbf_objective(log10g, X, f, v, N, e):
    """
    Objective function for the maximum likelihood heuristic for choosing the
    rbf shape parameters.

    :param float log10g: The log of the scaling factor for the rbf shape
        parameters.
    :param ndarray X: The ndarray of training points.
    :param ndarray f: The ndarray of training data.
    :param ndarray v: Contains the regularization parameters for the training
        data.
    :param int N: The order of polynomial approximation.
    :param ndarray e: Contains the eigenvalues from the active subspace
        analysis.

    :return: r, Objective function value. If you were training a Gaussian
        process, it would be the negative log likelihood. In this context, it's
        just a heuristic.
    :rtype: float
    """
    # TODO: I can probably make this implementation more efficient, but as of
    # now, I don't need to.
    g = 10**(log10g)

    M, m = X.shape
    if e is None:
        ell = g*np.ones((m,1))
        if v is None:
            v = 1e-6*np.ones(f.shape)
    else:
        ell = g*np.sum(e)/e[:m]
        if v is None:
            v = g*np.sum(e[m:])*np.ones(f.shape)

    # covariance matrix
    K = exponential_squared(X, X, 1.0, ell)
    K += np.diag(v.reshape((M,)))
    L = np.linalg.cholesky(K)

    # polynomial basis
    B = polynomial_bases(X, N)[0]
    A = np.dot(B.T, np.linalg.solve(K, B))
    z = np.dot(B.T, np.linalg.solve(K, f))
    beta = np.linalg.solve(A, z)

    # residual
    res = f - np.dot(B, beta)

    # variance
    sig2 = np.dot(res.T, np.linalg.solve(K, res))/M

    r = np.sum(np.log(np.diag(L))) + M*np.log(sig2)
    return r

def exponential_squared(X1, X2, sigma, ell):
    """
    Compute the matrix of radial basis functions.

    :param ndarray X1: Contains the centers of the radial functions.
    :param ndarray X2: The evaluation points of the radial functions.
    :param float sigma: Scales the radial functions.
    :param ndarray ell: Contains the length scales of each dimension.

    :return: C, The matrix of radial functions centered at `X1` and evaluated
        at `X2`. The shape of `C` is `X1.shape[0]`-by-`X2.shape[0]`.
    :rtype: ndarray
    """
    m = X1.shape[0]
    n = X2.shape[0]
    c = -1.0 / ell.flatten()
    C = np.zeros((m, n))
    for i in range(n):
        x2 = X2[i,:]
        B = X1 - x2
        C[:,i] = sigma*np.exp(np.dot(B*B, c))
    return C

def grad_exponential_squared(X1, X2, sigma, ell):
    """
    Compute the matrices of radial basis function gradients.

    :param ndarray X1:
        `X1` contains the centers of the radial functions.
    :param ndarray X2:
        `X2` is the evaluation points of the radial functions.
    :param float sigma:
        `sigma` scales the radial functions.
    :param ndarray ell:
        `ell` contains the length scales of each dimension.

    :return: dC, The matrix of radial function gradients centered at `X1` and
        evaluated at `X2`. The shape of `dC` is `X1.shape[0]`-by-`X2.shape[0]`-
        by-m. `dC` is a three-dimensional ndarray. The third dimension indexes
        the partial derivatives in each gradient.
    :rtype: ndarray
    """
    m, d = X1.shape
    n = X2.shape[0]
    c = -1.0 / ell.flatten()
    C = np.zeros((m, n, d))
    for k in range(d):
        for i in range(n):
            x2 = X2[i,:]
            B = X1 - x2
            C[:,i,k] = sigma*(-2.0*c[k]*B[:,k])*np.exp(np.dot(B*B, c))
    return C

def polynomial_bases(X, N):
    """
    Compute the monomial bases.

    :param ndarray X: Contains the points to evaluate the monomials.
    :param int N: The maximum degree of the monomial basis.

    :return: B, Contains the monomial evaluations.
    :rtype: ndarray
    :return: I, Contains the multi-indices that tell the degree of each
        univariate monomial term in the multivariate monomial.
    :rtype: ndarray
    """
    M, m = X.shape
    I = index_set(N, m)
    n = I.shape[0]
    B = np.zeros((M, n))
    for i in range(n):
        ind = I[i,:]
        B[:,i] = np.prod(np.power(X, ind), axis=1)
    return B, I

def grad_polynomial_bases(X, N):
    """
    Compute the gradients of the monomial bases.

    :param ndarray X: Contains the points to evaluate the monomials.
    :param int N: The maximum degree of the monomial basis.

    :return: dB, Contains the gradients of the monomials evaluate at `X`. `dB`
        is a three-dimensional ndarray. The third dimension indexes the partial
        derivatives in each gradient.
    :rtype: ndarray

    """
    M, m = X.shape
    I = index_set(N, m)
    n = I.shape[0]
    B = np.zeros((M, n, m))
    for k in range(m):
        for i in range(n):
            ind = I[i,:].copy()
            indk = ind[k]
            if indk==0:
                B[:,i,k] = np.zeros(M)
            else:
                ind[k] -= 1
                B[:,i,k] = indk*np.prod(np.power(X, ind), axis=1)
    return B

def _full_index_set(n, d):
    """
    A helper function for index_set.
    """
    if d == 1:
        I = np.array([[n]])
    else:
        II = _full_index_set(n, d-1)
        m = II.shape[0]
        I = np.hstack((np.zeros((m, 1)), II))
        for i in range(1, n+1):
            II = _full_index_set(n-i, d-1)
            m = II.shape[0]
            T = np.hstack((i*np.ones((m, 1)), II))
            I = np.vstack((I, T))
    return I

def index_set(n, d):
    """
    Enumerate multi-indices for a total degree of order `n` in `d` variables.

    :param int n:
    :param int d:

    :return: I
    :rtype: ndarray

    """
    I = np.zeros((1, d))
    for i in range(1, n+1):
        II = _full_index_set(i, d)
        I = np.vstack((I, II))
    return I[:,::-1]
