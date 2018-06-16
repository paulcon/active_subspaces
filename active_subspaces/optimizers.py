"""Utilities for exploiting active subspaces when optimizing."""
import numpy as np
from .domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain, \
                ActiveVariableMap
import scipy.optimize as scopt
from .utils.response_surfaces import PolynomialApproximation
from .utils.qp_solver import QPSolver
from .utils.misc import process_inputs_outputs

class MinVariableMap(ActiveVariableMap):
    """ActiveVariableMap for optimization
    
    This subclass is an domains.ActiveVariableMap specifically for optimization.

    See Also
    --------
    optimizers.BoundedMinVariableMap
    optimizers.UnboundedMinVariableMap

    Notes
    -----
    This class's train function fits a global quadratic surrogate model to the
    n+2 active variables---two more than the dimension of the active subspace.
    This quadratic surrogate is used to map points in the space of active
    variables back to the simulation parameter space for minimization.
    """

    def train(self, X, f):
        """Train the global quadratic for the regularization.

        Parameters
        ----------
        X : ndarray 
            input points used to train a global quadratic used in the 
            `regularize_z` function
        f : ndarray 
            simulation outputs used to train a global quadratic in the 
            `regularize_z` function
        """

        X, f, M, m = process_inputs_outputs(X, f)

        W1, W2 = self.domain.subspaces.W1, self.domain.subspaces.W2
        m, n = W1.shape
        W = self.domain.subspaces.eigenvecs

        # train quadratic surface on p>n active vars
        if m-n>2:
            p = n+2
        else:
            p = n+1

        Yp = np.dot(X, W[:,:p])
        pr = PolynomialApproximation(N=2)
        pr.train(Yp, f)
        br, Ar = pr.g, pr.H

        # get coefficients
        b = np.dot(W[:,:p], br)
        A = np.dot(W[:,:p], np.dot(Ar, W[:,:p].T))

        # some private attributes used in the regularize_z function
        self._bz = np.dot(W2.T, b)
        self._zAy = np.dot(W2.T, np.dot(A, W1))
        self._zAz = np.dot(W2.T, np.dot(A, W2)) + 0.01*np.eye(m-n)

class BoundedMinVariableMap(MinVariableMap):
    """This subclass is a MinVariableMap for bounded simulation inputs.

    See Also
    --------
    optimizers.MinVariableMap
    optimizers.UnboundedMinVariableMap
    """

    def regularize_z(self, Y, N=1):
        """Train the global quadratic for the regularization.

        Parameters
        ----------
        Y : ndarray 
            N-by-n matrix of points in the space of active variables
        N : int, optional
            merely there satisfy the interface of `regularize_z`. It should not 
            be anything other than 1

        Returns
        -------
        Z : ndarray 
            N-by-(m-n)-by-1 matrix that contains a value of the inactive 
            variables for each value of the inactive variables

        Notes
        -----
        In contrast to the `regularize_z` in BoundedActiveVariableMap and
        UnboundedActiveVariableMap, this implementation of `regularize_z` uses
        a quadratic program to find a single value of the inactive variables
        for each value of the active variables.
        """
        if N != 1:
            raise Exception('MinVariableMap needs N=1.')

        W1, W2 = self.domain.subspaces.W1, self.domain.subspaces.W2
        m, n = W1.shape
        NY = Y.shape[0]
        qps = QPSolver()

        Zlist = []
        A_ineq = np.vstack((W2, -W2))
        for y in Y:
            c = self._bz.reshape((m-n, 1)) + np.dot(self._zAy, y).reshape((m-n, 1))
            b_ineq = np.vstack((
                -1-np.dot(W1, y).reshape((m, 1)),
                -1+np.dot(W1, y).reshape((m, 1))
                ))
            z = qps.quadratic_program_ineq(c, self._zAz, A_ineq, b_ineq)
            Zlist.append(z)
        Z = np.array(Zlist).reshape((NY, m-n, N))
        return Z

class UnboundedMinVariableMap(MinVariableMap):
    """This subclass is a MinVariableMap for unbounded simulation inputs.

    See Also
    --------
    optimizers.MinVariableMap
    optimizers.BoundedMinVariableMap
    """

    def regularize_z(self, Y, N=1):
        """Train the global quadratic for the regularization.

        Parameters
        ----------
        Y : ndarray 
            N-by-n matrix of points in the space of active variables
        N : int, optional
            merely there satisfy the interface of `regularize_z`. It should not 
            be anything other than 1

        Returns
        -------
        Z : ndarray 
            N-by-(m-n)-by-1 matrix that contains a value of the inactive 
            variables for each value of the inactive variables

        Notes
        -----
        In contrast to the `regularize_z` in BoundedActiveVariableMap and
        UnboundedActiveVariableMap, this implementation of `regularize_z` uses
        a quadratic program to find a single value of the inactive variables
        for each value of the active variables.
        """
        if N != 1:
            raise Exception('MinVariableMap needs N=1.')

        m, n = self.domain.subspaces.W1.shape
        NY = Y.shape[0]

        Zlist = []
        for y in Y:
            c = self._bz.reshape((m-n, 1)) + np.dot(self._zAy, y).reshape((m-n, 1))
            z = np.linalg.solve(self._zAz, c)
            Zlist.append(z)
        return np.array(Zlist).reshape((NY, m-n, N))

def minimize(asrs, X, f):
    """Minimize a response surface constructed with the active subspace.
    
    Parameters
    ----------
    asrs : ActiveSubspaceResponseSurface 
        a trained response_surfaces.ActiveSubspaceResponseSurface
    X : ndarray 
        input points used to train the MinVariableMap
    f : ndarray 
        simulation outputs used to train the MinVariableMap
        
    Returns
    -------
    xstar : ndarray 
        the estimated minimizer of the function modeled by the
        ActiveSubspaceResponseSurface `asrs`
    fstar : float 
        the estimated minimum of the function modeled by `asrs`

    Notes
    -----
    This function has two stages. First it uses the scipy.optimize package to
    minimize the response surface of the active variables. Then it trains
    a MinVariableMap with the given input/output pairs, which it uses to map
    the minimizer back to the space of simulation inputs.

    This is very heuristic. 
    """
    X, f, M, m = process_inputs_outputs(X, f)

    # ActiveVariableDomain
    avdom = asrs.avmap.domain

    # wrappers
    def avfun(y):
        f = asrs.predict_av(y.reshape((1,y.size)))[0]
        return f[0,0]
    def avdfun(y):
        df = asrs.gradient_av(y.reshape((1,y.size)))
        return df.reshape((y.size,))

    if isinstance(avdom, UnboundedActiveVariableDomain):
        mvm = UnboundedMinVariableMap(avdom)
    elif isinstance(avdom, BoundedActiveVariableDomain):
        mvm = BoundedMinVariableMap(avdom)
    else:
        raise Exception('There is a problem with the avmap.domain.')

    ystar, fstar = av_minimize(avfun, avdom, avdfun=avdfun)
    mvm.train(X, f)
    xstar = mvm.inverse(ystar)[0]
    return xstar, fstar

def av_minimize(avfun, avdom, avdfun=None):
    """Minimize a response surface on the active variables.

    Parameters
    ----------
    avfun : function 
        a function of the active variables
    avdom : ActiveVariableDomain 
        information about the domain of `avfun`
    avdfun : function 
        returns the gradient of `avfun`

    Returns
    -------
    ystar : ndarray 
        the estimated minimizer of `avfun`
    fstar : float 
        the estimated minimum of `avfun`

    See Also
    --------
    optimizers.interval_minimize
    optimizers.zonotope_minimize
    optimizers.unbounded_minimize
    """
    if isinstance(avdom, UnboundedActiveVariableDomain):
        ystar, fstar = unbounded_minimize(avfun, avdom, avdfun)

    elif isinstance(avdom, BoundedActiveVariableDomain):
        n = avdom.subspaces.W1.shape[1]
        if n==1:
            ystar, fstar = interval_minimize(avfun, avdom)
        else:
            ystar, fstar = zonotope_minimize(avfun, avdom, avdfun)
    else:
        raise Exception('There is a problem with the avdom.')

    return ystar.reshape((1,ystar.size)), fstar

def interval_minimize(avfun, avdom):
    """Minimize a response surface defined on an interval.

    Parameters
    ----------
    avfun : function 
        a function of the active variables
    avdom : ActiveVariableDomain 
        contains information about the domain of `avfun`

    Returns
    -------
    ystar : ndarray 
        the estimated minimizer of `avfun`
    fstar : float 
        the estimated minimum of `avfun`

    See Also
    --------
    optimizers.av_minimize

    Notes
    -----
    This function wraps the scipy.optimize function fminbound.
    """

    yl, yu = avdom.vertY[0,0], avdom.vertY[1,0]
    result = scopt.fminbound(avfun, yl, yu, xtol=1e-9, maxfun=1e4, full_output=1)
    if result[2]:
        raise Exception('Max function values used in fminbound.')
        ystar, fstar = None, None
    else:
        ystar, fstar = np.array([[result[0]]]), result[1]
    return ystar, fstar

def zonotope_minimize(avfun, avdom, avdfun):
    """Minimize a response surface defined on a zonotope.
    
    Parameters
    ----------
    avfun : function 
        a function of the active variables
    avdom : ActiveVariableDomain 
        contains information about the domain of `avfun`
    avdfun : function 
        returns the gradient of `avfun`

    Returns
    -------
    ystar : ndarray 
        the estimated minimizer of `avfun`
    fstar : float 
        the estimated minimum of `avfun`

    See Also
    --------
    optimizers.av_minimize

    Notes
    -----
    This function wraps the scipy.optimize implementation of SLSQP with linear
    inequality constraints derived from the zonotope.
    """

    n = avdom.subspaces.W1.shape[1]

    opts = {'disp':False, 'maxiter':1e4, 'ftol':1e-9}

    # a bit of globalization
    curr_state = np.random.get_state()
    np.random.seed(42)
    minf = 1e100
    minres = []
    for i in range(10):
        y0 = np.random.normal(size=(1, n))
        cons = avdom.constraints
        result = scopt.minimize(avfun, y0, constraints=cons, method='SLSQP', \
                            jac=avdfun, options=opts)
        if not result.success:
            raise Exception('SLSQP failed with message: {}.'.format(result.message))
        if result.fun < minf:
            minf = result.fun
            minres = result

    np.random.set_state(curr_state)
    ystar, fstar = minres.x, minres.fun
    return ystar, fstar

def unbounded_minimize(avfun, avdom, avdfun):
    """Minimize a response surface defined on an unbounded domain.

    Parameters
    ----------
    avfun : function 
        a function of the active variables
    avdom  : ActiveVariableDomain 
        contains information about the domain of `avfun`
    avdfun : function 
        returns the gradient of `avfun`

    Returns
    -------
    ystar : ndarray 
        the estimated minimizer of `avfun`
    fstar : float 
        the estimated minimum of `avfun`

    See Also
    --------
    optimizers.av_minimize

    Notes
    -----
    If the gradient `avdfun` is None, this function wraps the scipy.optimize
    implementation of SLSQP. Otherwise, it wraps BFGS.
    """
    n = avdom.subspaces.W1.shape[1]
    opts = {'disp':False, 'maxiter':1e4}

    if avdfun == None:
        method = 'SLSQP'
    else:
        method = 'BFGS'

    # some tricks for globalization
    curr_state = np.random.get_state()
    np.random.seed(42)
    minf = 1e100
    minres = []
    for i in range(10):
        y0 = np.random.normal(size=(1, n))
        result = scopt.minimize(avfun, y0, method=method, jac=avdfun, options=opts)
        if not result.success:
            raise Exception('{} failed with message: {}.'.format(method, result.message))
        if result.fun < minf:
            minf = result.fun
            minres = result
    np.random.set_state(curr_state)
    ystar, fstar = minres.x, minres.fun
    return ystar, fstar
