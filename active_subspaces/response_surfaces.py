"""Utilities for exploiting active subspaces in response surfaces."""
import numpy as np
import utils.designs as dn
from utils.simrunners import SimulationRunner
from utils.misc import conditional_expectations
from utils.response_surfaces import RadialBasisApproximation
from domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain, \
                    ActiveVariableMap

class ActiveSubspaceResponseSurface():
    """
    A class for using active subspace with response surfaces.

    :param ResponseSurface respsurf: `respsurf` is a
        utils.response_surfaces.ResponseSurface.
    :param ActiveVariableMap avmap: a domains.ActiveVariableMap.

    **Notes**

    This class has several convenient functions for training and using a
    response surface with active subspaces. Note that the `avmap` must be
    given. This means that the active subspace must be computed already.
    """
    respsurf = None
    avmap = None

    def __init__(self, avmap, respsurf=None):
        """
        Initialize an ActiveSubspaceResponseSurface.

        :param ActiveVariableMap avmap: A domains.ActiveVariable map that
            includes the active variable domain, which includes the active and
            inactive subspaces.
        :param ResponseSurface respsurf: A
            utils.response_surfaces.ResponseSurface object. If a ResponseSurface
            is not given, a default RadialBasisApproximation is used.
        """
        if not isinstance(avmap, ActiveVariableMap):
            raise TypeError('avmap should be an ActiveVariableMap.')

        if respsurf == None:
            self.respsurf = RadialBasisApproximation()
        else:
            self.respsurf = respsurf
        self.avmap = avmap

    def _train(self, Y, f, v=None):
        """
        A private function for training the response surface with a set of
        active variable and function evaluations.
        """
        if isinstance(self.respsurf, RadialBasisApproximation):
            evals = self.avmap.domain.subspaces.eigenvalues
            self.respsurf.train(Y, f, v=v, e=evals)
        else:
            self.respsurf.train(Y, f)

    def train_with_data(self, X, f, v=None):
        """
        Train the response surface with input/output pairs.

        :param ndarray X: M-by-m matrix with evaluations of the simulation
            inputs.
        :param ndarray f: M-by-1 matrix with corresponding simulation quantities
            of interest.
        :param ndarray v: M-by-1 matrix that contains the regularization
            (i.e., errors) associated with `f`.

        **Notes**

        The training methods exploit the eigenvalues from the active subspace
        analysis to determine length scales for each variable when tuning
        the parameters of the radial bases.

        The method sets attributes of the object for further use.
        """
        Y = self.avmap.forward(X)[0]
        self._train(Y, f, v=v)

    def train_with_interface(self, fun, N, NMC=10):
        """
        Train the response surface with input/output pairs.

        :param function fun: A function that returns the simulation quantity of
            interest given a point in the input space as an 1-by-m ndarray.
        :param int N: The number of points used in the design-of-experiments for
            constructing the response surface.
        :param int NMC: The number of points used to estimate the conditional
            expectation and conditional variance of the function given a value
            of the active variables.

        **Notes**

        The training methods exploit the eigenvalues from the active subspace
        analysis to determine length scales for each variable when tuning
        the parameters of the radial bases.

        The method sets attributes of the object for further use.

        The method uses the response_surfaces.av_design function to get the
        design for the appropriate `avmap`.
        """
        Y, X, ind = av_design(self.avmap, N, NMC=NMC)

        if isinstance(self.avmap.domain, BoundedActiveVariableDomain):
            X = np.vstack((X, self.avmap.domain.vertX))
            Y = np.vstack((Y, self.avmap.domain.vertY))
            il = np.amax(ind) + 1
            iu = np.amax(ind) + self.avmap.domain.vertX.shape[0] + 1
            iind = np.arange(il, iu)
            ind = np.vstack(( ind, iind.reshape((iind.size,1)) ))

        # run simulation interface at all design points
        if isinstance(fun, SimulationRunner):
            f = fun.run(X)
        else:
            f = SimulationRunner(fun).run(X)

        Ef, Vf = conditional_expectations(f, ind)
        self._train(Y, Ef, v=Vf)

    def predict_av(self, Y, compgrad=False):
        """
        Compute the value of the response surface given values of the active
        variables.

        :param ndarray Y: M-by-n matrix containing points in the range of active
            variables to evaluate the response surface.
        :param bool compgrad: Determines if the gradient of the response surface
            with respect to the active variables is computed and returned.

        :return: f, contains the response surface values at the given `Y`.
        :rtype: ndarray

        :return: df, Contains the response surface gradients at the given `Y`.
            If `compgrad` is False, then `df` is None.
        :rtype: ndarray
        """
        f, df = self.respsurf.predict(Y, compgrad)
        return f, df

    def gradient_av(self, Y):
        """
        A convenience function for computing the gradient of the response
        surface with respect to the active variables.

        :param ndarray Y: M-by-n matrix containing points in the range of active
            variables to evaluate the response surface gradient.

        :return: df, Contains the response surface gradient at the given `Y`.
        :rtype: ndarray
        """
        df = self.respsurf.predict(Y, compgrad=True)[1]
        return df

    def predict(self, X, compgrad=False):
        """
        Compute the value of the response surface given values of the simulation
        variables.

        :param ndarray X: M-by-m matrix containing points in simulation's
            parameter space.
        :param bool compgrad: Determines if the gradient of the response surface
            is computed and returned.

        :return: f, Contains the response surface values at the given `X`.
        :rtype: ndarray

        :return: dfdx, An ndarray of shape M-by-m that contains the estimated
            gradient at the given `X`. If `compgrad` is False, then `dfdx` is
            None.
        :rtype: ndarray
        """
        Y = self.avmap.forward(X)[0]
        f, dfdy = self.predict_av(Y, compgrad)
        if compgrad:
            W1 = self.avmap.domain.subspaces.W1
            dfdx = np.dot(dfdy, W1.T)
        else:
            dfdx = None
        return f, dfdx

    def gradient(self, X):
        """
        A convenience function for computing the gradient of the response
        surface with respect to the simulation inputs.

        :param ndarray X: M-by-m matrix containing points in the space of
            simulation inputs.

        :return: df, Contains the response surface gradient at the given `X`.
        :rtype: ndarray
        """
        return self.predict(X, compgrad=True)[1]

    def __call__(self, X):
        return self.predict(X)[0]

def av_design(avmap, N, NMC=10):
    """
    A wrapper that returns the design for the response surface in the space of
    the active variables.

    :param ActiveVariableMap avmap: A domains.ActiveVariable map that includes
        the active variable domain, which includes the active and inactive
        subspaces.
    :param int N: The number of points used in the design-of-experiments for
        constructing the response surface.
    :param int NMC: The number of points used to estimate the conditional
        expectation and conditional variance of the function given a value
        of the active variables. (Default is 10)

    :return: Y, N-by-n matrix that contains the design points in the space of
        active variables.
    :rtype: ndarray

    :return: X, (N*NMC)-by-m matrix that contains points in the simulation input
        space to run the simulation.
    :rtype: ndarray

    :return: ind, Indices that map points in `X` to points in `Y`.
    :rtype: ndarray

    **See Also**

    utils.designs.gauss_hermite_design
    utils.designs.interval_design
    utils.designs.maximin_design
    """


    if not isinstance(avmap, ActiveVariableMap):
        raise TypeError('avmap should be an ActiveVariableMap.')

    # interpret N as total number of points in the design
    if not isinstance(N, int):
        raise Exception('N should be an integer.')

    if not isinstance(NMC, int):
        raise Exception('NMC should be an integer.')

    m, n = avmap.domain.subspaces.W1.shape

    if isinstance(avmap.domain, UnboundedActiveVariableDomain):
        NN = [int(np.floor(np.power(N, 1.0/n))) for i in range(n)]
        Y = dn.gauss_hermite_design(NN)

    elif isinstance(avmap.domain, BoundedActiveVariableDomain):

        if n==1:
            a, b = avmap.domain.vertY[0,0], avmap.domain.vertY[1,0]
            Y = dn.interval_design(a, b, N)
        else:
            vertices = avmap.domain.vertY
            Y = dn.maximin_design(vertices, N)
    else:
        raise Exception('There is a problem with the avmap.domain.')

    X, ind = avmap.inverse(Y, NMC)
    return Y, X, ind

