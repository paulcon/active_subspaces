""" A basic class for approximation, integration, and optimization with
active subspaces."""
import numpy as np
from utils.misc import process_inputs_outputs, process_inputs
from utils.simrunners import SimulationRunner, SimulationGradientRunner
from utils.plotters import eigenvalues, subspace_errors, eigenvectors, sufficient_summary
from response_surfaces import ActiveSubspaceResponseSurface
from integrals import integrate, av_integrate
from optimizers import minimize
from subspaces import Subspaces
from gradients import local_linear_gradients, finite_difference_gradients
from domains import UnboundedActiveVariableDomain, BoundedActiveVariableDomain, \
                    UnboundedActiveVariableMap, BoundedActiveVariableMap
import time

class ActiveSubspaceReducedModel():
    """
    A class for approximation, optimization, and integration with active
    subspaces.

    :cvar bool bounded_inputs: flag that tells if the simulation's inputs
        are bounded by an m-dimensional hypercube and equipped with a
        uniform probability density function (True) or if they are unbounded
        and equipped with a standard Gaussian density function (False).
    :cvar ndarray X: M-by-m matrix that contains a set of simulation inputs.
        Each row of `X` is a point in the simulations m-dimensional parameter
        space. These points are used to train response surfaces. They are also
        used in plots when analyzing the simulation.
    :cvar ndarray f: M-by-1 matrix that contains the simulation outputs that
        correspond to the rows of `X`. `f` is used as training data for
        response surfaces.
    :cvar int m: The dimension of the simulation inputs.
    :cvar int n: The dimension of the active subspace. Typically `n` is less
        than `m`---often much less.
    :cvar function fun: A function that interfaces with the simulation. It
        should take an ndarray of shape 1-by-m (e.g., a row of `X`), and it
        should return a scalar. That scalar is the quantity of interest from
        the simulation.
    :cvar function dfun: A function that interfaces with the simulation. It
        should take an ndarray of shape 1-by-m (e.g., a row of `X`), and it
        should return the gradient of the quantity of interest as an ndarray of
        shape 1-by-m.
    :cvar active-subspaces.response_surfaces.ActiveSubspaceResponseSurface as_respsurf:
        A response surface initialized and trained while building the model.
        Once trained, it can be used as a cheap surrogate for the simulation
        code.
    :cvar float Rsqr: The R-squared coefficient for the response surface.

    **Notes**

    This class contains several convenient methods for working with active
    subspaces. These methods are mostly wrappers to other parts of the library.
    The methods are structured to be used with relatively little knowledge of
    how active subspaces work. At best, this class provides a useful surrogate
    for an expensive simulation with many input parameters.

    There are two ways to build the model: one that uses given data (see the
    method `build_from_data`) and the other that uses given interfaces to the
    simulation (see the method `build_from_interface`). With a simulation
    interface, the code is able to build response surfaces on its own design
    sets, i.e., its own choice of points. Also, the choices for building
    response surfaces are better with a given interface. However, one can still
    learn a great deal about the simulation from a set of input/output pairs.

    Once the model is built and the response surface is trained, one can use
    the methods to estimate the average, probabilities, and the minimum of the
    simulation quantity of interest as a function of the simulation inputs.

    If the model is built from given input/output pairs, then `Rsqr` is the
    R-squared coefficient from the response surface on the active variables. If
    the model is built from a given interface, the R-squared coefficient is
    computed from the output samples computed along with the gradients while
    estimating the active subspace. This is a big advantage of building the
    response surface with an interface.
    """
    bounded_inputs = None
    X, f = None, None
    m, n = None, None
    fun, dfun = None, None
    as_respsurf = None
    Rsqr = None

    def __init__(self, m, bounded_inputs):
        """
        Initialize the ActiveSubspaceReducedModel.

        :param bool bounded_inputs: A flag that tells if the simulation's
            inputs are bounded by an m-dimensional hypercube and equipped
            with a uniform probability density function (True) or if they
            are unbounded and equipped with a standard Gaussian density
            function (False).
        :param int m: The number of the simulation inputs, i.e., the dimension
            of the space of simulation inputs.

        **Notes**

        An ActiveSubspaceReducedModel cannot exist without knowing some
        information about the inputs to the simulation that the model is meant
        to represent. `m` tells the model how many inputs the simulation takes.

        There are two cases the code is set to handle respresented by the
        boolean variable `bounded_inputs`. If `bounded_inputs` is True, then the
        model assumes that the simulation inputs each take values in the
        interval [-1,1]. This implies that the domain of the simulation quantity
        of interest is the m-dimensional hypercube centered at the origin. In
        reality, the simulation inputs may take values in some other shifted
        and scaled hyperrectangle. We assume there is an implied shifting and
        scaling from the simulation's natural inputs to the hypercube. We
        assume that the hypercube is equipped with a uniform probability
        density function that is 2^(-m) on the hypercube and zero outside it.
        (We could talk for hours about whether or not this means that the
        simulation inputs are "random." But let's avoid that discussion for
        now.)

        The second case is when `bounded_inputs` is False. In this case, the
        simulation inputs are assumed unbounded. The simulation input space
        is equipped with a standard Gaussian density function. Again, this
        lets us compute integrals. The simulation's true inputs may be
        unbounded with some correlation. In this case, we assume that the
        inputs have been properly transformed to a standard Gaussian.
        """
        if not isinstance(m, int):
            raise TypeError('m must be an integer.')
        else:
            self.m = m

        if not isinstance(bounded_inputs, bool):
            raise TypeError('bounded_inputs must be a boolean.')
        else:
            self.bounded_inputs = bounded_inputs

    def build_from_data(self, X, f, df=None, avdim=None):
        """
        Build the active subspace-enabled model with input/output pairs.

        :param ndarray X: M-by-m matrix with evaluations of the m-dimensional
            simulation inputs.
        :param ndarray f: M-by-1 matrix with corresponding simulation quantities
            of interest.
        :param ndarray df: M-by-m matrix that contains the gradients of the
            simulation quantity of interest, oriented row-wise, that correspond
            to the rows of `X`. If `df` is not present, then it is estimated
            with crude local linear models using the pairs `X` and `f`.
        :param int avdim: The dimension of the active subspace. If `avdim`
            is not present, a crude heuristic is used to choose an active
            subspace dimension based on the given data `X` and
            `f`---and possible `df`.

        **Notes**

        This method follows these steps:

        #. If `df` is None, estimate it with local linear models using the \
        input/output pairs `X` and `f`.
        #. Compute the active and inactive subspaces using `df`.
        #. Train a response surface using `X` and `f` that exploits the active \
        subspace.

        """
        X, f, M, m = process_inputs_outputs(X, f)

        # check if the given inputs satisfy the assumptions
        if self.bounded_inputs:
            if np.any(X) > 1.0 or np.any(X) < -1.0:
                raise Exception('The supposedly bounded inputs exceed the \
                    bounds [-1,1].')
        else:
            if np.any(X) > 10.0 or np.any(X) < -10.0:
                raise Exception('There is a very good chance that your \
                    unbounded inputs are not properly scaled.')
        self.X, self.f, self.m = X, f, m

        if df is not None:
            df, M_df, m_df = process_inputs(df)
            if m_df != m:
                raise ValueError('The dimension of the gradients should be \
                                the same as the dimension of the inputs.')
        else:
            # if gradients aren't available, estimate them from data
            df = local_linear_gradients(X, f)


        # compute the active subspace
        ss = Subspaces()
        ss.compute(df=df, nboot=100)
        if avdim is not None:
            if not isinstance(avdim, int):
                raise TypeError('avdim should be an integer.')
            else:
                ss.partition(avdim)
        self.n = ss.W1.shape[1]
        print 'The dimension of the active subspace is {:d}.'.format(self.n)

        # set up the active variable domain and map
        if self.bounded_inputs:
            avdom = BoundedActiveVariableDomain(ss)
            avmap = BoundedActiveVariableMap(avdom)
        else:
            avdom = UnboundedActiveVariableDomain(ss)
            avmap = UnboundedActiveVariableMap(avdom)

        # build the response surface
        asrs = ActiveSubspaceResponseSurface(avmap)
        asrs.train_with_data(X, f)

        # set the R-squared coefficient
        self.Rsqr = asrs.respsurf.Rsqr
        self.as_respsurf = asrs

    def build_from_interface(self, fun, dfun=None, avdim=None):
        """
        Build the active subspace-enabled model with interfaces to the
        simulation.

        :param function fun: A function that interfaces with the simulation.
            It should take an ndarray of shape 1-by-m (e.g., a row of `X`), and
            it should return a scalar. That scalar is the quantity of interest from the simulation.
        :param function dfun: A function that interfaces with the simulation.
            It should take an ndarray of shape 1-by-m (e.g., a row of `X`), and it
            should return the gradient of the quantity of interest as an ndarray of shape 1-by-m.
        :param int avdim: The dimension of the active subspace. If `avdim` is not
            present, it is chosen after computing the active subspaces using
            the given interfaces.

        **Notes**

        This method follows these steps:

        #. Draw random points according to the weight function on the space\
        of simulation inputs.
        #. Compute the quantity of interest and its gradient at the sampled\
        inputs. If `dfun` is None, use finite differences.
        #. Use the collection of gradients to estimate the eigenvectors and\
        eigenvalues that determine and define the active subspace.
        #. Train a response surface using the interface, which uses a careful\
        design of experiments on the space of active variables. This design\
        uses about 5 points per dimension of the active subspace.

        """
        if not hasattr(fun, '__call__'):
            raise TypeError('fun should be a callable function.')

        if dfun is not None:
            if not hasattr(dfun, '__call__'):
                raise TypeError('dfun should be a callable function.')

        if avdim is not None:
            if not isinstance(avdim, int):
                raise TypeError('avdim should be an integer')

        m = self.m

        # number of gradient samples
        M = int(np.floor(6*(m+1)*np.log(m)))

        # sample points for gradients
        if self.bounded_inputs:
            X = np.random.uniform(-1.0, 1.0, size=(M, m))
        else:
            X = np.random.normal(size=(M, m))

        fun = SimulationRunner(fun)
        f = fun.run(X)
        self.X, self.f, self.fun = X, f, fun

        # sample the simulation's gradients
        if dfun == None:
            df = finite_difference_gradients(X, fun)
        else:
            dfun = SimulationGradientRunner(dfun)
            df = dfun.run(X)
            self.dfun = dfun

        # compute the active subspace
        ss = Subspaces()
        ss.compute(df=df, nboot=100)

        if avdim is not None:
            ss.partition(avdim)
        self.n = ss.W1.shape[1]
        print 'The dimension of the active subspace is {:d}.'.format(self.n)

        # set up the active variable domain and map
        if self.bounded_inputs:
            avdom = BoundedActiveVariableDomain(ss)
            avmap = BoundedActiveVariableMap(avdom)
        else:
            avdom = UnboundedActiveVariableDomain(ss)
            avmap = UnboundedActiveVariableMap(avdom)

        # build the response surface
        asrs = ActiveSubspaceResponseSurface(avmap)
        asrs.train_with_interface(fun, int(np.power(5,self.n)))

        # compute testing error as an R-squared
        self.Rsqr = 1.0 - ( np.linalg.norm(asrs.predict(X)[0] - f)**2 \
                            / np.var(f) )

        self.as_respsurf = asrs

    def diagnostics(self):
        """
        Make plots that help determine the quality of the active subspace-
        enabled response surface and approximation.

        **Notes**

        This method produces four useful plots for verifying the quality of the
        active subspace-enable approximation.

        #. A semilog plot of the first 10 eigenvalues with their bootstrap ranges. \
        One is typically looking for large gaps between the\
        eigenvalues in the log space.
        #. A semilog plot of the estimated errors in the estimated active\
        subspace. This plot uses a bootstrap to estimate the errors.
        #. A plot of the components of the first four eigenvectors. These\
        components often reveal insights into the simulation's important\
        parameters.
        #. A 1d and a 2d summary plot of the computed quantity of interest at\
        different values of the first and second active variables. These plots\
        can be very useful in revealing the structure in the quantity of\
        interest as a function of the inputs.

        """
        ss = self.as_respsurf.avmap.domain.subspaces
        evalmin = min(10, ss.eigenvalues.size)
        eigenvalues(ss.eigenvalues[:evalmin,0], e_br=ss.e_br[:evalmin,:])
        subspace_errors(ss.sub_br[:evalmin,:])
        evecmin = min(4, ss.eigenvalues.size)
        eigenvectors(ss.eigenvectors[:,:evecmin])
        Y = np.dot(self.X, ss.eigenvectors[:,:2])
        sufficient_summary(Y, self.f)

    def predict(self, X, compgrad=False):
        """
        Compute the value of the response surface at given values of the
        simulation inputs.

        :param ndarray X: M-by-m matrix containing points in simulation's
            input space.
        :param bool compgrad: Determines if the gradient of the response surface is
            computed and returned. (Default is False)

        :return: response surface values at the given `X`.
        :return: df :  estimated gradient at the given `X`. If `compgrad` is False, then `df` is
            None.

        :rtype: response_surface
        :rtype: ndarray (M-by-m)

        **See Also**

        response_surfaces.ActiveSubspaceResponseSurface

        **Notes**

        The default response surface is a radial basis function approximation
        using an exponential-squared (i.e., Gaussian) radial basis. The
        eigenvalues from the active subspace analysis are used to determine the
        characteristic length scales for each of the active variables. In other
        words the radial basis is anisotropic, and the anisotropy is determined
        by the eigenvalues.

        The response surface also has a quadratic monomial basis. The
        coefficients of the monomial basis are fit with weighted least-squares.

        In practice, this is equivalent to a kriging or Gaussian process
        approximation. However, such approximations bring several assumptions
        about noisy data that are difficult, if not impossible, to verify in
        computer simulations. I chose to avoid the difficulties that come with
        such methods. That means there is no so-called prediction variance
        associated with the response surface prediction. Personally, I think
        this is better. The prediction variance has no connection to the
        approximation error, except in very special cases. I prefer not to
        confuse the user with things that look and smell like error bars but
        aren't actually error bars.
        """
        if not isinstance(compgrad, bool):
            raise TypeError('compgrad should be a boolean')

        X, M, m = process_inputs(X)

        if m != self.m:
            raise Exception('The dimension of the points is {:d} but should \
                be {:d}.'.format(m, self.m))
        f, df = self.as_respsurf.predict(X, compgrad=compgrad)
        return f, df

    def average(self, N):
        """
        Estimate the average of the simulation over the input space.

        **Parameters**

        :param int N: The number of function calls to use when estimating the
            average.

        :return: mu, estimated average of the quantity of interest as a
            function of the simulation inputs.
        :return: lb, estimated lower bound on `mu`. It comes from the
            variance of the Monte Carlo estimates when the model is built
            with a simulation interface. When no simulation interface is
            present, `lb` is None.
        :return: ub, estimated upper bound on `mu`. It comes from the
            variance of the Monte Carlo estimates when the model is built
            with a simulation interface. When no simulation interface is
            present, `ub` is None.
        :rtype: float
        :rtype: float
        :rtype: float

        **See Also**

        integrals.integrate

        **Notes**

        When the model is built with a given simulation interface, the
        quadrature rule for estimating the average includes simple Monte Carlo
        integration over the inactive variables. These bounds do not include
        errors from the quadrature rules on the active variables---only the
        variance due to random sampling in Monte Carlo over the inactive
        variables. Therefore, they should not be treated as precise bounds.

        When the model is built from data, `mu` is the average of the
        response surface. It's probably better to compute the average of the
        training data directly.
        """
        if not isinstance(N, int):
            raise TypeError('N should be an integer.')

        if N < 1:
            raise ValueError('N should positive')

        if self.fun is not None:
            mu, lb, ub = integrate(self.fun, self.as_respsurf.avmap, N)
        else:
            mu = av_integrate(self.as_respsurf, self.as_respsurf.avmap, N)
            lb, ub = None, None
        return mu, lb, ub

    def probability(self, lb, ub, M=10000):
        """
        Estimate the probably that the quantity of interest is within a given
        range.

        :param float lb: The lower bound on the interval.
        :param float ub: The upper bound on the interval.
        :param int M: The number of samples of the response surface.

        :return: p, The estimated probability that the quantity of interest
            falls in the given interval.
        :return: plb, A central limit theorem-based estimate of the lower 99%
            confidence bound on `p`. Note that this lower bound is only from
            the Monte Carlo used to estimate `p`. It does not include errors
            in the response surface.
        :return: pub, A central limit theorem-based estimate of the upper 99%
            confidence bound on `p`. Note that this lower bound is only from
            the Monte Carlo used to estimate `p`. It does not include errors
            in the response surface.

        :rtype: float
        :rtype: float
        :rtype: float

        **Notes**

        This method estimates the probability
        P[lb <= f <= ub],
        where f is the response surface approximation to the simulation
        quantity of interest. It uses simple Monte Carlo to estimate this
        probabiliy, and it includes central limit theorem-based 99% confidence
        bounds.
        """
        if not isinstance(lb, float):
            if isinstance(lb, int):
                lb = float(lb)
            else:
                raise TypeError('lb should be a float')

        if not isinstance(ub, float):
            if isinstance(ub, int):
                ub = float(ub)
            else:
                raise TypeError('ub should be a float')

        if self.bounded_inputs:
            X = np.random.uniform(-1.0,1.0,size=(M,self.m))
        else:
            X = np.random.normal(size=(M,self.m))
        f = self.as_respsurf(X)
        c = np.all(np.hstack(( f>lb, f<ub )), axis=1)
        p = np.sum(c) / float(M)
        plb, pub = p+2.58*np.sqrt(p*(1-p)/M), p-2.58*np.sqrt(p*(1-p)/M)
        return p, plb, pub

    def minimum(self):
        """
        Estimate the minimum of the quantity of interest.

        :return: fstar, The estimated minimum of the quantity of interest over
            the simulation input space.
        :return: xstar, An ndarray of shape 1-by-m that is the approximate
            minimizer of the simulation quantity of interest.

        :rtype: float
        :rtype: ndarray

        **See Also**

        optimizers.minimize

        **Notes**

        This method uses the active subspace-enabled response surface to search
        for a minimizer in the simulation input space. It is only a heuristic.
        There is not guarantee that this method finds the true global
        minimum.
        """
        xstar, fstar = minimize(self.as_respsurf, self.X, self.f)
        return fstar, xstar

