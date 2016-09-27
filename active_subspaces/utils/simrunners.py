"""Utilities for running several simulations at different inputs."""

import numpy as np
import multiprocessing as mp
import time
from misc import process_inputs

class SimulationRunner():
    """A class for running several simulations at different input values.

    Attributes
    ----------
    fun : function 
        runs the simulation for a fixed value of the input parameters, given as
        an ndarray

    See Also
    --------
    utils.simrunners.SimulationGradientRunner

    Notes
    -----
    The function fun should take an ndarray of size 1-by-m and return a float.
    This float is the quantity of interest from the simulation. Often, the
    function is a wrapper to a larger simulation code.
    """
    fun = None

    def __init__(self, fun):
        """Initialize a SimulationRunner.

        Parameters
        ----------
        fun : function  
            a function that runs the simulation for a fixed value of the input 
            parameters, given as an ndarray. This function returns the quantity 
            of interest from the model. Often, this function is a wrapper to a 
            larger simulation code.
        """
        if not hasattr(fun, '__call__'):
            raise TypeError('fun should be a callable function.')

        self.fun = fun

    def run(self, X, parallel=True, num_cores=mp.cpu_count()-1):
        """Run the simulation at several input values.
        
        Parameters
        ----------
        X : ndarray
            contains all input points where one wishes to run the simulation. If
            the simulation takes m inputs, then `X` must have shape M-by-m, 
            where M is the number of simulations to run.
        parallel : bool
            boolean value indicating whether to use parallel computation (True) 
            or not (False). Defaults to True.
        num_cores : int
            The number of cores to use for parallel computation. Defaults to the 
            number of cpu's available minus 1.

        Returns
        -------
        F : ndarray 
            contains the simulation output at each given input point. The shape 
            of `F` is M-by-1.

        Notes
        -----
        In principle, the simulation calls can be executed independently and in
        parallel. Right now this function uses a sequential for-loop. Future
        development will take advantage of multicore architectures to
        parallelize this for-loop.
        """

        X, M, m = process_inputs(X)
        F = np.zeros((M, 1))
        
        # Use parallel computing if desired and num_cores makes sense
        if parallel and isinstance(num_cores, int)\
        and num_cores >= 1 and num_cores <= mp.cpu_count():
            pool = mp.Pool(processes=num_cores)
            F = np.array(pool.map(self.fun, X)).reshape((M, 1))
            pool.close()
            pool.join()
            return F


        # TODO: provide some timing information
        # start = time.time()
        
        for i in range(M):
            F[i] = self.fun(X[i,:].reshape((1,m)))
            
        # TODO: provide some timing information
        # end = time.time() - start

        return F

class SimulationGradientRunner():
    """Evaluates gradients at several input values.
    
    
    A class for running several simulations at different input values that
    return the gradients of the quantity of interest.

    Attributes
    ----------
    dfun : function 
        a function that runs the simulation for a fixed value of the input 
        parameters, given as an ndarray. It returns the gradient of the quantity
        of interest at the given input.

    See Also
    --------
    utils.simrunners.SimulationRunner

    Notes
    -----
    The function dfun should take an ndarray of size 1-by-m and return an
    ndarray of shape 1-by-m. This ndarray is the gradient of the quantity of
    interest from the simulation. Often, the function is a wrapper to a larger
    simulation code.
    """
    dfun = None

    def __init__(self, dfun):
        """Initialize a SimulationGradientRunner.

        Parameters
        ----------
        dfun : function 
            a function that runs the simulation for a fixed value of the input 
            parameters, given as an ndarray. It returns the gradient of the 
            quantity of interest at the given input.
        """
        if not hasattr(dfun, '__call__'):
            raise TypeError('fun should be a callable function.')

        self.dfun = dfun

    def run(self, X, parallel=True, num_cores=mp.cpu_count()-1):
        """Run at several input values.
        
        Run the simulation at several input values and return the gradients of
        the quantity of interest.

        Parameters
        ----------
        X : ndarray 
            contains all input points where one wishes to run the simulation. 
            If the simulation takes m inputs, then `X` must have shape M-by-m, 
            where M is the number of simulations to run.
        parallel : bool
            boolean value indicating whether to use parallel computation (True) 
            or not (False). Defaults to True.
        num_cores : int
            The number of cores to use for parallel computation. Defaults to the 
            number of cpu's available minus 1.


        Returns
        -------
        dF : ndarray 
            contains the gradient of the quantity of interest at each given 
            input point. The shape of `dF` is M-by-m.

        Notes
        -----
        In principle, the simulation calls can be executed independently and in
        parallel. Right now this function uses a sequential for-loop. Future
        development will take advantage of multicore architectures to
        parallelize this for-loop.
        """

        X, M, m = process_inputs(X)
        dF = np.zeros((M, m))

        # Use parallel computing if desired and num_cores makes sense
        if parallel and isinstance(num_cores, int)\
        and num_cores >= 1 and num_cores <= mp.cpu_count():
            pool = mp.Pool(processes=num_cores)
            dF = np.array(pool.map(self.dfun, X)).squeeze()
            pool.close()
            pool.join()
            return dF

        # TODO: provide some timing information
        # start = time.time()
        
        for i in range(M):
            df = self.dfun(X[i,:].reshape((1,m)))
            dF[i,:] = df.reshape((1,m))
        
        # TODO: provide some timing information
        # end = time.time() - start

        return dF
