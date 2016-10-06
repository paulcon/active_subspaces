"""Utilities for running several simulations at different inputs."""

import numpy as np
import time
from misc import process_inputs

# checking to see if system has multiprocessing
try:
    HAS_MP = True
    import multiprocessing as mp
except ImportError, e:
    HAS_MP = False
    pass

#These are for parallel computation with a class method
def target(): pass
def target_star(args): return target(*args)

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

    def run(self, X, parallel=True, num_cores=None):
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
        To use parallel computation, it is recommended that the function used to initialize the 
        SimulationRunner object be a top-level function in the script creating it. It can also 
        be a class method, but performance is worse in this case. Additionally, due to the 
        implementation of the multiprocessing module, complex functions may not be passable to 
        worker processes, in which case a serial loop is used for the computation.
        """

        X, M, m = process_inputs(X)
        F = np.zeros((M, 1))

        # TODO: provide some timing information
        # start = time.time()
        
        # Set num_cores to its default value if multiprocessing is present
        # and the user hasn't specified a value.
        if parallel and HAS_MP and num_cores is None: num_cores = mp.cpu_count() - 1
        
        # Use parallel computing if desired and num_cores makes sense.
        if parallel and HAS_MP and isinstance(num_cores, int)\
        and num_cores >= 1 and num_cores <= mp.cpu_count():
            try:# Try using parallel computing
                if hasattr(self.fun, 'im_class'): # This executes if the function is a class method
                    import itertools
                    arg_list_objects = []
                    arg_list_inputs = []
                    for i in range(M):
                        arg_list_objects.append(self.fun.im_self)
                        arg_list_inputs.append(X[i])
                    target.__code__ = self.fun.im_func.__code__
                    pool = mp.Pool(processes=num_cores)
                    F = np.array(pool.map(target_star, itertools.izip(arg_list_objects, arg_list_inputs))).reshape((M, 1))
                    pool.close()
                    pool.join()
                else: # This executes otherwise
                    pool = mp.Pool(processes=num_cores)
                    F = np.array(pool.map(self.fun, X)).reshape((M, 1))
                    pool.close()
                    pool.join()
            except:# If there is an error, use a serial loop
                for i in range(M):
                    F[i] = self.fun(X[i,:].reshape((1,m)))
        else: # Otherwise use a serial loop.        
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

    def run(self, X, parallel=True, num_cores=None):
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
        To use parallel computation, it is recommended that the function used to initialize the 
        SimulationRunner object be a top-level function in the script creating it. It can also 
        be a class method, but performance is worse in this case. Additionally, due to the 
        implementation of the multiprocessing module, complex functions may not be passable to 
        worker processes, in which case a serial loop is used for the computation.
        """

        X, M, m = process_inputs(X)
        dF = np.zeros((M, m))

        # TODO: provide some timing information
        # start = time.time()

        # Set num_cores to its default value if multiprocessing is present
        # and the user hasn't specified a value.
        if parallel and HAS_MP and num_cores is None: num_cores = mp.cpu_count() - 1

        # Use parallel computing if desired and num_cores makes sense
        if parallel and HAS_MP and isinstance(num_cores, int)\
        and num_cores >= 1 and num_cores <= mp.cpu_count():
            try: # Try using parallel computing
                if hasattr(self.dfun, 'im_class'): # This executes if the fucntion is a class method
                    import itertools
                    arg_list_objects = []
                    arg_list_inputs = []
                    for i in range(M):
                        arg_list_objects.append(self.dfun.im_self)
                        arg_list_inputs.append(X[i])
                    target.__code__=self.dfun.im_func.__code__
                    pool = mp.Pool(processes=num_cores)
                    dF = np.array(pool.map(target_star, itertools.izip(arg_list_objects, arg_list_inputs))).squeeze()
                    pool.close()
                    pool.join()
                else: # This executes otherwise
                    pool = mp.Pool(processes=num_cores)
                    dF = np.array(pool.map(self.dfun, X)).squeeze()
                    pool.close()
                    pool.join()
            except: # If there is an error, use a serial loop
                for i in range(M):
                    df = self.dfun(X[i,:].reshape((1,m)))
                    dF[i,:] = df.reshape((1,m))
        else: # Otherwise use a serial loop.        
            for i in range(M):
                df = self.dfun(X[i,:].reshape((1,m)))
                dF[i,:] = df.reshape((1,m))
        
        # TODO: provide some timing information
        # end = time.time() - start

        return dF
