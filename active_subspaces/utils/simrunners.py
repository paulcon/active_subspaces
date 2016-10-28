"""Utilities for running several simulations at different inputs."""

import numpy as np
import time
from misc import process_inputs
import warnings
import itertools
# checking to see if system has multiprocessing
try:
	import multiprocessing as mp
	HAS_MP = True
except ImportError, e:
	HAS_MP = False
	pass

try: 
	from celery_runner import celery_runner
	import marshal
	HAS_CELERY = True
except ImportError, e:
	HAS_CELERY = False	



class SimulationRunner():
	"""A class for running several simulations at different input values.

	Attributes
	----------
	fun : function 
		runs the simulation for a fixed value of the input parameters, given as
		an ndarray
	
	backend : {'loop', 'multiprocessing', 'celery'}
		Specifies how each evaluation of the function f should be run.
		
		* loop - use a for loop over fun
		* multiprocessing - distribute the function across multiple cores using 
			the multiprocessing library
		* celery - use the Celery distributed task queue to split up function
			evaluations

	See Also
	--------
	utils.simrunners.SimulationGradientRunner

	Notes
	-----
	The function fun should take an ndarray of size 1-by-m and return a float.
	This float is the quantity of interest from the simulation. Often, the
	function is a wrapper to a larger simulation code.
	"""

	def __init__(self, fun, backend = None, num_cores = None):
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

		# Default backend
		if backend is None:
			backend = 'loop'

		# Check the user has specified a valid backend
		if backend not in ['loop', 'multiprocessing', 'celery']:
			raise TypeError('Invalid backend chosen')

		# Check if the backend selected is avalible
		if backend == 'multiprocessing' and HAS_MP is False:
			backend = 'loop'
			warnings.warn('multiprocessing not avalible, defaulting to "loop" backend')
		elif backend == 'celery' and HAS_CELERY is False:
			backend = 'loop'
			warnings.warn('celery not avalible, defaulting to "loop" backend')

		self.backend = backend

		# Setup the selected backend
		if backend == 'loop':
			self.run = self._run_loop	
		elif backend == 'multiprocessing':
			if num_cores is None:
				num_cores = mp.cpu_count() - 1
			self.num_cores = num_cores
			self.run = self._run_multiprocessing
		elif backend == 'celery':
			self.run = self._run_celery


	def _format_output(self, output):
		# Format and store the output
		# We'll need to check the output size and build the matrix appropreately
		M = len(output)
		n_output = None
		for i, out in enumerate(output):
			# Find the dimenison of the output
			if n_output is None and out is not None:
				out = np.array(out).flatten()
				n_output = out.shape[0]
				F = np.zeros((M,n_output), dtype = out.dtype)
			if n_output is not None:
				if out is not None:
					F[i] = np.array(out).flatten()
				else: 
					F[i] = np.nan

		# If no evalution was successful
		if n_output is None:
			F = np.nan*np.ones((M,1))
		return F

	def _run_loop(self, X):
		""" Runs a simple for-loop over the target function
		"""
		X, M, m = process_inputs(X)
		# We store the output in a list so that we can handle failures of the function
		# to evaluate
		output = []
		for i in range(M):
			# Try to evaluate the function
			try:
				out = self.fun(X[i,:].reshape((1,m)))
			except:
				out = None
			output.append(out)

		return self._format_output(output)

	def _run_multiprocessing(self, X):
		pool = mp.Pool(processes = self.num_cores)
		try:
			if hasattr(self.fun, 'im_class'): 	# If the function is a member of a class
				arg_list_objects = []
				arg_list_inputs = []
				for i in range(M):
					arg_list_objects.append(self.fun.im_self)
					arg_list_inputs.append(X[i])
				#These are for parallel computation with a class method
				def target(): pass
				def target_star(args): return target(*args)
				target.__code__ = self.fun.im_func.__code__
				output = pool.map(target_star, itertools.izip(arg_list_objects, arg_list_inputs))
			else: 			# Just a plain function
				output = pool.map(self.fun, X)
		
			pool.close()
			pool.join()
			return self._format_output(output)	
		except:	# If there is a failure in multiprocessing, disable it and restart
			warnings.warn('multiprocessing failed; dropping to "loop" backend')
			self.run = self._run_loop
			self.backend = 'loop'
			return self.run(X)
			
	def _run_celery(self, X):
		X, M, m = process_inputs(X)
		# store the function
		marshal_func = marshal.dumps(self.fun.func_code)
		results = [celery_runner.delay(x, marshal_func) for x in X]

		# Time between checking for results
		tick = 0.1
		start_time = time.time()
		while True:
			# Check if everyone is done
			status = [res.ready() for res in results]
			if all(status):
				break
			else:
				time.sleep(tick - ((time.time() - start_time) % tick ))

		output = []
		for res in results:
			try:
				output.append(res.get())
			except:
				output.append(None)

		return self._format_output(output)
		

def SimulationGradientRunner(*args, **kwargs):
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

	NB JMH: This function is provided for compatability.  The output processing
	of SimulationRunner has been upgraded such that can handle gradient outputs
	just as well.  So, we've simply subclassed for now.
	"""
	from warnings import warn
	warn("this code is depricated")
	return SimulationRunner(*args, **kwargs)
