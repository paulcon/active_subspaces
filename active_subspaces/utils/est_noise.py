"""Utility for estimating the noise in a function at a point"""
from __future__ import division
import numpy as np
from simrunners import SimulationRunner
import warnings

def estimate_noise(f, x, p = None, nf = 9, h = 1e-2, max_recursion = 5, previous_h = None): 
	"""Estimate the noise in a function near x in direction p 

	This code follows the work of More and Wild, in particular, 
	the function ECNoise.m, avalible from:

		http://www.mcs.anl.gov/~wild/cnoise/

	The primary change has been to incorporate the advice on scaling the step size
	h so that an accurate estimation has been made.

	TODO: How do we ensure we do not step outside the domain of the function?

	Parameters:
	----------	
		f: SimulationRunner
			Function whose noise we are attempting to estimate
		x: np.ndarray
			Coordinates where to evaluate the function
		p: np.ndarray
			Direction in which to evaluate the function (optional)
		nf: int
			Number of function evaluations to perform (default 9)
		h: float
			Starting stepsize
		max_recursion : int
			Maximum number or recursions allowed. This is used internally
			to prevent infinite recursion.

	Returns:
	-------	
		sigma: float
			Estimation of the standard deviation of the nosie	
	"""


	if previous_h is None:
		previous_h = [h]
	else:
		previous_h.append(h)

	n = x.shape[0]

	if p is None:
		# Construct a random direction on the unit sphere
		p = np.random.randn(n)
		p /= np.linalg.norm(p)

	# Compute the points on which to evaluate the function
	X = np.zeros((nf, n))
	for i, delta in enumerate(np.linspace(-h/2., h/2., num = nf)):
		X[i,:] = x + delta*p

	# Run the simulation 
	# TODO: Running in parallel breaks the test b/c each instatiation starts with the same seed.
	F = f.run(X)
	
	# Check that the range of function values is sufficiently small
	fmin, fmax = min(F), max(F)
	# This may trigger a divide by zero warning, so we ignore it, being OK with infinite values
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', RuntimeWarning)
		# np.errstate(divide = 'ignore', invalid = 'ignore' ):
		norm_range = (fmax - fmin)/max(abs(fmax), abs(fmin))
	if norm_range > 1. and len(previous_h) < max_recursion:
		# In this case More and Wild consider that noise has not been detected (inform=3)
		# and recommend retrying with h = h/100
		if h/100. not in previous_h:
			print "h too large, re-running with smaller h"
			return estimate_noise(f, x, p = p, nf = nf, h = h/100., 
						max_recursion = max_recursion, previous_h = previous_h)
	
	# h is too small if half the function values are equal
	if np.sum(np.diff(F.flatten()) == 0) >= nf/2. and max_recursion > 0:
		if h*100. not in previous_h:
			print "h too small, re-running with larger h"
			return estimate_noise(f, x, p = p, nf = nf, h = h*100., 
				max_recursion = max_recursion, previous_h = previous_h)


	# Construct the finite difference table
	DF_table = np.nan * np.ones((nf, nf), dtype = np.float)
	DF_table[:,0] = F.flatten()
	gamma = 1.		
	noise_at_level = np.zeros((nf,), dtype = np.float)
	sign_change = np.zeros((nf,), dtype = np.bool)
	for i in range(1,nf):
		gamma *= 0.5*(i/(2.*i-1))
		for j in range(nf - i):
			DF_table[j,i] = DF_table[j+1,i-1] - DF_table[j,i-1]


		noise_at_level[i] = np.sqrt(gamma*np.mean(DF_table[:nf-i, i]**2))
		emin = np.min(DF_table[:nf-i,i])
		emax = np.max(DF_table[:nf-i,i])
		sign_change[i] = (emin * emax < 0)

	
	# Determine which level of noise to use
	for k in range(nf - 3):
		emin = min(noise_at_level[k:k+2])
		emax = max(noise_at_level[k:k+2])
		if (emax < 4*emin and sign_change[k]):
			return noise_at_level[k]

	# If none works, shrink the interval 
	if len(previous_h) < max_recursion:
		print "h too large, re-running with smaller h (post-eval check)"
		return estimate_noise(f, x, p = p, nf = 2*nf, h = h/10., max_recursion = max_recursion, 
				previous_h = previous_h)	
	else:
		raise StandardError('Could not find an appropreate step size for the More-Wild algorithm')
		print DF_table
		print sign_change
		print noise_at_level
		
	

