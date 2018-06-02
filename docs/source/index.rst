Python Active subspaces Utility Library
=======================================

`Active subspaces <http://activesubspaces.org/>`_ are part of an emerging set of tools for discovering low-dimensional structure in a given function of several variables. Interesting applications arise in deterministic computer simulations of complex physical systems, where the function is the map from the physical model's input parameters to its output quantity of interest. The active subspace is the span of particular directions in the input parameter space; perturbing the inputs along these *active* directions changes the output more, on average, than perturbing the inputs orthogonally to the active directions. By focusing on the model's response along active directions and ignoring the relatively inactive directions, we *reduce the dimension* for parameter studies---such as optimization and integration---that are essential to engineering tasks such as design and uncertainty quantification.

For more information on active subspaces, visit http://activesubspaces.org/ or purchase the book `Active Subspaces: Emerging Ideas in Dimension Reduction for Parameter Studies <http://bookstore.siam.org/sl02>`_ published by `SIAM <http://www.siam.org/>`_.

This library contains Python tools for discovering and exploiting a given model's active subspace. The user may provide a function handle to a complex model or its gradient with respect to the input parameters. Alternatively, the user may provide a set of input/output pairs from a previously executed set of runs (e.g., a Monte Carlo or Latin hypercube study).

Installation
^^^^^^^^^^^^

The github repository is https://github.com/paulcon/active_subspaces.git

To install the active subspaces package, open the terminal/command line and clone the repository with the command

::

	 git clone https://github.com/paulcon/active_subspaces.git

Navigate into the ``active_subspaces`` folder (where the ``setup.py`` file is located) and run the command

::

	 python setup.py install

You should now be able to import the active subspaces library in Python scripts and interpreters with the command ``import active_subspaces``.

Examples
^^^^^^^^

The ``tutorials`` directory contains several Jupyter notebooks with examples of the code usage. You may also visit the `Active Subspaces Data Sets <http://github.com/paulcon/as-data-sets/>`_ repository for examples of applying active subspaces to real science and engineering models.

For a quickstart, consider a bivariate quadratic function

::

	 import numpy as np
	 
	 def fun(x):
	     A = np.array([[4., 2.], [2., 1.1]])
	     return 0.5*np.dot(x.ravel(), np.dot(A, x.ravel()))

with gradient function

::
	 
	 def dfun(x):
	     A = np.array([[4., 2.], [2., 1.1]])
	     return np.dot(A, x.ravel()).reshape((2, 1))

Draw 50 samples from the function's domain, assumed to be the [-1,1]^2 box equipped with a uniform probability density function,

::
	 
	 X = np.random.uniform(-1., 1., size=(50, 2))

For each sample, compute the function and its gradient using the ``SimulationRunner`` and ``SimulationGradientRunner`` classes.

::
	 
	 import active_subspaces as acs

	 # evaluate the function
	 sr = acs.utils.simrunners.SimulationRunner(fun)
	 f = sr.run(X)

	 # evaluate the gradient
	 sgr = acs.utils.simrunners.SimulationGradientRunner(dfun)
	 df = sgr.run(X)

Compute the active subspace with the gradients.

::
	 
	 ss = acs.subspaces.Subspaces()
	 ss.compute(df=df, sstype='AS')

See the documentation for ``Subspaces.compute()`` for more details. Use the plotting routines to examine the estimated eigenvalues.

::
	 
	 acs.utils.plotters.eigenvalues(ss.eigenvals)

Make a one-dimensional summary plot

::
	 
	 y = np.dot(X, ss.W1)
	 acs.utils.plotters.sufficient_summary(y, f)

To exploit the one-dimensional active subspace, first set up the active variable domain and the map between the active variables and the full variables,

::
	 
	 # set up the active variable domain
	 avd = acs.domains.BoundedActiveVariableDomain(ss)

	 # set up the maps between active and full variables
	 avm = acs.domains.BoundedActiveVariableMap(avd)

To estimate an integral,

::
	 
	 N = 10 # number of active variable quadrature points
	 mu = acs.integrals.integrate(fun, avm, N)[0]
	 print 'Estimated integral: {:6.4f}'.format(mu)

To train and test a low-dimensional response surface,

::
	 
	 rs = acs.response_surfaces.ActiveSubspaceResponseSurface(avm)

	 # train with the interface
	 N = 10 # number of active variable training points
	 rs.train_with_interface(fun, N)

	 # or train with the existing runs
	 rs.train_with_data(X, f)
	 
	 # test
	 XX = np.random.uniform(-1., 1., size=(100, 2))
	 fXX = rs.predict(XX)
	 
Explore the documentation and Jupyter notebooks to see the code's full range of capabilities. 


Developer documentation
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 3

   code
   Contact
   LICENSE


Indexes
=======================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

