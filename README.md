# Note

This repository is a fork of the [`active-subspaces`](https://github.com/paulcon/active_subspaces) Python package by Paul Constantine.

**I do not claim ownership or authorship.**

The original package currently contains a bug which renders its use on Windows platforms unworkable.
I have contributed a fix and submitted a [PR](https://github.com/paulcon/active_subspaces/pull/49) in June 2018, but this has yet to be merged.

A working version for Windows users is provided in this repository.

Install with

```bash
pip install https://github.com/ConnectedSystems/active_subspaces/zipball/master
```

The original readme is below.


# Description

*Active Subspaces* is a method for discovering low-dimensional subspaces of input parameters 
that accurately characterize the output of high-dimensional models. This dimension reduction 
can make otherwise infeasible parameter studies possible. This library contains 
python utilities for constructing and exploiting active subspaces.

WARNING: Development is very active right now, so the interfaces are far from
stable. It should settle down soon. Feel free to follow development.

# Status

[![Build Status](https://travis-ci.org/paulcon/active_subspaces.svg?branch=master)](https://travis-ci.org/paulcon/active_subspaces)

# Requirements and Dependencies

* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/), >= 0.15.0
* [matplotlib](http://matplotlib.org/)
* linear and quadratic program solvers
    - This package will look for [Gurobi](#gurobi) first and fall back to scipy, see [qp_solver.py](https://github.com/paulcon/active_subspaces/blob/master/active_subspaces/utils/qp_solver.py)

# Installation

```bash
> pip install active-subspaces
```

# Usage

For detailed examples of usage and results, see the Jupyter notebooks contained in the [active subspaces website]
(http://activesubspaces.org/applications/), the 'tutorials/test_functions' and 'tutorials/AS_tutorial' 
folders in this repo, and the [as-data-sets repo](https://github.com/paulcon/as-data-sets).

The core class used in this library is the Subspaces class contained in the 'subspaces.py' file. An instance of this class can compute 
the active subspace with a variety of methods that take either an array of gradients or input/output pairs. It contains the estimated 
eigenvalues (and bootstrap ranges), subspace errors (and bootstrap ranges), eigenvectors, and an array of the eigenvectors defining the 
active subspace. The 'utils/plotters.py' file contains functions to make plots of these quantities and summary plots that show model 
output against the first 1 or 2 active variables. The 'utils/response_surfaces.py' file contains classes for polynomial and radial-basis 
approximations that can be trained with input/output pairs. Both classes can predict the value and gradient at input points and have an 
Rsqr value that measures goodness-of-fit. The 'integrals.py' and 'optimizers.py' files contain functions for integrating and optimizing 
functions of the active variables, and rely on classes from the 'domains.py' file. The 'base.py' file contains the ActiveSubspaceReducedModel 
class, which has a simple interface for the functionality just described, but does not give the user as much freedom in the construction of 
the active subspace and surrogate models.

# Documentation

`active_subspaces` uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation. To build the html versions of the docs simply:

```bash
> cd docs
> make html
```

The generated html can be found in `docs/buidl/html`. Open up the `index.html` you find there to browse

# Testing

We are using Travis CI for continusous intergration testing. You can check out the current status [here](https://travis-ci.org/paulcon/active_subspaces).

To run tests locally:

```bash
> python test.py
```

# Distribution

We are following the guidelines [here](https://packaging.python.org/en/latest/distributing/) for packaging and distributing our code. Our steps:

1. `pip install twine`
2. Bump version in `setup.py`
3. From root directory, `python setup.py sdist`
4. `twine upload dist/active_subspaces-<version>.tar.gz`
    * This requires pypi username and password

# Gurobi

We are using [Gurobi](http://www.gurobi.com/); you can see the [qp_solver.py](https://github.com/paulcon/active_subspaces/blob/master/active_subspaces/utils/qp_solver.py) that solves the problems I need. To get Gurobi's Python interface working with Enthought/Canopy, take a look at this thread:
https://groups.google.com/forum/#!searchin/gurobi/canopy/gurobi/ArCkf4a40uU/R9U1XFuMJEkJ

# Notes

Right now I'm using [Enthought's Python Distribution](https://www.enthought.com/products/epd/) and [Canopy](https://www.enthought.com/products/canopy/) for development. You'll need numpy and scipy for these tools.

# Community Guidelines

If you have contributions, questions, or feedback, use the [Github repo](https://github.com/paulcon/active_subspaces) or contact 
[*Paul Constantine*](http://inside.mines.edu/~pconstan/) at [Colorado School of Mines](https://www.mines.edu/).
