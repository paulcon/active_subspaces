# Description

Python utilities for working with active subspaces.

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

# Contact

*Paul Constantine* at [Colorado School of Mines](https://www.mines.edu/). Google me for contact info.
