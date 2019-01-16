# Description

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.158941.svg)](https://doi.org/10.5281/zenodo.158941)

[Active subspaces](http://activesubspaces.org) are part of an emerging set of tools for discovering low-dimensional structure in a given function of several variables. Interesting applications arise in deterministic computer simulations of complex physical systems, where the function is the map from the physical model's input parameters to its output quantity of interest. The active subspace is the span of particular directions in the input parameter space; perturbing the inputs along these *active* directions changes the output more, on average, than perturbing the inputs orthogonally to the active directions. By focusing on the model's response along active directions and ignoring the relatively inactive directions, we *reduce the dimension* for parameter studies---such as optimization and integration---that are essential to engineering tasks such as design and uncertainty quantification.

This library contains Python tools for discovering and exploiting a given model's active subspace. The user may provide a function handle to a complex model or its gradient with respect to the input parameters. Alternatively, the user may provide a set of input/output pairs from a previously executed set of runs (e.g., a Monte Carlo or Latin hypercube study). Essential classes and methods are documented; see documentation below. We also provide a set of [Jupyter](http://jupyter.org/) notebooks that demonstrate how to apply the code to a given model.

To see active subspace in action on real science and engineering applications, see the [Active Subspaces Data Sets](https://github.com/paulcon/as-data-sets) repository, which contains several Jupyter notebooks applying the methods to study input/output relationships in complex models.

# Testing

[![Build Status](https://travis-ci.org/paulcon/active_subspaces.svg?branch=master)](https://travis-ci.org/paulcon/active_subspaces)

We are using Travis CI for continuous integration testing. You can check out the current status [here](https://travis-ci.org/paulcon/active_subspaces).

To run tests locally:

```bash
> python test.py
```

# Requirements and Dependencies

* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/), >= 0.15.0
* [matplotlib](http://matplotlib.org/)
* [Gurobi](http://www.gurobi.com/) is an _optional_ dependency. The same functionality is accomplished with SciPy's [optimize](http://docs.scipy.org/doc/scipy/reference/optimize.html) package, albeit less accurately (particularly in the quadratic programs).

If you wish to use Gurobi, you will need to install it separately by following the instructions contained in their quick-start guides for [Windows](http://www.gurobi.com/documentation/6.5/quickstart_windows.pdf), [Linux](http://www.gurobi.com/documentation/6.5/quickstart_linux.pdf), or [Mac](http://www.gurobi.com/documentation/6.5/quickstart_mac.pdf). To test your installation of Gurobi, start a python interpreter and execute the command `import gurobipy`. If there is no import exception, the active subspaces library will be able to use Gurobi.

We had some initial trouble getting the Gurobi Python interface working with Enthought Python; Gurobi formally supports as subset of Python distributions, and Enthought is not one of them. However, we were able to get the interface working by following instructions in this [thread](https://groups.google.com/forum/#!searchin/gurobi/canopy/gurobi/ArCkf4a40uU/R9U1XFuMJEkJ).

# Installation

To install the active subspaces package, open the terminal/command line and clone the repository with the command

```bash
git clone https://github.com/paulcon/active_subspaces.git
```

Navigate into the `active_subspaces` folder (where the `setup.py` file is located) and run the command

```bash
python setup.py install
```

You should now be able to import the active subspaces library in Python scripts and interpreters with the command `import active_subspaces`.

This method was tested on Windows 7 Professional, and Ubuntu 14.04 LTS, and Mac OSX El Capitan with the [Enthought](https://www.enthought.com/) Python distribution (Python 2.7.11, NumPy 1.10.4, SciPy 0.17.0, and matplotlib 1.5.1).

This method was also tested on Ubuntu 18.04 with Python 3.6 and Python 2.7 using the [Conda](https://anaconda.org) Python distribution.

# Usage

For detailed examples of usage and results, see the Jupyter notebooks contained in the `tutorials` directory, the [active subspaces website]
(http://activesubspaces.org/applications/), and the Active Subspaces Data Sets [repo](https://github.com/paulcon/as-data-sets).

The core class is the Subspaces class contained in the `subspaces.py` file. An instance of this class can compute the active subspace with a variety of methods that take either an array of gradients or input/output pairs. It contains the estimated eigenvalues (and bootstrap ranges), subspace errors (and bootstrap ranges), eigenvalues, and an array of the eigenvectors defining the active subspace. The `utils/plotters.py` file contains functions to plot these quantities and produce summary plots that show model output against the first 1 or 2 active variables. The `utils/response_surfaces.py` file contains classes for polynomial and radial-basis approximations that can be trained with input/output pairs. Both classes can predict the value and gradient at input points and have a coefficient of determination (R-squared) value that measures goodness-of-fit. The `integrals.py` and `optimizers.py` files contain functions for integrating and optimizing functions of the active variables; these rely on classes from the `domains.py` file.

# Documentation

[![Documentation Status](https://readthedocs.org/projects/active-subspaces/badge/?version=latest)](http://active-subspaces.readthedocs.io/en/latest/?badge=latest)

Documentation can be found on [ReadTheDocs](http://active-subspaces.readthedocs.io/en/latest/).

# Community Guidelines

To contribute to this project, please follow these steps. Thanks to [Marco Tezzele](https://github.com/mtezzele) for providing this helpful template.

## Submitting a patch

1. Open a new issue describing the bug to fix or feature to add. Even if you think it's relatively minor, it's helpful to know what people are working on.
2. Follow the normal process of [forking][] the project, and set up a new branch to work in.  It's important that each group of changes be done in separate branches to ensure that a pull request only includes the commits related to that bug or feature.
3. Significant changes should be accompanied by tests. The project already has good test coverage, so look at some of the existing tests if you're unsure how to go about it. 
4. Push the commits to your fork and submit a [pull request][]. Please, remember to rebase properly in order to maintain a clean, linear git history.

[forking]: https://help.github.com/articles/fork-a-repo
[pull request]: https://help.github.com/articles/creating-a-pull-request

If you have questions or feedback, contact [*Paul Constantine*](http://inside.mines.edu/~pconstan/).

# Acknowledgments

This material is based upon work supported by the U.S. Department of Energy Office of Science, Office of Advanced Scientific Computing Research, Applied Mathematics program under Award Number DE-SC-0011077.
