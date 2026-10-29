# active_subspaces

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.158941.svg)](https://doi.org/10.5281/zenodo.158941)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.00079/status.svg)](https://doi.org/10.21105/joss.00079)
[![CI](https://github.com/paulcon/active_subspaces/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/paulcon/active_subspaces/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/active-subspaces/badge/?version=latest)](https://active-subspaces.readthedocs.io/en/latest/)

Python tools for discovering and exploiting **active subspaces**: low-dimensional structure in a
function of many inputs.

The active subspace of a function f(x) is spanned by the leading eigenvectors of the matrix
C = E[∇f ∇fᵀ], the average outer product of the gradient over the distribution of the inputs
(gradients taken with respect to the normalized inputs).
Perturbing the inputs along these directions changes f more, on average, than perturbing them
along the remaining directions. When a few eigenvalues dominate, parameter studies such as
integration, optimization and surrogate construction can work on a few active variables instead of
all the inputs.

The library

- estimates the subspace from gradients, from input/output pairs or from a model, with bootstrap
  intervals and rules for choosing its dimension; alternatives include sufficient dimension
  reduction estimators (OLS, QPHD, SIR, pHd, SAVE, OPG);
- describes the geometry of the active variables (intervals, zonotopes) and maps between full and
  active coordinates;
- fits polynomial and radial-basis surrogates on the active variables;
- integrates and minimizes functions by exploiting the subspace;
- builds a reduced model in one call;
- plots the results with matplotlib.

Inputs live in an explicit input space (`Uniform` or `Gaussian`). You pass data in physical
units; the library normalizes it, and every result states whether its arrays are in physical or
normalized coordinates. Functions are stateless and return frozen result objects that can be saved
and loaded.

## Install

Until version 1.0 is on PyPI, install it from GitHub; `pip install active_subspaces` still gives the
2016 release.

```sh
pip install "active_subspaces @ git+https://github.com/paulcon/active_subspaces@main"
```

Python 3.11 or later, with NumPy, SciPy and matplotlib.

## Example

```python
import numpy as np
import active_subspaces as ac

wing = ac.test_functions.WING_WEIGHT          # built-in model: 10 inputs, a Uniform space
rng = np.random.default_rng(42)
X = wing.space.sample(300, rng=rng)            # physical inputs, shape (300, 10)
G = ac.evaluate_grads(wing, X=X)               # physical gradients, shape (300, 10)

sub = ac.active_subspace(grads=G, space=wing.space, nboot=200, rng=rng)
print(sub.n, sub.eigenvalues[:3])              # chosen dimension and leading eigenvalues
fig, ax = ac.plotting.plot_eigenvalues(sub)    # a matplotlib Figure; nothing is shown or saved

res = ac.integrate(wing, domain=ac.active_domain(sub), n_nodes=10, rng=rng)
print(res.estimate, res.inactive_std_error)    # mean of f over the input space
```

## Documentation

- Documentation: <https://active-subspaces.readthedocs.io>
- Migration from the 2016 version: <https://active-subspaces.readthedocs.io/en/stable/migration.html>
- Changes: [CHANGELOG.md](https://github.com/paulcon/active_subspaces/blob/main/CHANGELOG.md)
- Applications on real models: [paulcon/as-data-sets](https://github.com/paulcon/as-data-sets)

## Citing

If you use this library, please cite the JOSS article:

> P. Constantine, R. Howard, A. Glaws, Z. Grey, P. Diaz, L. Fletcher. Python Active-subspaces
> Utility Library. *Journal of Open Source Software* 1(5):79, 2016.
> [doi:10.21105/joss.00079](https://doi.org/10.21105/joss.00079)

The software is archived on Zenodo ([doi:10.5281/zenodo.158941](https://doi.org/10.5281/zenodo.158941)).
[CITATION.cff](https://github.com/paulcon/active_subspaces/blob/main/CITATION.cff) holds the
full author list; GitHub's "Cite this repository" button exports it. For the methods, see P. G. Constantine, *Active Subspaces: Emerging Ideas for Dimension
Reduction in Parameter Studies*, SIAM, 2015.

## Contributing

Contributions are welcome: see
[CONTRIBUTING.md](https://github.com/paulcon/active_subspaces/blob/main/CONTRIBUTING.md) and the
[code of conduct](https://github.com/paulcon/active_subspaces/blob/main/CODE_OF_CONDUCT.md).
Report security issues as described in
[SECURITY.md](https://github.com/paulcon/active_subspaces/blob/main/SECURITY.md). Contributors are
listed in [AUTHORS.md](https://github.com/paulcon/active_subspaces/blob/main/AUTHORS.md).

## The 2016 version

Version 1.0 is a rewrite for Python 3. It changes the API and has no compatibility layer; the
[migration guide](https://active-subspaces.readthedocs.io/en/stable/migration.html) maps every 2016
name to its replacement and says which numbers change and why. The 2016 code, its tutorials and its
Python 2.7 requirements remain available at the git tags
[`legacy-master`](https://github.com/paulcon/active_subspaces/tree/legacy-master) (the last 2016
`master`), [`legacy-dev`](https://github.com/paulcon/active_subspaces/tree/legacy-dev) (the
`dev` branch) and [`0.1`](https://github.com/paulcon/active_subspaces/tree/0.1) (the original
release). The tag [`legacy-py3`](https://github.com/paulcon/active_subspaces/tree/legacy-py3/legacy)
holds unofficial Python 3 ports of both branches, used only to capture the regression baselines.

## AI assistance

Version 1.0 was developed with AI assistance (Anthropic's Claude) and reviewed by the maintainer.
The maintainer ruled on every design decision, and on every point where the code departs from a
published method. AI-assisted commits carry a `Co-Authored-By` trailer; AI tools are not listed as
authors.

## License and acknowledgments

MIT; see [LICENSE](https://github.com/paulcon/active_subspaces/blob/main/LICENSE).

The 2016 library was based upon work supported by the U.S. Department of Energy Office of Science,
Office of Advanced Scientific Computing Research, Applied Mathematics program under Award Number
DE-SC-0011077.
