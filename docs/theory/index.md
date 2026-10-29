# Theory

These pages state the mathematics behind `active_subspaces`: what each function estimates, under
which assumptions, at what cost, and where the library departs from the published method. They are
written for readers who know linear algebra and probability. The user guide explains how to call
the functions; these pages explain what the numbers mean.

```{toctree}
:maxdepth: 1

active_subspaces
estimators
dimension
geometry
surrogates_and_designs
integration_and_optimization
reduced_model
estimator_selection
```

## Notation

| Symbol | Meaning | In the code |
|---|---|---|
| $m$ | number of inputs | `space.dim`, `result.m` |
| $n$ | dimension of the active subspace, $1\le n\le m$ | `result.n` |
| $M$ | number of samples | rows of `X`, `f`, `grads` |
| $\xi\in\mathbb R^m$ | a **physical** input, in the units of your model | `X` passed to any function |
| $x\in\mathbb R^m$ | a **normalized** input | arrays stored in results |
| $A$, $b$ | the affine map of the input space, $\xi=Ax+b$ | `space.scale`, `space.shift` |
| $\rho$ | the density of $x$: uniform on $[-1,1]^m$, or $N(0,I_m)$ | `Uniform`, `Gaussian` |
| $\Sigma$ | covariance of $\rho$: $\tfrac13I$ (uniform) or $I$ (Gaussian) | `space.reference_covariance` |
| $f$, $\nabla f$ | the function and its gradient with respect to $x$ | `f`, `grads` |
| $\omega_i$ | sample (quadrature) weights, $\omega_i\ge0$, $\sum_i\omega_i=1$ | `quad_weights` |
| $C=W\Lambda W^\top$ | the estimated matrix and its eigendecomposition | `eigenvalues`, `eigenvectors` |
| $W=[W_1\ W_2]$ | active ($m\times n$) and inactive ($m\times(m-n)$) eigenvectors | `W1`, `W2` |
| $y=W_1^\top x$, $z=W_2^\top x$ | active and inactive variables | `forward_map` |
| $\mathcal Y$ | the active domain, the set of possible $y$ | `active_domain` |

Indices in formulas start at 1 ($\lambda_1$ is the largest eigenvalue); array indices in code start
at 0 (`eigenvalues[0]`). Unless stated otherwise, $\lVert\cdot\rVert$ is the Euclidean norm of a
vector and the spectral norm of a matrix.

## Coordinates

Every computation is done in normalized coordinates $x$, and every result is stored there. You pass
physical data together with an input space, and the library maps it on entry. The eigenvalues are
defined only in normalized coordinates. Directions have a physical view, explained in
{ref}`theory-physical-directions`.

## Departures from the published methods

`active_subspaces` follows the published method for each algorithm it implements. Where it does
not, the page says so in a box like this one:

:::{admonition} Departure from the published method
:class: important

The box gives the published formula, the formula the library uses, and the reason for the change.
:::

Some methods have no published reference: the normalized and cross-moment gradient estimators
(NAS, ASX, NASX), the swarm subspace (SS), the RBF length-scale heuristic, and the lift that maps an
optimal active point back to a full input. The pages say so where they describe them.

## How the references are cited

Literature is cited by author and year, with a list at the end of each page. Entries marked "(details
to verify)" have bibliographic details (volume, pages or DOI) that have not yet been checked against
the publisher.
