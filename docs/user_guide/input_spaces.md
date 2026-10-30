---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Input spaces and coordinates

An input space states what your inputs are: their density, their physical ranges or moments, and
their names and units. It also owns the affine map between **physical** inputs $\xi$ and
**normalized** inputs $x$,

$$\xi = A x + b,$$

in which every computation is done and every result is stored. Every function that takes data
needs a space, either directly (`space=`) or through the result or model you pass it. The library
never infers ranges from the data.

v1 has three spaces.

| Space | Physical density | $A$, $b$ | Normalized density | $\Sigma = \operatorname{Cov}(x)$ |
|---|---|---|---|---|
| `ac.Uniform(lower, upper)` | uniform on the box $\prod_j [\ell_j, u_j]$ | $A = \operatorname{diag}\bigl(\tfrac{u-\ell}{2}\bigr)$, $b = \tfrac{u+\ell}{2}$ | uniform on $[-1,1]^m$ | $\tfrac13 I$ |
| `ac.Gaussian(mean, cov)` | $N(\mu, \Gamma)$ | $A = L$ with $\Gamma = LL^\top$ (Cholesky), $b = \mu$ | $N(0, I_m)$ | $I$ |
| `ac.Product(blocks)` | independent `Uniform` and `Gaussian` blocks | $A = \operatorname{blockdiag}(A_1, \dots, A_r)$, $b = (b_1, \dots, b_r)$ | $U[-1,1]$ or $N(0,1)$ per coordinate | $\operatorname{blockdiag}(\Sigma_1, \dots, \Sigma_r)$ |

```{code-cell} python
import numpy as np
import active_subspaces as ac
```

## Uniform inputs

Give the lower and upper bounds in physical units, and optionally names and units. Names default
to `x1, …, xm` and appear on plots.

```{code-cell} python
space = ac.Uniform(
    lower=[0.05, 100.0, 1120.0],
    upper=[0.15, 50000.0, 1680.0],
    names=["rw", "r", "L"],
    units=["m", "m", "m"],
)
print(space)
print("m =", space.dim, "  kind =", space.kind)
```

`to_normalized` and `from_normalized` apply the map. Points are rows of an `(M, m)` array.

```{code-cell} python
X = np.array([[0.05, 100.0, 1120.0],          # lower corner
              [0.10, 25050.0, 1400.0],        # centre
              [0.15, 50000.0, 1680.0]])       # upper corner
x = space.to_normalized(X)
print(x)
print(np.allclose(space.from_normalized(x), X))
```

`sample` draws from the input density, in physical coordinates by default. `contains` checks the
support (with a relative tolerance of $10^{-12}$ of each range).

```{code-cell} python
Xs = space.sample(5, rng=0)                   # physical
xs = space.sample(5, rng=0, normalized=True)  # the same draws, normalized
print(np.allclose(space.to_normalized(Xs), xs))
print(space.contains([[0.2, 1000.0, 1500.0]]))    # rw outside [0.05, 0.15]
```

A point outside the support is an error wherever data enter the library (code
`input.out_of_space`). The message counts the rows outside and names the first one: its row, the
input's name and column, its value, and the bound it crosses, in physical units.

```{code-cell} python
Xbad = space.sample(10, rng=0)
Xbad[3, 0] = 0.2
try:
    ac.active_subspace(grads=np.ones((10, 3)), X=Xbad, space=space)
except ac.InputError as err:
    print(err)
    print({k: err.details[k] for k in ("n_outside", "row", "column", "name", "side", "bound")})
```

`details` also holds `value`. Gaussian coordinates never fail the check.

## Gaussian inputs

Give the mean and either a covariance matrix or a vector of variances. A vector means
independent inputs.

```{code-cell} python
indep = ac.Gaussian(mean=[1.0, -2.0], cov=[0.25, 4.0], names=["a", "b"])
print(indep.scale)                            # A = diag of standard deviations
```

### Correlated inputs

A full covariance matrix $\Gamma$ gives correlated inputs. The map uses the lower Cholesky
factor $L$ of $\Gamma$, so $x = L^{-1}(\xi - \mu)$ has independent standard normal entries.

```{code-cell} python
Gamma = np.array([[1.0, 0.8],
                  [0.8, 1.0]])
corr = ac.Gaussian(mean=[0.0, 0.0], cov=Gamma)
Xc = corr.sample(100_000, rng=1)
print(np.round(np.cov(Xc.T), 2))                      # physical: correlated
print(np.round(np.cov(corr.to_normalized(Xc).T), 2))  # normalized: identity
```

Any other square root $LQ$ of $\Gamma$, with $Q$ orthogonal, would rotate the normalized
eigenvectors by $Q^\top$ and leave the eigenvalues and the physical subspace unchanged. Because the
Cholesky factor is used, the normalized coordinates of a non-diagonal $\Gamma$ depend on the
order of the inputs; the eigenvalues and the physical active directions do not.

## Mixed inputs: `Product`

`ac.Product` joins independent `Uniform` and `Gaussian` blocks. Coordinates are concatenated in the
order given, each block keeps its own map, and $A$ is block diagonal. Names and units are the
blocks' own. A block with default names gets `x{j}`, with $j$ the position in the product.

```{code-cell} python
mixed = ac.Product([
    ac.Uniform([0.0, 10.0], [2.0, 20.0], names=["a", "b"]),
    ac.Gaussian([1.0], [4.0], names=["c"]),
])
print(mixed.dim, mixed.kind, mixed.names)
print(np.round(np.diag(mixed.reference_covariance), 3))     # 1/3 on uniform, 1 on Gaussian
Xm = mixed.sample(2000, rng=0)
print(mixed.contains(Xm).all(), mixed.contains([[1.0, 15.0, 100.0]]).all())   # c is never checked
a = np.array([1.0, 0.1, 1.0])                               # f(X) = sin(X @ a)
Gm = np.cos(Xm @ a)[:, None] * a                            # physical gradients
sub_m = ac.active_subspace(grads=Gm, space=mixed, n=1)
print(np.round(sub_m.W1[:, 0], 3))
```

Here $\Sigma$ is not a multiple of $I$. SIR, pHd and SAVE whiten with it and map their directions
back by $\Sigma^{-1/2}$. Their directions are therefore expressed in the right coordinates, but Li's
linearity and Gaussian assumptions still fail on the uniform coordinates, so on a mixed product they
issue a `ValidityWarning` (`validity.assumption`).

Default block names are renumbered to the product position: a block placed third has default names
`x3`, ... A name given explicitly that equals a block default (for example `names=["x1"]` on a block
placed third) is treated as the default and renamed (here to `x3`). Pass distinct names to keep
them.

A `Product` is accepted by subspaces, gradients, surrogates (on all inputs or on a subspace) and
plots. Everything these need is block-wise: the maps, `contains`, `reference_covariance` and
`sample`. Geometry, integration, optimization, designs and the reduced model need a single density
and a domain. They raise `unsupported.space`.

```{code-cell} python
try:
    ac.active_domain(sub_m)
except ac.UnsupportedError as err:
    print(err.code)
```

A log-normal input is passed as its logarithm: use a Gaussian block for $s = \log r$, let the model
take $s$ (or exponentiate before calling it), and apply the chain rule
$\partial f/\partial s = r\,\partial f/\partial r$. The {doc}`migration guide <../migration>`
(example E4) does this for the 2016 borehole.

## Gradients and the chain rule

Pass gradients in physical units too. With rows of `G` holding $\nabla_\xi f$, the normalized
gradients are $\nabla_x f = A^\top \nabla_\xi f$, or $G A$ in row form. The library applies this
for you; `grads_to_normalized` and `grads_to_physical` are there if you need them.

```{code-cell} python
G = np.array([[1.0, 1e-4, -2e-3]])            # df/d(rw), df/d(r), df/d(L), physical units
print(space.grads_to_normalized(G))           # each entry times (upper - lower) / 2
```

```{warning}
Do not scale gradients by hand. If you already multiplied them by $(u - \ell)/2$, as the 2016
tutorials did, pass them with `ac.Uniform.reference(m)` (or `ac.Gaussian.reference(m)`), whose map
is the identity, or they are scaled twice.
```

## Physical directions

The eigenvectors of a subspace result are in normalized coordinates. Normalized components are
comparable across inputs, so they are the scale-free measure of importance. To express an active
direction in the physical inputs, use `result.physical_directions()`. If
$f(\xi) \approx h(a^\top \xi)$, the normalized direction is $w \propto A^\top a$, so the physical
directions are an orthonormal basis of $\operatorname{span}(A^{-\top} W_1)$.

```{code-cell} python
ridge = ac.test_functions.ridge([1.0, 1.0, 0.0], profile="linear", space=space)
Xr = space.sample(50, rng=2)
sub = ac.active_subspace(grads=ac.evaluate_grads(ridge, X=Xr), space=space, n=1)
print("normalized w1:", np.round(sub.W1[:, 0], 3))
print("physical a:   ", np.round(sub.physical_directions()[:, 0], 6))
```

Physical components carry the inverse units of their inputs, so compare them across inputs
only when the units agree. The eigenvalues have no physical counterpart.

## Data already in normalized coordinates

`ac.Uniform.reference(m)` is $[-1,1]^m$ and `ac.Gaussian.reference(m)` is $N(0, I_m)$. Both have
$A = I$ and $b = 0$, so their maps are the identity. Use them when your data are already
normalized.

Both take keyword-only `names` and `units`, checked as for any space, so plots of normalized data
are labelled. The result equals the explicit constructor call.

```{code-cell} python
ref = ac.Uniform.reference(2, names=["a", "b"], units=["m", None])
print(ref == ac.Uniform([-1, -1], [1, 1], names=("a", "b"), units=("m", None)))
```

## Spaces v1 does not have

`Uniform`, `Gaussian` and `Product` are the only spaces. There is no correlation between blocks
(correlated inputs must be jointly Gaussian), no other marginal, and no correlated uniform space.
Two ways around this:

- **Reparametrize the model.** A log-normal input $r = e^{s}$ with $s \sim N(\mu_s, \sigma_s^2)$
  becomes a Gaussian block if the model takes $s$ and computes $r$ itself. The chain rule gives
  $\partial f/\partial s = r\,\partial f/\partial r$.
- **Normalize by hand.** Map each input to its own standard coordinate, compute the normalized
  gradients yourself, and pass them with a reference space. Prefer `Product` when the inputs are
  independent uniform and Gaussian. Under hand normalization the reference space misdescribes
  the density. What stays exact:
  - `estimator="as"` and local linear gradients, which use only the gradients and the weights.
  - OLS (`estimator="ols"`), which uses only the normalized $X$.

  What does not: QPHD, SIR, pHd and SAVE, which use $\Sigma$ of the declared space, and
  `sample`, `contains` and everything in geometry, integration, optimization and designs.

### Discrete inputs

An input with levels such as $\{1, 4, 4.5\}$ passes the support check of a uniform box over the
levels. That is fine for OLS, AS and local linear gradients, which do not use the density. It is
wrong for the Σ-based estimators (QPHD, SIR, pHd, SAVE), for `sample` and for geometry, because
the declared density is continuous.

## Saving a space

Spaces are immutable, compare by value, and convert to and from JSON-compatible dicts. Every result
stores its space, so saving a result saves the space with it (see {doc}`saving`).

```{code-cell} python
d = space.to_dict()
print(d["type"], d["names"])
print(ac.Uniform.from_dict(d) == space)
```

See also {class}`~active_subspaces.spaces.Uniform`, {class}`~active_subspaces.spaces.Gaussian` and
the theory page {doc}`../theory/active_subspaces`.
