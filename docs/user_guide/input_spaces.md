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

v1 has two spaces.

| Space | Physical density | $A$, $b$ | Normalized density | $\Sigma = \operatorname{Cov}(x)$ |
|---|---|---|---|---|
| `ac.Uniform(lower, upper)` | uniform on the box $\prod_j [\ell_j, u_j]$ | $A = \operatorname{diag}\bigl(\tfrac{u-\ell}{2}\bigr)$, $b = \tfrac{u+\ell}{2}$ | uniform on $[-1,1]^m$ | $\tfrac13 I$ |
| `ac.Gaussian(mean, cov)` | $N(\mu, \Gamma)$ | $A = L$ with $\Gamma = LL^\top$ (Cholesky), $b = \mu$ | $N(0, I_m)$ | $I$ |

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
`input.out_of_space`).

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

## Spaces v1 does not have

Both spaces are homogeneous: all inputs uniform, or all jointly Gaussian. There is no mixed space
(some inputs uniform, others Gaussian), no log-normal or other marginal, and no correlated uniform
space. Two ways around this:

- **Reparametrize the model.** A log-normal input $r = e^{s}$ with $s \sim N(\mu_s, \sigma_s^2)$
  becomes a Gaussian input if the model takes $s$ and computes $r$ itself. The chain rule gives
  $\partial f/\partial s = r\,\partial f/\partial r$.
- **Normalize by hand.** Map each input to its own standard coordinate, compute the normalized
  gradients yourself, and pass them with a reference space. The gradient-based estimator
  `estimator="as"` then gives the right eigenpairs, since it uses only the gradients and the
  weights. But the reference space misdescribes the density, so do not use that result for
  anything that depends on it: the value-based estimators, the active domain, designs,
  integration or optimization. The {doc}`migration guide <../migration>` (example E4) does this for
  the 2016 borehole.

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
