---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Geometry: the active domain and the maps

With the eigenvectors split as $W = [W_1\ W_2]$, every normalized input $x$ has **active
variables** $y = W_1^\top x \in \mathbb R^n$ and **inactive variables**
$z = W_2^\top x \in \mathbb R^{m-n}$, and $x = W_1 y + W_2 z$. This page covers the set of
possible $y$ (the active domain), the forward map $x \mapsto (y, z)$, and the inverse map, which
draws $x$ given $y$. Everything here is in **normalized** coordinates except where `X` or
`X_physical` says otherwise. The geometry needs $1 \le n \le m-1$. The theory page
{doc}`../theory/geometry` states the constructions.

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
```

```{code-cell} python
import numpy as np
import active_subspaces as ac

wing = ac.test_functions.builtin_models().get("wing_weight")
space = wing.space
X = space.sample(400, rng=0)
f = ac.evaluate(wing, X=X)
sub = ac.active_subspace(grads=ac.evaluate_grads(wing, X=X), space=space)
sub1, sub2 = ac.with_dimension(sub, 1), ac.with_dimension(sub, 2)
```

## The active domain

`ac.active_domain(subspace)` builds the set $\mathcal Y$ of active variables [Constantine 2015].
Its `kind` follows from the space and $n$:

| Space | $n$ | $\mathcal Y$ | `kind` |
|---|---|---|---|
| `Uniform` | 1 | the interval $[-\lVert w_1\rVert_1,\ \lVert w_1\rVert_1]$ | `"interval"` |
| `Uniform` | $\ge 2$ | the zonotope $W_1^\top[-1,1]^m$ | `"zonotope"` |
| `Gaussian` | any | $\mathbb R^n$, with $y \sim N(0, I_n)$ | `"unbounded"` |

```{code-cell} python
dom1 = ac.active_domain(sub1)
print(dom1.kind, dom1.vertices.ravel())
dom2 = ac.active_domain(sub2)
print(dom2.kind, dom2.vertices.shape, "vertices,", dom2.facet_normals.shape[0], "facets")
```

The zonotope is computed exactly and deterministically from its generators, the rows of $W_1$
[Ziegler 1995, ch. 7]. Its facets give the inequalities
$\mathcal Y = \lbrace y : N y \le c \rbrace$, with `facet_normals` $N$ (unit outward normals)
and `facet_offsets` $c$. `vertex_preimages` holds a corner of the cube that maps onto each vertex.

```{code-cell} python
fig, ax = ac.plotting.plot_active_domain(dom2, points=ac.forward_map(sub2, X=X).Y, values=f)
fig
```

### Size

A zonotope in general position has

$$N_v = 2\sum_{i=0}^{n-1}\binom{m-1}{i}$$

vertices, which grows like $m^{n-1}$. `ac.zonotope_vertex_count(m, n)` returns this count, so you
can check the cost before you build the domain:

```{code-cell} python
for n in (2, 3, 4):
    print(n, [ac.zonotope_vertex_count(m, n) for m in (10, 30, 100)])
```

The domain needs about $8 N_v (m + n)$ bytes. Bounded geometry is practical while
$N_v(m+n) \le 10^8$: $n=2$ always, $n=3$ up to $m \approx 300$, $n=4$ up to $m \approx 100$. See
{doc}`problem_sizes`.

## The forward map

`ac.forward_map(subspace, X=X)` takes **physical** points, normalizes them with the subspace's
space, and returns $Y = x W_1$ and $Z = x W_2$ (row-wise). The result unpacks as a pair.

```{code-cell} python
Y, Z = ac.forward_map(sub2, X=X)
print(Y.shape, Z.shape)
print(ac.in_active_domain(dom2, Y=Y).all())
```

`ac.in_active_domain(domain, Y=Y)` tests points against the facets, with the domain's relative
tolerance. Every $Y$ that comes from the forward map is inside.

## The inverse map

Many inputs map to the same $y$. `ac.inverse_map(subspace, Y=Y, n_draws=k, rng=…)` draws $k$
inputs $x = W_1 y + W_2 z$ for each row of `Y`, with $z$ from its conditional distribution
given $y$ under the input density:

- **Gaussian space:** $z \mid y \sim N(0, I_{m-n})$, independent of $y$. Draws are exact.
- **Uniform space:** $x$ is uniform on the fiber
  $\lbrace x \in [-1,1]^m : W_1^\top x = y \rbrace$. When $m - n = 1$ the fiber is a segment and
  the draws are exact (`sampler == "exact_interval"`). Otherwise the library runs a hit-and-run
  chain on the fiber [Smith 1984], started at its Chebyshev center (`sampler == "hit_and_run"`).

```{code-cell} python
y = Y[:3]                                      # three active points
inv = ac.inverse_map(sub2, Y=y, n_draws=5, rng=1)
print(inv.sampler, "burn_in =", inv.burn_in, "thin =", inv.thin)
print(inv.X.shape, inv.index)                  # row k * n_draws + j is draw j for y[k]
print(np.allclose(inv.X @ sub2.W1, y[inv.index]))
print(space.contains(inv.X_physical).all())    # physical draws, inside the box
```

`inv.X` is normalized and `inv.X_physical` is physical. Pass `inv.X_physical` to your model.

### Hit-and-run settings

Each step of the chain picks a uniformly random direction in the fiber and moves to a uniform
point on the chord through the current point. The defaults are `burn_in = 10 (m - n)` steps and
`thin = m - n` steps between recorded draws. Draws from one chain are correlated. With `thin=1`
and more than one draw, the library warns `accuracy.correlated_draws`. Raise `burn_in` and `thin`
when $m - n$ is large and you need nearly independent draws.

A point on the boundary of $\mathcal Y$ has a fiber with empty interior. The draws are then copies
of one point, the row is flagged in `inv.degenerate`, and an `accuracy.degenerate_fiber`
warning is issued. A point outside $\mathcal Y$ raises `ac.InputError` with code
`input.out_of_domain`.

```{code-cell} python
inv1 = ac.inverse_map(sub1, Y=Y[:2, :1], n_draws=3, rng=2)    # m - n = 9: hit-and-run
print(inv1.sampler, inv1.degenerate)
```

API reference: {mod}`active_subspaces.geometry`.

## References

- P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM, 2015.
- R. L. Smith. Efficient Monte Carlo procedures for generating points uniformly distributed over
  bounded regions. *Oper. Res.* 32(6):1296–1308, 1984.
- G. M. Ziegler. *Lectures on Polytopes.* Graduate Texts in Mathematics 152, Springer, 1995
  (ch. 7, zonotopes).
