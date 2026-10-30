---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Discovering a subspace

One function, `ac.active_subspace`, estimates a subspace from samples. What you pass decides
what it can do:

| You have | Pass | Default estimator |
|---|---|---|
| gradients at sample points | `grads=G` (physical) and `space=` | `"as"`, the active subspace |
| a model with a gradient | `grads=ac.evaluate_grads(model, X=X)` | `"as"` |
| a model without a gradient | `grads=ac.finite_difference_gradients(model, X=X)` | `"as"` |
| values only | `X=`, `f=`, `space=` and `estimator=` one of the value-based estimators | — |

The result is a frozen `ac.SubspaceResult`: signed eigenvalues `eigenvalues` `(m,)`, orthonormal
eigenvectors `W` `(m, m)` in **normalized** coordinates, the dimension `n`, and the active and
inactive bases `W1` `(m, n)` and `W2` `(m, m - n)`.

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
```

```{code-cell} python
import warnings
import numpy as np
import active_subspaces as ac

registry = ac.test_functions.builtin_models()
wing = registry.get("wing_weight")            # 10 uniform inputs, analytic gradient
space = wing.space
X = space.sample(400, rng=0)                  # physical
```

## From gradients

The active subspace of $f$ is spanned by the leading eigenvectors of

$$C = \int \nabla f(x)\,\nabla f(x)^\top \rho(x)\,dx \approx \sum_{i=1}^M w_i\,\nabla f(x_i)\,\nabla f(x_i)^\top ,$$

with $x$ normalized, $\rho$ the normalized input density and quadrature weights $w_i$ (by
default $1/M$, for Monte Carlo samples) [Constantine 2015]. See the theory page
{doc}`../theory/active_subspaces`.

```{code-cell} python
G = ac.evaluate_grads(wing, X=X)              # (400, 10), physical gradients
sub = ac.active_subspace(grads=G, space=space)
print("n =", sub.n)
print(np.array2string(sub.eigenvalues, precision=3))
print(sub.W1.shape, sub.W2.shape)
```

Each eigenvector is defined up to sign. The library makes each one's largest-magnitude component
positive; see [Signs and orientation](#signs-and-orientation) to choose another convention.

### Without an analytic gradient

`ac.finite_difference_gradients` differences a model along the normalized axes. It costs
$(m+1)M$ evaluations with the default forward scheme and never evaluates outside the support of a
uniform space. The step `h` is relative and in normalized units; its default `1e-6` suits a model
computed to near machine precision. For a noisy model, choose `h` to balance the truncation error
$O(h)$ against the rounding error $O(\varepsilon_f/h)$, where $\varepsilon_f$ is the noise level.

```{code-cell} python
fd = ac.finite_difference_gradients(wing, X=X[:100])
print(type(fd).__name__, fd.method, fd.n_evaluations)
sub_fd = ac.active_subspace(grads=fd)         # the space comes with the GradientResult
print(np.array2string(sub_fd.eigenvalues[:3], precision=3))
```

`ac.local_linear_gradients` needs no model at all: it fits a linear function to the $p$ nearest
samples around each sample (in normalized coordinates) and takes its slope.

```{code-cell} python
f = ac.evaluate(wing, X=X)
llg = ac.local_linear_gradients(X=X, f=f, space=space)
sub_ll = ac.active_subspace(grads=llg)
print("p =", llg.p, "  n =", sub_ll.n)
print(np.array2string(sub_ll.eigenvalues[:3], precision=3))
```

## From values only

When you have only values $f_i$ at points $x_i$, choose a value-based estimator. Several of them
come from sufficient dimension reduction, and their theory assumes Gaussian inputs, or conditions
that Gaussian inputs satisfy. Here is a ridge function of ten Gaussian inputs:

```{code-cell} python
ridge = registry.get("ridge_sin_gaussian_10")     # f = sin(a^T x), x ~ N(0, I)
Xg = ridge.space.sample(500, rng=1)
fg = ac.evaluate(ridge, X=Xg)
a = ridge.reference_results.eigenvectors[:, 0]    # the true direction

for est in ["ols", "sir", "qphd"]:
    r = ac.active_subspace(X=Xg, f=fg, space=ridge.space, estimator=est)
    print(f"{est:5s} n = {r.n}   |w1 . a| = {abs(r.W1[:, 0] @ a):.3f}")
```

Options go on an estimator instance, such as `ac.subspaces.SIR(n_slices=5)`:

```{code-cell} python
r = ac.active_subspace(X=Xg, f=fg, space=ridge.space, estimator=ac.subspaces.SIR(n_slices=5))
r.estimator, r.estimator_options
```

### Validity warnings

An estimator used outside its assumptions still runs, and issues an `ac.ValidityWarning` with
code `validity.assumption` that says why. OLS on uniform inputs is one such case:

```{code-cell} python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ac.active_subspace(X=X, f=f, space=space, estimator="ols")
for w in caught:
    print(type(w.message).__name__, w.message.code)
    print(" ", w.message.message[:110], "...")
```

See {doc}`errors` for how to filter warnings by code.

## The estimators

| `estimator` | Needs | Matrix | Uniform inputs | Reference |
|---|---|---|---|---|
| `"as"` | `grads` | $\sum_i w_i g_i g_i^\top$ | valid | Constantine (2015) |
| `"nas"` | `grads` | as `"as"` with unit gradients $g_i/\lVert g_i\rVert$ | valid | none (library heuristic) |
| `"asx"` | `X`, `grads` | symmetrized $\sum_i w_i g_i x_i^\top$; indefinite | warns | none (library heuristic) |
| `"nasx"` | `X`, `grads` | as `"asx"` with unit $g_i$ and unit $x_i$; indefinite | warns | none (library heuristic) |
| `"ols"` | `X`, `f` | $uu^\top$, $u$ the slope of a linear least-squares fit | warns | see {doc}`../theory/estimators` |
| `"qphd"` | `X`, `f` | $gg^\top + H\Sigma H$ from a quadratic least-squares fit | warns | Li (1992), Corollary 3.1 with $q=2$ |
| `"sir"` | `X`, `f` | covariance of whitened slice means | warns | Li (1991) |
| `"phd"` | `X`, `f` | whitened $\sum_i w_i (f_i-\bar f)(x_i-\bar x)(x_i-\bar x)^\top$; indefinite | warns | Li (1992) |
| `"save"` | `X`, `f` | $\sum_h \hat p_h (I-\hat V_h)^2$, $\hat V_h$ the covariance of the whitened inputs in slice $h$ | warns | Cook and Weisberg (1991) |
| `"ss"` | `X`, `f` | weighted outer products of pairwise difference directions | warns (also on Gaussian inputs) | none (library heuristic) |
| `"opg"` | `X`, `f` | `"as"` with local linear gradients at every sample | valid | Hristache et al. (2001); Xia et al. (2002) |

$g_i$ is the normalized gradient at $x_i$, $\Sigma$ the covariance of the normalized density
($\tfrac13 I$ uniform, $I$ Gaussian), and $w_i$ the quadrature weights. For the indefinite
matrices (`"asx"`, `"nasx"`, `"phd"`) the eigenvalues are ordered by magnitude and reported with
their signs.

The mathematics of each estimator is on the {doc}`estimators <../theory/estimators>` theory page,
and the {doc}`estimator-selection guide <../theory/estimator_selection>` helps you choose. In short: use `"as"` when
you have gradients; with values only, `"opg"` makes no distributional assumption, and the
inverse-regression estimators are most reliable on Gaussian inputs.

### Where v1 departs from the published methods

- **Whitening (SIR, pHd, SAVE).** The published methods whiten with the sample covariance,
  $z_i = \hat\Sigma^{-1/2}(x_i - \bar x)$. v1 uses the covariance of the declared space,
  $z_i = \Sigma^{-1/2}(x_i - \bar x_w)$ with the weighted mean $\bar x_w = \sum_i w_i x_i$.
- **Slices (SIR, SAVE).** Slice probabilities and means are weighted,
  $\hat p_h = \sum_{i\in h} w_i$; slices have about equal counts, ties in $f$ stay in one slice,
  and single-point slices are merged.
- **QPHD.** v1 forms $gg^\top + H\Sigma H$ with $\Sigma$ from the space. The 2016 code used
  $gg^\top + \gamma HH^\top$ with $\gamma$ guessed from the data range.
- **OPG.** The published OPG weights each local fit with a fixed-bandwidth kernel. v1 uses the
  uniform kernel on the $p$ nearest distinct samples (an adaptive bandwidth) at every sample.

## Weighted data

If your points come from a quadrature rule rather than Monte Carlo, pass its weights as
`quad_weights` `(M,)`. They must be non-negative and sum to 1.

```{code-cell} python
nodes, wts = np.polynomial.legendre.leggauss(7)
x1, x2 = np.meshgrid(nodes, nodes, indexing="ij")
Xq = np.column_stack([x1.ravel(), x2.ravel()])         # tensor Gauss-Legendre on [-1, 1]^2
wq = np.outer(wts, wts).ravel() / 4.0                  # sums to 1
Gq = np.column_stack([np.cos(Xq[:, 0] + 0.5 * Xq[:, 1]), 0.5 * np.cos(Xq[:, 0] + 0.5 * Xq[:, 1])])
r = ac.active_subspace(grads=Gq, space=ac.Uniform.reference(2), quad_weights=wq)
np.round(r.W1[:, 0], 4)
```

## The bootstrap

`nboot=B` recomputes the estimate on $B$ resamples of the rows, each on its own random stream,
and stores the replicates in the result: `bootstrap_eigenvalues` `(B, m)`, and the subspace
distances `bootstrap_distances` `(B, m-1)`. Pass `rng` to make it reproducible (see
{doc}`randomness`).

```{code-cell} python
sub_b = ac.active_subspace(grads=G, space=space, nboot=200, rng=2)
lo, hi = ac.bootstrap_interval(sub_b.bootstrap_eigenvalues)      # 95% percentile interval
print(np.array2string(np.vstack([lo[:4], sub_b.eigenvalues[:4], hi[:4]]), precision=2))
print("mean subspace distance:", np.round(ac.mean_distance(sub_b)[:4], 4))
```

`ac.bootstrap_interval(..., kind="minmax")` gives the 2016 min/max band instead; it widens as $B$
grows. The plots draw the percentile band:

```{code-cell} python
fig, ax = ac.plotting.plot_subspace_errors(sub_b)
fig
```

### Bands for eigenvector components

Eigenvector replicates are stored only on request, because they take $B\,m\,k$ numbers.
`bootstrap_vectors=k` keeps the leading $k$ eigenvectors in `bootstrap_eigenvectors` `(B, m, k)`.
Each replicate column is multiplied by $\operatorname{sign}(w_j^\top w_j^{(b)})$, so the replicates
point the same way as the estimate. If some replicate has $|w_j^\top w_j^{(b)}|<0.5$, the sign
alignment is unreliable and the result carries an `accuracy.unaligned_replicates` warning. The
replacement of the 2016 `W_br` is `plot_eigenvectors` with `interval="percentile"` (the default
for bands is a 95% interval, `level=0.95`) or `interval="minmax"`:

```{code-cell} python
sub_v = ac.active_subspace(grads=G, space=space, nboot=200, bootstrap_vectors=2, rng=2)
print(sub_v.bootstrap_eigenvectors.shape)
fig, axes = ac.plotting.plot_eigenvectors(sub_v, indices=[0, 1], interval="percentile")
fig
```

Two limits apply. The eigenvalue bootstrap is inconsistent at repeated or zero eigenvalues
[Beran and Srivastava 1985; Eaton and Tyler 1991], so a band there is descriptive only. And the
bootstrap assumes the rows are independent draws, which is why non-uniform `quad_weights` with
`nboot > 0` issue `validity.bootstrap_weights`.

Replicates can run on an executor (`executor=`); the result is the same with or without one
(see {doc}`models`).

## Signs and orientation

An eigenvector is defined up to sign. The library gives each column of `W` the sign that makes its
largest-magnitude component positive. For `estimator="ols"` the first column is the signed slope
$\hat u/\lVert\hat u\rVert$ instead, so $f$ increases along $w_1$ [Constantine 2015, Algorithm 1.3].
`plot_eigenvectors`, `y = W1ᵀx` and `physical_directions()` all use the signs the result carries;
nothing is re-signed.

`ac.orient` returns a new result with other signs. `by="reference"` makes each leading column agree
with a reference direction, which aligns a sequence of results, for example estimates from
separate batches of samples:

```{code-cell} python
batches = [G[:200], G[200:]]
subs = [ac.active_subspace(grads=g, space=space) for g in batches]
aligned = [ac.orient(s, by="reference", reference=subs[0]) for s in subs]
print([round(float(aligned[0].W1[:, 0] @ a.W1[:, 0]), 4) for a in aligned])
```

The other criteria are `by="covariance"` (with `X=` and `f=`: the sign makes the sufficient summary
$y_j=w_j^\top x$ positively correlated with $f$) and `by="mean_gradient"` (with `grads=`: the sign
makes $\bar g^\top w_j$ positive, $\bar g$ the mean normalized gradient). `columns=c` re-signs the leading $c$ columns; the default is
`n`. The replicates in `bootstrap_eigenvectors` are re-signed with the same signs.

## Activity scores

`ac.activity_scores(sub, n=None)` returns $\alpha_i(n)=\sum_{j=1}^{n}|\lambda_j|\,w_{ij}^2$, a
global sensitivity measure for input $i$ in normalized coordinates [Constantine and Diaz 2017].
With $n=m$ the scores sum to $\operatorname{tr}C$. The default $n$ is `sub.n`.

```{code-cell} python
scores = ac.activity_scores(sub)
order = np.argsort(scores)[::-1]
print(order[:3], np.round(scores[order[:3]], 1))
print(np.isclose(ac.activity_scores(sub, n=10).sum(), sub.eigenvalues.sum()))
```

## From a matrix you accumulated

A tensor quadrature rule can have too many nodes to hold its gradients in one array. Accumulate
$C=\sum_i\omega_i g_ig_i^\top$ yourself, in normalized coordinates, and pass it to
`ac.subspace_from_matrix`. The result has `estimator="matrix"`, no samples and no bootstrap, but it
takes `n=` or the eigenvalue-gap rule; the rules `"rs"` and `"li"` need bootstrap replicates, which it does not have.

```{code-cell} python
nodes, wts = np.polynomial.legendre.leggauss(7)
C = np.zeros((2, 2))
for xi, wi in zip(nodes, wts):                         # one slab of the tensor rule at a time
    Gs = np.column_stack([np.cos(xi + 0.5 * nodes), 0.5 * np.cos(xi + 0.5 * nodes)])
    ws = wi * wts / 4.0
    C += (Gs * ws[:, None]).T @ Gs
r_mat = ac.subspace_from_matrix(C, space=ac.Uniform.reference(2), n=1)
print(r_mat.estimator, np.round(r_mat.W1[:, 0], 4))
```

For a matrix $C_\xi$ of physical gradients, pass $A^\top C_\xi A$ with `A = space.scale`.

API reference: {mod}`active_subspaces.subspaces`, {mod}`active_subspaces.gradients`.

## References

- R. Beran, M. S. Srivastava. Bootstrap tests and confidence regions for functions of a covariance
  matrix. *Ann. Statist.* 13(1):95–115, 1985 (details to verify).
- P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM, 2015.
- P. G. Constantine, P. Diaz. Global sensitivity metrics from active subspaces. *Reliab. Eng.
  Syst. Saf.* 162:1–13, 2017.
- R. D. Cook, S. Weisberg. Discussion of "Sliced inverse regression for dimension reduction".
  *J. Amer. Statist. Assoc.* 86(414):328–332, 1991.
- M. L. Eaton, D. E. Tyler. On Wielandt's inequality and its application to the asymptotic
  distribution of the eigenvalues of a random symmetric matrix. *Ann. Statist.* 19(1):260–271,
  1991 (details to verify).
- M. Hristache, A. Juditsky, J. Polzehl, V. Spokoiny. Structure adaptive approach for dimension
  reduction. *Ann. Statist.* 29(6):1537–1566, 2001.
- K.-C. Li. Sliced inverse regression for dimension reduction. *J. Amer. Statist. Assoc.*
  86(414):316–327, 1991.
- K.-C. Li. On principal Hessian directions for data visualization and dimension reduction:
  another application of Stein's lemma. *J. Amer. Statist. Assoc.* 87(420):1025–1039, 1992.
- Y. Xia, H. Tong, W. K. Li, L.-X. Zhu. An adaptive estimation of dimension reduction space.
  *J. R. Statist. Soc. B* 64(3):363–410, 2002.
