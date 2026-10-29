---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Choosing the dimension

The dimension $n$ of a `SubspaceResult` comes from one of three rules, or from you. Changing it
later costs nothing: `ac.with_dimension` returns a new result with the same eigenpairs and
bootstrap replicates.

| Rule | `dimension_rule` | Needs a bootstrap | Chooses |
|---|---|---|---|
| Eigenvalue gap (default) | `"evg"`, or `ac.subspaces.EVG(gap="log")` | no | the largest gap $\mu_k-\mu_{k+1}$ (or $\log\mu_k-\log\mu_{k+1}$) |
| Response-surface bound | `"rs"` | yes | the smallest error bound of a response surface in $k$ variables |
| Ladle | `"li"` | yes | the smallest ladle criterion of Luo and Li (2016) |
| Fixed | `n=k` | no | $k$ |

All rules work with the magnitudes $\mu_k = \lvert\lambda_k\rvert$ and return
$1\le n\le m-1$. Give at most one of `n` and `dimension_rule`. The theory page {doc}`../theory/dimension` derives
the rules.

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
```

```{code-cell} python
import numpy as np
import active_subspaces as ac

wing = ac.test_functions.builtin_models().get("wing_weight")
X = wing.space.sample(400, rng=0)
G = ac.evaluate_grads(wing, X=X)
sub = ac.active_subspace(grads=G, space=wing.space, nboot=200, rng=2)   # EVG by default
print("n =", sub.n, " rule =", sub.dimension_rule, dict(sub.dimension_options))
```

## Comparing the rules

`ac.dimension_criterion` evaluates a rule without changing the result. It returns the candidate
dimensions, the criterion at each, and the rule's choice $\hat k$.

```{code-cell} python
for rule in ["evg", ac.subspaces.EVG(gap="log"), "rs", "li"]:
    ks, values, k_hat = ac.dimension_criterion(sub, rule=rule)
    print(f"{str(rule):18s} k_hat = {k_hat}   criterion: {np.array2string(values[:5], precision=3)} ...")
```

The rules answer different questions, and here they disagree.

- **EVG** looks only at the eigenvalues. The wing weight has one dominant eigenvalue, so the largest
  gap, linear or logarithmic, is after $\mu_1$.
- **RS** ranks
  $$\varepsilon_k=\Bigl(\sum_{j\le k}\mu_j\Bigr)^{1/2}\hat\delta_k+\Bigl(\sum_{j>k}\mu_j\Bigr)^{1/2},$$
  where $\hat\delta_k$ is the bootstrap mean distance between the estimated $k$-dimensional
  subspace and its replicates. The first term is the error from estimating the subspace, the second
  the error from ignoring the remaining directions [Constantine 2015]. With accurate gradients
  every subspace is estimated well, and the tail $\mu_{10}$ is tiny, so $\varepsilon_9$ is
  smallest. The Poincaré constant of the bound is dropped, so $\varepsilon_k$ ranks dimensions and
  is not an error estimate.
- **LI**, the ladle, combines the eigenvalues with the bootstrap variability of the eigenvectors
  (below).

A large $n$ is costly to exploit: the active domain and the default designs grow quickly with
$n$ (see {doc}`problem_sizes`). Plot the criterion before you accept a choice:

```{code-cell} python
fig, ax = ac.plotting.plot_dimension_criterion(sub, rule="rs")
fig
```

Then fix the dimension you want:

```{code-cell} python
sub2 = ac.with_dimension(sub, 2)
sub_rs = ac.with_dimension(sub, dimension_rule="rs")
print(sub2.n, sub2.W1.shape, "|", sub_rs.n, sub_rs.dimension_rule)
print(np.array_equal(sub2.eigenvalues, sub.eigenvalues))        # eigenpairs unchanged
```

## The ladle

With $d_k^{(b)}=\lvert\det(W_{1:k}^\top W^{(b)}_{1:k})\rvert$ from replicate $b$,
$k_n = m-1$ for $m\le10$ and $k_n=\lfloor m/\ln m\rfloor$ otherwise,

$$f^0(0)=0,\quad f^0(k)=\frac1B\sum_{b=1}^B\bigl(1-d_k^{(b)}\bigr),\qquad
g(k)=\frac{f^0(k)}{1+\sum_{i=0}^{k_n}f^0(i)}+\frac{\mu_{k+1}}{1+\sum_{i=0}^{k_n}\mu_{i+1}},$$

and $\hat k$ minimizes $g$ over $0\le k\le k_n$. The library uses $n=\hat k$ (the 2016 code
returned $\hat k+1$). If $\hat k=0$, it sets $n=1$ and issues an `ac.AccuracyWarning` with code
`accuracy.zero_dimension`; the full criterion is kept in `result.criterion`. Both denominators
contain $1+\sum$, as in the paper, so the criterion depends on the scale of $\lambda$: multiplying
$f$ by a constant can change $\hat k$.

**Departure.** Luo and Li average $f^0$ over $n$ replicates, $n$ being the sample size $M$. v1
averages over $B$ = `nboot` replicates, as 2016 did. Their theory takes $B=M$.

The ladle assumes a positive semidefinite matrix. v1 applies it to $\lvert\lambda\rvert$ for the
indefinite estimators (`"asx"`, `"nasx"`, `"phd"`) and issues `validity.assumption`. RS likewise
warns with every estimator except `"as"` and `"opg"`, because its bound holds for
$C=\mathbb E[\nabla f\nabla f^\top]$ only.

## Bootstrap intervals

The eigenvalue bands help judge a gap. A gap between two eigenvalues whose 95% intervals overlap
is not resolved by the data.

```{code-cell} python
lo, hi = ac.bootstrap_interval(np.abs(sub.bootstrap_eigenvalues))
for k in range(4):
    print(f"lambda_{k + 1}: {sub.eigenvalues[k]:9.3f}   [{lo[k]:9.3f}, {hi[k]:9.3f}]")
```

`level=` changes the level of the percentile interval, and `kind="minmax"` gives the 2016 band.

## Ties

On an exact ridge, or with repeated eigenvalues, a criterion can take the same value at several
$k$ in exact arithmetic, and only rounding separates them. Each rule therefore counts values within
$8\,m\,\epsilon\,s$ of the extreme as tied ($\epsilon$ the machine epsilon, $s$ the scale of the
criterion), and the **smallest** tied $k$ wins. On an exact ridge in ten variables every rule
chooses $n=1$:

```{code-cell} python
ridge = ac.test_functions.builtin_models().get("ridge_sin_uniform_10")
Xr = ridge.space.sample(200, rng=0)
sr = ac.active_subspace(grads=ac.evaluate_grads(ridge, X=Xr), space=ridge.space, nboot=100, rng=1)
for rule in ["evg", "rs", "li"]:
    print(rule, ac.dimension_criterion(sr, rule=rule)[2])
```

The tolerance does not change the criteria; it only makes the choice well defined in floating
point. `dimension_criterion` returns the same $\hat k$ as the rule.

API reference: {func}`~active_subspaces.subspaces.with_dimension`, {func}`~active_subspaces.subspaces.dimension_criterion`, {func}`~active_subspaces.subspaces.bootstrap_interval`.

## References

- P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM, 2015.
- W. Luo, B. Li. Combining eigenvalues and variation of eigenvectors for order determination.
  *Biometrika* 103(4):875–887, 2016 (details to verify).
