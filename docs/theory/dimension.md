(theory-dimension)=
# Choosing the dimension

The dimension $n$ of the active subspace is chosen by a rule applied to the eigenvalues, or fixed
by you. Three rules are built in: the eigenvalue gap (EVG), the response-surface bound (RS) and the
ladle (LI). All work with the magnitudes $\mu_k=\lvert\lambda_k\rvert$, so they apply to indefinite
estimators as well, and all return $n\in\lbrace1,\dots,m-1\rbrace$ (for $m=1$, $n=1$). A fixed
integer $1\le n\le m$ is also allowed; $n=m$ means no reduction.

`active_subspace` uses EVG with the linear gap when neither `n` nor `dimension_rule` is given.
`with_dimension(result, ...)` applies a different rule or a fixed $n$ to a stored result without
recomputing the eigenpairs or the bootstrap, and `dimension_criterion(result, rule=...)` returns
the criterion values for plotting without changing $n$.

## Eigenvalue gap (EVG)

With the linear gap (`EVG(gap="linear")`, the default):

$$c_k=\mu_k-\mu_{k+1},\qquad n=\min\arg\max_{1\le k\le m-1}c_k .$$

With the logarithmic gap (`EVG(gap="log")`):

$$c_k=\log\tilde\mu_k-\log\tilde\mu_{k+1},\qquad\tilde\mu_k=\max(\mu_k,\tau),\qquad\tau=m\,\epsilon\,\mu_1,$$

with $\epsilon$ the machine epsilon; the floor keeps numerically zero eigenvalues from producing an
infinite or undefined gap. The two can disagree. For $\mu=(1.97,\ 0.051,\ 1.2\times10^{-6},\ 10^{-7})$
the linear gap gives $n=1$, because $\mu_1-\mu_2$ is the largest difference, while the log gap gives
$n=2$, because $\mu_2/\mu_3$ is the largest ratio. The linear gap favours a small $n$ when $\lambda_1$
dominates; the log gap is invariant to the scale of the eigenvalues and responds to relative drops.
EVG needs no bootstrap and is a heuristic for every estimator.

## Response-surface bound (RS)

The second error bound on the [Active subspaces](active_subspaces.md) page has the form
$\bigl(\varepsilon(\lambda_1+\dots+\lambda_n)^{1/2}+(\lambda_{n+1}+\dots+\lambda_m)^{1/2}\bigr)^2$ up to
a constant: a larger $n$ leaves out less of the spectrum but has a harder subspace to estimate.
RS estimates the subspace error by the bootstrap mean of the distances,
$\hat\delta_k=\frac1B\sum_b\delta_k^{(b)}$, and picks the minimizer

$$\varepsilon_k=\Bigl(\sum_{j\le k}\mu_j\Bigr)^{1/2}\hat\delta_k+\Bigl(\sum_{j>k}\mu_j\Bigr)^{1/2},\qquad n=\min\arg\min_{1\le k\le m-1}\varepsilon_k$$

[Constantine 2015]. The Poincaré constant is dropped, so $\varepsilon_k$ ranks the candidate
dimensions but is not an estimate of the error. RS needs a bootstrap (`nboot > 0`). The bound holds
for $C=\mathbb E[\nabla f\nabla f^\top]$ only, which AS and OPG estimate; with any other estimator RS
issues a validity warning.

## Ladle (LI)

The ladle combines the eigenvalues with the bootstrap variability of the eigenvectors [Luo and Li
2016]. Let $k_n=m-1$ if $m\le10$ and $k_n=\lfloor m/\ln m\rfloor$ otherwise, and use the bootstrap
statistics $d_k^{(b)}=\lvert\det(W_{1:k}^\top W^{(b)}_{1:k})\rvert$:

$$f^0(0)=0,\qquad f^0(k)=\frac1B\sum_{b=1}^B\bigl(1-d_k^{(b)}\bigr)\quad(1\le k\le k_n),$$

$$g(k)=\frac{f^0(k)}{1+\sum_{i=0}^{k_n}f^0(i)}+\frac{\mu_{k+1}}{1+\sum_{i=0}^{k_n}\mu_{i+1}},\qquad\hat k=\min\arg\min_{0\le k\le k_n}g(k).$$

The first term grows when the leading $k$ eigenvectors vary between replicates; the second is the
first eigenvalue left out. $n=\hat k$. The denominators $1+\sum$ follow the paper, so the criterion
depends on the scale of the eigenvalues, exactly as published. If $\hat k=0$ (no reduction found),
the library sets $n=1$, keeps the whole criterion $g(0),\dots,g(k_n)$ in the result, and issues an
`AccuracyWarning` (`accuracy.zero_dimension`).

:::{admonition} Departure from the published method: replicate count and indefinite matrices
:class: important

**Replicates.** Luo and Li average $f^0$ over $n$ bootstrap replicates, $n$ being the sample size,
and their theory assumes that choice. v1 averages over $B$ = `nboot` replicates, chosen by you.
Reason: the cost of the bootstrap is $B$ times the cost of the estimator, and for large $M$ the
published choice is expensive. Pass `nboot=M` to follow the paper.

**Indefinite matrices.** The ladle assumes a positive semidefinite candidate matrix. v1 applies it
to $\mu_k=\lvert\lambda_k\rvert$ for every estimator, and issues a validity warning with ASX, NASX
and pHd.
:::

## Ties

On an exact ridge, several candidate dimensions can have criterion values that are equal in exact
arithmetic and differ only by rounding, so an exact comparison would let rounding error choose $n$.
Every $\min\arg\max$ and $\min\arg\min$ above is therefore taken
with a rounding-level tolerance: a value within $8\,m\,\epsilon\,s$ of the extreme counts as tied,
and the smallest $k$ wins. The scale $s$ is $\mu_1$ for the linear gap, $\max_k\lvert c_k\rvert$ for
the log gap, $(\sum_j\mu_j)^{1/2}$ for RS and 1 for the ladle. This does not change the criteria; it
makes them well defined in floating point.

## Validity of each rule

| Rule | Bootstrap | No warning with | Warns with | Basis |
|---|---|---|---|---|
| `evg` | no | every estimator | — | a heuristic on $\lvert\lambda\rvert$ |
| `rs` | yes | `as`, `opg` | all others | the bound holds for $\mathbb E[\nabla f\nabla f^\top]$ only |
| `li` | yes | PSD estimators | `asx`, `nasx`, `phd` | the ladle assumes a PSD matrix |

A fixed $n$ triggers no warning.

## Bootstrap intervals

The bootstrap replicates are stored in the result: `bootstrap_eigenvalues` ($B\times m$, signed,
each row ordered by $\lvert\lambda\rvert$), `bootstrap_distances` ($B\times(m-1)$, the
$\delta_k^{(b)}$) and `bootstrap_dets` ($B\times(m-1)$, the $d_k^{(b)}$). `bootstrap_interval`
turns any such array into intervals:

- `kind="percentile"` (default) at level $1-\alpha$ (default 0.95) takes the empirical
  $\alpha/2$ and $1-\alpha/2$ quantiles of each column;
- `kind="minmax"` takes the column minimum and maximum, as the 2016 code reported. These intervals
  widen as $B$ grows, so they are not confidence intervals.

`bootstrap_interval` is a pure array function: pass `np.abs(res.bootstrap_eigenvalues)` for
intervals on magnitudes. `mean_distance(result)` returns the $\hat\delta_k$ that RS uses.

The intervals describe sampling variability only. They inherit the limits of the bootstrap: they
are unreliable for eigenvalues that are repeated or near zero [Beran and Srivastava 1985; Eaton and
Tyler 1991], and they assume independent samples.

**Reading the evidence.** A well-separated active subspace shows a gap between $\lambda_n$ and
$\lambda_{n+1}$ that is large compared with the bootstrap intervals of both, and a bootstrap
distance $\hat\delta_n$ that is small. If $\hat\delta_k$ is large for every $k$, the sample is too
small for any subspace to be estimated reliably.

## References

- Beran, R. and Srivastava, M. S. (1985). Bootstrap tests and confidence regions for functions of a
  covariance matrix. *Annals of Statistics* 13(1):95–115. (details to verify)
- Constantine, P. G. (2015). *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM.
- Eaton, M. L. and Tyler, D. E. (1991). On Wielandt's inequality and its application to the
  asymptotic distribution of the eigenvalues of a random symmetric matrix. *Annals of Statistics*
  19(1):260–271. (details to verify)
- Luo, W. and Li, B. (2016). Combining eigenvalues and variation of eigenvectors for order
  determination. *Biometrika* 103(4):875–887. (details to verify)
