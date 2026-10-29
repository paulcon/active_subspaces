---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Integration

The mean of $f$ over the input density, $\mu = \int f(x)\,\rho(x)\,dx$, splits into an integral
over the active variables and one over the inactive variables [Constantine 2015]:

$$\mu = \int_{\mathcal Y} G(y)\,\pi(y)\,dy, \qquad G(y) = \int f(W_1 y + W_2 z)\,\pi(z \mid y)\,dz,$$

where $\pi(y)$ is the density of $y = W_1^\top x$ and $\pi(z\mid y)$ the conditional density of
$z$. `ac.integrate` applies a quadrature rule $\lbrace(y_k, \omega_k)\rbrace$ in $y$ and Monte
Carlo in $z$:

$$\hat G_k = \frac1{N_{MC}}\sum_{j=1}^{N_{MC}} f(W_1 y_k + W_2 z_{kj}), \qquad
\hat\mu = \sum_{k=1}^{N_q} \omega_k \hat G_k ,$$

with the $z_{kj}$ drawn by the inverse map (see {doc}`geometry`). It costs $N_q N_{MC}$ model
runs. When $f$ varies little along $z$, a few nodes in $y$ and a modest $N_{MC}$ go a long way.
The theory page {doc}`../theory/integration_and_optimization` gives the error analysis.

```{code-cell} python
import numpy as np
import active_subspaces as ac

wing = ac.test_functions.builtin_models().get("wing_weight")
space = wing.space
X = space.sample(300, rng=0)
f = ac.evaluate(wing, X=X)
sub = ac.active_subspace(grads=ac.evaluate_grads(wing, X=X), space=space)   # n = 1
dom = ac.active_domain(sub)
ref = wing.reference_results
print(f"reference mean {ref.mean:.3f} (Monte Carlo, standard error {ref.mean_se:.3f})")
```

## Integrating a model

```{code-cell} python
res = ac.integrate(wing, domain=dom, n_nodes=8, rng=1)       # n_mc = 50 by default
print(f"estimate {res.estimate:.3f}, inactive standard error {res.inactive_std_error:.3f}")
print("model runs:", res.f.size, "  sampler:", res.design.inactive_sampler)
```

`model` is an `ac.Model` on the domain's space, or a batch callable of physical points
`(k, m) -> (k,)`. `executor=` and `batch_size=` are passed to the evaluation (see {doc}`models`).

## Rules on the active variables

`n_nodes` means slightly different things for each kind of domain.

| Domain | Rule | `n_nodes` | Exact for |
|---|---|---|---|
| Gaussian | tensor Gauss–Hermite | total count $N$; $k = \lfloor N^{1/n}\rfloor$ per dimension, or a list of per-dimension counts | degree $\le 2k-1$ in each $y_i$ |
| interval ($n=1$) | Gauss rule of the density $\pi$ of $y = w^\top x$ | the node count $N_q$ | degree $\le 2N_q - 1$ against $\pi$ |
| zonotope ($n\ge2$) | centroids of a Delaunay triangulation of the vertices and $N_d$ maximin points; weights are Monte Carlo cell masses from `n_weight_samples` draws (default $10^5$) | $N_d$ | constants; error $O(h)$ in the largest cell diameter |

`ac.active_quadrature_rule(domain, n_nodes=…)` returns the rule itself:

```{code-cell} python
rule = ac.active_quadrature_rule(dom, n_nodes=8)
print(rule.kind, "degree", rule.degree)
print(np.round(rule.nodes.ravel(), 3))
print(np.round(rule.weights, 4), rule.weights.sum())
```

For an interval, $\pi$ is the density of a sum of independent uniform variables, a box spline,
and the rule is its $N_q$-point Gauss rule [Golub and Welsch 1969; Gautschi 2004]. The 2016 code
used midpoints of equal subintervals with Monte Carlo histogram weights instead. On a zonotope,
the weights are Monte Carlo estimates; if some samples fall outside an incomplete triangulation,
the weights are renormalized and an `accuracy.weight_deficit` warning reports the fraction.

## Standard errors

`inactive_std_error` is the standard error of the Monte Carlo in $z$, given the rule:

$$\hat\sigma_{\text{in}} = \Bigl(\sum_k \omega_k^2\,\hat\tau\,\hat V_k / N_{MC}\Bigr)^{1/2}, \qquad
\hat V_k = \frac1{N_{MC}-1}\sum_j \bigl(f(x_{kj}) - \hat G_k\bigr)^2 .$$

$\hat\tau = 1$ when the draws are independent: Gaussian spaces, and uniform spaces with
$m - n = 1$. For hit-and-run draws, $\hat\tau$ is the integrated autocorrelation time of $f$
along the chains, estimated with Geyer's initial positive sequence [Geyer 1992] and reported as
`autocorrelation_time`. No confidence interval is formed; multiply by a quantile yourself if you
want one.

The rule's own error is not in $\hat\sigma_{\text{in}}$. To see every random component, run
`n_replicates` $= R \ge 2$ independent replicates of the whole procedure. Then `estimate` is their
mean and `total_std_error` $= s_R/\sqrt R$, with $s_R$ the standard deviation of the replicate
estimates:

```{code-cell} python
res4 = ac.integrate(wing, domain=dom, n_nodes=8, n_replicates=4, rng=2)
print(np.round(res4.replicate_estimates, 3))
print(f"estimate {res4.estimate:.3f}, total {res4.total_std_error:.3f}, "
      f"inactive {res4.inactive_std_error:.3f}")
```

Replicates cannot see a bias: neither the error of a fixed rule in $y$ (Gaussian and interval
rules), nor the discretization error of the Delaunay rule on a zonotope.

## How many inactive samples

`n_mc` defaults to 50. Hit-and-run draws are correlated, and with short chains Geyer's estimate
of $\hat\tau$ is too small, so `inactive_std_error` understates the error. With
$2 \le$ `n_mc` $< 20$ on hit-and-run, the library warns `accuracy.correlated_draws`.

In a calibration on $f = \lVert x\rVert^2$ over a box with $n=1$ and 8 nodes, the reported
standard error at `n_mc=50` was within about 10% of the actual spread of $\hat\mu$ for $m - n$ up
to 19; at `n_mc=10` it was 16–38% too small. The shortfall grows with $m - n$, because hit-and-run
mixes more slowly in more dimensions. Raise `n_mc`, or set `thin` above its default of $m - n$,
when $m - n$ is large or when you rely on the standard error. Use `n_replicates` to check it.

## Integrating a surrogate

For a function of $y$ alone, such as a surrogate, no inactive draws are needed:
$\int g(W_1^\top x)\rho(x)\,dx \approx \sum_k \omega_k g(y_k)$.

```{code-cell} python
poly = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)
rs = ac.integrate_surrogate(poly, domain=dom, n_nodes=8)
print(f"{rs.estimate:.3f}", rs.source, rs.inactive_std_error)
```

The 8-point Gauss rule is exact for this quadratic, so the only error is that of the surrogate.

## Running the model yourself

If your model runs outside Python, or on a cluster, split `integrate` into its three steps:

```{code-cell} python
design = ac.integration_design(dom, n_nodes=8, rng=3)   # nodes and inactive draws
Xrun = design.X_physical                                 # (Q * n_mc, m) physical points to run
f_run = ac.evaluate(wing, X=Xrun)                        # ... or your own runs, in row order
res_d = ac.estimate_integral(design, f=f_run)
print(Xrun.shape, f"{res_d.estimate:.3f}")
```

API reference: {mod}`active_subspaces.integration`.

## References

- P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM, 2015.
- W. Gautschi. *Orthogonal Polynomials: Computation and Approximation.* Oxford University Press,
  2004.
- C. J. Geyer. Practical Markov chain Monte Carlo. *Statist. Sci.* 7(4):473–483, 1992.
- G. H. Golub, J. H. Welsch. Calculation of Gauss quadrature rules. *Math. Comp.*
  23(106):221–230, 1969.
