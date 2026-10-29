---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(theory-integration-optimization)=
# Integration and optimization

This page covers what the library does with an active subspace once it has one: estimate the mean
$\mu=\int f\rho\,dx$ with a rule on the active variables and Monte Carlo on the inactive ones, and
minimize over the active variables and then lift the minimizer back to a full input. Everything is
in normalized coordinates.

## Integration by active and inactive variables

With $x=W_1y+W_2z$, let $\pi(y)$ be the marginal density of $y=W_1^\top x$ under $\rho$ and
$\pi(z\mid y)$ the conditional density of $z$. Then [Constantine 2015]

$$\mu=\int_{\mathcal Y}G(y)\,\pi(y)\,dy,\qquad G(y)=\int f(W_1y+W_2z)\,\pi(z\mid y)\,dz .$$

| Space | $\pi(y)$ | $\pi(z\mid y)$ |
|---|---|---|
| Gaussian | $N(0,I_n)$ | $N(0,I_{m-n})$, independent of $y$ |
| Uniform | the push-forward of the uniform density on $\mathcal Y$; for $n=1$ the box spline of $w=W_1$ | uniform on the fiber polytope $\mathcal P(y)$ (see [Geometry](geometry.md)) |

**Estimator.** A rule $\lbrace(y_k,\omega_k)\rbrace_{k=1}^{N_q}$ on $\mathcal Y$ with $\omega_k\ge0$ and
$\sum_k\omega_k=1$, and $N_{MC}$ draws $z_{kj}$ from $\pi(z\mid y_k)$ by the inverse map, give

$$\hat G_k=\frac1{N_{MC}}\sum_{j=1}^{N_{MC}}f(W_1y_k+W_2z_{kj}),\qquad\hat\mu=\sum_{k=1}^{N_q}\omega_k\hat G_k .$$

The pairs $\bigl(x_{kj},\ \omega_k/N_{MC}\bigr)$ form a quadrature rule on $x$ whose weights sum to 1,
so they can be passed as `quad_weights` to the estimators. A surrogate $g$ of $y$ alone needs no
inactive draws: $\int g(W_1^\top x)\rho(x)\,dx\approx\sum_k\omega_kg(y_k)$ (`integrate_surrogate`).

## Rules on the active variables

| Domain | Nodes | Weights | Exact for | `kind` |
|---|---|---|---|---|
| Gaussian | tensor product of probabilists' Gauss–Hermite rules, $k_i$ points in $y_i$ | products of the 1-D weights | polynomials of degree $\le2k_i-1$ in $y_i$ | `"gauss_hermite"` |
| Uniform, $n=1$ | the $N_q$-point Gauss rule of the box-spline density $\pi$ | its Christoffel weights | polynomials of degree $\le2N_q-1$ against $\pi$ | `"box_spline_gauss"` |
| Uniform, $n\ge2$ | centroids of Delaunay simplices | Monte Carlo cell masses | constants; error $O(h)$ for Lipschitz $G$ | `"delaunay_centroid"` |

For the Gaussian rule an integer `n_nodes` $N$ gives $k_i=k$, the largest integer with $k^n\le N$
(computed exactly), and a sequence gives $k_i$ per dimension. One-dimensional Gauss rules are
computed by the Golub–Welsch method from the three-term recurrence [Golub and Welsch 1969]; rules
of symmetric measures are symmetrized, so the middle node of an odd rule is exactly 0.

### One active variable: the box-spline Gauss rule

For uniform inputs and $n=1$, $y=\sum_{j=1}^mw_jx_j$ with independent $x_j\sim U[-1,1]$. Its density
$\pi$ is a box spline, supported on $[-\lVert w\rVert_1,\lVert w\rVert_1]$ and symmetric about 0. The
rule is the unique $N_q$-point Gauss rule of $\pi$, exact for polynomials of degree up to $2N_q-1$,
with deterministic nodes and weights.

It is computed by convolution. Start from the $N_q$-point Gauss–Legendre rule scaled by $w_{j_1}$.
For each further nonzero $w_j$, convolve the current discrete measure with the scaled Legendre rule
(an $N_q^2$-atom measure) and compress it back to $N_q$ atoms by the Gauss rule of that discrete
measure, computed by the Lanczos process with full reorthogonalization. Each step preserves the
moments up to degree $2N_q-1$, and the moments of a sum up to a given degree depend only on the
summands' moments up to that degree. By induction the final rule has the moments of $\pi$ up to
degree $2N_q-1$, so in exact arithmetic it is the Gauss rule of $\pi$, the same rule that the
moments of $\pi$ define. Each step is a stable Lanczos process on a discrete measure
[Gautschi 2004], whereas the map from moments to the recurrence is exponentially ill-conditioned.
The cost is $O(mN_q^4)$.

### Two or more active variables: the Delaunay-centroid rule

For uniform inputs and $n\ge2$, the rule triangulates the zonotope. The points are its vertices
together with a maximin design of $N_d$ interior points (`n_nodes`; see
[Surrogates and designs](surrogates_and_designs.md)). The nodes are the centroids $c_k$ of the
Delaunay simplices $T_k$, and the weights are Monte Carlo estimates of the cell masses:

$$\omega_k=\frac{\#\lbrace i:\ W_1^\top x_i\in T_k\rbrace}{N_X},\qquad x_i\sim\rho\ \text{independent},$$

with $N_X$ = `n_weight_samples` (default $10^5$).

Since every $W_1^\top x_i$ lies in the zonotope and the vertex set is complete, the weights sum to 1.
Simplices with zero weight are dropped. The rule integrates constants exactly; for Lipschitz $G$ its
error is $O(h)$ in the largest cell diameter $h$. The Monte Carlo weights add an error floor of
about $\sigma_G/\sqrt{N_X}$, about $3\times10^{-3}\sigma_G$ at the default, with $\sigma_G$ the standard
deviation of $G(y)$, whatever the number of nodes.

:::{admonition} Departure from the 2016 method: the active-variable rules
:class: important

**One active variable.** The 2016 code used the midpoints of $N$ equal cells of the interval with
weights from a Monte Carlo histogram of $10^4$ samples. v1 uses the Gauss rule of the box spline
above. Reason: the histogram weights left an error floor $\sigma_G/\sqrt{10^4}$ whatever the number of
nodes; the Gauss rule is deterministic and exact for polynomials of degree up to $2N_q-1$.

**Two or more active variables.** The 2016 code fixed $N_X=10^4$, and samples that fell outside an
incomplete vertex hull were silently lost, so the weights summed to less than 1 and $\hat\mu$ was
biased toward 0. v1 makes $N_X$ a parameter (default $10^5$). If some samples still fall outside the
triangulation (a fraction $d$), the weights are renormalized,
$\omega_k=\#\lbrace i:W_1^\top x_i\in T_k\rbrace/\bigl(N_X(1-d)\bigr)$, and an `AccuracyWarning`
(`accuracy.weight_deficit`) reports $d$. Renormalizing estimates $\mathbb E[G\mid y\in\text{hull}]$
rather than $\mu$; with the exact vertex enumeration of v1 the hull is complete and $d=0$.
:::

## Error estimates

**Inactive Monte Carlo standard error.** With the conditional variances

$$\hat V_k=\frac1{N_{MC}-1}\sum_{j=1}^{N_{MC}}\bigl(f(x_{kj})-\hat G_k\bigr)^2,$$

the standard error of $\hat\mu$ for a fixed rule is

$$\hat\sigma_{\text{in}}=\Bigl(\sum_k\omega_k^2\,\hat\tau\,\hat V_k/N_{MC}\Bigr)^{1/2}$$

(`inactive_std_error`). $\hat\tau=1$ when the draws are independent: Gaussian inputs, and uniform
inputs with one inactive variable. For hit-and-run draws, $\hat\tau$ (`autocorrelation_time`) is the
integrated autocorrelation time of $f$ along the chains, pooled over the nodes and estimated by
Geyer's initial positive sequence [Geyer 1992]. With $c$ indexing the chains (one per node) and
$\bar f_c$ the chain means,

$$\hat\gamma_\ell=\frac1{N_qN_{MC}}\sum_c\sum_{j=1}^{N_{MC}-\ell}(f_{c,j}-\bar f_c)(f_{c,j+\ell}-\bar f_c),\qquad\hat r_\ell=\frac{\hat\gamma_\ell}{\hat\gamma_0},\qquad\Gamma_t=\hat r_{2t}+\hat r_{2t+1},$$

and, with $T$ the last $t$ before the first $\Gamma_t\le0$,
$\hat\tau=\min\bigl(N_{MC},\max(1,\,-1+2\sum_{t=0}^T\Gamma_t)\bigr)$. Pooling assumes that the chains
mix similarly at every node. No confidence interval is formed; the 2016 code reported
$\hat\mu\pm1.96\hat\sigma$ with variances computed with divisor $N_{MC}$ and with correlated draws
treated as independent.

**How many inactive draws.** With very short chains Geyer's $\hat\tau$ underestimates the
autocorrelation time, so the standard error is too small. Hit-and-run designs with $N_{MC}<20$ warn
(`accuracy.correlated_draws`), and for $N_{MC}<4$ the library sets $\hat\tau=1$. The default is
$N_{MC}=50$. A calibration on $f=\lVert x\rVert^2$ on a random box ($n=1$, 8 nodes, hit-and-run,
400 seeds) gave these ratios of the root-mean-square standard error to the empirical standard
deviation of $\hat\mu$, with the coverage of $\hat\mu\pm1.96\,\hat\sigma_{\text{in}}$:

| $m-n$ | $N_{MC}=10$ | 20 | 50 | 100 |
|---|---|---|---|---|
| 3 | 0.84, 86.5% | 0.91, 91.5% | 1.08, 96.8% | 1.10, 96.3% |
| 9 | 0.62, 75.7% | 0.79, 85.0% | 0.92, 91.2% | 1.03, 96.0% |
| 19 | | | 0.98, 93.5% | 1.03, 95.3% |

At 50 draws the standard error was within about 10% in these cases. Hit-and-run mixes more slowly
as $m-n$ grows, so for a large inactive dimension raise `n_mc` or set `thin` above its default.

**Total error from replicates.** With $R\ge2$ independent replicates of the whole procedure (design,
weights and all inactive draws), giving $\hat\mu_1,\dots,\hat\mu_R$, the library reports their mean
and

$$\hat\sigma_{\text{tot}}=\frac{s_R}{\sqrt R},\qquad s_R^2=\frac1{R-1}\sum_r(\hat\mu_r-\hat\mu)^2$$

(`total_std_error`). It includes every random component: the inactive Monte Carlo, the Monte Carlo
weights and the random design. It does not include bias, which replicates cannot see: the
deterministic error of a fixed rule in $y$ (Gaussian and $n=1$), and the discretization error of the
Delaunay-centroid rule, whose expected value $\sum_k\pi(T_k)\,G(c_k)$ differs from $\mu$ unless $G$
is constant on each cell.

The cell below integrates $f(x)=(a^\top x)^2+\lVert x\rVert^2$ on $[-1,1]^5$ with the exact
direction $w=a/\lVert a\rVert$, so $\mu=\lVert a\rVert^2/3+5/3=20$. The ridge part is integrated exactly
by the box-spline rule; the remainder varies along the four inactive variables, which hit-and-run
samples.

```{code-cell} python
import numpy as np
import active_subspaces as ac

m = 5
space = ac.Uniform.reference(m)              # physical = normalized here
a = np.arange(1.0, m + 1)
domain = ac.active_domain(ac.subspace_from_basis(a[:, None], space=space))

rule = ac.active_quadrature_rule(domain, n_nodes=3)
print(rule.kind, "nodes", rule.nodes[:, 0].round(4), "weights", rule.weights.round(4))

def f(X):
    return (X @ a) ** 2 + np.sum(X**2, axis=1)

res = ac.integrate(f, domain=domain, n_nodes=4, n_mc=50, n_replicates=10, rng=0)
print(f"estimate {res.estimate:.4f} (exact 20)   inactive SE {res.inactive_std_error:.4f}"
      f"   total SE {res.total_std_error:.4f}   tau {res.autocorrelation_time:.2f}"
      f"   evaluations {res.n_evaluations}")
```

## Minimizing over the active variables

Given a function $g$ of $y$ (a surrogate, or any callable), the library computes
$y^\ast\in\arg\min_{y\in\mathcal Y}g(y)$:

| Domain | Method | Starts |
|---|---|---|
| interval $[\ell,u]$ | bounded Brent | one run |
| zonotope $\lbrace y:Ay\le b\rbrace$ | SLSQP with the linear constraints, and $\nabla g$ when available | `n_starts` (default 10) points $y_0=W_1^\top x_0$, $x_0\sim\rho$ |
| Gaussian, $\mathbb R^n$ | BFGS, with $\nabla g$ when available | as above, so $y_0\sim N(0,I_n)$ |

Starts drawn as $W_1^\top x_0$ always lie in $\mathcal Y$. $y^\ast$ is the best value over the runs that
did not fail; failed runs are reported, not fatal, unless all fail. In the Gaussian case the
minimization is over all of $\mathbb R^n$ and can leave the region where the surrogate was fitted;
the library warns (`accuracy.extrapolation`) when $\lVert y^\ast\rVert>\sqrt{\chi^2_{n,\,0.999}}$, the
radius of the 99.9% probability ball of $N(0,I_n)$, but does not constrain $y$.

## The lift: from $y^\ast$ to a full input

A minimizer of $g$ fixes only $y$. The lift chooses the inactive variables by minimizing a quadratic
model of $f$ fitted to the stored samples. No published reference exists for the lift; it is the
heuristic of the 2016 code with its defects corrected. Let $p$ be the lift dimension,
$n+1\le p\le m$ (default $n+2$ if $m-n>2$, else $n+1$), $W_p=[w_1\cdots w_p]$ and
$u=W_p^\top x=(y,\zeta)$ with $\zeta\in\mathbb R^{p-n}$.

1. **Fit** by least squares over the $(p+1)(p+2)/2$ monomials of degree at most 2 in $u$:
   $$q(u)=a_0+\gamma^\top u+\tfrac12u^\top Hu,\qquad\gamma=\begin{bmatrix}\gamma_y\\\gamma_\zeta\end{bmatrix},\quad H=\begin{bmatrix}H_{yy}&H_{y\zeta}\\H_{\zeta y}&H_{\zeta\zeta}\end{bmatrix}.$$
2. **The model in $z$.** Write $z=W_2^\top x=(\zeta,z_b)$ with $z_b\in\mathbb R^{m-p}$. For fixed $y$,
   $q=\text{const}+c(y)^\top\zeta+\tfrac12\zeta^\top Q\zeta$ with $c(y)=\gamma_\zeta+H_{\zeta y}\,y$ and
   $Q=H_{\zeta\zeta}$; $z_b$ does not enter $q$.
3. **Curvature.** With $Q=V\mathrm{diag}(\theta)V^\top$ and $\tau=\delta\max_i\lvert\theta_i\rvert$
   ($\delta$ = `curvature_floor`, default $10^{-2}$), set
   $Q_+=V\mathrm{diag}\bigl(\max(\theta_i,\tau)\bigr)V^\top$. Any clipped eigenvalue issues
   `accuracy.curvature_clipped`. If the fitted curvature is at rounding level,
   $\max_i\lvert\theta_i\rvert\le64\,\epsilon\max(1,\lVert H\rVert,\lVert\gamma\rVert)$, the model is
   treated as constant in $z$: $c=0$ and $\tau=1$, which selects the minimum-norm $z$.
4. **Minimize over $z$** with $\tilde H=\mathrm{blockdiag}(Q_+,\ \tau I_{m-p})$ and $\tilde c=(c(y^\ast),0)$:
   - Gaussian: $z^\ast=-\tilde H^{-1}\tilde c$, that is, $\zeta^\ast=-Q_+^{-1}c$ and $z_b^\ast=0$;
   - uniform: the convex quadratic program
     $$z^\ast=\arg\min_z\ \tfrac12z^\top\tilde Hz+\tilde c^\top z\quad\text{s.t.}\quad-\mathbf 1\le W_1y^\ast+W_2z\le\mathbf 1 .$$
     At a vertex of $\mathcal Y$, where rounding can make this infeasible, $y^\ast$ is first projected
     onto $\mathcal Y$ by a linear program and the distance is reported.
5. $x^\ast=W_1y^\ast+W_2z^\ast$. If a model is passed, $f(x^\ast)$ is evaluated once and reported next
   to $g(y^\ast)$; no further optimization in the full space is done.

:::{admonition} Departure from the 2016 code: the lift
:class: important

**Sign and factor.** The 2016 code returned $z=+Q^{-1}c$ in the Gaussian case and minimized
$c^\top z+z^\top Qz$ in the uniform case; neither is the minimizer of the fitted model
$c^\top\zeta+\tfrac12\zeta^\top Q\zeta$. v1 uses $-Q_+^{-1}c$ and the quadratic program with the
$\tfrac12$. Reason: these are errors, and the corrected forms minimize the fitted model.

**Curvature.** The 2016 code used $Q+0.01I$, which can be indefinite, so its "minimizer" could be a
saddle point and the uniform-case program non-convex; and $0.01$ does not scale with $f$. v1 clips
the eigenvalues at $\delta\max_i\lvert\theta_i\rvert$. Reason: the program stays convex, and the floor
scales with the fitted curvature.

**Unmodelled directions.** The 2016 code added $0.01\lVert z_b\rVert^2$; v1 adds
$\tfrac12\tau\lVert z_b\rVert^2$. Reason: it breaks ties in the directions the model does not see,
toward $z_b=0$, on the same scale as the rest of the model.
:::

## References

- Constantine, P. G. (2015). *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM.
- Gautschi, W. (2004). *Orthogonal Polynomials: Computation and Approximation.* Oxford University
  Press.
- Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical Science* 7(4):473–483.
- Golub, G. H. and Welsch, J. H. (1969). Calculation of Gauss quadrature rules. *Mathematics of
  Computation* 23(106):221–230.
- Smith, R. L. (1984). Efficient Monte Carlo procedures for generating points uniformly distributed
  over bounded regions. *Operations Research* 32(6):1296–1308.
