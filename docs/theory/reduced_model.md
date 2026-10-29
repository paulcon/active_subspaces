(theory-reduced-model)=
# The reduced model

`build_reduced_model` runs the whole pipeline in one call and returns a `ReducedModel`: the
subspace, the active domain, a surrogate on the active variables, the samples the subspace was
estimated from, and a measure of fit. Four operations then answer questions about $f$: prediction,
the mean, a probability and the minimum. The reduced model adds no mathematics of its own; this page
states how it combines the pieces described on the other pages, and what its defaults are.

## The approximation

The reduced model approximates

$$f(\xi)\approx g\bigl(W_1^\top A^{-1}(\xi-b)\bigr),$$

with $g:\mathbb R^n\to\mathbb R$ a surrogate (see [Surrogates and designs](surrogates_and_designs.md)).
For fixed $W_1$ the best approximation of this form in $L^2(\rho)$ is the conditional mean $G(y)$,
whose mean-squared error is at most $C_P(\lambda_{n+1}+\dots+\lambda_m)$ [Constantine, Dow and Wang
2014; Constantine 2015]. The two ways of building the model estimate $G$ differently:

- **From data** (`X`, `f`, optionally gradients): $g$ is fitted to the pairs $(W_1^\top x_i,f_i)$.
- **From a model**: the builder samples $f$ and its gradient to find the subspace, then fits $g$ to
  conditional means $\hat G(y_k)$ at a design on the active domain, with the variance of each mean as
  its noise.

A reduced model needs at least two inputs and at least one inactive variable: $m\ge2$ and
$1\le n\le m-1$. For a function of one input, fit a surrogate on all inputs instead.

## Defaults

**Subspace.** The estimator is AS, the dimension rule is EVG with the linear gap, and the bootstrap
has $B=100$ replicates. On the data path without gradients, gradients for a gradient-based
estimator are estimated by local linear fits at every sample. On the model path, a model without a
gradient is differentiated by forward finite differences. The record says which source was used.

**Sample size on the model path.** The builder draws

$$M=\bigl\lfloor\alpha\,(m+1)\ln m\bigr\rfloor,\qquad\alpha=6,$$

points from $\rho$, the rule $M=\alpha k\ln m$ of [Constantine 2015; Constantine and Gleich 2014]
with $k=m+1$: $M=12$ for $m=2$, 151 for $m=10$ and 2790 for $m=100$.

**Surrogate on the model path.** A design of $5^n$ points, each with $N_{MC}=50$ inactive draws
($N_{MC}\ge2$ is required, so that every node has a variance). These are defaults, not adaptive
rules.

**Model evaluations.** The model path costs $M$ values and $M$ gradients (or, with forward
differences, $Mm$ further values), plus $5^n\cdot N_{MC}$ values for the surrogate, plus the vertex
evaluations of a bounded domain. For example, $m=10$, $n=2$, forward differences, inputs in general
position: $151+1510+5^2\cdot50+20=2931$ evaluations.

**Dimension envelope.** When the dimension rule chooses $n>4$ for uniform inputs or $n>6$ for
Gaussian inputs, the builder warns (`accuracy.large_dimension`) and continues: geometry and the
default design grow quickly with $n$. A fixed `n` never warns.

## Fit quality

With weights $\omega_i$, values $f_i$ and **out-of-sample** predictions $\hat g_i$,

$$R^2=1-\frac{\sum_i\omega_i(f_i-\hat g_i)^2}{\sum_i\omega_i(f_i-\bar f)^2},\qquad\bar f=\sum_i\omega_if_i .$$

| Path | $\hat g_i$ | $\omega_i$ | `r_squared_kind` |
|---|---|---|---|
| model | $g(W_1^\top x_i)$ at the $M$ subspace samples, which the surrogate was not fitted to | $1/M$ | `"holdout"` |
| data | leave-one-out predictions $f_i-e_i$, hyperparameters held fixed | the quadrature weights | `"loo"` |

With quadrature weights, $R^2$ estimates $1-\lVert f-g\rVert^2_{L^2(\rho)}/\mathrm{Var}_\rho f$. An in-sample
$R^2$ would be near 1 for any interpolating surrogate, so it is not used. $R^2$ is `None` when $f$ is
constant or the leave-one-out residuals are undefined.

## Operations

**Prediction.** For physical points, $x=A^{-1}(\xi-b)$, $y=W_1^\top x$, the prediction is $g(y)$, and
the gradient in physical coordinates is

$$\nabla_\xi\bigl[g(W_1^\top A^{-1}(\xi-b))\bigr]=A^{-\top}W_1\nabla_yg(y).$$

**Mean.** By default, the mean of the surrogate, $\int g(W_1^\top x)\rho\,dx\approx\sum_k\omega_kg(y_k)$,
with the active-variable rule of [Integration and optimization](integration_and_optimization.md); no
model evaluations are needed. With `model=`, the mean of $f$ itself by active-variable quadrature and
inactive Monte Carlo, with the standard errors described there. The surrogate's mean estimates
$\mathbb E[G]=\mu$ only as well as $g$ approximates $G$.

**Probability.** The probability **under the surrogate**,

$$p=\mathbb P_\rho\bigl[\ell\le g(W_1^\top x)\le u\bigr],\qquad\hat p=\frac1N\sum_{i=1}^N\mathbf 1\bigl[\ell\le g(W_1^\top x_i)\le u\bigr],\quad x_i\sim\rho,$$

by plain Monte Carlo (default $N=10^4$), with standard error $\sqrt{\hat p(1-\hat p)/N}$.
`confidence_interval(level)` returns the Wilson score interval [Wilson 1927]: with
$q=\Phi^{-1}\bigl(\tfrac{1+c}2\bigr)$ for level $c$ (default 0.99),

$$\tilde p=\frac{\hat p+q^2/(2N)}{1+q^2/N},\qquad h=\frac q{1+q^2/N}\sqrt{\frac{\hat p(1-\hat p)}N+\frac{q^2}{4N^2}},\qquad[\tilde p-h,\ \tilde p+h]\subseteq[0,1].$$

Unlike the normal interval $\hat p\pm q\sqrt{\hat p(1-\hat p)/N}$ (which the 2016 code also returned
with its bounds swapped), it stays in $[0,1]$ and does not collapse to a point when $\hat p$ is 0 or 1.

The interval covers only the Monte Carlo error of $\hat p$. Even with $g=G$ exactly,
$\mathbb P[\ell\le G\le u]$ differs from $\mathbb P[\ell\le f\le u]$, because $G$ averages out the
variation of $f$ along the inactive variables. That difference is controlled by the error of the
ridge approximation, not by $N$.

**Minimum.** Minimize $g$ over the active domain and lift the minimizer to a full input with the
quadratic model fitted to the stored samples (see
[Integration and optimization](integration_and_optimization.md)). With `model=`, $f(x^\ast)$ is
evaluated and reported next to $g(y^\ast)$.

## Relation to the sufficient dimension reduction checks

The 2016 development branch had two quick checks. The linear check fitted $f\approx u_0+u^\top x$ and
reported $u/\lVert u\rVert$: this is the first eigenvector of the OLS estimator. The quadratic check
formed $gg^\top+H\,\mathrm{diag}(\gamma)\,H$ from a fitted quadratic in physical coordinates: this is
QPHD. If $\xi=Ax$ with $A$ diagonal and $A\Sigma A=\mathrm{diag}(\gamma)$, then the normalized matrix
is $C_x=A\,C_{\text{dev}}\,A$, whose physical directions span the same subspace as the development
branch's result when $C_{\text{dev}}$ has exact low rank. See [Estimators](estimators.md).

## References

- Constantine, P. G. (2015). *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM.
- Constantine, P. G., Dow, E. and Wang, Q. (2014). Active subspace methods in theory and practice:
  applications to kriging surfaces. *SIAM Journal on Scientific Computing* 36(4):A1500–A1524.
- Constantine, P. G. and Gleich, D. F. (2014). Computing active subspaces with Monte Carlo.
  arXiv:1408.0545. (details to verify)
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
  *Journal of the American Statistical Association* 22(158):209–212.
