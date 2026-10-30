(theory-surrogates)=
# Surrogates and designs

A surrogate approximates $f(x)\approx g(W_1^\top x)$ by a cheap function $g$ of the active
variables. Fitted to the conditional means of $f$, it estimates the best ridge approximation
$G(y)=\mathbb E[f\mid W_1^\top x=y]$ of the [Active subspaces](active_subspaces.md) page. The library
has two kinds: polynomials fitted by least squares, and radial basis functions with a polynomial
tail fitted as a Gaussian process. A surrogate's input coordinates are the normalized active
variables $y$ (when built on a subspace) or the normalized inputs $x$ (when built on all inputs);
$k$ denotes their number. Training points are the rows $y_i$, $i=1,\dots,M$.

## Polynomial least squares

The basis is the monomials of total degree at most $d$,

$$y^\alpha=\prod_{j=1}^ky_j^{\alpha_j},\qquad\alpha\in\mathcal I_{k,d}=\lbrace\alpha\in\mathbb N^k:\lvert\alpha\rvert\le d\rbrace,\qquad p=\binom{k+d}{k}\ \text{terms},$$

ordered by degree and then lexicographically descending (for $k=2$, $d=2$: $1$, $y_1$, $y_2$,
$y_1^2$, $y_1y_2$, $y_2^2$). With design matrix $B_{ia}=y_i^{\alpha_a}$ and sample weights
$s_i\ge0$,

$$\hat\beta=\arg\min_\beta\sum_{i=1}^Ms_i\bigl(f_i-b(y_i)^\top\beta\bigr)^2,$$

solved with rows scaled by $\sqrt{s_i}$, so an integer weight is equivalent to repeating a row.
(The 2016 code scaled rows by $s_i$, which does not minimize this objective.) The columns are scaled
to unit norm before a singular value decomposition; a basis that is rank-deficient on the points
raises an error. The basis is the raw monomial basis, which is ill-conditioned beyond degree about
6.

For $d\ge1$ the coefficients of degree at most 2 give the Taylor terms of $g$ at $y=0$,
$g(y)=c+a^\top y+\tfrac12y^\top Hy+\dots$, with $c=\hat\beta_0$, $a_j=\hat\beta_{e_j}$,
$H_{jj}=2\hat\beta_{2e_j}$ and $H_{jl}=\hat\beta_{e_j+e_l}$ ($j\ne l$); `quadratic_terms` returns
them.

## Radial basis functions with a polynomial tail

**Model.** The RBF surrogate is the posterior mean of a Gaussian process with a polynomial mean,

$$f=B\beta+\varepsilon,\qquad\varepsilon\sim N\bigl(0,\ \sigma^2(R_\eta+\Delta)\bigr),\qquad(R_\eta)_{il}=\exp\Bigl(-\sum_{j=1}^k\frac{(y_{ij}-y_{lj})^2}{\ell_j}\Bigr),$$

with an anisotropic squared-exponential correlation, $B$ the polynomial basis of degree $d$ (the
tail, default 2), an amplitude $\sigma^2$ and a diagonal nugget $\Delta=\mathrm{diag}(\delta)$.
Write $K_\eta=R_\eta+\Delta$.

**Length scales.** On a subspace with eigenvalues $\lambda_1,\dots,\lambda_m$ (magnitudes are used),

$$\ell_j=\eta\,\frac{\sum_{q=1}^m\lvert\lambda_q\rvert}{\max\bigl(\lvert\lambda_j\rvert,\ \epsilon\lvert\lambda_1\rvert\bigr)},\qquad j=1,\dots,n,$$

so active directions with larger eigenvalues get shorter length scales, and one scalar $\eta$ is
left to fit. This is the rule of Constantine, Dow and Wang (2014, §4.2–4.3, eq. (4.16)), whose
parameter $\alpha$ is $\eta=2\alpha$ here [D-194]. It is also the rule of the 2016 code.
On all inputs, or on a subspace given only by a basis (no eigenvalues), $\ell_j=\eta$.

**Nugget.** The nugget is dimensionless, relative to the amplitude. With
$s_f^2=\frac1M\sum_i(f_i-\bar f)^2$:

- given noise variances $v_i$ in units of $f^2$, $\delta_i=v_i/s_f^2$;
- otherwise, on a subspace with eigenvalues, the inactive spectral fraction
  $\delta_i=\eta\sum_{q>n}\lvert\lambda_q\rvert\big/\sum_{q=1}^m\lvert\lambda_q\rvert$;
- otherwise $\delta_i=0$;

and every $\delta_i$ is then floored at $10^{-6}$, in the objective and in the fit. Each $\delta_i$ is
unchanged when $f$ is replaced by $af+b$, so the fitted surrogate transforms the same way: it is
affine-equivariant in $f$.

**Hyperparameter.** For fixed $\eta$, the generalized least-squares estimates are

$$\hat\beta_\eta=(B^\top K_\eta^{-1}B)^{-1}B^\top K_\eta^{-1}f,\qquad r_\eta=f-B\hat\beta_\eta,\qquad\hat\sigma^2_\eta=\tfrac1M\,r_\eta^\top K_\eta^{-1}r_\eta$$

(floored at $5\epsilon s_f^2$), and $\eta$ minimizes the concentrated negative log-likelihood

$$J(\eta)=\tfrac12\log\det K_\eta+\tfrac M2\log\hat\sigma^2_\eta,$$

which is $-\log$ of the Gaussian likelihood at $(\hat\beta_\eta,\hat\sigma^2_\eta)$ up to a constant
[Rasmussen and Williams 2006]. By default the search is over $\log_{10}\eta\in[-10,1]$; with
`f_variance` it is over the published interval below. The search evaluates $J$ on a grid of $\log_{10}\eta$ with step at
most 0.25, then refines by a bounded scalar minimization around the best grid point. It is
deterministic.

**Published search interval.** Constantine, Dow and Wang (2014, eq. (4.13)) bound $\alpha$, so
with $\eta=2\alpha$ the interval for $\eta$ is

$$\frac{2\hat\sigma_f^2}{\sum_q\lvert\lambda_q\rvert}\ \le\ \eta\ \le\ 2C_1,\qquad
\log_{10}\eta\in\Bigl[\log_{10}\frac{2\hat\sigma_f^2}{\sum_q\lvert\lambda_q\rvert},\ \log_{10}(2C_1)\Bigr].$$

Here $\hat\sigma_f^2$ is the biased variance of $f$ at the **gradient samples** (the samples that
gave the eigenvalues), not of the training values, and $C_1$ is the Poincaré constant of the input
density: $C_1=1$ for a Gaussian and $C_1=4/\pi^2$ for the uniform density on a cube, which is sharp
[D-193, D-204]. The paper's (4.10) prints $2\sqrt m/\pi$ for the uniform case and cites Bebendorf;
Bebendorf's bound uses the diameter of the domain and gives $4m/\pi^2$ for $[-1,1]^m$.
Both are valid but loose, and v1 uses the sharp constant. This is an erratum to the paper, not a
departure from it. The interval is opt-in through `f_variance` [D-192]. If it is empty, v1 uses its
upper end and issues `accuracy.empty_bounds`. If the likelihood is flat over four or more decades of
$\log_{10}\eta$, the selected value is arbitrary and v1 issues `accuracy.flat_objective`. Explicit
`log10_eta_bounds` or `log10_eta` take precedence over `f_variance`. The bounds are derived for the eigenvalues of
$E[\nabla f\nabla f^\top]$ (the `as` and `opg` estimators); with another estimator the
eigenvalues have other units and the interval is only a heuristic.

:::{admonition} Departure from the 2016 code: objective and nugget
:class: important

The 2016 code minimized $\tfrac12\log\det K+M\log\hat\sigma^2$, with a nugget in units of $f^2$ added to
a unit-amplitude kernel and the eigenvalue default $\eta\sum_{q>n}\lambda_q$. v1 minimizes the
concentrated likelihood $\tfrac12\log\det K+\tfrac M2\log\hat\sigma^2$ and uses the dimensionless
nugget above. Reason: the 2016 fit was not invariant to the units of $f$ (multiplying $f$ by 100
changed its predictions by more than a rescaling), and its objective was not the concentrated
likelihood. The numbers differ from 2016 whenever $\sum_q\lambda_q\ne1$ or $s_f^2\ne1$. On one test problem
($f=a\sin3y$ on $[-1,1]$, 25 training points) v1's relative prediction error is 0.0117 at both $a=1$
and $a=100$; the 2016 code gives 0.0074 at $a=1$ and 0.554 at $a=100$.
:::

:::{admonition} Departures from the published method
:class: important

v1 keeps two departures from Constantine, Dow and Wang (2014) [D-194]. Both are deliberate.

1. **Amplitude.** The paper ties the amplitude to the length scale, $\sigma^2=\alpha\sum_q\lambda_q$
   (4.10). v1 profiles it out: $\hat\sigma^2_\eta=r_\eta^\top K_\eta^{-1}r_\eta/M$, the maximum
   likelihood value for each $\eta$.
2. **Nugget.** The paper's nugget is $\sum_{q>n}\lambda_q/\sum_q\lambda_q$ (4.14), independent of
   $\alpha$. v1 uses $\delta=\eta\sum_{q>n}\lvert\lambda_q\rvert/\sum_q\lvert\lambda_q\rvert$, which grows
   with $\eta=2\alpha$, and takes magnitudes of the eigenvalues.
:::

**Fit.** At the selected $\eta$ the coefficients solve

$$\begin{bmatrix}K_\eta&B\\B^\top&0\end{bmatrix}\begin{bmatrix}\alpha\\\beta\end{bmatrix}=\begin{bmatrix}f\\0\end{bmatrix},$$

that is, $\beta=\hat\beta_\eta$ and $\alpha=K_\eta^{-1}(f-B\beta)$, and

$$g(y)=\sum_{i=1}^M\alpha_i\,\kappa(y,y_i)+b(y)^\top\beta,\qquad\kappa(y,y')=\exp\Bigl(-\sum_j\frac{(y_j-y'_j)^2}{\ell_j}\Bigr).$$

The nugget enters only the diagonal of $K_\eta$, so $g$ smooths rather than interpolates:
$g(y_i)=f_i-\delta_i\alpha_i$. The cost is $O(M^3)$ per trial $\eta$ and $O(M^2)$ memory, practical up
to a few thousand points.

## Prediction and confidence intervals

For a polynomial surrogate, `ac.predict_interval` and `PolynomialSurrogate.interval` return
Gaussian-error least-squares limits [D-200]. With basis $b(y)$, $p$ coefficients, weights
$S=\mathrm{diag}(s_i)$ and residuals $r_i$, let

$$s^2=\frac{\sum_i s_i r_i^2}{\nu},\qquad\nu=\#\lbrace s_i>0\rbrace-p.$$

The limits at $y$ are

$$\hat g(y)\pm t_{\nu,(1+\text{level})/2}\;s\,\sqrt{c+b(y)^\top(B^\top SB)^{-1}b(y)},$$

with $c=1$ for a prediction interval (a new observation, whose weight is 1) and $c=0$ for a
confidence interval (the mean of $f$ given $y$). A one-sided limit uses $t_{\nu,\text{level}}$. The
interval needs $\nu>0$, and it is not defined for an RBF. It is conditional on $W_1$: it ignores
the error in estimating the subspace. See Montgomery, Peck and Vining (2012, §3.4–3.5).

## Fit quality: leave-one-out

Both kinds report the leave-one-out residuals $e_i=f_i-g^{(-i)}(y_i)$, where $g^{(-i)}$ is refitted
without point $i$ and with the hyperparameters held fixed. Both have closed forms:

- polynomial: $e_i=r_i/(1-h_{ii})$, with $r=f-B\hat\beta$ and $h_{ii}$ the leverages of the weighted fit;
- RBF: $e_i=\alpha_i/(S^{-1})_{ii}$, with $S$ the saddle-point matrix above [Rippa 1999]; for a
  positive nugget this is the Gaussian-process leave-one-out identity [Rasmussen and Williams 2006].

The summary is

$$R^2_{\text{LOO}}=1-\frac{\sum_is_ie_i^2}{\sum_is_i(f_i-\bar f_s)^2},\qquad\bar f_s=\frac{\sum_is_if_i}{\sum_is_i},$$

with $s_i=1$ for the RBF. It is undefined (`None`, with a warning) when a point determines its own
fit, for example with as many points as basis functions. An in-sample $R^2$ is close to 1 for any
interpolant, which is why it is not reported.

## Training from a model: conditional means

To fit $G$ rather than $f$, the model is evaluated at $N_{MC}$ inputs per design point $y_k$, with
inactive variables drawn from $\pi(z\mid y_k)$ by the inverse map (see [Geometry](geometry.md)), and the
surrogate is fitted to the conditional means with their sampling variance as noise:

$$\bar f_k=\frac1{N_{MC}}\sum_{j=1}^{N_{MC}}f(W_1y_k+W_2z_{kj}),\qquad\hat V_k=\frac1{N_{MC}-1}\sum_j(f_{kj}-\bar f_k)^2,\qquad v_k=\hat V_k/N_{MC}.$$

$v_k$ is the variance of the mean, not of $f$ (the 2016 code passed $\hat V_k$ with divisor
$N_{MC}$). It treats the draws as independent, which they are for Gaussian inputs and for
one inactive variable; hit-and-run draws are correlated, which thinning reduces. The default is
$N_{MC}=50$, and at least 2 are required.

**Vertices.** On a bounded domain the design is augmented with the vertices of $\mathcal Y$. If a
vertex has a unique preimage (no inert input), $f$ is evaluated once there, with $\hat V=0$. If some
inputs are inert, the preimage is a face of the cube, and $N_{MC}$ points are drawn with the inert
coordinates uniform on $[-1,1]$, as for any design point.

## Designs on the active domain

| Domain | Design of $N$ points | `method` |
|---|---|---|
| interval $[a,b]$ | $y_k=a+k\,\frac{b-a}{N+1}$, $k=1,\dots,N$ (equispaced, endpoints excluded) | `"interval"` |
| zonotope, $N(N-1)/2+N N_f\le2000$ | maximin by SLSQP in epigraph form | `"maximin_slsqp"` |
| zonotope, $N(N-1)/2+N N_f>2000$ | greedy farthest point | `"maximin_greedy"` |
| Gaussian | tensor Gauss–Hermite nodes, $q$ per dimension with $q=\max\lbrace q:q^n\le N\rbrace$ | `"gauss_hermite"` |

The default design size on the model path is $5^n$.

**Maximin.** With vertices $\hat y_v$ and facets $\nu_l^\top y\le b_l$,

$$\max_{y_1,\dots,y_N,\,t}\ t\quad\text{s.t.}\quad\lVert y_i-y_j\rVert^2\ge t\ (i<j),\qquad\lVert y_i-\hat y_v\rVert^2\ge t\ (\text{all }i,v),\qquad\nu_l^\top y_i\le b_l\ (\text{all }i,l).$$

This maximizes the smallest squared distance among the design points and to the vertices, as in
the 2016 code, but in epigraph form: the constraints are smooth, whereas the 2016 objective
$\min_{i<j}\lVert y_i-y_j\rVert$ is not differentiable, a likely cause of the 2016 code's failed
starts. The problem is
non-convex; the library solves it from several starts drawn uniformly in $\mathcal Y$ (default 3) and
returns the best design that satisfies the facet constraints, warning if some starts failed.

**Greedy farthest point.** The epigraph problem has $N(N-1)/2$ pair constraints and $N N_f$ facet
constraints ($N_f$ facets), and SLSQP's dense subproblems make it impractical beyond about 2000
constraints in all. Above that, the design starts from the vertex set $S$, draws $50N$ candidates uniformly in
$\mathcal Y$, and $N$ times adds the candidate farthest from $S$:
$c^\ast=\arg\max_c\min_{s\in S}\lVert c-s\rVert$. This is the standard greedy 2-approximation to the
$k$-centre problem, relative to the candidate set. It costs $O(50N^2n)$.

## References

- Constantine, P. G. (2015). *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM.
- Constantine, P. G., Dow, E. and Wang, Q. (2014). Active subspace methods in theory and practice:
  applications to kriging surfaces. *SIAM Journal on Scientific Computing* 36(4):A1500–A1524.
- Montgomery, D. C., Peck, E. A. and Vining, G. G. (2012). *Introduction to Linear Regression
  Analysis*, 5th ed. Wiley. §3.4–3.5.
- Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT
  Press. §5.4.2. (details to verify)
- Rippa, S. (1999). *Advances in Computational Mathematics* 11:193–210. (title and details to
  verify)
