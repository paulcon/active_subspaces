---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Surrogates and designs

A surrogate on the active variables approximates $f(x) \approx g(W_1^\top x)$ with a function
$g$ of only $n$ variables [Constantine, Dow and Wang 2014; Constantine 2015]. v1 has two kinds:

- `kind="poly"`: a least-squares polynomial of total degree `degree`
  (`ac.PolynomialSurrogate`);
- `kind="rbf"` (the default): a Gaussian radial-basis interpolant with a polynomial tail, its
  hyperparameter chosen by maximum likelihood (`ac.RBFSurrogate`).

Fit one to data with `ac.fit_surrogate`, or let the library choose the points and run your model
with `ac.fit_surrogate_from_model`. The theory page {doc}`../theory/surrogates_and_designs` states
the methods in full.

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
```

```{code-cell} python
import warnings
import numpy as np
import active_subspaces as ac

wing = ac.test_functions.builtin_models().get("wing_weight")
space = wing.space
X = space.sample(300, rng=0)                  # physical
f = ac.evaluate(wing, X=X)
sub = ac.active_subspace(grads=ac.evaluate_grads(wing, X=X), space=space)
sub2 = ac.with_dimension(sub, 2)
```

## Fitting to data

Pass **physical** `X`, the values `f`, and the subspace. The surrogate's input coordinates are
the normalized active variables $y = W_1^\top x$.

```{code-cell} python
poly = ac.fit_surrogate(X=X, f=f, subspace=sub2, kind="poly", degree=2)
rbf = ac.fit_surrogate(X=X, f=f, subspace=sub2, kind="rbf")
print(f"poly  leave-one-out R^2 = {poly.loo_r_squared:.4f}")
print(f"rbf   leave-one-out R^2 = {rbf.loo_r_squared:.4f}")
```

`subspace=None` fits on all $m$ normalized inputs instead (then `space=` is required). That is
useful as a baseline:

```{code-cell} python
full = ac.fit_surrogate(X=X, f=f, space=space, subspace=None, kind="poly", degree=2)
print(f"all 10 inputs: {full.coefficients.size} coefficients, LOO R^2 = {full.loo_r_squared:.4f}")
```

## Predicting

`ac.predict` and `ac.predict_gradient` take **physical** points and return values `(M,)` and
**physical** gradients `(M, m)`, $\nabla_\xi g = A^{-\top} W_1 \nabla_y g$. A surrogate's own
`evaluate(Y)` and `gradient(Y)` take normalized active coordinates `(K, n)`.

```{code-cell} python
Xt = space.sample(4, rng=1)
print(np.round(ac.predict(rbf, Xt), 2))
print(np.round(ac.evaluate(wing, X=Xt), 2))
print(ac.predict_gradient(rbf, Xt).shape)
Yt, _ = ac.forward_map(sub2, X=Xt)
print(np.allclose(rbf.evaluate(Yt), ac.predict(rbf, Xt)))
```

```{code-cell} python
fig, ax = ac.plotting.plot_surrogate_fit(rbf, X=X, f=f)
fig
```

## Fit quality: leave-one-out

`loo_r_squared` is computed from the leave-one-out residuals $e_i = f_i - g^{(-i)}(y_i)$, where
$g^{(-i)}$ is refitted without point $i$ with the hyperparameters held fixed:

$$R^2_{\text{LOO}} = 1 - \frac{\sum_i s_i e_i^2}{\sum_i s_i (f_i - \bar f_s)^2}, \qquad \bar f_s = \frac{\sum_i s_i f_i}{\sum_i s_i},$$

with sample weights $s_i$ (all 1 for the RBF). Both kinds have closed forms: $e_i = r_i/(1-h_{ii})$
with the leverages $h_{ii}$ for the polynomial, and the analogous formula for the RBF's linear
system. It is lower than
an in-sample $R^2$, which an interpolant drives to 1. `loo_r_squared` is `None` when the
residuals are undefined, for example when there are as many points as polynomial coefficients
(warning `accuracy.loo_undefined`).

## Options

- **Polynomial.** `sample_weights` `(M,)` weight the least-squares fit,
  $\min_\beta \sum_i s_i (f_i - b(y_i)^\top \beta)^2$; integer weights act like repeated rows.
  `ac.quadratic_terms(poly)` returns $(c, a, H)$ with
  $g(y) = c + a^\top y + \tfrac12 y^\top H y + \dots$
- **RBF.** `noise_variance` `(M,)`, in units of $f^2$, tells the fit how noisy each value is.
  `log10_eta` fixes the hyperparameter $\eta$ instead of searching `log10_eta_bounds`.
  `degree` is the degree of the polynomial tail.

```{code-cell} python
c, a, H = ac.quadratic_terms(poly)
print(round(c, 2), np.round(a, 2), np.round(H, 2), sep="\n")
```

### The RBF model

The RBF surrogate treats $f = B\beta + \varepsilon$ with
$\varepsilon \sim N\bigl(0, \sigma^2(R_\eta + \Delta_\eta)\bigr)$, where

$$(R_\eta)_{il} = \exp\Bigl(-\sum_{j=1}^n \frac{(y_{ij}-y_{lj})^2}{\ell_j}\Bigr), \qquad
\ell_j = \eta\,\frac{\sum_{q=1}^m \lvert\lambda_q\rvert}{\lvert\lambda_j\rvert},$$

so directions with larger eigenvalues get shorter length scales, and $\Delta_\eta$ is a
dimensionless nugget (the given noise variances divided by the sample variance of $f$, or $\eta$
times the inactive fraction $\sum_{q>n}\lvert\lambda_q\rvert / \sum_q \lvert\lambda_q\rvert$, at
least $10^{-6}$). $\eta$ minimizes the concentrated negative log-likelihood

$$J(\eta) = \tfrac12 \log\det(R_\eta + \Delta_\eta) + \tfrac M2 \log \hat\sigma^2_\eta$$

over $\log_{10}\eta \in [-10, 1]$. Three points differ from the 2016 code. The 2016 objective
weighted the second term by $M$ instead of $M/2$. The 2016 nugget had the units of $f^2$, so the
fit changed when $f$ was rescaled; v1's fit is unchanged by $f \mapsto af + b$. And the 2016
"$R^2$" of the RBF was an in-sample quantity. The length-scale rule is the 2016 heuristic and has
no published reference.

## Fitting from a model

`ac.fit_surrogate_from_model(domain=…, model=…)` chooses design points $y_k$ on the active domain,
draws `n_mc` inputs on each fiber with the inverse map (see {doc}`geometry`), runs the model there,
and fits the surrogate to the conditional means

$$\bar f_k = \frac1{N_{MC}}\sum_j f(x_{kj}), \qquad v_k = \frac{\hat V_k}{N_{MC}},
\quad \hat V_k = \frac1{N_{MC}-1}\sum_j (f(x_{kj}) - \bar f_k)^2,$$

passing $v_k$ to the RBF as its noise variance. The defaults are $5^n$ design points, the
domain's vertices on a bounded domain, and `n_mc=50`, so $(5^n + N_v)$ points cost about
$50 \cdot 5^n$ model runs (one run at a vertex whose preimage is a single corner).

```{code-cell} python
dom2 = ac.active_domain(sub2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ac.AccuracyWarning)   # see the note below
    s_model = ac.fit_surrogate_from_model(domain=dom2, model=wing, n_design=10, n_mc=20, rng=2)
cs = s_model.samples                                      # the ac.ConditionalSamples used
print(type(s_model).__name__, cs.Y.shape, "points,", cs.f_values.size, "model runs")
print(f"LOO R^2 = {s_model.loo_r_squared:.4f}")
```

```{note}
Maximin designs on a zonotope place some points on its boundary. The fiber of a boundary point is
a single input, so the inverse map returns copies of it and warns `accuracy.degenerate_fiber`.
The conditional variance there is 0, which is correct for that point.
```

## Designs on the active domain

`ac.active_design(domain=…, n_points=N)` builds the design that `fit_surrogate_from_model` uses:

| `domain.kind` | Design |
|---|---|
| `"interval"` $[a, b]$ | $N$ equispaced interior points $a + k(b-a)/(N+1)$ |
| `"zonotope"` | maximin: maximize the smallest distance among the points and to the vertices, with the points inside the zonotope; a greedy farthest-point design when $N(N-1)/2 + N N_f > 2000$ ($N_f$ facets) |
| `"unbounded"` | tensor Gauss–Hermite nodes, $q^n \le N$ points |

```{code-cell} python
design = ac.active_design(domain=dom2, n_points=12, rng=3)
print(design.method, design.points.shape, round(design.min_distance, 3))
fig, ax = ac.plotting.plot_active_domain(dom2, design=design.points, triangulate=True)
fig
```

`ac.conditional_samples(domain=…, model=…, design=…)` runs the model on the fibers of any design
and returns the conditional means and variances, if you want to fit something else to them.

API reference: {mod}`active_subspaces.surrogates`, {mod}`active_subspaces.designs`.

## References

- P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM, 2015.
- P. G. Constantine, E. Dow, Q. Wang. Active subspace methods in theory and practice: applications
  to kriging surfaces. *SIAM J. Sci. Comput.* 36(4):A1500–A1524, 2014.
