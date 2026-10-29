---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# The reduced model

`ac.build_reduced_model` runs the whole pipeline in one call: it estimates the subspace, chooses
$n$, builds the active domain and fits a surrogate, and returns an `ac.ReducedModel`,

$$f(\xi) \approx g\bigl(W_1^\top A^{-1}(\xi - b)\bigr),$$

with physical inputs $\xi$. The reduced model predicts like a surrogate and offers three
operations: `average`, `probability` and `minimum`. See also the theory page
{doc}`../theory/reduced_model`.

```{code-cell} python
import warnings
import numpy as np
import active_subspaces as ac

otl = ac.test_functions.builtin_models().get("otl_circuit")   # 6 uniform inputs, output in V
```

## Building from a model

With `model=`, the library draws $M$ inputs from the input density, evaluates $f$ and
$\nabla f$ there, estimates the subspace (`"as"`, with 100 bootstrap replicates and the EVG rule
by default), and fits an RBF surrogate to conditional means on a design of $5^n$ points with
`n_mc=50` inactive draws each (see {doc}`surrogates`). The default $M$ is the 2016 heuristic
$M = \lfloor 6\,(m+1)\ln m \rfloor$ [Constantine 2015], so v1 requires $m \ge 2$.

```{code-cell} python
rm = ac.build_reduced_model(model=otl, rng=0)
print("n =", rm.n, " M =", rm.X.shape[0], " gradients from:", rm.gradient_source)
print(f"R^2 = {rm.r_squared:.4f} ({rm.r_squared_kind})")
```

A model without a gradient is differenced (`gradient_source == "finite_difference"`, step
`fd_step`). `n`, `estimator`, `dimension_rule`, `nboot`, `surrogate_kind`, `degree`,
`n_samples`, `n_design` and `n_mc` override the defaults. `subspace=` reuses a subspace you
already computed.

## Building from data

With `X`, `f` and `space` (and `grads` if you have them), nothing is evaluated. Without gradients,
the subspace comes from local linear gradients:

```{code-cell} python
X = otl.space.sample(200, rng=1)              # physical
f = ac.evaluate(otl, X=X)
rm_d = ac.build_reduced_model(space=otl.space, X=X, f=f, rng=2)
print("n =", rm_d.n, " gradients from:", rm_d.gradient_source)
print(f"R^2 = {rm_d.r_squared:.4f} ({rm_d.r_squared_kind})")
```

## Fit quality

`r_squared` is an out-of-sample coefficient of determination,

$$R^2 = 1 - \frac{\sum_i \omega_i (f_i - \hat g_i)^2}{\sum_i \omega_i (f_i - \bar f)^2}, \qquad \bar f = \sum_i \omega_i f_i ,$$

with out-of-sample predictions $\hat g_i$:

- **model path** (`"holdout"`): $\hat g_i = g(W_1^\top x_i)$ at the $M$ gradient samples, which
  the surrogate never saw, with $\omega_i = 1/M$;
- **data path** (`"loo"`): leave-one-out predictions of the surrogate, with the quadrature weights.

The 2016 values differ: its model path omitted a factor $1/M$, which inflated the error term by
$M$, and its data path reported the RBF's in-sample value, which is near 1 for any interpolant.

## Predicting

```{code-cell} python
Xt = otl.space.sample(3, rng=3)
print(np.round(ac.predict(rm, Xt), 3), np.round(ac.evaluate(otl, X=Xt), 3))
print(ac.predict_gradient(rm, Xt).shape)       # physical gradients (3, 6)
```

The plots accept a reduced model wherever they accept a surrogate, and its subspace wherever they
accept a subspace: `ac.plotting.plot_eigenvalues(rm.subspace)`,
`ac.plotting.plot_surrogate_fit(rm, X=X, f=f)`.

## Average

`rm.average(n_nodes=N)` integrates the surrogate with the rule of {doc}`integration` (no model
runs). With `model=`, it integrates $f$ itself, with inactive Monte Carlo and its standard error.

```{code-cell} python
print(f"surrogate: {rm.average(n_nodes=10).estimate:.4f}")
avg = rm.average(n_nodes=10, model=otl, rng=4)
print(f"model:     {avg.estimate:.4f} +/- {avg.inactive_std_error:.4f}")
print(f"reference: {otl.reference_results.mean:.4f}")
```

## Probability

`rm.probability(lower, upper, n_samples=N)` estimates
$p = \mathbb P\bigl[\ell \le g(W_1^\top x) \le u\bigr]$ by plain Monte Carlo with $N$ draws
(default $10^4$). `-np.inf` and `np.inf` give one-sided probabilities. `confidence_interval(c)`
gives the Wilson score interval [Wilson 1927] at level $c$ (default 0.99): with
$q = \Phi^{-1}\bigl(\tfrac{1+c}{2}\bigr)$,

$$\tilde p = \frac{\hat p + q^2/(2N)}{1 + q^2/N}, \qquad
h = \frac{q}{1+q^2/N}\sqrt{\frac{\hat p(1-\hat p)}{N} + \frac{q^2}{4N^2}}, \qquad [\tilde p - h,\ \tilde p + h].$$

```{code-cell} python
p = rm.probability(-np.inf, 4.0, rng=5)
lo, hi = p.confidence_interval(0.99)
print(f"P[V <= 4] = {p.p:.4f}, 99% interval [{lo:.4f}, {hi:.4f}]")
```

```{note}
The probability is that of the **surrogate**. The interval covers only the Monte Carlo error.
Averaging over the inactive variables smooths $f$, so $\mathbb P[\ell \le g \le u]$ can differ
from $\mathbb P[\ell \le f \le u]$ by more than the interval suggests.
```

## Minimum

`rm.minimum()` minimizes the surrogate over the active domain and lifts the minimizer with the
stored samples (see {doc}`optimization`). With `model=`, it also evaluates $f(x^\ast)$.

```{code-cell} python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ac.AccuracyWarning)     # accuracy.curvature_clipped
    mn = rm.minimum(model=otl, rng=6)
print(f"g(y*) = {mn.f_star:.4f}, f(x*) = {mn.f_model:.4f}")
print("x* (physical):", np.round(mn.x_star_physical, 3))
```

The operations are also functions of the module: `ac.reduced_model.average(rm, ...)`,
`ac.reduced_model.probability(rm, ...)`, `ac.reduced_model.minimum(rm, ...)`.

API reference: {mod}`active_subspaces.reduced_model`.

## References

- P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM, 2015.
- E. B. Wilson. Probable inference, the law of succession, and statistical inference. *J. Amer.
  Statist. Assoc.* 22(158):209–212, 1927.
