---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Optimization

To approximate $\min_x f(x)$, minimize a surrogate $g(y)$ over the active domain, then **lift** the
minimizer $y^\ast$ back to a full input $x^\ast$ [Constantine 2015]. `ac.minimize_active` does the
first step, `ac.lift` the second, and `ac.minimize` both, optionally checking $f(x^\ast)$ with
one model run. The theory page {doc}`../theory/integration_and_optimization` covers the method.

```{code-cell} python
import warnings
import numpy as np
import active_subspaces as ac

wing = ac.test_functions.builtin_models().get("wing_weight")
space = wing.space
X = space.sample(300, rng=0)
f = ac.evaluate(wing, X=X)
sub = ac.active_subspace(grads=ac.evaluate_grads(wing, X=X), space=space)
sub2 = ac.with_dimension(sub, 2)
dom2 = ac.active_domain(sub2)
g = ac.fit_surrogate(X=X, f=f, subspace=sub2, kind="poly", degree=2)
```

## Minimizing over the active domain

`ac.minimize_active(g, domain=…)` takes a surrogate, or any callable on normalized active
coordinates `(k, n) -> (k,)` (with an optional `grad=`). The method follows the domain:

| Domain | Method | Starts |
|---|---|---|
| interval | bounded Brent | one run, no randomness |
| zonotope | SLSQP with the facet inequalities | `n_starts` (default 10), $y_0 = W_1^\top x_0$ with $x_0 \sim \rho$, so always feasible |
| unbounded (Gaussian) | BFGS | `n_starts`, $y_0 \sim N(0, I_n)$ |

```{code-cell} python
act = ac.minimize_active(g, domain=dom2, rng=1)
print(act.method, np.round(act.y_star, 4), round(act.value, 3))
print(act.run_status.count("converged"), "of", len(act.run_status), "runs converged; best run", act.best_run)
```

If some starts fail, the best of the rest is returned with a `convergence.failed_starts` warning;
only if all fail is there an error (`ac.SolverError`, `solver.failed`). On a Gaussian space the
search is over all of $\mathbb R^n$, so it can run into a region where the surrogate extrapolates.
The library warns `accuracy.extrapolation` when $\lVert y^\ast\rVert$ exceeds the radius of the
99.9% probability ball of $N(0, I_n)$; it does not constrain $y$.

## Minimize and lift

`ac.minimize` runs `minimize_active`, lifts $y^\ast$, and, with `model=`, evaluates $f(x^\ast)$
once. The lift needs the data `X`, `f` that the subspace came from.

```{code-cell} python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res = ac.minimize(g, domain=dom2, X=X, f=f, model=wing, rng=1)
print("surrogate g(y*) =", round(res.f_star, 3), "  model f(x*) =", round(res.f_model, 3))
print("smallest f in the data:", round(f.min(), 3))
print("x* (physical):", np.round(res.x_star_physical, 3))
print([w.message.code for w in caught])
```

`res.x_star` is normalized and `res.x_star_physical` physical. `res.active` holds the
`ActiveMinimizationResult` and `res.lift` the `LiftResult`. The gap between $g(y^\ast)$ and
$f(x^\ast)$ shows how much the surrogate and the lift can be trusted; the library does no local
refinement in the full space.

## The lift

Many $x$ share the active coordinates $y^\ast$. The lift chooses $z$ by minimizing a quadratic
model of $f$ along the inactive directions. It is a heuristic, the 2016 one with its defects
fixed, and has no published reference.

1. Fit $q(u) = a_0 + \gamma^\top u + \tfrac12 u^\top H u$ by least squares to the data, in the
   first $p$ eigen-coordinates $u = W_p^\top x = (y, \zeta)$, with `lift_dim` $= p$ (default
   $n+2$, or $n+1$ when $m - n \le 2$).
2. For fixed $y^\ast$, $q$ is a quadratic in $\zeta$ with Hessian $Q = H_{\zeta\zeta}$ and linear
   term $c = \gamma_\zeta + H_{\zeta y}\,y^\ast$.
3. Raise the eigenvalues $\theta_i$ of $Q$ to at least $\tau = \delta\max_i\lvert\theta_i\rvert$,
   with $\delta$ = `curvature_floor` (default $10^{-2}$), giving $Q_+$. Any clip issues
   `accuracy.curvature_clipped`.
4. Minimize $\tfrac12 z^\top \tilde H z + \tilde c^\top z$ with
   $\tilde H = \operatorname{blockdiag}(Q_+, \tau I)$ and $\tilde c = (c, 0)$: in closed form for a
   Gaussian space, and as a convex quadratic program subject to $-1 \le W_1 y^\ast + W_2 z \le 1$
   for a uniform space. The $\tfrac12\tau\lVert z_b\rVert^2$ term picks $z_b = 0$ in the
   directions the model does not see.
5. $x^\ast = W_1 y^\ast + W_2 z^\ast$.

The 2016 lift moved $z$ in the wrong direction ($+Q^{-1}c$ instead of $-Q^{-1}c$) and minimized a
quadratic without the $\tfrac12$, which was not the fitted model. `ac.lift` runs the lift on its
own, for any $y$ in the domain:

```{code-cell} python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ac.AccuracyWarning)
    L = ac.lift(act.y_star, domain=dom2, X=X, f=f)
print("p =", L.lift_dim, " clipped:", L.n_clipped, " q(x*) =", round(L.model_value, 3))
print(np.round(L.x, 3))
```

## Solvers

The quadratic programs of the lift, and the linear programs behind the inverse map, go through a
solver backend; the built-in `ac.ScipyBackend` is the default. `backend=` accepts any object that
implements `ac.SolverBackend`. A failed solve raises `ac.SolverError` with a `solver.*` code.

API reference: {mod}`active_subspaces.optimization`, {mod}`active_subspaces.solvers`.

## References

- P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM, 2015.
