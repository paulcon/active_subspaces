---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Migrating from the 2016 package

## Who needs this page

`active_subspaces` v1.0 is not backward compatible with the 2016 package (v0.1, Python 2.7). There
is no compatibility layer. To keep running 2016 code unchanged, pin the git tags `legacy-master` or
`legacy-dev` in a Python 2.7 environment. v1 needs Python 3.11 or later. The tag `legacy-py3` holds
mechanical Python 3 ports of both branches under `legacy/`; they were used to capture the
regression baselines and are not maintained.

In the code below, `ac` is `import active_subspaces as ac`. A 2016 name such as `ac.utils` or
`ac.domains` raises an `AttributeError` that names its replacement and links to this page.

## Five ideas that change every call

1. **Functions return frozen results.** Nothing is computed by mutating an object
   (`Subspaces().compute(...)` becomes `ac.active_subspace(...)`). Results are immutable, carry
   their provenance, and save and load with `result.save` and `ac.load`.
2. **Pass physical data with an input space.** You give `X` and gradients in the units of your
   model, together with `ac.Uniform(lower, upper)` or `ac.Gaussian(mean, cov)`. The library
   normalizes them. Do not normalize `X` or scale gradients by hand (see
   [Coordinates](#coordinates)).
3. **NumPy shapes and keyword arrays.** `f` is `(M,)` (an `(M, 1)` input is accepted), eigenvalues
   are `(m,)`, and arrays are passed by keyword: `ac.active_subspace(grads=G, space=space)`.
4. **Randomness goes through `rng`.** Every stochastic function takes `rng=` (an integer seed, a
   `SeedSequence` or a `Generator`). `np.random.seed` has no effect on the library.
5. **Nothing prints, shows or saves.** Plots return `(fig, ax)`. Problems are reported with
   warnings and with exceptions that carry a stable `code`.

(coordinates)=
## Coordinates

v1 normalized coordinates are the 2016 ones, so normalized results compare directly.

- **Uniform.** 2016 `BoundedNormalizer(lb, ub)` computed `x = 2 (xi - lb) / (ub - lb) - 1`.
  `ac.Uniform(lb, ub).to_normalized(X)` is the same map.
- **Gaussian.** 2016 `UnboundedNormalizer(mu, C)` used the Cholesky factor of `C`;
  `ac.Gaussian(mu, C)` uses the same factor.
- **Gradients.** v1 applies the chain rule itself: for a uniform space it multiplies physical
  gradients by `(ub - lb) / 2`, exactly the tutorials' `df *= (ub - lb)/2.`. Gradients that you
  already scaled by hand must be passed with `ac.Uniform.reference(m)` (or
  `ac.Gaussian.reference(m)`), or they are scaled twice. The same goes for any array already in
  normalized coordinates.
- **Physical directions.** `result.physical_directions()` is new: the directions of a ridge in the
  physical inputs.

## Quick reference

### Modules

| 2016 module | v1 | Notes |
|---|---|---|
| `active_subspaces` (`import utils, domains, …`) | `ac` (flat top level); `ac.plotting` and `ac.test_functions` load on first use | `import active_subspaces.experimental` explicitly |
| `subspaces` | `ac.subspaces` | same name, new contents |
| `gradients` | `ac.gradients` | same name, new signatures |
| `domains` | `ac.geometry` | |
| `response_surfaces`, `utils.response_surfaces` | `ac.surrogates` | |
| `integrals` | `ac.integration` | |
| `optimizers` | `ac.optimization` | |
| `utils` | removed; its modules are mapped below | |
| `utils.designs` | `ac.designs` | |
| `utils.quadrature` | private | use `ac.active_quadrature_rule` |
| `utils.qp_solver` | `ac.solvers` | Gurobi dropped |
| `utils.misc` | `ac.spaces` (normalizers) | |
| `utils.simrunners` | `ac.models` | |
| `utils.plotters` | `ac.plotting` | |
| *(dev)* `base` | `ac.reduced_model` | |
| *(dev)* `sdr` | merged into `ac.subspaces` | |
| *(dev)* `utils.est_noise` | dropped | ECNoise is not in v1; use Moré and Wild's ECNoise (Moré and Wild, 2011) |
| *(dev)* `gradfree` | `active_subspaces.experimental` | warns once on import |
| *(dev)* `utils.celery_runner` | removed | unsafe; use `executor=` |
| `tutorials/test_functions/*` | `ac.test_functions` | installed with the package |

### `subspaces`

| 2016 | v1 | Notes |
|---|---|---|
| `Subspaces()`, `.compute(X, f, df, weights, sstype, ptype, nboot)` | `ac.active_subspace(space=, X=, f=, grads=df, quad_weights=weights, estimator=, dimension_rule=, nboot=, rng=)` → `ac.SubspaceResult` | `n=` fixes the dimension |
| `.partition(n)` | `ac.with_dimension(result, n)` | returns a new result |
| `.eigenvals` `(m,1)`, `.eigenvecs` | `result.eigenvalues` `(m,)`, `result.eigenvectors` (`result.W`) | new sign rule |
| `.W1`, `.W2` | `result.W1`, `result.W2` | |
| `.W_br` *(dev)* | `ac.active_subspace(..., nboot=B, bootstrap_vectors=k)` → `result.bootstrap_eigenvectors` `(B, m, k)` | sign-aligned replicates of the leading $k$ eigenvectors; `ac.plotting.plot_eigenvectors(sub, interval="percentile" \| "minmax", level=0.95)` draws the bands |
| `.e_br` | `ac.bootstrap_interval(abs(result.bootstrap_eigenvalues), kind="minmax")` | `(2, m)`; default is a 95% percentile interval |
| `.sub_br` | `ac.bootstrap_interval(result.bootstrap_distances, kind="minmax")`, `ac.mean_distance(result)` | |
| `sstype='AS'`, …, `'OPG'` | `estimator="as"`, …, `"opg"`; or `ac.subspaces.SIR(n_slices=…)`, `ac.subspaces.SAVE(n_slices=…)`, `ac.subspaces.OPG(p=…)` | names in `ac.subspaces.ESTIMATORS` |
| `ptype='EVG'`, `'RS'`, `'LI'` | `dimension_rule="evg"`, `"rs"`, `"li"`; `ac.subspaces.EVG(gap="log")` | |
| `active_subspace(df, weights)`, `ols_subspace`, `qphd_subspace`, `opg_subspace`, … | `ac.active_subspace(..., estimator=...)` | same name, keyword-only, returns a result |
| `eig_partition`, `errbnd_partition`, `ladle_partition` | `ac.dimension_criterion(result, rule="evg" \| "rs" \| "li")` | ladle returns $\hat k$, not $\hat k+1$ |
| `sorted_eigh(C)` | removed | `numpy.linalg.eigh` |
| — | `ac.subspace_from_basis(basis, space=)` | wraps a basis computed elsewhere |
| a hand-accumulated $C$ (no 2016 equivalent) | `ac.subspace_from_matrix(C, space=, n=)` | new: a symmetric $C$ in normalized coordinates becomes a result with eigenvalues (`estimator="matrix"`) |
| activity scores coded by hand | `ac.activity_scores(result, n=)` | new: $\sum_{j\le n}\lvert\lambda_j\rvert w_{ij}^2$ in normalized coordinates |
| hand-flipped eigenvector signs | `ac.orient(result, by="reference" \| "covariance" \| "mean_gradient", …)` | new: returns a new result; plots and `physical_directions()` follow its signs |
| `sstype='MAVE'` *(dev)* | removed | the 2016 stub returned `None` |

### `gradients`

| 2016 | v1 | Notes |
|---|---|---|
| `local_linear_gradients(X, f, p=None, weights=None)` | `ac.local_linear_gradients(X=, f=, space=, p=None)` → `ac.GradientResult` | one gradient per row of `X`; pass it as `grads=` |
| `finite_difference_gradients(X, fun, h=1e-6)` | `ac.finite_difference_gradients(model, X=, space=, h=1e-6, scheme="forward")` | relative step in normalized coordinates; never steps outside the support |

### `domains` → `geometry`

| 2016 | v1 | Notes |
|---|---|---|
| `BoundedActiveVariableDomain(ss)`, `UnboundedActiveVariableDomain(ss)` | `ac.active_domain(subspace)` → `ac.ActiveDomain` | `kind` is `"interval"`, `"zonotope"` or `"unbounded"` |
| `.compute_boundary()` | `ac.active_domain(ac.with_dimension(sub, n))` | |
| `.vertY`, `.vertX` | `domain.vertices`, `domain.vertex_preimages` | exact vertices |
| `.convhull`, `.constraints` ($Ay-b\ge0$) | `domain.facet_normals`, `domain.facet_offsets` ($Ny\le c$) | sign reversed |
| `BoundedActiveVariableMap(avdom)`; `.forward(X)` | `ac.forward_map(subspace, X=X)` → `ac.ActiveCoordinates` | unpacks as `Y, Z`; `X` physical |
| `.inverse(Y, N=1)` | `ac.inverse_map(subspace, Y=Y, n_draws=N, rng=)` → `ac.InverseMapResult` | `.X`, `.X_physical`, `.index` |
| `nzv(m, n)` | `ac.zonotope_vertex_count(m, n)` | exact `int` |
| `zonotope_vertices`, `interval_endpoints` | `ac.active_domain(sub).vertices` | |
| `sample_z`, `hit_and_run_z`, `rejection_sampling_z` | `ac.inverse_map(...)` | samplers are private |
| `unique_rows`, `random_walk_z` | removed | `numpy.unique(S, axis=0)` |

### `response_surfaces` → `surrogates`

| 2016 | v1 | Notes |
|---|---|---|
| `ActiveSubspaceResponseSurface(avmap)`; `.train_with_data(X, f, v)` | `ac.fit_surrogate(X=X, f=f, subspace=sub, kind="rbf", noise_variance=v)` | `X` physical |
| `.train_with_interface(fun, N, NMC)` | `ac.fit_surrogate_from_model(domain=dom, model=model, n_design=N, n_mc=NMC, rng=)` | |
| `.predict(X, compgrad)` | `ac.predict(s, X)`, `ac.predict_gradient(s, X)` | `(M,)` and `(M, m)`, physical |
| `.predict_av(Y)`, `.gradient_av(Y)` | `s.evaluate(Y)`, `s.gradient(Y)` | normalized active coordinates |
| `PolynomialApproximation(N).train(X, f, weights)` | `ac.fit_surrogate(X=, f=, space=, subspace=None, kind="poly", degree=N, sample_weights=weights)` → `ac.PolynomialSurrogate` | weighted fit corrected |
| `RadialBasisApproximation(N).train(X, f, v)` | `ac.fit_surrogate(..., kind="rbf", degree=N, noise_variance=v)` → `ac.RBFSurrogate` | new likelihood |
| `ResponseSurface` (base class) | `ac.Surrogate` (protocol) | |
| `.Rsqr` | `s.loo_r_squared` | leave-one-out, so lower |
| `.poly_weights`, `.g`, `.H` | `s.coefficients`, `ac.quadratic_terms(s)` | |
| `av_design(avmap, N, NMC)` | `ac.active_design(domain=, n_points=N)`, then `ac.inverse_map`; or `ac.conditional_samples(domain=, model=, n_design=N, n_mc=NMC)` | |

### `integrals` → `integration`

| 2016 | v1 | Notes |
|---|---|---|
| `integrate(fun, avmap, N, NMC)` → `(mu, lb, ub)` | `ac.integrate(model, domain=dom, n_nodes=N, n_mc=NMC, rng=)` → `ac.IntegrationResult` | `.estimate`, `.inactive_std_error` |
| `av_integrate(avfun, avmap, N)` | `ac.integrate_surrogate(surrogate, domain=dom, n_nodes=N)` | |
| `quadrature_rule(avmap, N, NMC)` | `ac.integration_design(dom, n_nodes=N, n_mc=NMC, rng=)`, then `ac.estimate_integral(design, f=)` | |
| `av_quadrature_rule`, `interval_quadrature_rule`, `zonotope_quadrature_rule` | `ac.active_quadrature_rule(dom, n_nodes=N)` → `ac.ActiveQuadratureRule` | Gauss rule on intervals |

### `optimizers` → `optimization`

| 2016 | v1 | Notes |
|---|---|---|
| `minimize(asrs, X, f)` → `(xstar, fstar)` | `ac.minimize(surrogate, domain=dom, X=X, f=f, model=None, rng=)` → `ac.MinimizationResult` | lift corrected |
| `av_minimize(avfun, avdom)` → `(ystar, fstar)` | `ac.minimize_active(g, domain=dom, rng=)` → `ac.ActiveMinimizationResult` | `.y_star`, `.value` |
| `interval_minimize`, `zonotope_minimize`, `unbounded_minimize` | `ac.minimize_active` | |
| `MinVariableMap` and subclasses | `ac.lift(y, domain=dom, X=X, f=f)` → `ac.LiftResult` | |

### `utils.designs` → `designs`

| 2016 | v1 | Notes |
|---|---|---|
| `interval_design(a, b, N)`, `maximin_design(vert, N)`, `gauss_hermite_design(N)` | `ac.active_design(domain=dom, n_points=N, rng=)` → `ac.ActiveDesign` | method follows `domain.kind`: `"interval"`, `"maximin_slsqp"` (zonotope) or `"gauss_hermite"` |

### `utils.qp_solver` → `solvers`

| 2016 | v1 | Notes |
|---|---|---|
| `QPSolver(solver='GUROBI')` | `ac.ScipyBackend()`, `ac.get_backend(None)`, or any `ac.SolverBackend` | |
| `.linear_program_eq(c, A, b, lb, ub)` | `ac.solve_lp(c, A_eq=A, b_eq=b, bounds=np.column_stack([lb, ub]))` → `ac.SolverOutput` | |
| `.linear_program_ineq(c, A, b)` ($Ax\ge b$) | `ac.solve_lp(c, A_ub=-A, b_ub=-b)` | v1 uses $A_{ub}x\le b_{ub}$ |
| `.quadratic_program_bnd(c, Q, lb, ub)` | `ac.solve_qp(2*Q, c, bounds=np.column_stack([lb, ub]))` | v1 minimizes $\tfrac12x^\top Hx+g^\top x$ |
| `.quadratic_program_ineq(c, Q, A, b)` | `ac.solve_qp(2*Q, c, A_ub=-A, b_ub=-b)` | |
| failure writes `bad_*.npz` | `ac.SolverError` | no files |

### `utils.misc`, `utils.simrunners`

| 2016 | v1 | Notes |
|---|---|---|
| `Normalizer`, `BoundedNormalizer(lb, ub)`, `UnboundedNormalizer(mu, C)` | `ac.InputSpace`, `ac.Uniform(lb, ub)`, `ac.Gaussian(mu, C)` | `.to_normalized`, `.from_normalized` |
| `process_inputs`, `atleast_2d*` | removed | 1-D `f` accepted |
| `conditional_expectations(f, ind)` | `ac.ConditionalSamples` `.means`, `.variances` | ddof 1 |
| `SimulationRunner(fun).run(X)` | `ac.evaluate(model, X=, executor=None)` | wrap a per-point function with `ac.pointwise(fun)` |
| `SimulationGradientRunner(dfun).run(X)` | `ac.evaluate_grads(model, X=)`, `ac.evaluate_with_grads(model, X=)` | |
| `backend=`, `num_cores=` | `executor=` (any `concurrent.futures.Executor`), `batch_size=` | |
| — | `ac.Model`, `ac.ModelRegistry`, `ac.test_functions.builtin_models()` | |

### `utils.plotters` → `plotting`

| 2016 | v1 | Notes |
|---|---|---|
| `plot_opts`, `show_plot` | removed; `fig.savefig(...)` or `ac.plotting.figure_to_bytes(fig)` | |
| `eigenvalues(e, e_br)` | `ac.plotting.plot_eigenvalues(sub, interval=None)` | `interval="minmax"` draws the 2016 band |
| `subspace_errors(sub_br)` | `ac.plotting.plot_subspace_errors(sub)` | |
| `eigenvectors(W, in_labels=)` | `ac.plotting.plot_eigenvectors(sub, labels=None)` | labels from `space.names` |
| `sufficient_summary(y, f)` | `ac.plotting.plot_sufficient_summary(sub, X=X, f=f)` | takes physical `X`, not `y` |
| `zonotope_2d_plot(vertices, design, y, f)` | `ac.plotting.plot_active_domain(dom, design=, points=, values=)` | |
| — | `ac.plotting.plot_surrogate_fit`, `ac.plotting.plot_dimension_criterion`, `ac.plotting.PLOTS` | new |

### *(dev)* `base` → `reduced_model`, and `sdr`

| 2016 | v1 | Notes |
|---|---|---|
| `ActiveSubspaceReducedModel(m, bounded_inputs)` | `ac.build_reduced_model(...)` → `ac.ReducedModel` | `space=` replaces `m`, `bounded_inputs` |
| `.build_from_data(X, f, df, avdim)` | `ac.build_reduced_model(space=, X=X, f=f, grads=df, n=avdim)` | |
| `.build_from_interface(fun, dfun, avdim)` | `ac.build_reduced_model(model=model, n=avdim, rng=)` | |
| `.predict(X, compgrad)` | `ac.predict(rm, X)`, `ac.predict_gradient(rm, X)` | |
| `.average(N)` | `ac.reduced_model.average(rm, n_nodes=N, model=None)` or `rm.average(...)` | integrates the surrogate unless `model=` |
| `.probability(lb, ub, M)` | `ac.reduced_model.probability(rm, lb, ub, n_samples=M)` → `ac.ProbabilityResult` | Wilson interval |
| `.minimum()` → `(fstar, xstar)` | `ac.reduced_model.minimum(rm, model=None)` → `ac.MinimizationResult` | |
| `.diagnostics()` | `ac.plotting.plot_eigenvalues(rm.subspace)` and the other plots | |
| `.Rsqr` | `rm.r_squared`, `rm.r_squared_kind` | holdout $R^2$ |
| `linear_gradient_check(X, f)` | `ac.active_subspace(X=, f=, space=, estimator="ols", nboot=, rng=)` | |
| `quadratic_model_check(X, f, gamma)` | `ac.active_subspace(X=, f=, space=, estimator="qphd")` | `gamma` comes from the space |

### *(dev)* `gradfree` → `experimental`

| 2016 | v1 | Notes |
|---|---|---|
| `simple_projection(dfm, E)` | `ac.experimental.subspace_from_sketches(sketches=dfm, directions=E, space=, method="projection")` | after `import active_subspaces.experimental`; unpacks as `W, V, s` |
| `alternating_minimization(dfm, E, r)` | `ac.experimental.subspace_from_sketches(..., method="alternating", rank=r)` | |
| `gradient_measurements(sr, X, E)` | `ac.experimental.sketched_active_subspace(model, space=, k=, X=X, rng=)` | |

### Tutorial test functions

| 2016 (normalized inputs) | v1 (physical inputs) | Notes |
|---|---|---|
| `borehole`, `borehole_grad` | `ac.test_functions.borehole`, `ac.test_functions.BOREHOLE` | uniform $r_w$, $r$: see E4 |
| `circuit`, `circuit_grad` | `ac.test_functions.otl_circuit`, `ac.test_functions.OTL_CIRCUIT` | |
| `piston`, `piston_grad` | `ac.test_functions.piston`, `ac.test_functions.PISTON` | |
| `robot`, `robot_grad` | `ac.test_functions.robot_arm`, `ac.test_functions.ROBOT_ARM` | |
| `wing`, `wing_grad` | `ac.test_functions.wing_weight`, `ac.test_functions.WING_WEIGHT` | gradient in degrees of sweep |

## Parameters and return values

Parameters that kept their name but changed meaning:

| 2016 | v1 | What changes |
|---|---|---|
| `X`, `df` | `X`, `grads` | physical, with a space |
| `weights` (subspaces) | `quad_weights` | must sum to 1 |
| `weights` (polynomial fit) | `sample_weights` | rows scaled by $\sqrt w$ |
| `weights` (`local_linear_gradients`) | removed | it had no effect |
| `p` (`local_linear_gradients`) | `p` | counts distinct locations and includes the center |
| `h` (`finite_difference_gradients`) | `h` | relative, in normalized coordinates |
| `N` | `degree`, `n_nodes`, `n_points`, `n_design`, `n_draws` | one name per meaning |
| `NMC`, `NX`, `M` | `n_mc`, `n_weight_samples`, `n_samples` | |
| `sstype`, `ptype`, `avdim` | `estimator`, `dimension_rule`, `n` | lower-case names |
| `compgrad` | removed | call `ac.predict_gradient` |
| `in_labels`, `out_label`, `opts` | `space.names`, `qoi_label`, `figsize`, `ax` | |

Tuple returns become results with named fields. Only `ac.forward_map` (`Y, Z`) and
`subspace_from_sketches` (`W, V, s`) still unpack.

| 2016 return | v1 field |
|---|---|
| `integrate`, `average` → `(mu, lb, ub)` | `.estimate`, `.inactive_std_error` (no fixed 1.96 interval) |
| `inverse` → `(X, ind)` | `.X` (normalized), `.X_physical`, `.index` |
| `minimize` → `(xstar, fstar)`; `minimum()` → `(fstar, xstar)` | `.x_star`, `.x_star_physical`, `.f_star` |
| `av_minimize` → `(ystar, fstar)` | `.y_star`, `.value` |
| `probability` → `(p, plb, pub)` | `.p`, `.confidence_interval(0.99)` |

## What changes in your numbers

- **Identical.** The normalization maps; active-subspace eigenpairs from the same gradients and
  weights (equal to rounding, up to the sign rule: v1 takes an SVD of $\mathrm{diag}(\sqrt\omega)G$
  through a QR factor when $M>m$, and the eigenvalues agree with the direct SVD to $10^{-13}$
  relative); the default dimension rule (EVG with the linear gap); interval and
  Gauss–Hermite design points.
- **Same quantity, different convention.**
  - Each eigenvector's largest-magnitude component is positive (2016: the first), so a column of
    `W`, `y` and the sufficient summary may be mirrored. `ac.orient` sets any other convention.
  - OLS: $w_1=\hat u/\lVert\hat u\rVert$, the signed slope of the published method, so $f$
    increases along $w_1$ (the largest-magnitude rule is not applied to it). An analysis that
    re-signed OLS by hand no longer needs to. A result saved with the development builds keeps the
    sign it was saved with.
  - `physical_directions()` keeps the signs of `W1`: $q_j^\top A^{-\top}w_j>0$. The development
    builds re-applied the largest-magnitude rule to them, so an oriented `W1` and
    `physical_directions()` could disagree.
  - ASX, NASX and pHd eigenvalues keep their sign.
  - Bootstrap bands are 95% percentile intervals, narrower than the 2016 min/max band. With
    `bootstrap_vectors=k`, `bootstrap_eigenvectors` `(B, m, k)` holds the sign-aligned replicates of
    the leading $k$ eigenvectors, and `plot_eigenvectors(interval=…)` draws their bands (`nboot=100`,
    $m=1000$, $k=1$: 0.8 MB).
  - `quad_weights` must sum to 1.
  - Shapes are `(m,)` and `(M,)`, not `(m, 1)` and `(M, 1)`.
  - Fit quality is a leave-one-out or holdout $R^2$, lower than the 2016 in-sample value.
- **Same quantity, different estimate.** Every stochastic output (bootstrap replicates, designs,
  inverse-map draws, optimizer starts, quadrature weights) differs from 2016 at any seed. Local
  linear gradients use the right neighbors [C-02]. Interval quadrature is a Gauss rule with exact
  weights. Zonotope vertices are exact. Inactive standard errors use ddof 1. The default `n_mc` is 50
  (2016: `NMC=10`).
- **Corrected bugs**, not reproducible in v1: weighted polynomial fits [C-04], probability bounds
  [C-06], reduced-model $R^2$ [C-07], the lift in `minimize` [C-16], the ladle dimension [C-17], and
  the units dependence of the RBF nugget [C-18].
- **Method choices ruled by the author.** SAVE slice weights (eigenvalues about $1/S$ of 2016), the
  pHd scaling ($\sqrt3$ times 2016 for uniform inputs), whitening from the declared space for SIR,
  pHd and SAVE, the RBF likelihood, the lift curvature floor, and the uniform borehole variant.
  Three of these also correct a 2016 form that contradicted the published method, so the 2016
  value is not reproducible: SAVE now weights slice $h$ by its probability,
  $C=\sum_h\hat p_h(I-\widehat{\mathrm{Cov}}(z\mid h))^2$ (Cook and Weisberg, 1991; 2016 used
  unit weights); pHd is whitened with $\Sigma^{-1}$, not $\gamma^{-1/2}$ (Li, 1992); and the RBF
  hyperparameters maximize the concentrated likelihood
  $-\tfrac12\log\lvert K\rvert-\tfrac M2\log\hat\sigma^2$ (2016 weighted the two terms
  differently by a factor of 2).

## Reproducing 2016 numbers

- **With settings:**
  - `ac.bootstrap_interval(..., kind="minmax")`, `ac.plotting.plot_eigenvalues(sub, interval="minmax")` and `ac.plotting.plot_eigenvectors(sub, interval="minmax")` give the 2016 bands.
  - `dimension_rule="evg"` (the default, linear gap) gives the 2016 EVG choice.
  - For unnormalized 2016 weights, multiply the eigenvalues by $\sum_i w_i$.
  - Multiply SAVE eigenvalues by the number of slices.
  - Multiply pHd eigenvalues by $\sqrt{1/3}$ for uniform inputs.
  - Negate an eigenvector whose first component is negative to restore the 2016 sign.
- **Different estimate.** Random draws are statistically equivalent but never bitwise equal; no
  seed maps the 2016 global stream onto v1's `rng`.
- **Not reproducible.** Where the 2016 value was wrong (the corrected bugs above), use the
  `legacy-*` tags.

No v1 function takes a "legacy" or "compat" flag.

## Worked examples

The "before" code is abridged 2016 tutorial code (Python 2), shown as text. The "after" code runs
with fixed seeds. Its numbers are not compared with 2016; the previous sections explain why they
differ.

### E1. Values, gradients and a 1-D surrogate (`tutorials/basic.ipynb`)

Before:

```python
ss = ac.subspaces.Subspaces()
ss.compute(X=XX, f=f, nboot=100, sstype='OLS')         # XX normalized by hand
ac.utils.plotters.eigenvalues(ss.eigenvals, ss.e_br)
ac.utils.plotters.sufficient_summary(XX.dot(ss.W1), f)
df = wing_grad(x)
df *= (ub - lb)/2.
df[:,3] *= np.pi/180.
ss.compute(df=df, nboot=100)
RS = ac.utils.response_surfaces.PolynomialApproximation(2)
RS.train(XX.dot(ss.W1), f)
print 'The R^2 value of the response surface is {:.4f}'.format(RS.Rsqr)
avdom = ac.domains.BoundedActiveVariableDomain(ss)
avmap = ac.domains.BoundedActiveVariableMap(avdom)
int_I = ac.integrals.av_integrate(lambda x: RS.predict(x)[0], avmap, 1000)
ystar, fstar = ac.optimizers.av_minimize(lambda x: RS.predict(x.reshape(1,1))[0], avdom)
```

After (the OLS and QPHD calls issue the expected `ValidityWarning`: on a uniform space these
estimators are outside their assumptions):

```{code-cell} python3
import numpy as np
import active_subspaces as ac

wing = ac.test_functions.WING_WEIGHT       # Model; Uniform space with the tutorial's bounds
space = wing.space
rng = np.random.default_rng(2016)
X = space.sample(1000, rng=rng)            # physical inputs: no hand normalization
f = ac.evaluate(wing, X=X)                 # (1000,)

ols = ac.active_subspace(X=X, f=f, space=space, estimator="ols", nboot=100, rng=rng)
ac.plotting.plot_eigenvalues(ols)          # percentile band; interval="minmax" draws the 2016 band
ac.plotting.plot_subspace_errors(ols)
ac.plotting.plot_sufficient_summary(ols, X=X, f=f)
qphd = ac.active_subspace(X=X, f=f, space=space, estimator="qphd", nboot=100, rng=rng)

llg = ac.local_linear_gradients(X=X, f=f, space=space)   # one gradient per row of X
ll = ac.active_subspace(grads=llg, nboot=100, rng=rng)   # the space comes from llg

G = ac.evaluate_grads(wing, X=X)           # physical gradients, sweep in degrees
sub = ac.active_subspace(grads=G, space=space, nboot=100, rng=rng)   # no (ub-lb)/2, no pi/180

poly = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)
print(f"leave-one-out R^2 = {poly.loo_r_squared:.4f}")
ac.plotting.plot_surrogate_fit(poly, X=X, f=f)

dom = ac.active_domain(sub)
mean = ac.integrate_surrogate(poly, domain=dom, n_nodes=10)   # Gauss rule, exact for a quadratic
opt = ac.minimize_active(poly, domain=dom, rng=rng)
print(mean.estimate, opt.y_star, opt.value)

assert f.shape == (1000,) and llg.grads.shape == (1000, 10) and sub.eigenvalues.shape == (10,)
assert isinstance(poly, ac.PolynomialSurrogate) and poly.loo_r_squared <= 1
assert opt.y_star.shape == (sub.n,) and ac.in_active_domain(dom, Y=opt.y_star[None, :])[0]
```

The 2016 tutorial swapped the names `lb` and `ub`, so its `XX` was $-x$; the v1 sign rule makes
that irrelevant. 2016 `av_integrate` needed 1000 nodes because its quadrature weights had a Monte
Carlo error floor; v1 uses a Gauss rule.

### E2. The low-level pipeline with a function handle (`tutorials/interface_bounded.ipynb`, dev)

Before:

```python
dfX0 = asub.utils.simrunners.SimulationGradientRunner(quad_dfun).run(X0)
ss = asub.subspaces.Subspaces()
ss.compute(df=dfX0, nboot=100)
ss.partition(1)
avdom = asub.domains.BoundedActiveVariableDomain(ss)
avmap = asub.domains.BoundedActiveVariableMap(avdom)
fX1 = asub.utils.simrunners.SimulationRunner(quad_fun).run(X1)
Y1 = avmap.forward(X1)[0]
pr = asub.utils.response_surfaces.PolynomialApproximation(N=2)
pr.train(Y1, fX1)
avavg = asub.integrals.integrate(quad_fun, avmap, 50, NMC=10)        # (mu, lb, ub)
asrs = asub.response_surfaces.ActiveSubspaceResponseSurface(avmap)
asrs.train_with_interface(quad_fun, 21)
fx, dfx = asrs.predict(x, compgrad=True)
avmin = asub.optimizers.minimize(asrs, X1, fX1)                      # (xstar, fstar)
ss.partition(2); avdom.compute_boundary()
```

After:

```{code-cell} python3
A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
              [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
              [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])

def quad_fun(X):            # batch: (k, 3) -> (k,)
    return 0.5 * np.einsum("ij,jk,ik->i", X, A, X)

def quad_dfun(X):           # (k, 3) -> (k, 3)
    return X @ A

space = ac.Uniform.reference(3)                 # the tutorial's inputs already live in [-1, 1]^3
model = ac.Model(space=space, value=quad_fun, grad=quad_dfun)
rng = np.random.default_rng(0)

X0 = space.sample(20, rng=rng)
sub = ac.active_subspace(grads=ac.evaluate_grads(model, X=X0), space=space, nboot=100, rng=rng)
sub1 = ac.with_dimension(sub, 1)
dom = ac.active_domain(sub1)

X1 = space.sample(20, rng=rng)
fX1 = ac.evaluate(model, X=X1)
Y1, Z1 = ac.forward_map(sub1, X=X1)
pr = ac.fit_surrogate(X=X1, f=fX1, subspace=sub1, kind="poly", degree=2)

avg = ac.integrate(model, domain=dom, n_nodes=50, rng=rng)    # n_mc defaults to 50
print(avg.estimate, avg.inactive_std_error)     # no fixed 1.96 interval

asrs = ac.fit_surrogate_from_model(domain=dom, model=model, n_design=21, rng=rng)
x = space.sample(1, rng=rng)
fx, dfx = ac.predict(asrs, x), ac.predict_gradient(asrs, x)
mres = ac.minimize(asrs, domain=dom, X=X1, f=fX1, model=model, rng=rng)
print(mres.f_star, mres.f_model, mres.x_star_physical)

sub2 = ac.with_dimension(sub, 2)                # a new result; rebuild what depends on it
ac.plotting.plot_sufficient_summary(sub2, X=X1, f=fX1)

assert Y1.shape == (20, 1) and Z1.shape == (20, 2) and fx.shape == (1,) and dfx.shape == (1, 3)
assert avg.inactive_std_error >= 0 and sub2.n == 2
assert np.all(np.abs(mres.x_star_physical) <= 1 + 1e-9)
```

A per-point 2016 function can be kept:
`ac.Model(space=space, value=ac.pointwise(fun), grad=ac.pointwise(lambda x: dfun(x).ravel(), output="grad"))`.

### E3. The reduced model (`interface_bounded.ipynb`, "quick start", dev)

Before:

```python
model = asub.ActiveSubspaceReducedModel(3, True)
model.build_from_interface(quad_fun, dfun=quad_dfun, avdim=1)
fx, dfx = model.predict(x, compgrad=True)
avavg = model.average(20)            # (mu, lb, ub)
avmin = model.minimum()              # (fstar, xstar)
```

After (with `model`, `space` and `x` from E2):

```{code-cell} python3
rm = ac.build_reduced_model(model=model, n=1, rng=1)
fx, dfx = ac.predict(rm, x), ac.predict_gradient(rm, x)
avg = rm.average(n_nodes=20, model=model, rng=2)          # integrates f, as 2016 did when it had `fun`
mn = rm.minimum(model=model, rng=3)
print(avg.estimate, avg.inactive_std_error, mn.f_star, mn.x_star_physical)
p = rm.probability(-np.inf, 0.1, rng=4)
lo, hi = p.confidence_interval(0.99)                      # Wilson interval; bounds ordered
print(p.p, (lo, hi))

assert isinstance(rm, ac.ReducedModel) and rm.n == 1 and fx.shape == (1,)
assert 0 <= p.p <= 1 and lo <= p.p <= hi
```

Without `model=`, `rm.average` integrates the surrogate. `rm.r_squared` is a holdout $R^2$.

### E4. The 2016 borehole, a mixed density (`tutorials/test_functions/borehole`)

The 2016 borehole had a Gaussian $r_w$ and a log-normal $r$; v1 has no input space for that mix.

Before:

```python
rw = np.random.normal(.1, .0161812, (M, 1)); r = np.exp(np.random.normal(7.71, 1.0056, (M, 1)))
# ... six uniform inputs, normalized with BoundedNormalizer, rw and r standardized by hand
f = borehole(XX); df = borehole_grad(XX)     # gradients with respect to the normalized inputs
ss = ac.subspaces.Subspaces(); ss.compute(df=df, nboot=500)
ss.partition(2)
ac.utils.plotters.eigenvectors(ss.W1, in_labels=in_labels)
```

After, (a) reproducing the 2016 analysis by mapping the inputs by hand:

```{code-cell} python3
tf = ac.test_functions
lo, hi = tf.BOREHOLE.space.lower, tf.BOREHOLE.space.upper   # columns 3..8 are the 2016 bounds
rng = np.random.default_rng(2016)
x = np.hstack([rng.standard_normal((1000, 2)), rng.uniform(-1, 1, (1000, 6))])   # 2016 normalized sample
xi = np.empty_like(x)
xi[:, 0] = 0.0161812 * x[:, 0] + 0.1
xi[:, 1] = np.exp(1.0056 * x[:, 1] + 7.71)
xi[:, 2:] = lo[2:] + 0.5 * (hi[2:] - lo[2:]) * (x[:, 2:] + 1)
f = tf.borehole(xi)                              # no support check, so r_w and r may leave the box
g = tf.borehole_grad(xi)                         # physical gradients; chain rule to 2016 coordinates:
g[:, 0] *= 0.0161812
g[:, 1] *= 1.0056 * xi[:, 1]
g[:, 2:] *= 0.5 * (hi[2:] - lo[2:])
ref = ac.Gaussian(np.zeros(8), np.ones(8), names=tf.BOREHOLE.space.names)   # identity map
sub = ac.with_dimension(ac.active_subspace(grads=g, space=ref, nboot=500, rng=rng), 2)
ac.plotting.plot_eigenvectors(sub)
ac.plotting.plot_sufficient_summary(sub, X=x, f=f)

assert sub.n == 2 and sub.W1.shape == (8, 2) and np.all(np.diff(sub.eigenvalues) <= 0)
```

The identity space passes the 2016 normalized gradients through unchanged, so the eigenpairs are
those of 2016 up to sampling. It misdescribes the density of the six uniform inputs, so do not use
this subspace for geometry, integration or surrogates.

(b) The supported v1 problem, with a different input density:

```{code-cell} python3
G = ac.evaluate_grads(tf.BOREHOLE, X=tf.BOREHOLE.space.sample(1000, rng=rng))
sub_u = ac.active_subspace(grads=G, space=tf.BOREHOLE.space, nboot=500, rng=rng)

assert sub_u.eigenvalues.shape == (8,) and np.all(sub_u.eigenvalues >= 0)
```

Its eigenpairs differ from the 2016 tutorial's because the input density differs.

## Errors and warnings

| 2016 | v1 |
|---|---|
| bare `Exception`, `TypeError` or `ValueError` with free text | `ac.InputError` (a `ValueError`) with a `code` such as `input.shape` or `input.weights` |
| NaN propagated silently | `input.nonfinite` at entry |
| inputs overwritten (dev NAS, NASX) | inputs never modified |
| `print` of warnings and progress | `ac.ValidityWarning`, `ac.AccuracyWarning`, `ac.ConvergenceWarning`; `logging` |
| solver failure writes `bad_*.npz` | `ac.SolverError` (`solver.*`) |
| model failure gives a NaN row (dev) | `ac.ModelError` (`model.raised`, `model.nonfinite`) |
| any failed optimizer start raises | `convergence.failed_starts` warning; an error only if every start fails |
| estimator used outside its assumptions | `validity.assumption` warning |

2016 results depended on the global `np.random` state, which some functions reseeded with 42. Code
that called `np.random.seed(42)` once at the top must now pass `rng=42`, or one `Generator`, to
each stochastic call. `rng=None` draws fresh entropy and records it in the result's provenance.

## Removed without replacement

- MAVE (`sstype='MAVE'`): the 2016 code was a stub that returned `None`.
- The Celery runner: it executed serialized code received from a message broker. Use `executor=`.
- Gurobi: the SciPy backend is the only built-in solver; a `SolverBackend` can wrap another.
- `random_walk_z`: defective and unused.
- The 2016 tutorial data files.
- ECNoise (`utils.est_noise`, dev): estimates of the noise level of a computed function are
  outside the scope of v1. Use Moré and Wild's ECNoise, the method it ported (Moré and Wild, 2011).

## Where to get help

The [API reference](api/index.md) documents every public name, and the
[error and warning code table](codes.md) lists every `code`. Report problems on the [issue tracker](https://github.com/paulcon/active_subspaces/issues).

## References

- Cook, R. D., Weisberg, S. (1991). Discussion of "Sliced inverse regression for dimension
  reduction". *Journal of the American Statistical Association* 86(414):328–332.
- Li, K.-C. (1992). On principal Hessian directions for data visualization and dimension
  reduction: another application of Stein's lemma. *Journal of the American Statistical
  Association* 87(420):1025–1039.
- Moré, J. J., Wild, S. M. (2011). Estimating computational noise. *SIAM Journal on Scientific
  Computing* 33(3):1292–1314. [doi:10.1137/100786125](https://doi.org/10.1137/100786125)
