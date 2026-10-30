# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) with PEP 440 version strings.

Entries cite the IDs of the project's design records: findings (`P-`, `C-`, `N-`, `R-`, `A-`) are
defects and design inputs found in the 2016 code, and decisions (`D-`) are the maintainer's rulings.
Both are kept in `design-specs/` (`findings-register.md`, `decisions.md`). "Changed results" lists
changes that alter computed numbers for the same inputs. The migration guide
(<https://active-subspaces.readthedocs.io/en/stable/migration.html>) maps every 2016 name to its
replacement and explains each change in the numbers in more detail.

## [Unreleased]

These changes will be released as 1.0.0. Version 1.0 is a rewrite of the 2016 library (0.1, git
tags `legacy-master` and `legacy-dev`) for Python 3. It merges the 2016 `master` and `dev` branches,
changes the public API without a compatibility layer, and fixes the defects listed below (D-01,
D-02, D-02a, D-03).

### Added

- **Models with several outputs.** `ac.evaluate_multi` evaluates a model whose value has shape
  `(k, outputs)`, and `ac.finite_difference_gradients_multi` returns one `GradientResult` per
  output from one set of stencil evaluations, each equal bitwise to the single-output call on that
  column (D-199).
- **Python 3 packaging.** `pyproject.toml` built with hatchling, a `src` layout, the version
  single-sourced in `_version.py`, PEP 561 type hints (`py.typed`), and dependency floors
  `numpy>=2.2`, `scipy>=1.15`, `matplotlib>=3.10` (P-10, D-05, D-147, D-151).
- **Explicit input spaces** `ac.Uniform(lower, upper)` and `ac.Gaussian(mean, cov)`, behind the
  `ac.InputSpace` protocol. Data enter in physical units; the space normalizes inputs and applies
  the chain rule to gradients. Results store normalized arrays and expose physical views
  (A-01, D-02b, D-11, D-12, D-58, D-62, D-67).
- **A functional API with frozen results.** Stateless keyword-only functions return immutable,
  typed result objects (`SubspaceResult`, `ActiveDomain`, `IntegrationResult`, ...) that carry
  their input space and provenance, serialize with `to_dict` / `save` and reload with `ac.load` /
  `ac.from_dict`, with versioned schemas and no pickle (A-05, A-07, R-09, D-02, D-18, D-19, D-20,
  D-65, D-68, D-70).
- **Reproducible randomness.** Every stochastic function takes `rng` (an integer seed or a
  `numpy.random.Generator`); results record the generator state in their provenance, so a run can
  be replayed, and the library never touches the global NumPy random state (R-03, D-14, D-15, D-71, D-72).
- **Errors and warnings with stable codes.** `ac.ActiveSubspacesError` and its subclasses
  (`InputError`, `ModelError`, `NumericalError`, `SolverError`, `SchemaError`,
  `UnsupportedError`), warning classes (`AccuracyWarning`, `ConvergenceWarning`, `ValidityWarning`,
  `ExperimentalWarning`, `ActiveSubspacesDeprecationWarning`), and a code such as
  `input.out_of_space` on each; the library logs through the `active_subspaces` logger and never
  prints (R-04, R-05, D-16, D-17, D-73, D-74, D-152).
- **`ac.active_subspace`**, one entry point for all estimators: `"as"` (default), and from the
  `dev` branch `"nas"`, `"asx"`, `"nasx"`, `"ss"`, `"sir"`, `"phd"`, `"save"`, together with
  `"ols"`, `"qphd"` and `"opg"`. Estimator instances (`ac.subspaces.SIR(n_slices=...)`, ...) set
  options. Validity warnings flag estimator and input-space combinations whose assumptions do not
  hold (A-03, D-03, D-41, D-44, N-17).
- Bootstrap replicates stored in the result, with `ac.bootstrap_interval` (`kind="percentile"` by
  default, or `"minmax"`), `ac.mean_distance`, `ac.dimension_criterion` and `ac.with_dimension`
  (D-30, D-79, D-80).
- The log-gap option of the eigenvalue-gap rule, `ac.subspaces.EVG(gap="log")` (N-06, D-38).
- **Eigenvector bootstrap bands.** `ac.active_subspace(..., nboot=B, bootstrap_vectors=k)` stores
  `SubspaceResult.bootstrap_eigenvectors` `(B, m, k)`, each replicate column multiplied by
  $\operatorname{sign}(w_j^\top w_j^{(b)})$; the warning `accuracy.unaligned_replicates` flags
  replicates with $|w_j^\top w_j^{(b)}|<0.5$. `ac.plotting.plot_eigenvectors(sub,
  interval="percentile" | "minmax", level=0.95)` draws the component bands. They replace the 2016
  `W_br`. `SubspaceResult` is now schema version 2; version-1 files load with
  `bootstrap_eigenvectors=None` through a migration (D-196).
- `ac.orient(subspace, by="reference" | "covariance" | "mean_gradient", ...)` returns a result with
  user-chosen eigenvector signs (D-197).
- `ac.activity_scores(subspace, n=None)`, $\sum_{j\le n}\lvert\lambda_j\rvert w_{ij}^2$ in
  normalized coordinates (Constantine and Diaz 2017) (D-201).
- `ac.subspace_from_matrix(C, space=, n=, dimension_rule=, qoi=)` makes a result with eigenvalues
  (`estimator="matrix"`) from a symmetric matrix accumulated by hand, in normalized coordinates
  (D-202).
- `ac.subspace_from_basis` to use an externally computed basis with geometry and surrogates
  (D-92).
- `ac.local_linear_gradients` and `ac.finite_difference_gradients` return a `GradientResult`
  aligned with the inputs; finite differences gain `scheme="central"` and never evaluate the model
  outside its input space (D-32, D-82, D-83, D-88).
- **Geometry**: `ac.active_domain`, `ac.forward_map`, `ac.inverse_map`, `ac.in_active_domain` and
  `ac.zonotope_vertex_count`, with exact zonotope vertices and facets (R-06, D-46, D-90, D-92,
  D-93, D-94, D-95).
- **Designs and surrogates**: `ac.active_design`, `ac.conditional_samples`, `ac.fit_surrogate`,
  `ac.fit_surrogate_from_model`, `ac.predict`, `ac.predict_gradient`, `ac.quadratic_terms`;
  polynomial and RBF surrogates with a leave-one-out $R^2$ (`loo_r_squared`) and an
  `ac.Surrogate` protocol (D-97, D-98, D-99, D-102, D-104).
- **Integration and optimization**: `ac.integrate`, `ac.integrate_surrogate`,
  `ac.integration_design`, `ac.estimate_integral`, `ac.active_quadrature_rule`, `ac.minimize`,
  `ac.minimize_active` and `ac.lift`, with a `lift_dim` option for the lift (D-33, D-47, D-53,
  D-105, D-107, D-108, D-109, D-111, D-113, D-114).
- **Reduced model** (from the `dev` branch, redesigned): `ac.build_reduced_model` returns an
  `ac.ReducedModel` with `average`, `minimum` and `probability` (Wilson interval) (D-03, D-22,
  D-115, D-116, D-118, D-119, D-120, D-121).
- **Models and evaluation**: `ac.Model`, `ac.as_model`, `ac.pointwise`, `ac.evaluate`,
  `ac.evaluate_grads` and `ac.evaluate_with_grads` with vectorized batches, an optional combined
  value-and-gradient function, any `concurrent.futures.Executor`, and results independent of the
  number of workers; `ac.ModelRegistry` (A-06, D-23, D-24, D-25, D-26, D-27, D-123, D-124,
  D-125, D-127).
- **Built-in test functions** in `ac.test_functions` (borehole, OTL circuit, piston, robot arm,
  wing weight, and the known-answer problems `quadratic`, `ridge`, `dct_basis`), with physical
  inputs and stored reference results (A-11, D-21, D-122, D-126, D-128).
- **Solver backends**: the `ac.SolverBackend` protocol, `ac.ScipyBackend` (HiGHS for linear
  programs), `ac.solve_lp`, `ac.solve_qp`, `ac.get_backend` (D-04, D-28).
- **Plotting** in `ac.plotting`, loaded on first use: every function returns `(fig, ax)` and never
  shows, saves or changes global matplotlib state; new `plot_surrogate_fit` and
  `plot_dimension_criterion` (D-06, D-129, D-130, D-131, D-132, D-133, D-134, D-135, D-136).
- **Experimental gradient-free subspaces** (from the `dev` branch) in
  `active_subspaces.experimental`: `sketched_active_subspace`, `subspace_from_sketches`,
  `SketchedSubspace`. Importing it issues an `ExperimentalWarning`; its API may change in any minor
  release (D-03, D-35, D-73, D-138, D-140, D-141).
- An explicit `__all__` in every public module, enforced import layering, and an
  `AttributeError` message that points to the migration guide for removed 2016 names (A-09, P-02,
  D-148, D-149, D-168).
- Documentation on Read the Docs with a user guide, theory pages, executed tutorials, an API
  reference, a table of error and warning codes, and a migration guide (D-144, D-145, D-167).
- A test suite in six tiers (unit, known-answer, property, statistical, regression against the 2016
  results, acceptance use cases) and GitHub Actions CI on Linux, macOS and Windows for Python
  3.11 to 3.14 (D-37, D-77, D-78, D-143, D-159, D-163).
- `CITATION.cff`, `AUTHORS.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` and `SECURITY.md`
  (D-146, D-155).

### Changed

- **The whole public API.** Classes called in a fixed order (`Subspaces`, `ActiveVariableDomain`,
  `ActiveVariableMap`, `ActiveSubspaceResponseSurface`, `ActiveSubspaceReducedModel`) are replaced
  by functions and results; tuple returns become named fields; modules are renamed (`domains` →
  `geometry`, `response_surfaces` → `surrogates`, `integrals` → `integration`, `optimizers` →
  `optimization`, `utils.qp_solver` → `solvers`, `utils.simrunners` → `models`,
  `utils.plotters` → `plotting`, dev `base` → `reduced_model`); the `utils` package is gone
  (A-03, A-05, A-07, A-09, A-10, D-02, D-02a).
- **Inputs are physical.** `X` and `grads` are passed in physical units with their input space,
  instead of being normalized by hand; do not rescale gradients yourself (A-01, D-58).
- **Shapes.** `f`, eigenvalues, weights and predictions are 1-D, `(M,)` and `(m,)`; 1-D `f` is
  accepted, and when $m=1$ so is 1-D `X` (A-04, D-08, D-09, D-61, D-64).
- **Weights are named for their role.** `quad_weights` (quadrature weights, which must sum to 1)
  and `sample_weights` (regression weights) replace the overloaded `weights`; the weights argument
  of `local_linear_gradients` is removed (A-02, C-03, D-10, D-60).
- Parameter names: `sstype` → `estimator`, `ptype` → `dimension_rule`, `avdim` → `n`, `N` →
  `degree`, `n_nodes`, `n_points`, `n_design` or `n_draws` by meaning, `NMC` → `n_mc`, `compgrad`
  → a separate `predict_gradient` (D-104).
- **Eigenvalues are signed** for the indefinite estimators ASX, NASX and pHd; ranking is still by
  magnitude (N-07, D-31).
- **Eigenvector sign rule**: the largest-magnitude component of each eigenvector is positive (2016:
  the first component). Components within $\sqrt{\varepsilon}$ of the largest count as tied, and
  the first of them sets the sign, so the sign does not depend on the platform when two components
  are equal in exact arithmetic (D-81, D-188).
- **OLS eigenvector sign**: for `estimator="ols"`, $w_1=\hat u/\lVert\hat u\rVert$ keeps the sign
  of the least-squares slope, so $f$ increases along $w_1$, as in Constantine (2015), Algorithm 1.3,
  eqs. (1.5)–(1.6), and Constantine et al., *J. Comput. Phys.* 302 (2015), Algorithms 1–2, eq. (22).
  The other columns keep the largest-magnitude rule (D-195).
- **Physical directions keep the signs of `W1`**: `physical_directions()` and
  `space.directions_to_physical` give $q_j^\top A^{-\top}w_j>0$ instead of re-applying the
  largest-magnitude rule, so they agree with an oriented `W1` (D-197).
- **Default bootstrap interval**: a 95% percentile interval replaces the min/max band in plots and
  in `ac.bootstrap_interval`; `kind="minmax"` gives the 2016 statistic (N-08, D-30).
- **Fit quality**: the in-sample polynomial `Rsqr` and the RBF "R²" are replaced by a
  leave-one-out $R^2$, which is lower (N-05, D-97); the reduced model reports a holdout $R^2$
  (C-07, D-120).
- **Integration error**: `IntegrationResult.inactive_std_error` replaces the `(lb, ub)` pair of
  ±1.96 standard errors (D-33).
- **Probability**: `ProbabilityResult.confidence_interval(level)` gives a Wilson interval
  (C-06, D-116, D-119).
- **Model evaluation**: a failed evaluation raises `ac.ModelError` instead of producing a NaN row;
  `KeyboardInterrupt` is never caught (C-11, D-25).
- Zonotope vertices are computed exactly, so the `Nsamples` and `maxcount` arguments are gone
  (R-06, C-15, D-90).
- The default number of samples for zonotope quadrature weights is $10^5$ (2016: $10^4$) (N-10,
  D-105).

### Changed results

Computed values differ from the 2016 code for the same inputs. Each item is either a bug fix (the
2016 value contradicted the method) or a departure from the 2016 method ruled on by the maintainer
(D-03b). The migration guide gives the size of each change.

- **AS, NAS and OPG factorization.** With $M>m$ the estimators take the SVD of the triangular factor
  of a blocked QR factorization of the weighted gradient rows, with the AS rows scaled in place
  when there is no bootstrap, so peak memory is about $1+2/m$ times the gradient array (1.25 on the Ebola $7^8$ rule)
  instead of about three times; with a bootstrap it is about 2.57 times at $m=8$ (D-213).
  Eigenvalues agree with the direct SVD to $10^{-13}$ relative (D-202).
- **OLS and QPHD least squares** use a QR solve when the design is well conditioned and `gelsd`
  otherwise; QPHD is about three times faster at large sizes, and results change only by rounding
  (V1-11, D-203). The `gelsd` rank cutoff is now $\max(M,p)\,\epsilon\,\sigma_1$ (was
  $\epsilon\,\sigma_1$), so a duplicated column warns `accuracy.rank_deficient` on every platform.
- **Random streams.** Every stochastic output (bootstrap replicates, designs, inactive samples,
  optimization starts, weight samples) differs from 2016 at any seed (R-03, D-14).
- **Default inactive sample count** $N_{MC}=50$ (2016: 10) in `integration_design`, `integrate`,
  `conditional_samples`, `fit_surrogate_from_model` and the reduced model; model-path costs grow
  fivefold (D-171).
- **Inactive variances** use ddof 1, and the standard error accounts for the autocorrelation of
  hit-and-run draws (N-15, D-48, D-114).
- **Interval quadrature** ($n=1$): Gauss nodes of the box-spline density with exact weights replace
  midpoints with Monte Carlo weights, removing the 2016 error floor (N-10, D-47, D-53, D-113).
  Zonotope quadrature ($n\ge2$) renormalizes a weight deficit instead of biasing the estimate toward
  zero (C-15, N-10, D-105).
- **Inverse map**: exact for $m-n=1$; otherwise hit-and-run with burn-in and thinning, and points
  on the boundary are flagged (N-09, C-22, D-46, D-94, D-95).
- **Local linear gradients**: one gradient per sample at the correct $p$ nearest neighbours,
  including the center (C-02, C-03, C-19, D-32, D-82). Finite differences use one-sided steps near
  the faces of a bounded space and relative steps (D-83).
- **Maximin designs**: epigraph formulation with feasible starts; designs whose SLSQP program
  would exceed 2000 constraints (point pairs plus point–facet rows) use a greedy construction
  (N-19, D-51, D-98, D-170, D-177).
- **Lift curvature**: eigenvalues of the fitted quadratic are clipped at $10^{-2}$ of the largest
  magnitude instead of adding $0.01I$ (N-16, D-49, D-110).
- **RBF surrogates**: concentrated Gaussian-process likelihood with $\tfrac M2\log\hat\sigma^2$, a
  grid search followed by refinement, and a nugget relative to the fitted amplitude, so the fit is
  invariant under $f\mapsto af+b$ (N-05, C-18, D-36, D-100, D-101, D-172).
- **SAVE** weights each slice by its probability, $C=\sum_h\hat p_hV_h^2$ (N-01, D-42).
- **pHd** uses the scale $\Sigma^{-1/2}\Sigma_{fxx}\Sigma^{-1/2}$; for uniform inputs its eigenvalues
  are $\sqrt3$ times the 2016 values (N-02, D-43).
- **SIR, pHd and SAVE** whiten with the covariance of the declared input space instead of a scalar
  guessed from the range of the data (N-03, D-39); SIR and SAVE slicing uses weighted slice
  probabilities and centering, equal-count slices that keep ties together, and a configurable
  number of slices (N-13, D-44).
- **QPHD** is $C=gg^\top+H\Sigma H$ from a global quadratic fit (Li 1992, Corollary 3.1 with $q=2$);
  the dev `quadratic_model_check` is merged into it (N-11, D-40, D-117, D-173).
- **Swarm subspace** skips only exactly coincident pairs (N-12, D-45, D-86); **NAS and NASX** use a
  relative threshold for zero gradient rows and no longer modify the caller's arrays (C-21, D-87).
- **Ladle** uses $1+\sum$ in both denominators and truncates at $k_n$ as in Luo and Li (2016)
  (D-52).
- The shipped borehole function is uniform in $r_w$ and $r$, so its eigenpairs differ from the 2016
  tutorial (D-12, D-21). The robot-arm gradient is 0 instead of NaN where $f=0$ (D-128).

### Deprecated

- Nothing. Version 1.0 deprecates nothing; the deprecation policy starts with 1.0 (D-152).

### Removed

- Python 2 support (P-01 to P-09, D-05).
- The Gurobi solver path; SciPy is the only backend (A-08, D-04).
- The Celery backend (`utils/celery_runner.py`, `celeryconfig.py`, `start_celery.sh`), the MATLAB
  code, the tutorial data sets (see `paulcon/as-data-sets`) and the MAVE stub (C-12, R-08, D-03a).
- ECNoise noise estimation (dev `utils/est_noise.py`); use Moré and Wild's ECNoise. The step of
  `finite_difference_gradients` is chosen by the user (C-23, R-11, N-04, D-174).
- `random_walk_z`, which scrambled its samples and was unused (C-10, D-91).
- Public 1-D quadrature rules (`utils.quadrature`); NumPy and SciPy provide them (D-106).
- Plot options that saved files (`plot_opts`, `savefigs`), `show_plot`, and the eigenvector
  bootstrap bands `W_br` that no code produced (R-02, R-12, A-12, D-131, D-133).
- `setup.py`, `requirements.txt`, `test.py`, `MANIFEST.in`, Travis CI and nose (P-10).
- The 2016 tutorials and Sphinx sources; they remain at the tags `legacy-master` and `legacy-dev`
  (D-01, D-144).
- `legacy/`: the 2016 master tree and the Python 3 reference ports used to capture the regression
  baselines, with the `legacy` CI job and the legacy mode of `tools/capture_baselines.py`. They
  remain at the tag `legacy-py3`; the baselines stay in `tests/data/baselines/` (D-37, D-150).
- No compatibility shim is provided for 2016 code (D-02a).

### Fixed

- Hit-and-run draws (`inverse_map`, `integrate`) are bitwise repeatable for a given `rng`; the
  chain start no longer uses an alignment-dependent BLAS product (D-15; review 2026-10-29-phase5
  #1).
- A bootstrap replicate computes its per-$k$ distances and determinants on the smaller side of
  each split, about 2.5 times faster at $m=1000$ with the same results up to rounding (D-178).
- The package imports and runs under Python 3 and NumPy 2 (P-01 to P-09).
- QPHD, SIR, pHd and SAVE no longer crash for data in $[-1,1]^m$ (C-01).
- The lift $y^\ast\to x^\ast$ minimizes the fitted quadratic over $z$: 2016 had the wrong sign in
  the unbounded case and a missing $\tfrac12$ in the bounded case (C-16, D-107).
- The ladle estimate returns $\hat k$ instead of $\hat k+1$ (C-17).
- Weighted polynomial fits scale rows by $\sqrt w$, not $w$ (C-04).
- The RBF nugget is scale-equivariant, and the model-path nugget uses the variance of the
  conditional mean (C-18, D-48, D-100).
- `local_linear_gradients` no longer fails when $M\le1.7m$ and works for $m=1$ (C-19, C-03, D-82).
- The reduced model's input check fires on data outside the space (C-05); probability bounds are
  ordered (C-06); the model-path $R^2$ includes the missing $1/M$ (C-07); $m=1$ raises an error
  instead of drawing no samples (C-08, D-118); messages about the dimension are formatted
  correctly instead of raising `ValueError` (C-09).
- Zonotope vertices are complete and free of duplicates with inert or parallel inputs (C-15,
  C-20, D-90, D-103).
- Multistart minimization tolerates failed starts, and reaching the evaluation limit warns instead
  of raising (C-13, C-14, D-109).
- Hit-and-run flags and warns about repeated points where it cannot move, instead of repeating
  them silently (C-22, D-95).
- Gradient-free subspaces fix the signs of the singular vectors jointly (C-24, D-140).
- Constraints of the active domain use the outward convention of `ConvexHull` (R-07).
- Maximin designs no longer pass on SciPy's "Values in x were outside bounds" warning, which
  SciPy versions before 1.16 emit when SLSQP clips a step to the bounds (D-189).

### Security

- No function writes files unless asked: plots no longer create `figs/`, and failed solves no
  longer write `bad_*.npz` to the working directory (R-01, R-02).
- Results are saved as JSON and NPZ, never pickle, and loading checks the schema (D-19).
- The Celery runner, which executed `marshal`-ed code received from a broker, is removed (R-08,
  D-03a).
- A Redis password in the 2016 `dev` history is inert: the service it protected no longer exists.
  The history was not rewritten (R-08, D-175).
