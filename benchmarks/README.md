# Envelope benchmarks

Plain Python scripts that time and size the operations whose cost limits the
supported problem sizes (the envelope of `design-specs/target/00` §2.3 and the
docs page `docs/user_guide/problem_sizes.md`). They are run by hand on the
reference runner, in Phase 3 and before each minor release. They are **not**
pytest tests, are **not** packaged, are **not** run in CI and gate nothing
(spec 15 §5.5).

## Running

From the repository root, with NumPy, SciPy and matplotlib installed:

```sh
python benchmarks/run_benchmarks.py --runner "reference runner: <machine>"   # every case
python benchmarks/run_benchmarks.py --quick      # skip the long edge-of-envelope cases
python benchmarks/run_benchmarks.py --only geometry.active_domain --only bootstrap
python benchmarks/run_benchmarks.py --list       # case names; [boundary] marks the long ones
```

By default the scripts benchmark the checkout's `src/`; `--installed` uses the
installed package instead. On the 2026 reference machine (an Intel i7-1068NG7
laptop, one thread, shared with other work) the full set took about 23 minutes,
about 10 of them in the `boundary` cases; `--quick` skips those and takes about 8.

Each case runs in a fresh child process with

```
OPENBLAS_NUM_THREADS=1  MKL_NUM_THREADS=1  OMP_NUM_THREADS=1  VECLIB_MAXIMUM_THREADS=1  MPLBACKEND=agg
```

as in CI (`VECLIB_MAXIMUM_THREADS` covers Apple Accelerate), so the timings are
single-threaded and one case's memory never counts towards another's. A
multithreaded BLAS speeds up the dense-algebra cases (subspace estimation, the
bootstrap, local linear gradients) but not the Python-level loops.

## Output

Each run writes `benchmarks/results/<date>.json` (`<date>-2.json`, … for a
second run the same day; `--output` overrides). The file holds:

- `environment`: Python, NumPy, SciPy and active_subspaces versions, the import
  path, git commit and dirty flag, platform, CPU model and count, memory, BLAS
  and LAPACK names, the thread variables, and `threadpoolctl` pools if that
  package is installed;
- `method`: how each figure is measured (below);
- `cases`: one record per case with `name`, `group`, `params` (the sizes `m`,
  `M`, `n`, …), `wall_s`, `wall_s_all`, `tracemalloc_peak_bytes`,
  `rss_after_setup_bytes`, `rss_peak_bytes`, the warning codes raised, an
  `info` dict of facts about the output (vertex counts, node counts, bytes
  stored, the counter-clockwise check, …), `status` and a `note`.

Measurement:

- `wall_s` is the minimum over `repeat` timed calls (`time.perf_counter`) of
  the operation alone; building the synthetic inputs is not timed.
- `tracemalloc_peak_bytes` is the peak of Python and NumPy allocations during
  one extra call under `tracemalloc`, made only when a call takes at most 30 s.
- `rss_peak_bytes` is the process high-water resident set size
  (`getrusage`); it includes the interpreter, the imports and the inputs, so
  compare it with `rss_after_setup_bytes`.

## Cases

| Group | What | Sizes |
|---|---|---|
| `subspace_estimation` | `active_subspace` with each estimator, `n = 2` fixed | $m=1000$, $M=2000$ (and $M=10^4$ for `as`); QPHD at $m=30$, SS and OPG at $m=100$ |
| `bootstrap` | `active_subspace(..., nboot=B)` | $m=100$ ($B=100$), $m=300$ ($B=20$), $m=1000$ ($B=3$, scaled down) |
| `local_linear_gradients` | `local_linear_gradients` | $m=100, 300$ (all centers); $m=1000$ (10 centers, scaled down) |
| `zonotope_vertex_count` | the target/00 §2.3 vertex table | $n=2,3,4$; $m=30,100,300,1000$ |
| `active_domain` | uniform zonotope from a random orthonormal basis | $n=2$: $m\le1000$; $n=3$: $m\le450$; $n=4$: $m\le130$ |
| `inverse_map` | `n_draws = 50` (the default `n_mc`) per point | uniform $m=100, 1000$; Gaussian $m=1000$ |
| `active_design` | maximin design: SLSQP ($N_q\le63$) or greedy | $n=2$, $m=100, 1000$, $N_q=10, 25$; $n=3$, $m=30, 100$, $N_q=10$; greedy $N_q=100, 125$ ($n=3$) and $625$ ($n=4$, $m=20$) |
| `active_quadrature_rule` | Delaunay rule on the vertices, 100 000 weight samples | $n=2$, $m=100, 1000$; $n=3$, $m=30, 100$ |
| `integrate` | `integrate` with `n_mc = 50` and a quadratic model | uniform $n=2$, $m\le1000$; Gaussian $m=1000$ |

Cases marked `boundary` sit at the edge of the envelope and are long; `--quick`
skips them. Scaled-down cases (fewer bootstrap replicates or gradient centers
than a real run) say so in their `note`; multiply the per-replicate or
per-center cost.

To add a case, append a `Case` to the lists in `cases.py`: `setup(params)`
builds seeded synthetic inputs (untimed) and `run(state)` makes the timed call
and returns a small JSON-serializable dict.

## After a run

Compare the results with the envelope table of target/00 §2.3, and update the
measured-timings table of `docs/user_guide/problem_sizes.md` (it names the
results file, machine and date it was taken from).
