---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Supported problem sizes

The library enforces no size limits. This page states the sizes it is designed and tested for,
explains which operations become slow or memory-hungry beyond them and why, and shows how to check
a problem before starting an expensive step. The measured timings at the end come from the
benchmark scripts in the repository's `benchmarks/` directory.

The sizes are:

| Symbol | Meaning |
|---|---|
| $m$ | number of inputs |
| $M$ | number of samples (rows of `X`, `f`, `grads`) |
| $n$ | dimension of the active subspace |
| $N_v$, $N_f$ | numbers of vertices and facets of the active domain (uniform inputs, $n \ge 2$) |
| $N_q$ | number of nodes of a quadrature rule or points of a design on the active variables |
| $N_{MC}$ | inactive draws per node, the `n_mc` argument (default 50) |

## The envelope

| Area | Supported sizes |
|---|---|
| Subspace estimation, gradients, bootstrap | $m \le 1000$; $M$ limited by memory |
| Geometry, integration, designs and optimization on the active variables | $n \le 4$ for uniform inputs, $n \le 6$ for Gaussian inputs; for uniform inputs also $N_v\,(m+n) \le 10^8$ |

Inside the envelope the operations are designed to work, but not all of them are fast or small
at its edge. The bootstrap and local linear gradients at large $m$, and maximin designs on large
zonotopes, take minutes; the default design of `build_reduced_model` at $n = 4$ needs more memory
than a workstation has unless $m$ is small. The sections below give the costs.

`build_reduced_model` warns `accuracy.large_dimension` and continues when its dimension rule
chooses an $n$ above the envelope; an $n$ you pass explicitly never warns.

## Bounded geometry depends on $m$ as well as $n$

For uniform inputs the active domain $\mathcal Y = W_1^\top[-1,1]^m$ is a zonotope. In general
position it has

$$N_v(m, n) = 2\sum_{i=0}^{n-1}\binom{m-1}{i}$$

vertices, which grows like $m^{n-1}$. The `ActiveDomain` stores the vertices $(N_v, n)$, one cube
preimage per vertex $(N_v, m)$ and about as many facets as vertices, so it holds about
$8 N_v (m + n)$ bytes. The guidance $N_v(m+n) \le 10^8$ keeps that below 0.8 GB:

| $n$ | $N_v(m,n)$ | $m=30$ | $m=100$ | $m=300$ | $m=1000$ |
|---|---|---|---|---|---|
| 2 | $2m$ | 60 | 200 | 600 | 2 000 |
| 3 | $m^2-m+2$ | 872 | 9 902 | 89 702 | 999 002 |
| 4 | $2\sum_{i<4}\binom{m-1}{i}$ | 8 180 | 323 600 | 8.9 M | 332 M |

That covers $n=2$ for every $m$ in the table, $n=3$ up to $m \approx 460$ and $n=4$ up to
$m \approx 130$. Building a domain needs several times its stored size while it runs (about
5 to 6 times at the edge of the guidance, see the measurements below), and takes one to two and a
half minutes there.
The Gaussian domain is all of $\mathbb R^n$ and has no vertices, so only $n \le 6$ applies to it.

### Check before you build

`ac.zonotope_vertex_count(m, n)` returns $N_v$ exactly (an upper bound when rows of $W_1$ are
zero or parallel), for $m \ge 2$ and $1 \le n \le m-1$, without building anything. Use it
before `active_domain`, or before any function that builds one, such as `build_reduced_model`:

```{code-cell} python
import active_subspaces as ac

def domain_size(m, n):
    """Vertex count and approximate storage of a uniform active domain."""
    Nv = ac.zonotope_vertex_count(m, n)
    return Nv, 8 * Nv * (m + n) / 1e9  # GB

for m, n in [(100, 2), (1000, 2), (300, 3), (1000, 3), (100, 4), (300, 4)]:
    Nv, gb = domain_size(m, n)
    verdict = "fine" if Nv * (m + n) <= 1e8 else "too large: reduce n or m"
    print(f"m = {m:4d}, n = {n}:  N_v = {Nv:>11,d}  ~ {gb:8.3f} GB   {verdict}")
```

When the count is too large, use a smaller $n$ for the geometry and everything built on it; the
subspace estimate itself is not affected.

## What is slow, and why

The costs below are for one thread; a multithreaded BLAS speeds up the dense linear algebra but
not the Python-level loops.

**Subspace estimation.** Every estimator forms an $m \times m$ matrix from the samples or takes
the SVD of an $M \times m$ array, at cost $O(Mm^2)$, which is a second or two at $m = 1000$. Three
estimators cost more:

- `"opg"` estimates a local linear gradient at every sample (below), so it costs as much as
  `local_linear_gradients` with every row as a center.
- `"ss"` forms all $M^2$ pairwise difference quotients: $O(M^2 m^2)$ time.
- `"qphd"` fits a full quadratic and needs $M \ge (m+1)(m+2)/2$ samples (501 501 at $m=1000$).

The inputs themselves take $8Mm$ bytes per array: 80 MB each for `X` and `grads` at
$M = 10^4$, $m = 1000$.

**Bootstrap.** Each of the `nboot` replicates resamples the rows, repeats the decomposition, and
computes, for every candidate dimension $k=1,\dots,m-1$, a subspace distance (the spectral norm
of a $k \times (m-k)$ block) and a determinant. Both are computed on the smaller side of each
split (exact rewrites, equal to the definitions up to rounding), but the per-replicate cost still
grows like $m^4$, and this loop, not the decomposition, dominates at large $m$. Replicates are
independent: pass a `concurrent.futures` executor as `executor=` to `active_subspace` to run them
in parallel. The result does not depend on the executor or the number of workers.

**Local linear gradients.** At each center, a least-squares fit uses the $p$ nearest distinct
samples ($p = \lfloor 1.7m \rfloor$ by default) and $m+1$ unknowns, so the cost is about
$O(pm^2) = O(m^3)$ per center and $O(Mm^3)$ for all of them, plus a nearest-neighbour search.
($p$ is capped by the number of distinct samples, which must be at least $m+1$.) Pass `centers=`
to estimate gradients at a subset of rows.

**Vertex enumeration.** `active_domain` for $n \ge 3$ enumerates facets over subsets of $n-1$
generators and removes duplicate vertices. Time and peak memory grow roughly like $N_v\,m$; the
peak is a few times the stored size.

**Maximin designs.** `active_design` on a zonotope, the interior points of the zonotope
quadrature rule, and the model path of `build_reduced_model` (which uses $5^n$ design points)
optimize a maximin design with SLSQP while the optimization is small: $N_q(N_q-1)/2$ pair
constraints plus one constraint per point and facet, $N_q N_f$ with $N_f \approx N_v$, at most 2000
in all. That keeps SLSQP to small domains and few points (for example 10 points at $n = 2$,
$m = 30$). Above that the library switches to a greedy farthest-point design from $50 N_q$ random
candidates. It is fast, but it computes the distances
from every candidate to every vertex at once, an array of $50 N_q N_v n$ doubles.

**Inverse map and integration.** For uniform inputs with $m - n \ge 2$, `inverse_map` solves one
linear program (with $2m$ inequality constraints) per active point for a starting point, then runs
a hit-and-run chain of `burn_in + n_draws * thin` steps, by default $10(m-n) + 50(m-n)$ at
`n_draws = 50`, each step costing $O(mn)$. `integrate` and `integration_design` call it with
`n_draws = n_mc` for every node, and then evaluate the model $N_q N_{MC}$ times, which for a real
simulation is usually the largest cost. For Gaussian inputs the inactive draws are exact and
cheap.

**Model-path surrogates.** `fit_surrogate_from_model`, and with it the model path of
`build_reduced_model`, also evaluates the model at every vertex of a bounded domain by default
(`include_vertices=True`): one run per vertex in general position, so $N_v$ runs in addition to
the $N_q N_{MC}$ at the design points. That is 2 000 extra runs at $n = 2$, $m = 1000$ and
9 902 at $n = 3$, $m = 100$. With an expensive model and many vertices, call
`fit_surrogate_from_model` with `include_vertices=False`.

## Measured timings

Measured on 2026-10-29 by `benchmarks/run_benchmarks.py` (results files
`benchmarks/results/2026-10-29.json` and `2026-10-29-2.json`; the bootstrap and design rows
from `2026-10-29-3.json`, rerun after the faster bootstrap distances and the design switch that
counts facet constraints). Machine: a 2020 13-inch MacBook Pro
(Intel Core i7-1068NG7, 4 cores, 32 GB), macOS 14.8, Python 3.14.0, NumPy 2.5.4, SciPy 1.18.1,
Apple Accelerate BLAS, **one thread** (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`OMP_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` set to 1). This laptop serves as the reference
runner for now. It was not idle during the run (load average between 2 and 12 on 4 cores), so read
the times as upper estimates of their order of magnitude, not as precise figures.

Inputs are synthetic and seeded; $n = 2$ unless stated. Gaussian inputs with $n > 2$ and the
optimization functions were not benchmarked. *Time* is the wall time of the call alone.
*Memory* is the peak of NumPy and Python allocations during the call (`tracemalloc`); a figure
marked † is instead the growth of the process's peak resident memory, used for calls longer than
30 s.

**Subspace estimation, bootstrap and gradients**

| Call | Sizes | Time | Memory |
|---|---|---|---|
| `active_subspace`, `"as"` | $m=1000$, $M=2000$ | 1.1 s | 136 MB |
| `active_subspace`, `"as"` | $m=1000$, $M=10^4$ | 2.8 s | 456 MB |
| `"nas"`, `"asx"`, `"ols"`, `"sir"`, `"phd"`, `"save"` | $m=1000$, $M=2000$ | 0.6–1.5 s | 56–136 MB |
| `"qphd"` | $m=30$, $M=1000$ | 0.11 s | 11 MB |
| `"ss"` | $m=100$, $M=1000$ | 1.7 s | 66 MB |
| `"opg"` | $m=100$, $M=1000$ | 3.0 s | 71 MB |
| bootstrap, `nboot=100` | $m=100$, $M=1000$ | 1.7 s (0.017 s per replicate) | 6 MB |
| bootstrap, `nboot=20` | $m=300$, $M=1000$ | 3.6 s (0.18 s per replicate) | 20 MB |
| bootstrap, `nboot=3` | $m=1000$, $M=2000$ | 23 s (**about 8 s per replicate**: about 13 min for `nboot=100`) | 160 MB |
| `local_linear_gradients`, all centers | $m=100$, $M=1000$ | 2.9 s | 70 MB |
| `local_linear_gradients`, all centers | $m=300$, $M=1000$ | 20 s | 74 MB |
| `local_linear_gradients`, 10 centers | $m=1000$, $M=2000$ | 4.0 s (**0.4 s per center**: about 13 min for all 2000) | 97 MB |

**Active domain** (uniform inputs; every count equals `zonotope_vertex_count`)

| $n$ | $m$ | $N_v$ | Time | Stored | Memory |
|---|---|---|---|---|---|
| 2 | 100 | 200 | 0.003 s | 0.17 MB | 1 MB |
| 2 | 1000 | 2 000 | 0.14 s | 16 MB | 98 MB |
| 3 | 100 | 9 902 | 0.52 s | 8.5 MB | 35 MB |
| 3 | 300 | 89 702 | 15 s | 220 MB | 911 MB |
| 3 | 450 | 202 052 | 56 s | 739 MB | 4.4 GB † |
| 4 | 30 | 8 180 | 0.35 s | 2.5 MB | 11 MB |
| 4 | 60 | 68 560 | 5.6 s | 38 MB | 156 MB |
| 4 | 100 | 323 600 | 46 s | 282 MB | 1.6 GB † |
| 4 | 130 | 715 780 | 142 s | 796 MB | 4.3 GB † |

**Inverse map, designs, quadrature and integration** (inverse map and integration at 50 draws
per point, the default `n_mc`)

| Call | Sizes | Time | Memory |
|---|---|---|---|
| `inverse_map`, uniform (hit-and-run, 5 880 steps per chain) | $m=100$, 64 points | 5.0 s | 31 MB |
| `inverse_map`, uniform (hit-and-run, 59 880 steps per chain) | $m=1000$, 10 points | 74 s | 244 MB † |
| `inverse_map`, Gaussian | $m=1000$, 64 points | 0.25 s | 63 MB |
| `active_design`, greedy, $N_q=10$ | $n=2$, $m=100$ | 0.007 s | 2 MB |
| `active_design`, greedy, $N_q=25$ | $n=2$, $m=100$ | 0.017 s | 6 MB |
| `active_design`, greedy, $N_q=10$ | $n=2$, $m=1000$ | 0.05 s | 24 MB |
| `active_design`, greedy, $N_q=25$ | $n=2$, $m=1000$ | 0.13 s | 60 MB |
| `active_design`, greedy, $N_q=10$ | $n=3$, $m=30$ | 0.025 s | 14 MB |
| `active_design`, greedy, $N_q=10$ | $n=3$, $m=100$ | 0.29 s | 158 MB |
| `active_design`, greedy, $N_q=100$ | $n=3$, $m=100$ | 2.8 s | 1.6 GB |
| `active_design`, greedy, $N_q=125$ | $n=3$, $m=100$ | 3.6 s | 2.0 GB |
| `active_design`, greedy, $N_q=625$ | $n=4$, $m=20$ | 6.9 s | 2.9 GB |

All nine design cases above now use the greedy design: each exceeds 2000 SLSQP constraints. Before
the switch counted facet constraints, the same SLSQP designs took up to 286 s ($N_q=25$, $n=2$,
$m=1000$).
| `active_quadrature_rule`, vertices only ($N_q = 8$ cells) | $n=2$, $m=1000$ | 1.4 s | 526 MB |
| `active_quadrature_rule`, vertices only ($N_q = 71$ cells) | $n=3$, $m=100$ | 0.73 s | 395 MB |
| `integrate`, uniform, $N_q=38$ | $n=2$, $m=10$ | 0.28 s | 6 MB |
| `integrate`, uniform, $N_q=8$ | $n=2$, $m=100$ | 0.9 s | 54 MB |
| `integrate`, uniform, $N_q=8$ | $n=2$, $m=1000$ | 80 s | 717 MB † |
| `integrate`, Gaussian, $5\times5$ Gauss–Hermite | $n=2$, $m=1000$ | 0.15 s | 50 MB |

The model here is a cheap quadratic, so the integration times are almost all sampling; with a
real simulation the $N_q N_{MC}$ model runs usually dominate. In short, on one thread:

- subspace estimation takes seconds at $m = 1000$, except with `"opg"`, `"ss"` and `"qphd"`;
- at $m = 1000$ the bootstrap (about 8 s per replicate) and local linear gradients
  (about half a second per center, so also `"opg"`) are the slow steps;
- bounded geometry is fast for $n = 2$ at every $m$, and for $n = 3, 4$ takes up to a few minutes
  and a few GB at the edge of the vertex guidance;
- maximin designs are fast: SLSQP is used only while it has at most 2000 constraints, and the
  greedy design otherwise;
- the greedy design holds $50 N_q N_v n$ doubles at once, so the
  $5^n$-point default design of `build_reduced_model` needs about 2 GB at $n = 3$, $m = 100$ and,
  by the same count, 13 GB at $n = 3$, $m = 300$ and 320 GB at $n = 4$, $m = 100$. For these
  sizes use the data path of
  `build_reduced_model`, or call `fit_surrogate_from_model` with your own `design=` points;
- hit-and-run inactive draws at $m = 1000$ and `n_mc = 50` take over a minute for 8 to 10 nodes
  (the chains of up to 64 nodes advance together, one vectorized step at a time).

## Reproducing the measurements

From a checkout of the repository:

```sh
python benchmarks/run_benchmarks.py           # about 25 minutes
python benchmarks/run_benchmarks.py --quick   # about 8 minutes: skips the long edge cases
```

Each run writes `benchmarks/results/<date>.json` with the environment (Python, NumPy, SciPy and
BLAS versions, CPU, thread settings), and for each case its sizes, wall time and peak memory.
`benchmarks/README.md` describes the cases and the measurements.
