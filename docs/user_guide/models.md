---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Models, executors and the registry

A **model** is your simulation as the library sees it: a scalar quantity of interest $f$ of
$m$ inputs, given by batch callables in **physical** coordinates, together with its input space.
The library calls the model whenever it needs values or gradients at points it chose: in
`evaluate`, finite differences, integration, surrogate fitting from a model, and the reduced model.

```{code-cell} python
import concurrent.futures
import numpy as np
import active_subspaces as ac
```

## Defining a model

`ac.Model` takes the space and up to three callables, each receiving a `(k, m)` array of physical
points:

| Callable | Returns |
|---|---|
| `value` | `(k,)` (or `(k, 1)`) values |
| `grad` | `(k, m)` physical gradients |
| `value_and_grad` | the pair, from one call |

At least one of `value` and `value_and_grad` is required. Metadata are optional: `name`,
`description`, `qoi` (the label of $f$), `qoi_units`, `cost` (seconds per point, a hint) and
`batch_size`.

```{code-cell} python
space = ac.Uniform([0.0, 1.0], [2.0, 5.0], names=["a", "b"], units=["m", "s"])

def value(X):                                  # (k, 2) -> (k,)
    return np.exp(0.7 * X[:, 0] + 0.1 * X[:, 1])

def grad(X):                                   # (k, 2) -> (k, 2)
    return value(X)[:, None] * np.array([0.7, 0.1])

model = ac.Model(space=space, value=value, grad=grad, name="toy", qoi="growth", qoi_units="kg")
print(model.dim, model.has_grad)
```

## Evaluating

`ac.evaluate`, `ac.evaluate_grads` and `ac.evaluate_with_grads` run a model at many physical
points. The points are checked against the space first (`input.out_of_space` if one is outside).

```{code-cell} python
X = space.sample(1000, rng=0)
f = ac.evaluate(model, X=X)                    # (1000,)
f2, G = ac.evaluate_with_grads(model, X=X)     # (1000,), (1000, 2)
print(f.shape, G.shape, np.array_equal(f, f2))
```

Every function that takes a model also accepts a bare value callable together with `space=`:

```{code-cell} python
f3 = ac.evaluate(value, X=X, space=space)
print(np.array_equal(f, f3))
```

### A function of one point

If your code evaluates one point at a time, wrap it with `ac.pointwise`. `output=` says what the
function returns: `"value"`, `"grad"` or `"value_and_grad"`.

```{code-cell} python
def one_point(x):                              # (m,) -> float
    return float(np.exp(0.7 * x[0] + 0.1 * x[1]))

m1 = ac.Model(space=space, value=ac.pointwise(one_point))
print(np.allclose(ac.evaluate(m1, X=X[:5]), f[:5]))
```

### A model with several outputs

One run of a simulation may yield several outputs, for example a trajectory at $P$ times. Give the
callable values of shape `(k, P)`. `ac.evaluate_multi` runs it, and
`ac.finite_difference_gradients_multi` returns one `GradientResult` per output from a single set
of runs. Each stencil point is evaluated once, so the cost is that of one output.

```{code-cell} python
def trajectory(X):                             # (k, 2) -> (k, 3)
    t = np.array([0.5, 1.0, 2.0])
    return np.exp(-t * (0.7 * X[:, :1] + 0.1 * X[:, 1:]))

traj = ac.evaluate_multi(trajectory, X=X, outputs=3, space=space)
res = ac.finite_difference_gradients_multi(
    trajectory, X=X[:50], space=space, outputs=3, qoi_names=("t0.5", "t1", "t2")
)
print(traj.shape, [r.qoi for r in res], res[0].n_evaluations)
```

Result $p$ equals, bit for bit, `ac.finite_difference_gradients` on a model that returns column
$p$. A callable that returns `(k,)` raises `model.shape`. Without `qoi_names` the labels are
`"f0"`, `"f1"`, and so on.

## Batches and executors

Rows are split into batches of `batch_size` rows (the argument, else `model.batch_size`, else all
rows in one call). Each call receives a fresh copy of its rows, so a model cannot change your `X`.

`executor=` takes any `concurrent.futures.Executor`. Batches are submitted to it and their results
collected **in batch order**, so the output, and any error, is the same for every executor and
every number of workers. The executor belongs to you; the library never shuts it down.

```{code-cell} python
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    f_par = ac.evaluate(model, X=X, executor=pool, batch_size=100)
print(np.array_equal(f_par, f))
```

A thread pool helps when the model releases the GIL, for example when it waits on an external
program. For pure-Python models use a `ProcessPoolExecutor`; the model's callables must then be
picklable, which means defined at module level in an importable file:

```python
# my_sim.py defines value() and grad(); run under `if __name__ == "__main__":`
with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
    G = ac.evaluate_grads(my_sim.MODEL, X=X, executor=pool, batch_size=50)
```

The same `executor` and `batch_size` arguments appear on every function that runs a model, and
`executor` alone on the bootstrap of `ac.active_subspace`.

## When a model fails

The library does not retry, substitute NaN or skip points. The first failing batch, in batch
order, raises an `ac.ModelError`:

| Code | Cause |
|---|---|
| `model.raised` | the model raised an exception (it is attached as `__cause__`) |
| `model.nonfinite` | it returned NaN or ±inf |
| `model.shape` | it returned the wrong shape |
| `model.dtype` | it returned a non-numeric dtype |
| `model.executor` | the executor failed, for example a worker process died |

```{code-cell} python
def fragile(X):
    out = X[:, 0] ** 2
    out[X[:, 1] > 4.9] = np.nan                # fails in part of the space
    return out

try:
    ac.evaluate(fragile, X=X, space=space, batch_size=100)
except ac.ModelError as err:
    print(err.code, err.details)
```

`details["batch"]` gives the rows of the failing batch, so you can rerun them.

## The registry and test functions

`ac.test_functions.builtin_models()` returns an immutable `ac.ModelRegistry` of the built-in
problems. Each entry is an `ac.Model` with an analytic gradient and stored reference results.

```{code-cell} python
reg = ac.test_functions.builtin_models()
for name in reg:
    mdl = reg.get(name)
    print(f"{name:24s} m = {mdl.dim:2d}  {mdl.space.kind:8s}  {mdl.description}")
```

- **Engineering functions**: borehole, OTL circuit, piston, robot arm and wing weight, in physical
  units on the ranges of Surjanovic and Bingham, on uniform spaces. (All eight borehole inputs
  are uniform; the 2016 tutorial version had a Gaussian $r_w$ and a log-normal $r$, see the
  {doc}`migration guide <../migration>`.) They are also importable as
  constants (`ac.test_functions.WING_WEIGHT`) and as plain functions of physical points
  (`ac.test_functions.wing_weight(X)`, `ac.test_functions.wing_weight_grad(X)`).
- **Known-answer problems**: `ac.test_functions.quadratic(eigenvalues)` builds
  $f(x) = \tfrac12 x^\top H x$ whose matrix $C$ has exactly the given eigenvalues, and
  `ac.test_functions.ridge(direction, profile=…)` builds $f(x) = h(a^\top x)$. Their reference
  results are exact. The registry holds four of them in ten dimensions.

`model.reference_results` holds the eigenvalues and eigenvectors of $C$ in normalized coordinates
and the mean and variance of $f$; for the engineering functions these are Monte Carlo estimates
with standard errors.

```{code-cell} python
q = ac.test_functions.quadratic([1.0, 0.1, 0.01])
rr = q.reference_results
print(rr.exact, rr.eigenvalues)
pist = reg.get("piston")
print(pist.space.names, pist.space.units)
print(f"mean {pist.reference_results.mean:.5f} +/- {pist.reference_results.mean_se:.5f}")
```

`registry.with_models(...)` returns a new registry with your models added (`replace=True` to
override a name), and `registry.describe()` gives every entry's metadata as JSON-compatible
dicts. An unknown name raises `ac.InputError` with code `input.unknown_model`.

```{code-cell} python
mine = reg.with_models(model)
print("toy" in mine.names(), len(mine.names()))
```

API reference: {mod}`active_subspaces.models`, {mod}`active_subspaces.registry`, {mod}`active_subspaces.test_functions`.

## References

- S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments: Test Functions and
  Datasets. <https://www.sfu.ca/~ssurjano/>.
