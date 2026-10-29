---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Randomness and reproducibility

Every function that draws random numbers takes a keyword argument `rng`, and no function uses any
other source of randomness. The library never calls `np.random.seed` and never reads NumPy's
global random state, so seeding that state has no effect on it.

```{code-cell} python
import numpy as np
import active_subspaces as ac

wing = ac.test_functions.builtin_models().get("wing_weight")
space = wing.space
X = space.sample(200, rng=0)
G = ac.evaluate_grads(wing, X=X)
```

## The `rng` argument

`rng` follows the Scientific Python convention (SPEC 7); its meaning is that of
`np.random.default_rng(rng)`:

| `rng` | Meaning |
|---|---|
| `None` (default) | fresh entropy from the operating system, recorded in the result's provenance |
| an integer $\ge 0$ | `np.random.SeedSequence(rng)` |
| a `np.random.SeedSequence` | used as given |
| a `np.random.Generator` | used as given; your generator advances |

A legacy `np.random.RandomState`, a bool or a negative integer raises `ac.InputError`.

The functions that take `rng` are `space.sample`, `active_subspace` (only when `nboot > 0`),
`inverse_map`, `active_design` (on a zonotope), `conditional_samples`,
`fit_surrogate_from_model`, `active_quadrature_rule` (on a zonotope), `integration_design`,
`integrate`, `integrate_surrogate`, `minimize_active`, `minimize`, `build_reduced_model` and the
reduced model's `average`, `probability` and `minimum`. Everything else is deterministic and has
no `rng`.

```{code-cell} python
a = ac.active_subspace(grads=G, space=space, nboot=100, rng=7)
b = ac.active_subspace(grads=G, space=space, nboot=100, rng=np.random.default_rng(7))
print(a == b)                                  # an int and default_rng(int) are the same
```

## Seeding a whole analysis

Passing the same integer to every call reuses one stream for unrelated purposes. Instead, spawn
one child `SeedSequence` per step from a single root seed:

```{code-cell} python
root = np.random.SeedSequence(20261029)
s_sample, s_boot, s_inverse = root.spawn(3)

X1 = space.sample(200, rng=s_sample)
sub = ac.active_subspace(grads=ac.evaluate_grads(wing, X=X1), space=space, nboot=100, rng=s_boot)
inv = ac.inverse_map(sub, Y=[[0.5]], n_draws=4, rng=s_inverse)
print(sub.n, inv.X.shape)
```

Passing one `Generator` to several calls also works; each call then advances it, so the calls get
different numbers and the results depend on the order of the calls.

## Independent tasks get child streams

Inside a function, independent tasks such as bootstrap replicates, optimizer starts, inverse-map
chains for different points, and integration replicates each get their own child stream, spawned
from the function's generator before any draw. The number of children depends only on the problem
(for example `nboot`), never on the number of workers. Results are reduced in task order, so a
result is the same with or without an executor, for any number of workers:

```{code-cell} python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    c = ac.active_subspace(grads=G, space=space, nboot=100, rng=7, executor=pool)
print(c == a)
```

## Replaying a run

With `rng=None` the library draws fresh entropy, and it records the generator's state on entry in
`result.provenance.rng`, so you can repeat the run later.

```{code-cell} python
fresh = ac.active_subspace(grads=G, space=space, nboot=100)    # rng=None
fresh.provenance.rng
```

There is no replay function in the library. This recipe rebuilds the generator the function saw
on entry. A description of kind `"seed_sequence"` (from `rng=None`, an integer or a
`SeedSequence`) holds the seed sequence. One of kind `"generator"` (from a `Generator`) also holds
the generator's state at entry, whose integers are stored as decimal strings so that they survive
JSON; they are converted back before the state is restored.

```{code-cell} python
def _ints(obj):
    """Decimal strings back to integers, recursively (the generator name stays a string)."""
    if isinstance(obj, dict):
        if "__uint_array__" in obj:
            return np.array([int(v) for v in obj["data"]], dtype=obj["__uint_array__"])
        return {k: v if k == "bit_generator" else _ints(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ints(v) for v in obj]
    if isinstance(obj, str) and obj.lstrip("-").isdigit():
        return int(obj)
    return obj


def replay_rng(desc):
    """Rebuild the generator described by a result's provenance.rng."""
    ss = desc if desc["kind"] == "seed_sequence" else desc["seed_sequence"]
    bit_generator = getattr(np.random, desc["bit_generator"])
    if ss is None:  # a generator without a seed sequence: restore the state only
        bg = bit_generator()
    else:
        ent = ss["entropy"]                      # a decimal string, or a list of them
        entropy = int(ent) if isinstance(ent, str) else [int(e) for e in ent]
        seq = np.random.SeedSequence(
            entropy,
            spawn_key=tuple(ss["spawn_key"]),
            n_children_spawned=ss["n_children_spawned"],
            pool_size=ss.get("pool_size", 4),
        )
        bg = bit_generator(seq)
    if desc["kind"] == "generator":              # rng was a Generator: restore its state too
        bg.state = _ints(desc["state"])
    return np.random.Generator(bg)


again = ac.active_subspace(grads=G, space=space, nboot=100, rng=replay_rng(fresh.provenance.rng))
print(again == fresh)
```

The same recipe works when the original call was given a `Generator`:

```{code-cell} python
gen = np.random.default_rng(3)
gen.standard_normal(5)                          # advance it, as earlier work would
r1 = ac.active_subspace(grads=G, space=space, nboot=50, rng=gen)
r2 = ac.active_subspace(grads=G, space=space, nboot=50, rng=replay_rng(r1.provenance.rng))
print(r1.provenance.rng["kind"], r1 == r2)
```

A deterministic function, or `active_subspace` with `nboot=0`, records `provenance.rng = None`.

## What is reproducible

- **Bitwise.** Result fields are bitwise identical when the inputs, the `rng` description, the
  versions of the library, NumPy, SciPy and Python, the platform, the BLAS build and the BLAS
  thread count all match. The provenance timestamp differs.
- **Workers.** For any executor and any number of workers, results equal sequential execution.
- **BLAS.** A multithreaded BLAS may change the order of sums; the promise holds for a fixed
  thread count. Some BLAS builds (Apple Accelerate is one) round differently depending on where
  their operands sit in memory. On such a build, some fields can differ in the last bits between
  runs, and agree only to the library's numerical tolerances.
- **Across platforms, BLAS builds or library versions**: agreement to tolerances only. A change
  to how a function uses its random streams is noted in the changelog.

Results comparisons with `==` are bitwise, field by field, excluding provenance.
