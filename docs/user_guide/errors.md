---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Errors, warnings and codes

The library reports problems in two ways. An **error** means it could not compute what you asked
for. A **warning** means it computed a result, but the result may be less accurate than you expect
or the method is used outside its assumptions. Nothing is printed.

Every error and warning carries a stable `code` of the form `family.name`, such as
`input.shape` or `validity.assumption`, and a JSON-compatible `details` dict. Codes are permanent:
a code is never renamed or reused, so you can rely on them in your own code. The
{doc}`table of codes <../codes>` lists all of them.

```{code-cell} python
import warnings
import numpy as np
import active_subspaces as ac

wing = ac.test_functions.WING_WEIGHT
space = wing.space
X = space.sample(100, rng=0)
f = ac.evaluate(wing, X=X)
```

## Errors

```text
ActiveSubspacesError(Exception)
├── InputError(ValueError)              input.*        bad arrays, parameters, spaces, sample counts
├── SchemaError(ValueError)             schema.*       a saved result cannot be read
├── UnsupportedError(NotImplementedError)  unsupported.*  not defined for this kind of input
├── NumericalError(RuntimeError)        numerical.*    a factorization failed, or non-finite values
├── SolverError(RuntimeError)           solver.*       an optimization backend failed
└── ModelError(RuntimeError)            model.*        your model raised or returned bad values
```

Each class also derives from the builtin exception shown, so `except ValueError` still catches an
`InputError`. Catch by class, then branch on the code if you need to:

```{code-cell} python
try:
    ac.active_subspace(grads=np.ones((5, 3)), space=space)     # wrong number of columns
except ac.InputError as err:
    print(err.code)
    print(err.details)
    print(err)                                  # the message, with the code in brackets
```

Data are checked on entry. NaN or infinite values (`input.nonfinite`), points outside the space
(`input.out_of_space`) and shape mismatches are errors; the library never propagates NaN silently
and never modifies your arrays.

The `input.out_of_space` message counts the rows outside the support and names the first one. Its
`details` hold `param`, `n_outside`, `row`, `column` (also `coordinate`), `name` (from the space),
`value`, `bound` (in physical units) and `side` (`"lower"` or `"upper"`), all plain Python values.

```{code-cell} python
Xo = X.copy()
Xo[7, 0] = space.upper[0] * 1.1
try:
    ac.active_subspace(grads=np.ones_like(Xo), X=Xo, space=space)
except ac.InputError as err:
    print(err)
    print(err.details["n_outside"], err.details["name"], err.details["side"], err.details["bound"])
```

## Warnings

```text
ActiveSubspacesWarning(UserWarning)
├── ValidityWarning                     validity.*     a method used outside its assumptions
├── AccuracyWarning                     accuracy.*     the result is valid but likely degraded
├── ConvergenceWarning                  convergence.*  an iteration stopped early, or some starts failed
├── ExperimentalWarning                 experimental.* the experimental module was imported
└── ActiveSubspacesDeprecationWarning   deprecated.*   a name or default will change (also a DeprecationWarning)
```

Warnings are issued with Python's `warnings` module and point at the line of your code that made
the call. A warning instance carries `code` and `details` like an error:

```{code-cell} python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ac.active_subspace(X=X, f=f, space=space, estimator="ols")
w = caught[0].message
print(type(w).__name__, w.code)
print(w.details["cells"])
```

### Filtering warnings

Filter by class with the standard `warnings` filters:

```python
warnings.filterwarnings("ignore", category=ac.ValidityWarning)
```

To filter one code, match the end of the message, which always ends with the code in brackets:

```{code-cell} python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    warnings.filterwarnings("ignore", message=r".*\[validity\.assumption\]$")
    ac.active_subspace(X=X, f=f, space=space, estimator="ols")
print(len(caught), "warnings shown")
```

Use a `with warnings.catch_warnings():` block, as here, to keep a filter local. To make a warning
fatal, for example in a test or a production pipeline, turn it into an error:

```{code-cell} python
with warnings.catch_warnings():
    warnings.filterwarnings("error", category=ac.ValidityWarning)
    try:
        ac.active_subspace(X=X, f=f, space=space, estimator="ols")
    except ac.ValidityWarning as exc:
        print("stopped:", exc.code)
```

Each warning is issued at most once per call, with counts in `details` where several items are
affected.

## Logging

The library logs progress and diagnostics to the `logging` logger `"active_subspaces"`: the steps of
`build_reduced_model` at `INFO` level, and details such as slice sizes or zeroed gradient rows at
`DEBUG` level. It installs no handler, so nothing appears unless you configure logging:

```python
import logging
logging.basicConfig()
logging.getLogger("active_subspaces").setLevel(logging.DEBUG)
```

API reference: {mod}`active_subspaces.errors`.
