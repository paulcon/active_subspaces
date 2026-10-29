# Error and warning codes

Every exception and warning raised by `active_subspaces` carries a stable `code` of the form
`<family>.<name>` and a `details` dictionary with JSON-compatible values. The family matches the
class: `input.*` codes are raised as {py:class}`~active_subspaces.errors.InputError`,
`accuracy.*` codes are issued as {py:class}`~active_subspaces.errors.AccuracyWarning`, and so on.
Codes are permanent: a code is never renamed or given a new meaning, and removing one needs a new
major version. New codes may appear in minor releases. Branch on the code, not on the message text, which may change.

```python
import warnings
import active_subspaces as ac

try:
    sub = ac.active_subspace(grads=G, space=space)
except ac.InputError as err:
    print(err.code, err.details)

with warnings.catch_warnings():
    warnings.simplefilter("error", ac.ValidityWarning)   # turn one warning class into errors
    ...
```

All errors derive from {py:class}`~active_subspaces.errors.ActiveSubspacesError` and all warnings
from {py:class}`~active_subspaces.errors.ActiveSubspacesWarning`. Each error class also derives from
the matching built-in exception (`InputError` is a `ValueError`, `SolverError` a `RuntimeError`), so
existing `except ValueError:` clauses still catch it.

The table is generated from the package's code registry when these pages are built, so it lists
exactly the codes of this release.

```{as-codes-table}
```
