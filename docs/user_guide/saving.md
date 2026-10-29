---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Saving and loading results

Every function returns an immutable **result**: `SubspaceResult`, `ActiveDomain`,
`IntegrationResult`, `ReducedModel` and the others. Results own read-only copies of their arrays,
compare by value, and convert to JSON-compatible dicts and to `.npz` files. Each carries the space
it was computed in and a record of how it was produced.

```{code-cell} python
import io
import json
import os
import tempfile

import numpy as np
import active_subspaces as ac

wing = ac.test_functions.WING_WEIGHT
X = wing.space.sample(200, rng=0)
sub = ac.active_subspace(grads=ac.evaluate_grads(wing, X=X), space=wing.space, nboot=50, rng=1)
```

## Files: `save` and `load`

`result.save(file)` writes one uncompressed `.npz` file; `ac.load(file)` reads any result back,
whatever its type. `file` is a path or a binary file object.

```{code-cell} python
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "wing_subspace.npz")
    sub.save(path)
    loaded = ac.load(path)
print(type(loaded).__name__, loaded == sub)

buf = io.BytesIO()                             # in memory, e.g. to send over a network
sub.save(buf)
buf.seek(0)
print(ac.load(buf) == sub)
```

`load` never unpickles: it opens the file with `allow_pickle=False`, so a file holding Python
objects is rejected with `ac.SchemaError` (`schema.invalid`) instead of running code. Results
pickle, because process pools need that, but pickle is not a storage format and carries no
compatibility promise. Use `save` and `load`.

## JSON: `to_dict` and `from_dict`

`result.to_dict()` gives a dict that `json.dumps` accepts, with arrays as tagged objects holding
dtype, shape and the flat values. Floats are written so that they decode exactly, and NaN and ±inf
as strings. `ac.from_dict(d)` rebuilds the result.

```{code-cell} python
d = sub.to_dict()
text = json.dumps(d, allow_nan=False)
print(d["type"], d["schema_version"], list(d["fields"])[:6], "...")
print(ac.from_dict(json.loads(text)) == sub)
```

`to_dict(binary=True)` stores the array data as base64 instead of lists, which is smaller:

```{code-cell} python
print(len(text), len(json.dumps(sub.to_dict(binary=True))))
```

Input spaces have their own `to_dict` and `from_dict` (see {doc}`input_spaces`). A `Model` holds
your functions, so it is not serialized; save its results instead.

## Provenance

`result.provenance` records how the result was produced: the function, its parameters, the input
space, the random generator's state on entry, the versions of the library, NumPy, SciPy and
Python, the platform and a timestamp. Input arrays are recorded by shape, dtype and SHA-256 digest,
not by value.

```{code-cell} python
prov = sub.provenance
print(prov.function, prov.library_version, prov.numpy_version)
print(prov.parameters["nboot"], prov.parameters["grads"])
print(prov.rng["kind"], prov.rng["entropy"])
```

The digest lets you check that a saved result came from the data you have:

```{code-cell} python
import hashlib

G = ac.evaluate_grads(wing, X=X)
print(hashlib.sha256(np.ascontiguousarray(G).tobytes()).hexdigest() == prov.parameters["grads"]["sha256"])
```

The rng record is enough to rerun a stochastic call; see {doc}`randomness`. Provenance is excluded
from `==`, so a rerun compares equal to the original.

## Versions

Each result type has a schema version. A file written by an older library version loads in newer
ones, which migrate it. A file from a newer schema than your library knows raises `ac.SchemaError`
with code `schema.too_new`; an unknown result type raises `schema.unknown_type`.

API reference: {mod}`active_subspaces.results`.
