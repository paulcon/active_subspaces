---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting started

## Install

`active_subspaces` needs Python 3.11 or later, NumPy, SciPy and matplotlib.

Until version 1.0 is on PyPI, install it from GitHub; `pip install active_subspaces` still gives the
2016 release.

```bash
pip install "active_subspaces @ git+https://github.com/paulcon/active_subspaces@main"
```

To work from a clone of the repository instead:

```bash
git clone https://github.com/paulcon/active_subspaces.git
cd active_subspaces
pip install -e .
```

Every example in this documentation imports the package as `ac`.

## An analysis in ten lines

The piston cycle time is one of the built-in test problems. It has seven inputs, each uniform on a
physical range, and an analytic gradient. The cell below samples the inputs, evaluates gradients,
estimates the active subspace, plots its eigenvalues, and saves and reloads the result.

```{code-cell} python
:tags: [remove-cell]
%matplotlib inline
```

```{code-cell} python
import os, tempfile
import numpy as np
import active_subspaces as ac

model = ac.test_functions.builtin_models().get("piston")      # an ac.Model on an ac.Uniform space
X = model.space.sample(300, rng=0)                             # (300, 7) physical inputs
G = ac.evaluate_grads(model, X=X)                              # (300, 7) physical gradients
sub = ac.active_subspace(grads=G, space=model.space, nboot=200, rng=1)
fig, ax = ac.plotting.plot_eigenvalues(sub)
with tempfile.TemporaryDirectory() as tmp:
    sub.save(os.path.join(tmp, "piston.npz"))
    assert ac.load(os.path.join(tmp, "piston.npz")) == sub
fig
```

## What happened

- `X` and `G` are in **physical** units: kilograms, square metres, kelvin. The library maps
  them to **normalized** coordinates, where each input ranges over $[-1, 1]$, and computes and
  stores everything there. You never scale gradients by hand.
- `ac.active_subspace` estimated $C = \mathbb E[\nabla f\,\nabla f^\top]$ in normalized
  coordinates from the 300 gradients and took its eigendecomposition. The plot shows the
  eigenvalues on a log scale; the shaded band is a 95% bootstrap percentile interval from the 200
  replicates, and the dotted line marks the chosen dimension.
- The dimension `sub.n` was chosen by the largest gap between consecutive eigenvalues.
- `sub` is an immutable result. It saves to an `.npz` file, and `ac.load` reads it back without
  pickle.

```{code-cell} python
print("n =", sub.n)
print("eigenvalues:", np.array2string(sub.eigenvalues, precision=2))
print("first eigenvector (normalized):", np.round(sub.W1[:, 0], 3))
```

The first eigenvector shows how strongly each normalized input contributes to the dominant
direction. Its components are comparable across inputs because every normalized input has the
same range:

```{code-cell} python
fig, axes = ac.plotting.plot_eigenvectors(sub)
fig
```

The sufficient summary plot shows $f$ against the active variable $y = w_1^\top x$. A tight
one-dimensional trend means that $f$ is close to a function of $y$ alone.

```{code-cell} python
f = ac.evaluate(model, X=X)                   # (300,)
fig, ax = ac.plotting.plot_sufficient_summary(sub, X=X, f=f)
fig
```

## Where to go next

- The {doc}`user_guide/index` explains each step: {doc}`input spaces <user_guide/input_spaces>`,
  {doc}`discovering a subspace <user_guide/discovering>`,
  {doc}`choosing the dimension <user_guide/choosing_n>`, and what to do with the subspace
  ({doc}`surrogates <user_guide/surrogates>`, {doc}`integration <user_guide/integration>`,
  {doc}`optimization <user_guide/optimization>`, the {doc}`reduced model <user_guide/reduced_model>`).
- To analyze your own simulation, wrap it as a model: {doc}`user_guide/models`.
- The {doc}`theory pages <theory/index>` state the mathematics behind each method.
- If you used the 2016 package, read the {doc}`migration guide <migration>`.
