# API reference

The reference has one page per public module. Each page lists exactly the names in that module's
`__all__`; a name that is not listed there is private and carries no compatibility promise.

Most names are also available at the top level of the package:

```python
import active_subspaces as ac

sub = ac.active_subspace(grads=G, space=space, rng=0)
```

The top-level names are documented under the module that defines them; the
[table below](#top-level-names) gives the module for each one. Some names stay module-qualified,
such as `ac.reduced_model.average` and the estimator classes `ac.subspaces.SIR`. Two modules are
loaded on first use rather than at `import active_subspaces`: `ac.plotting` (which imports
matplotlib) and `ac.test_functions`. The experimental module is never loaded by the package; import
it explicitly.

## Data, errors and results

```{eval-rst}
.. autosummary::
   :toctree: generated

   active_subspaces.spaces
   active_subspaces.results
   active_subspaces.errors
```

## Models and test problems

```{eval-rst}
.. autosummary::
   :toctree: generated

   active_subspaces.models
   active_subspaces.registry
   active_subspaces.test_functions
```

## Discovering a subspace

```{eval-rst}
.. autosummary::
   :toctree: generated

   active_subspaces.gradients
   active_subspaces.subspaces
```

## Exploiting a subspace

```{eval-rst}
.. autosummary::
   :toctree: generated

   active_subspaces.geometry
   active_subspaces.designs
   active_subspaces.surrogates
   active_subspaces.integration
   active_subspaces.optimization
   active_subspaces.reduced_model
   active_subspaces.solvers
```

## Plotting

```{eval-rst}
.. autosummary::
   :toctree: generated

   active_subspaces.plotting
```

## Experimental

```{eval-rst}
.. autosummary::
   :toctree: generated

   active_subspaces.experimental
```

(top-level-names)=
## Top-level names

Every name of `active_subspaces.__all__` other than the modules and `__version__`, with the page
that documents it.

```{as-toplevel-table}
```
