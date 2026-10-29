# Tutorials

Each tutorial is a Jupyter notebook that runs from top to bottom in a few seconds, with fixed
seeds and only the built-in test functions or synthetic data. The notebook files are in the
`docs/tutorials` directory of the repository; the pages below show them with their output.

```{toctree}
:maxdepth: 1

first_active_subspace
values_only
physical_inputs
exploiting_the_subspace
reduced_model
reproducibility
```

- {doc}`first_active_subspace`: the wing weight function from its gradients: bootstrap,
  choosing $n$, and the standard plots.
- {doc}`values_only`: OLS, QPHD, SIR and local linear gradients on a data set of values,
  compared by subspace distance.
- {doc}`physical_inputs`: physical and normalized coordinates, gradients and directions, the
  effect of the input density, and saving results with their provenance (borehole function).
- {doc}`exploiting_the_subspace`: the active domain, surrogates, integration with standard
  errors, and optimization with the lift (wing weight function).
- {doc}`reduced_model`: `build_reduced_model` and its average, probability and minimum (OTL
  circuit).
- {doc}`reproducibility`: `rng`, replaying a result from its provenance, and determinism under
  parallel evaluation.
