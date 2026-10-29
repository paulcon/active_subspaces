# User guide

Each page covers one task, with short examples that run as written. The examples use the built-in
test functions and fixed seeds, so their output is reproducible.

A typical analysis follows the pages in order:

1. Describe the inputs with an **input space**, in physical units.
2. **Discover** the active subspace from gradients, from values, or from data you already have,
   and **choose its dimension** $n$.
3. **Exploit** it: the active domain and the maps between $x$ and $y$, a surrogate on the active
   variables, integration, optimization, or all of these packaged as a reduced model.

The remaining pages cover what every step shares: models and executors, randomness, errors and
warnings, saving results, and the problem sizes the library supports.

Two coordinate systems appear throughout. **Physical** coordinates $\xi$ are the units of your
model. **Normalized** coordinates $x$ are those of the reference density: $[-1,1]^m$ for uniform
inputs and $N(0, I_m)$ for Gaussian inputs. You pass physical data; the library computes and
stores results in normalized coordinates, and every page says which one an array is in.

```{toctree}
:maxdepth: 1

input_spaces
discovering
choosing_n
geometry
surrogates
integration
optimization
reduced_model
models
randomness
errors
saving
problem_sizes
```
