# active_subspaces

Python tools for discovering and exploiting **active subspaces**: low-dimensional structure in a
function of many inputs.

The active subspace of a function $f$ of $m$ inputs is spanned by the leading eigenvectors of

$$
C = \int \nabla f(x)\, \nabla f(x)^\top \rho(x)\, dx,
$$

the average outer product of the gradient with respect to the normalized inputs $x$, whose density
is $\rho$. Perturbing $x$ along these eigenvectors changes $f$ more, on average, than perturbing it
along the others. When a few eigenvalues dominate, integration, optimization and surrogate
construction can work with a few active variables instead of all $m$ inputs.

```bash
pip install "active_subspaces @ git+https://github.com/paulcon/active_subspaces@main"
```

- [Getting started](getting_started.md): install and a first analysis.
- [User guide](user_guide/index.md): the library task by task.
- [Theory](theory/index.md): the methods, their assumptions and how to choose an estimator.
- [Tutorials](tutorials/index.md): worked analyses of the built-in test problems.
- [API reference](api/index.md) and the [error and warning codes](codes.md).
- [Migrating from the 2016 package](migration.md).

```{toctree}
:maxdepth: 2
:hidden:

getting_started
user_guide/index
theory/index
tutorials/index
api/index
codes
migration
changelog
citing
contributing
```
