# Authors and contributors

Maintainer: **Paul G. Constantine**.

The order below is the order of the software authors in [CITATION.cff](CITATION.cff): the
authors of the 2016 JOSS article in their paper order, then the other 2016 contributors by number
of commits, then contributors to version 1.0 as they join. Contributions are summarized from the
git history of the 2016 `master` and `dev` branches (git tags `legacy-master` and `legacy-dev`).

## The 2016 library (version 0.1)

- **Paul G. Constantine**: creator and principal author. The methods, the design of the library
  and most of its code on both branches.
- **Ryan Howard** (JOSS 2016 author): the tutorial test functions (borehole, OTL circuit, piston,
  robot arm, wing weight). On the `dev` branch: alternative subspace estimators, multiprocessing
  for the simulation runners, the PyMinkSum notebook, and edits to the gradient-free subspaces.
- **Andrew Glaws** (JOSS 2016 author): the sufficient dimension reduction methods, in Python and
  MATLAB.
- **Zachary Grey** (JOSS 2016 author): the MATLAB version of the library, including its gradient-free
  methods.
- **Paul Diaz** (JOSS 2016 author): the alternative-subspace tutorials and their MATLAB
  implementation, and work on the tutorial test functions.
- **Leslie Fletcher** (JOSS 2016 author): the test suite and its data, the documentation, packaging
  and continuous integration.
- **Jeffrey M. Hokanson** (`dev` branch): a Python port of the ECNoise noise estimator, a
  refactoring of the simulation runners, and a Celery backend for them; edits to the gradient-free
  subspaces. These parts of the 2016 code are not carried into version 1.0.
- **Marco Tezzele**: tests of polynomial approximation, the default logging handler, and the
  contribution guidelines of the 2016 README.
- **Chris Maes**: a fix to the Gurobi interface.

## Version 1.0

- **Paul G. Constantine**: direction, design rulings and review.

Version 1.0 was developed with AI assistance (Anthropic's Claude) and reviewed by the maintainer.
AI tools are not listed as authors.

Code ported from the 2016 `dev` branch keeps its authors through `Co-authored-by` commit trailers.
New contributors are added here, and to `CITATION.cff`, with their first merged contribution.
