# Contributing to active_subspaces

Thank you for helping. Bug reports, documentation fixes, tests and new features are all welcome.
Everyone who takes part follows the
[code of conduct](https://github.com/paulcon/active_subspaces/blob/main/CODE_OF_CONDUCT.md).

The maintainer is **Paul G. Constantine**. Report security problems privately as described in
[SECURITY.md](https://github.com/paulcon/active_subspaces/blob/main/SECURITY.md), not in an issue.

## Before you start

- For a bug, open an issue with a minimal example that uses a fixed `rng`.
- For a new feature or a change to an existing method, open an issue first to discuss it.
- The design records in `design-specs/` explain why the library is the way it is.
  `decisions.md` holds the maintainer's rulings (`D-` IDs) and `findings-register.md` the defects
  and design inputs found in the 2016 code (`P-`, `C-`, `N-`, `R-`, `A-` IDs).

## Development setup

You need Python 3.11 or later and pip 25.1 or later (for `--group`).

```sh
git clone https://github.com/paulcon/active_subspaces.git
cd active_subspaces
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install --group dev -e .
pre-commit install                 # optional: runs ruff and the file checks on each commit
```

## Running the checks

Tests (set `MPLBACKEND=agg` if you have no display):

```sh
pytest -m "not slow"               # the default CI run
pytest -m slow                     # tests that take more than 5 s
pytest tests/unit                  # one tier: unit, known_answer, property, statistical,
                                   # regression, acceptance
pytest --stat-seeds=random         # statistical checks with fresh seeds; a failure prints the
                                   # entropy, and --stat-seeds=<entropy> replays it
```

Lint, format and type checks:

```sh
ruff check src tests
ruff format src tests
mypy
lint-imports
validate-pyproject pyproject.toml
cffconvert --validate
```

Documentation (the tutorials and code pages are executed during the build):

```sh
sphinx-build -W --keep-going -b html docs docs/_build/html
pytest --doctest-modules src/active_subspaces \
    -o doctest_optionflags="NORMALIZE_WHITESPACE ELLIPSIS NUMBER" \
    -W "ignore::active_subspaces.errors.ExperimentalWarning"
```

## Pull requests

- **One change per pull request**, on its own branch.
- **Cite the IDs** it resolves (finding, decision or question IDs, or the issue number) in the
  description. The pull-request template lists what to fill in.
- **Add tests** in the right tier: `unit` (contracts), `known_answer` (analytic or published
  values), `property` (invariances), `statistical` (stochastic claims with a controlled
  false-failure rate), `regression` (stored baselines), `acceptance` (end-to-end use cases). Mark a
  test `slow` if it takes more than 5 s. Tests use explicit seeds, never the global NumPy random
  state, and write no files.
- **Add a changelog entry** under "Unreleased" in
  [CHANGELOG.md](https://github.com/paulcon/active_subspaces/blob/main/CHANGELOG.md), in the
  right section (Added, Changed, Changed results, Deprecated, Removed, Fixed, Security), with its
  IDs. A change in computed values for the same inputs goes under "Changed results".
- **Update the documentation**: numpydoc docstrings for every public function (state whether each
  array is in physical or normalized coordinates), the user guide, and the migration guide if a
  public name changes.
- Keep the public API stable within a major version; see the versioning and deprecation policy in
  the documentation.

## Changes to the mathematics

A change to the mathematics of a method (an estimator, a formula, a default that changes results,
or any departure from the published method) needs a ruling by the maintainer before it is merged.
Open an issue that states the current formula, the proposed formula, a published reference, and the
effect on results. The ruling is recorded in `design-specs/decisions.md` and cited in the pull
request. This applies to bug fixes as well when the fix changes what a method computes.

## Credit

- Contributors are listed in [AUTHORS.md](https://github.com/paulcon/active_subspaces/blob/main/AUTHORS.md) and
  [CITATION.cff](https://github.com/paulcon/active_subspaces/blob/main/CITATION.cff); add yourself in your
  first pull request, or ask the maintainer to.
- Code ported from another author, including the 2016 `dev` branch, keeps that author: use a
  `Co-authored-by: Name <email>` trailer in the commit message.

## AI assistance

State in the pull-request description if you used AI tools substantially, and for what. Commits
written with AI assistance carry a `Co-Authored-By` trailer that names the tool, as the project's
own commits do. You remain responsible for the code you submit: review it, test it, and check every
reference it cites. AI tools are not listed as authors.
