"""Pytest setup for the doctests in ``src/active_subspaces`` (``pytest --doctest-modules src``).

With ``--import-mode=importlib`` (set in ``pyproject.toml``), pytest imports a module
given by path under its canonical name only if that name is already in
``sys.modules``; otherwise it loads a second copy, whose result classes then
replace the registered ones. Importing the package here first makes every
doctest run against the one copy of each module, whether a single file or the
whole directory is collected.

``active_subspaces.experimental`` issues ``ExperimentalWarning`` once, on first
import. It is imported here with that warning ignored, so collecting its
module does not fail under ``filterwarnings = error``.

This file is not part of the installed package.
"""

import warnings

import active_subspaces  # canonical modules in sys.modules before collection
import active_subspaces.plotting
import active_subspaces.test_functions
from active_subspaces.errors import ExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ExperimentalWarning)
    import active_subspaces.experimental
    import active_subspaces.experimental.gradfree  # noqa: F401
