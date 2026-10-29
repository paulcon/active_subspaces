"""Experimental features; their API and results may change in any release.

Importing this package issues an :class:`~active_subspaces.ExperimentalWarning`
(code ``experimental.module``) once per process. ``import active_subspaces``
never imports it; loading a saved experimental result with
:func:`~active_subspaces.load` does, and then issues the warning.

Contents: :func:`sketched_active_subspace` and :func:`subspace_from_sketches`
estimate an active subspace from :math:`k < m` directional derivatives per
sample and return a :class:`SketchedSubspace`.

Examples
--------
>>> import warnings
>>> import active_subspaces as ac
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore", ac.ExperimentalWarning)
...     from active_subspaces import experimental
>>> experimental.__all__
('SketchedSubspace', 'sketched_active_subspace', 'subspace_from_sketches')
"""

from active_subspaces.errors import ExperimentalWarning, _warn

_warn(
    ExperimentalWarning,
    "active_subspaces.experimental is experimental; its API and results may change in any "
    "release.",
    code="experimental.module",
)

from active_subspaces.experimental.gradfree import (  # noqa: E402 (after the import-time warning)
    SketchedSubspace,
    sketched_active_subspace,
    subspace_from_sketches,
)

__all__ = ("SketchedSubspace", "sketched_active_subspace", "subspace_from_sketches")
