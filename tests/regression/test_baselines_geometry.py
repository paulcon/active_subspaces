"""v1 active domains against the D-37 baselines ``geometry/*`` (spec 17 §5.5; V08.17; [C-20]).

v1 builds its own orthonormal basis of ``range(W1)``, so its vertices are mapped into the
2016 coordinates with ``P = W1_v1^T W1_2016`` before the set comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from helpers.baselines import assert_matches_baseline, load_baseline


def _domain(W1: np.ndarray) -> tuple:
    sub = asub.subspace_from_basis(W1, space=asub.Uniform.reference(W1.shape[0]))
    dom = asub.active_domain(sub)
    P = np.asarray(sub.W1).T @ W1
    return dom, np.asarray(dom.vertices) @ P


@pytest.mark.parametrize(
    ("case", "name"), [("interval", "endpoints"), ("zonotope_m3n2", "vertices")]
)
def test_vertices(case, name):
    b = load_baseline("geometry", case)
    dom, Y = _domain(np.asarray(b.inputs["W1"]))
    assert_matches_baseline(b, name, Y.ravel() if case == "interval" else Y)
    assert_matches_baseline(b, "preimages", np.asarray(dom.vertex_preimages))


def test_zonotope_inert():
    """[C-20] ``hull_of rows``: 2016 returned 16 rows (searching for nzv(5, 2) = 10); v1
    returns the 8 hull vertices.
    """
    b = load_baseline("geometry", "zonotope_inert")
    _dom, Y = _domain(np.asarray(b.inputs["W1"]))
    assert Y.shape[0] == asub.zonotope_vertex_count(4, 2)
    assert_matches_baseline(b, "rows", Y)
