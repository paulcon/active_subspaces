"""v1 maximin design against the D-37 summary baseline ``designs/maximin`` (spec 17 §5.5; V09.11;
G-11).

The 2016 ``maximin_design`` ran on the square zonotope of W1 = [e1 e2] (m = 3) at 20 seeds;
its ``min_distance`` summary is stored. v1's mean over 20 seeds must be at least the 2016
mean, by the one-sided two-sample check of spec 17 §5.5 with z = 5.
"""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from helpers.baselines import assert_matches_baseline, load_baseline


@pytest.mark.parametrize("N", [5, 10])
def test_maximin_vs_2016(N, stat_rng):
    b = load_baseline("designs", "maximin")
    sub = asub.subspace_from_basis(b.inputs["W1"], space=asub.Uniform.reference(3))
    dom = asub.active_domain(sub)
    R = int(b.record(f"min_distance_N{N}")["stats"]["R"])
    seeds = stat_rng(f"N{N}").spawn(R)
    values = np.array(
        [asub.active_design(domain=dom, n_points=N, rng=s).min_distance for s in seeds]
    )
    assert_matches_baseline(b, f"min_distance_N{N}", values)
