"""Stored Monte Carlo reference results against a fresh sample (spec 12 §3.4, §8; V12.13).

Construction 2 of spec 17 §3.3, Bonferroni k = 2 (z = 5.03), with
se^2 = se_fresh^2 + se_stored^2; fixed seed 1 [D-78]; guards D-126.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import test_functions as tf

M_FRESH = 2 * 10**4
Z2 = 5.03


@pytest.mark.parametrize("name", ["borehole", "otl_circuit", "piston", "robot_arm", "wing_weight"])
def test_reference_results_fresh_sample(name):
    model = tf.builtin_models()[name]
    rr = model.reference_results
    space = model.space
    X = space.from_normalized(space.sample(M_FRESH, rng=1, normalized=True))
    g = space.grads_to_normalized(asub.evaluate_grads(model, X=X))
    _, W = np.linalg.eigh(g.T @ g / M_FRESH)
    W = W[:, ::-1]
    S = (g @ W[:, :2]) ** 2
    lam = S.mean(axis=0)
    se = S.std(axis=0, ddof=1) / math.sqrt(M_FRESH)
    for k in range(2):
        bound = Z2 * math.sqrt(se[k] ** 2 + rr.eigenvalue_se[k] ** 2)
        assert abs(lam[k] - rr.eigenvalues[k]) <= bound, (
            name,
            k,
            lam[k],
            rr.eigenvalues[k],
            bound,
        )
