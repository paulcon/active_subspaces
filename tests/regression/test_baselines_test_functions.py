"""v1 test functions against the 2016 tutorial functions, baselines ``test_functions/*`` (spec 17
§5.5; V12.3, V12.4; [A-11]).

Values: F14 on the scale max|f| (spec 12: 1e-13 relative). Gradients: F12 relative to each row's
max.
The borehole baseline uses the 2016 inputs (x1, x2 standard normal); the v1 side applies the
migration recipe of spec 12 §3.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_subspaces import test_functions as tf
from helpers.baselines import assert_matches_baseline, load_baseline

MODELS = {
    "otl_circuit": tf.OTL_CIRCUIT,
    "piston": tf.PISTON,
    "robot_arm": tf.ROBOT_ARM,
    "wing_weight": tf.WING_WEIGHT,
}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_vs_2016(name):
    b = load_baseline("test_functions", name)
    model = MODELS[name]
    space = model.space
    X = space.from_normalized(b.inputs["X"])
    assert_matches_baseline(b, "values", model.value(X))
    assert_matches_baseline(b, "grads", space.grads_to_normalized(model.grad(X)))


def test_borehole_migration():
    b = load_baseline("test_functions", "borehole")
    x = np.asarray(b.inputs["X"])
    lo, up = tf.BOREHOLE.space.lower, tf.BOREHOLE.space.upper
    half = 0.5 * (up - lo)
    X = np.empty_like(x)
    X[:, 0] = 0.0161812 * x[:, 0] + 0.1
    X[:, 1] = np.exp(1.0056 * x[:, 1] + 7.71)
    X[:, 2:] = lo[2:] + half[2:] * (x[:, 2:] + 1.0)
    G = tf.borehole_grad(X)
    g = np.empty_like(G)
    g[:, 0] = 0.0161812 * G[:, 0]
    g[:, 1] = 1.0056 * X[:, 1] * G[:, 1]
    g[:, 2:] = half[2:] * G[:, 2:]
    assert_matches_baseline(b, "values", tf.borehole(X))
    assert_matches_baseline(b, "grads", g)
