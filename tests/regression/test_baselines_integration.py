"""v1 quadrature and active-variable integration against the D-37 baselines (spec 17 §5.5; V10.25).

``quadrature/gauss``: 2016 Gauss-Hermite and Gauss-Legendre rules (probability weights), k = 1..10,
F14.
``integration/av_unbounded``: ``av_integrate`` of a degree-3 polynomial on n = 1, Gaussian, 10
nodes, F12.
"""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import _quadrature as Q
from helpers.baselines import assert_matches_baseline, load_baseline


@pytest.mark.parametrize(
    ("label", "rule"), [("hermite", Q.gauss_hermite), ("legendre", Q.gauss_legendre)]
)
def test_gauss_rules(label, rule):
    b = load_baseline("quadrature", "gauss")
    rules = [rule(int(k)) for k in b.inputs["k"]]
    assert_matches_baseline(b, f"{label}_nodes", np.concatenate([t for t, _ in rules]))
    assert_matches_baseline(b, f"{label}_weights", np.concatenate([w for _, w in rules]))


def test_av_unbounded():
    b = load_baseline("integration", "av_unbounded")
    W1 = np.asarray(b.inputs["W1"])
    sub = asub.subspace_from_basis(W1, space=asub.Gaussian.reference(W1.shape[0]))
    s = asub.fit_surrogate(X=b.inputs["X"], f=b.inputs["f"], subspace=sub, kind="poly", degree=3)
    res = asub.integrate_surrogate(
        s, domain=asub.active_domain(sub), n_nodes=int(b.inputs["n_nodes"])
    )
    assert_matches_baseline(b, "estimate", np.float64(res.estimate))


def test_bounded_weights_not_stored():
    """The 2016 interval weights are Monte Carlo histograms, replaced by deterministic weights
    [N-10], [D-47].
    """
    b = load_baseline("integration", "bounded_weights")
    assert b.outputs == {}
    assert "weights" in b.meta["not_stored"]
