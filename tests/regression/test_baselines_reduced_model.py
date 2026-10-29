"""v1 reduced-model probability against the dev baseline ``reduced_model/probability`` (spec 17
§5.5; V11.18; [C-06]).

dev ``probability`` returned ``(p, plb, pub)`` with ``plb = p + 2.58 se > pub = p - 2.58 se``.
v1 returns an ordered Wilson interval [D-119]; at the 2016 estimate it is the reversed 2016 pair
up to O(1/N). The v1 estimate itself agrees with the 2016 one within the Monte Carlo error.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import active_subspaces as asub
from active_subspaces import reduced_model as R
from helpers.baselines import assert_matches_baseline, load_baseline


def test_probability_swap(stat_rng):
    b = load_baseline("reduced_model", "probability")
    a = np.asarray(b.inputs["a"])
    lb, ub = (float(v) for v in b.inputs["bounds"])
    N = int(b.inputs["n_samples"])
    m = a.size
    gen = np.random.default_rng(stat_rng("data"))
    X = gen.standard_normal((50, m))
    rm = R.build_reduced_model(
        space=asub.Gaussian.reference(m),
        X=X,
        f=X @ a,
        grads=np.tile(a, (50, 1)),
        surrogate_kind="poly",
        degree=1,
        rng=0,
    )
    res = R.probability(rm, lb, ub, n_samples=N, rng=stat_rng("probability"))
    assert_matches_baseline(b, "p", {"mean": res.p, "sd": res.std_error, "R": 1})
    lo, hi = res.confidence_interval(0.99)
    assert lo <= res.p <= hi
    rel = b.record("interval")["relation"]
    at_2016_p = dataclasses.replace(
        res, p=float(rel["p"]), std_error=float(np.sqrt(rel["p"] * (1 - rel["p"]) / N))
    )
    assert_matches_baseline(b, "interval", np.array(at_2016_p.confidence_interval(rel["level"])))
