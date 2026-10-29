"""UC-09: reduced model end to end (spec 01 §5; spec 17 V01.45–V01.50).

KA-6 on ``U5``: s = x_1 + x_2, f = s² + s; E f = 2/3, P(f <= 3/4) = 11/16, min f = −1/4.
``accuracy.curvature_clipped`` is allowed in ``minimum`` (f is flat in z); no other warning
(``average`` with a model uses the default n_mc = 50 >= 20, so no ``accuracy.correlated_draws``
[D-114, D-171]).
"""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from helpers.metrics import subspace_distance
from helpers.usecases import allow_warnings, physical_model, space

E12 = np.array([1.0, 1.0, 0, 0, 0]) / np.sqrt(2)


def phi(x):
    s = x[:, 0] + x[:, 1]
    return s**2 + s


def dphi(x):
    s = x[:, 0] + x[:, 1]
    return np.outer(2 * s + 1, [1.0, 1.0, 0, 0, 0])


MODEL, F = physical_model("U5", phi, dphi, model_name="ka6")


def _build(stat_rng):
    return asub.build_reduced_model(model=MODEL, rng=stat_rng("UC-09/build"))


def test_subspace_and_prediction(stat_rng):
    """V01.45: n == 1; dist(W_1, (1,1,0,0,0)/√2) <= 1e-10; predictions equal f to 1e-6
    (max f − min f) (the conditional means lie in the degree-2 tail, so α = 0); holdout
    ``r_squared >= 1 − 1e-10``."""
    rm = _build(stat_rng)
    assert rm.n == 1
    assert subspace_distance(rm.W1, E12) <= 1e-10
    X_test = space("U5").sample(100, rng=stat_rng("UC-09/test_points"))
    f = F(X_test)
    assert np.max(np.abs(rm.predict(X_test) - f)) <= 1e-6 * (f.max() - f.min())
    assert np.array_equal(rm.predict(X_test), asub.predict(rm, X_test))
    assert rm.r_squared_kind == "holdout"
    assert rm.r_squared >= 1 - 1e-10


def test_average(stat_rng):
    """V01.46: the surrogate and the model integrate to 2/3 (rel. 1e-8, F8); the model path has
    ``inactive_std_error <= 1e-12`` (KA-2: a ridge); the surrogate path has none."""
    rm = _build(stat_rng)
    avg = rm.average(n_nodes=5)
    avg_f = rm.average(n_nodes=5, model=MODEL, rng=stat_rng("UC-09/average"))
    assert abs(avg.estimate - 2 / 3) <= 1e-8 * 2 / 3
    assert abs(avg_f.estimate - 2 / 3) <= 1e-8 * 2 / 3
    assert avg_f.inactive_std_error <= 1e-12
    assert avg.inactive_std_error is None


def test_probability(stat_rng):
    """V01.47 [C-06]: T = |p̂ − 0.6875| / std_error, n = 1e5 binomial; construction 1, z = 5;
    ``lo <= p <= hi`` for ``confidence_interval(0.99)`` (the 2016 bounds were swapped)."""
    rm = _build(stat_rng)
    pr = rm.probability(-np.inf, 0.75, n_samples=100_000, rng=stat_rng("UC-09/probability"))
    assert pr.std_error > 0
    assert abs(pr.p - 0.6875) <= 5 * pr.std_error
    lo, hi = pr.confidence_interval(0.99)
    assert lo <= pr.p <= hi


def test_minimum(stat_rng):
    """V01.48: ``f_star`` = −1/4 (surrogate); f(x*) = −1/4 by the test; x* in U5 (1e-6, F6)."""
    rm = _build(stat_rng)
    with allow_warnings("accuracy.curvature_clipped"):
        mn = rm.minimum(rng=stat_rng("UC-09/minimum"))
    assert abs(mn.f_star + 0.25) <= 1e-6
    assert abs(F(mn.x_star_physical[None])[0] + 0.25) <= 1e-6
    assert space("U5").contains(mn.x_star_physical[None])[0]


def test_values_only_model(stat_rng):
    """V01.49: a Model without ``grad``: ``gradient_source == "finite_difference"``, n == 1,
    dist <= 1e-8.
    """
    model_v = asub.Model(space=space("U5"), value=F)
    rm = asub.build_reduced_model(model=model_v, rng=stat_rng("UC-09/build"))
    assert rm.gradient_source == "finite_difference"
    assert rm.n == 1
    assert subspace_distance(rm.W1, E12) <= 1e-8


def test_data_path(stat_rng):
    """V01.49: data only, M = 200: ``gradient_source == "local_linear"``,
    ``r_squared_kind == "loo"`` [D-97];
    finite predictions (smoke)."""
    X = space("U5").sample(200, rng=stat_rng("UC-09/data_path"))
    rm = asub.build_reduced_model(X=X, f=F(X), space=space("U5"), rng=stat_rng("UC-09/data_path"))
    assert rm.gradient_source == "local_linear"
    assert rm.r_squared_kind == "loo"
    assert np.all(np.isfinite(rm.predict(X[:10])))


def test_bounds_and_silence(stat_rng, capsys):
    """V01.50 [C-05], [R-04]: predicting outside U5 -> ``input.out_of_space``; nothing printed
    during the build.
    """
    rm = _build(stat_rng)
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    x_out = np.asarray(space("U5").upper, dtype=float) + 1.0
    with pytest.raises(asub.InputError) as e:
        rm.predict(x_out[None])
    assert e.value.code == "input.out_of_space"
