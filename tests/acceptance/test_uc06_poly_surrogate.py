"""UC-06: polynomial surrogate on a 1-D active variable (spec 01 §5; spec 17 V01.30–V01.34).

``R7``; KA-2 with ``w = c_1/‖c_1‖`` (UC-02) and ``h(y) = 1 + 2y − 3y²``; M = 50.
"""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from helpers.tolerances import assert_close
from helpers.usecases import R7, physical_model, space

W = np.array([1, 0.5, 0, -0.3, 0, 0.1, 0]) / np.linalg.norm([1, 0.5, 0, -0.3, 0, 0.1, 0])


def h(y):
    return 1 + 2 * y - 3 * y**2


MODEL, F = physical_model("R7", lambda x: h(x @ W), lambda x: np.outer(2 - 6 * (x @ W), W))


@pytest.fixture
def setup(stat_rng):
    X = space("R7").sample(50, rng=stat_rng("UC-06/sample"))
    f, G = asub.evaluate_with_grads(MODEL, X=X)
    sub = asub.active_subspace(grads=G, space=space("R7"))
    return X, f, sub


def _fit(X, f, sub, **kw):
    return asub.fit_surrogate(subspace=sub, X=X, f=f, kind="poly", degree=2, **kw)


def test_fit(setup):
    """V01.30 [D-97]: ``loo_r_squared >= 1 − 1e-12``; residuals <= 1e-12 max|f| (F12). ``f`` as
    (M,) or (M, 1) [A-04].
    """
    X, f, sub = setup
    assert sub.n == 1
    surr = _fit(X, f, sub)
    assert _fit(X, f[:, None], sub) == surr
    assert surr.loo_r_squared >= 1 - 1e-12
    assert np.max(np.abs(surr.residuals)) <= 1e-12 * np.max(np.abs(f))


def test_active_values(setup):
    """V01.31: ``surr.evaluate`` at 21 active coordinates equals h(s y), s = sign(w^T ŵ_1)
    (1e-9 max|h|).
    """
    X, f, sub = setup
    s = np.sign(W @ sub.W1[:, 0])
    y = np.linspace(-1.0, 1.0, 21)
    assert_close(_fit(X, f, sub).evaluate(y[:, None]), h(s * y), rtol=1e-9)


def test_prediction(setup, stat_rng):
    """V01.31 [D-104]: ``predict`` on 100 new physical points equals f (1e-9 max|f|)."""
    X, f, sub = setup
    X_new = space("R7").sample(100, rng=stat_rng("UC-06/new_points"))
    assert_close(asub.predict(_fit(X, f, sub), X_new), F(X_new), rtol=1e-9)


def _noisy(setup, stat_rng):
    X, f, sub = setup
    e = np.random.default_rng(stat_rng("UC-06/noise")).standard_normal(len(f))
    X_new = space("R7").sample(100, rng=stat_rng("UC-06/new_points"))
    return X, f + 0.1 * e, sub, X_new


def test_replication_equivalence(setup, stat_rng):
    """V01.32 [C-04], [D-60]: integer weights k_i ≡ rows repeated k_i times (√w row scaling);
    rel. 1e-10.
    """
    X, fn, sub, X_new = _noisy(setup, stat_rng)
    k = 1 + np.arange(len(fn)) % 3
    weighted = asub.predict(_fit(X, fn, sub, sample_weights=k), X_new)
    repeated = asub.predict(_fit(np.repeat(X, k, axis=0), np.repeat(fn, k), sub), X_new)
    assert_close(weighted, repeated, rtol=1e-10)


def test_weight_scale_and_zero(setup, stat_rng):
    """V01.33: weights 7k ≡ k (F12); zero weights on rows 40–49 ≡ dropping them (F10)."""
    X, fn, sub, X_new = _noisy(setup, stat_rng)
    k = 1.0 + np.arange(len(fn)) % 3
    base = asub.predict(_fit(X, fn, sub, sample_weights=k), X_new)
    assert_close(asub.predict(_fit(X, fn, sub, sample_weights=7 * k), X_new), base, rtol=1e-12)
    w = np.ones(len(fn))
    w[40:50] = 0.0
    zero = asub.predict(_fit(X, fn, sub, sample_weights=w), X_new)
    dropped = asub.predict(_fit(X[:40], fn[:40], sub), X_new)
    assert_close(zero, dropped, rtol=1e-10)


def test_too_few(setup):
    """V01.34: degree 10 with M = 5 -> ``input.too_few_samples``."""
    X, f, sub = setup
    with pytest.raises(asub.InputError) as e:
        asub.fit_surrogate(subspace=sub, X=X[:5], f=f[:5], kind="poly", degree=10)
    assert e.value.code == "input.too_few_samples"


def test_space_mismatch(setup):
    """V01.34 [D-76]: a ``space`` different from the subspace's -> ``input.parameter``, reason
    ``space_mismatch``.
    """
    X, f, sub = setup
    other = asub.Uniform(R7["lower"], 2 * np.asarray(R7["upper"]))
    with pytest.raises(asub.InputError) as e:
        _fit(X, f, sub, space=other)
    assert e.value.code == "input.parameter"
    assert e.value.details["reason"] == "space_mismatch"
