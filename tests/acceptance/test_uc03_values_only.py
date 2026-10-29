"""UC-03: values only: OLS, QPHD, SIR and pHd (spec 01 §5; spec 17 V01.16–V01.21).

Every estimator here except QPHD on ``G5`` and SIR on ``G6`` runs on a uniform space, where spec 07
§3.5
[D-41] issues ``validity.assumption``; the tests require that warning (spec 01 lists only the pHd
one).
"""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from helpers.metrics import subspace_distance
from helpers.tolerances import assert_eigenvalues_close
from helpers.usecases import allow_warnings, expect_warning, ka1, physical_model, space

G_OLS = np.array([3, -1, 0.5, 0, 0.2, 0, -2.0])
Q5, _ = ka1(5, np.ones(5))
G_QPHD = np.array([1, 0, -1, 0.5, 0.0])
H_QPHD = Q5 @ np.diag([1, -0.5, 0.2, 0, 0]) @ Q5
A_SIR = np.array([1, 2, 0, 0, -1, 0.5]) / np.linalg.norm([1, 2, 0, 0, -1, 0.5])


def _ols_data(M, stat_rng):
    _, f = physical_model("R7", lambda x: 0.7 + x @ G_OLS)
    X = space("R7").sample(M, rng=stat_rng("UC-03/ols"))
    return X, f(X)


def _quad(name):
    return physical_model(
        name, lambda x: 0.3 + x @ G_QPHD + 0.5 * np.einsum("ij,jk,ik->i", x, H_QPHD, x)
    )[1]


def _validity():
    return expect_warning(asub.ValidityWarning, "validity.assumption")


def test_ols(stat_rng):
    """V01.16: KA-3, M = 15 on R7: λ̂_1 = ‖g‖² = 14.29, the rest <= 1e-10 λ̂_1, dist(W_1, g) <=
    1e-10, n == 1 (F10).
    """
    X, f = _ols_data(15, stat_rng)
    with _validity():
        s = asub.active_subspace(X=X, f=f, space=space("R7"), estimator="ols")
    assert abs(s.eigenvalues[0] - 14.29) <= 1e-10 * 14.29
    assert np.all(np.abs(s.eigenvalues[1:]) <= 1e-10 * s.eigenvalues[0])
    assert subspace_distance(s.W1, G_OLS) <= 1e-10
    assert s.n == 1


def test_ols_too_few(stat_rng):
    """V01.17: M = 7 < m + 1 -> ``input.too_few_samples``."""
    X, f = _ols_data(7, stat_rng)
    with pytest.raises(asub.InputError) as e, allow_warnings("validity.assumption"):
        asub.active_subspace(X=X, f=f, space=space("R7"), estimator="ols")
    assert e.value.code == "input.too_few_samples"


def test_qphd_uniform(stat_rng):
    """V01.18 [C-01], [D-40]: QPHD runs on bounded inputs; eigenvalues of gg^T + H Σ H with Σ = I/3
    from the space.

    Tolerance 1e-9 relative to λ_1 (least squares on 60 points). KA-4 six-figure values also
    checked.
    """
    X = space("U5").sample(60, rng=stat_rng("UC-03/qphd"))
    with _validity():
        s = asub.active_subspace(X=X, f=_quad("U5")(X), space=space("U5"), estimator="qphd")
    exact = np.sort(np.linalg.eigvalsh(np.outer(G_QPHD, G_QPHD) + H_QPHD @ H_QPHD / 3))[::-1]
    assert_eigenvalues_close(s.eigenvalues, exact, rtol=1e-9)
    assert np.allclose(exact, [2.41563, 0.178459, 0.0833333, 0.00257744, 0], rtol=0, atol=6e-6)


def test_qphd_gaussian(stat_rng):
    """V01.18: QPHD on G5: eigenvalues of gg^T + H² (rel. 1e-9 to λ_1)."""
    X = space("G5").sample(60, rng=stat_rng("UC-03/qphd"))
    s = asub.active_subspace(X=X, f=_quad("G5")(X), space=space("G5"), estimator="qphd")
    exact = np.sort(np.linalg.eigvalsh(np.outer(G_QPHD, G_QPHD) + H_QPHD @ H_QPHD))[::-1]
    assert_eigenvalues_close(s.eigenvalues, exact, rtol=1e-9)


def test_qphd_too_few(stat_rng):
    """V01.19: M = 20 < 21 -> ``input.too_few_samples``."""
    X = space("U5").sample(20, rng=stat_rng("UC-03/qphd"))
    with pytest.raises(asub.InputError) as e, allow_warnings("validity.assumption"):
        asub.active_subspace(X=X, f=_quad("U5")(X), space=space("U5"), estimator="qphd")
    assert e.value.code == "input.too_few_samples"


def test_sir(stat_rng):
    """V01.20: KA-9 on G6, M = 2000: n == 1 (calibrated gap ratio, construction 6);
    T = dist(W_1, a), construction 6, calibrated bound 0.2 (spec 17 §5.4)."""
    _, f = physical_model("G6", lambda x: (x @ A_SIR) + (x @ A_SIR) ** 3 / 4)
    X = space("G6").sample(2000, rng=stat_rng("UC-03/sir"))
    s = asub.active_subspace(X=X, f=f(X), space=space("G6"), estimator="sir")
    assert s.n == 1
    assert subspace_distance(s.W1, A_SIR) <= 0.2


def test_phd_outside_assumptions(stat_rng):
    """V01.21 [D-41], [N-17], [D-31]: pHd on U5 warns ``validity.assumption`` and returns signed
    eigenvalues
    ranked by |λ|. T = min λ̂ < 0 at M = 4000, construction 6 ([D-161]: the 1e-6 quantile is
    −0.069)."""
    X = space("U5").sample(4000, rng=stat_rng("UC-03/phd"))
    with _validity():
        s = asub.active_subspace(X=X, f=_quad("U5")(X), space=space("U5"), estimator="phd")
    assert isinstance(s, asub.SubspaceResult)
    assert np.all(np.diff(np.abs(s.eigenvalues)) <= 0)
    assert s.eigenvalues.min() < 0
