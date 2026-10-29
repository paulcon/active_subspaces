"""v1 subspace estimators against the D-37 baselines ``subspaces/*`` (spec 17 §5.5; V07.8, V07.14,
V07.31).

[C-01] QPHD on bounded inputs raised; [C-17] ladle n shifted by one; [N-01] SAVE without p_h;
[N-02] pHd scaled by gamma^{-1/2}; [N-07] |eigenvalues| of indefinite pHd.
"""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces.errors import ValidityWarning
from helpers.baselines import assert_matches_baseline, load_baseline


def _check_eigs(b, res, prefix: str = "") -> None:
    assert_matches_baseline(b, prefix + "e", np.asarray(res.eigenvalues))
    assert_matches_baseline(b, prefix + "W", np.asarray(res.eigenvectors))


@pytest.mark.parametrize("case", ["as_exact_rule", "as_mc"])
def test_active_subspace(case):
    """AS on KA-1 gradients: eigenvalues F12/lambda_1, eigenvectors sign-free F10 (gap-scaled),
    EVG n exact.
    """
    b = load_baseline("subspaces", case)
    w = b.inputs.get("weights")
    res = asub.active_subspace(
        space=asub.Uniform.reference(5), grads=b.inputs["grads"], quad_weights=w
    )
    _check_eigs(b, res)
    assert_matches_baseline(b, "n", np.int64(res.n))


def test_ols():
    b = load_baseline("subspaces", "ols")
    res = asub.active_subspace(
        space=asub.Gaussian.reference(5), X=b.inputs["X"], f=b.inputs["f"], estimator="ols", n=1
    )
    _check_eigs(b, res)


def test_qphd_gaussian():
    b = load_baseline("subspaces", "qphd_gaussian")
    res = asub.active_subspace(
        space=asub.Gaussian.reference(4), X=b.inputs["X"], f=b.inputs["f"], estimator="qphd", n=1
    )
    _check_eigs(b, res)


def test_qphd_bounded():
    """[C-01] ``raises``: 2016 raised TypeError; v1 returns C = H Sigma H with Sigma = I/3
    (KA-4, g = 0) [D-40].
    """
    b = load_baseline("subspaces", "qphd_bounded")
    c = b.inputs["c"]
    with pytest.warns(ValidityWarning):
        res = asub.active_subspace(
            space=asub.Uniform.reference(4),
            X=b.inputs["X"],
            f=b.inputs["f"],
            estimator="qphd",
            n=1,
        )
    assert_matches_baseline(b, "C", res)
    H = 2.0 * np.outer(c, c)
    C = H @ (np.eye(4) / 3.0) @ H
    C_v1 = (np.asarray(res.W) * np.asarray(res.eigenvalues)) @ np.asarray(res.W).T
    assert np.linalg.norm(C_v1 - C, 2) <= 1e-10 * np.linalg.norm(C, 2)


def test_ladle_ridge():
    """[C-17] ``shift -1``: on an exact ridge every replicate has the same W1 (KA-2); 2016
    returned n = 2, v1 returns 1.
    """
    b = load_baseline("subspaces", "li_ridge")
    res = asub.active_subspace(
        space=asub.Uniform.reference(5),
        grads=b.inputs["grads"],
        dimension_rule="li",
        nboot=int(b.inputs["nboot"]),
        rng=0,
    )
    assert_matches_baseline(b, "n", np.int64(res.n))


def test_sir():
    """SIR on Gaussian X: the 2016 percentile slices equal v1's slices (M divisible by S), so C
    agrees (F10).
    """
    b = load_baseline("subspaces", "sir")
    res = asub.active_subspace(
        space=asub.Gaussian.reference(4), X=b.inputs["X"], f=b.inputs["f"], estimator="sir", n=1
    )
    _check_eigs(b, res)


def test_save():
    """[N-01] ``scale 1/S`` [D-42]: v1 weights each slice by p_h = 1/S; eigenvectors unchanged."""
    b = load_baseline("subspaces", "save")
    res = asub.active_subspace(
        space=asub.Gaussian.reference(4), X=b.inputs["X"], f=b.inputs["f"], estimator="save", n=1
    )
    _check_eigs(b, res)


def test_phd():
    """pHd: reference on Gaussian X; on uniform X [N-02] ``scale gamma^{-1/2}`` (2016 run with
    Python 2 semantics of [C-01]) [D-43].
    """
    b = load_baseline("subspaces", "phd")
    res = asub.active_subspace(
        space=asub.Gaussian.reference(4),
        X=b.inputs["X_gaussian"],
        f=b.inputs["f_gaussian"],
        estimator="phd",
        n=1,
    )
    _check_eigs(b, res, "gaussian_")
    with pytest.warns(ValidityWarning):
        res = asub.active_subspace(
            space=asub.Uniform.reference(4),
            X=b.inputs["X_uniform"],
            f=b.inputs["f_uniform"],
            estimator="phd",
            n=1,
        )
    _check_eigs(b, res, "uniform_")


def test_phd_signs():
    """[N-07] ``abs_of`` [D-31]: 2016 reported |lambda|; v1 keeps the signs of an indefinite
    pHd matrix.
    """
    b = load_baseline("subspaces", "phd_signs")
    res = asub.active_subspace(
        space=asub.Gaussian.reference(4), X=b.inputs["X"], f=b.inputs["f"], estimator="phd", n=1
    )
    assert np.any(np.asarray(res.eigenvalues) < 0)
    _check_eigs(b, res)


def test_bootstrap_minmax_not_stored():
    """The 2016 min/max bootstrap ranges are stochastic and replaced by replicates [N-08],
    [D-30]: nothing is stored.
    """
    b = load_baseline("subspaces", "bootstrap_minmax")
    assert b.outputs == {}
    assert set(b.meta["not_stored"]) == {"e_br", "sub_br"}
