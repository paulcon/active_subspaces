"""v1 surrogates against the D-37 baselines ``surrogates/*`` (spec 17 §5.5; [C-04], [C-18],
[N-05]).
"""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from helpers.baselines import assert_matches_baseline, load_baseline


def poly_function(X: np.ndarray) -> np.ndarray:
    """``tools/capture_baselines.py``: exp(0.5 x1) (+ sin(2 x2) when k = 2)."""
    f = np.exp(0.5 * X[:, 0])
    if X.shape[1] > 1:
        f = f + np.sin(2.0 * X[:, 1])
    return f


def _fit(X, degree, weights=None):
    return asub.fit_surrogate(
        X=X,
        f=poly_function(X),
        space=asub.Uniform.reference(X.shape[1]),
        subspace=None,
        kind="poly",
        degree=degree,
        sample_weights=weights,
    )


def _coef_in_2016_order(s, indices_2016: np.ndarray) -> np.ndarray:
    pos = {tuple(int(v) for v in row): i for i, row in enumerate(np.asarray(s.indices))}
    return np.asarray(s.coefficients)[[pos[tuple(int(v) for v in row)] for row in indices_2016]]


@pytest.mark.parametrize(("k", "d"), [(1, 2), (1, 3), (2, 2), (2, 3)])
def test_poly(k, d):
    b = load_baseline("surrogates", "poly")
    s = _fit(b.inputs[f"X_k{k}"], d)
    assert_matches_baseline(
        b, f"coef_k{k}_d{d}", _coef_in_2016_order(s, b.inputs[f"indices_k{k}_d{d}"])
    )
    assert_matches_baseline(b, f"pred_k{k}_d{d}", s.evaluate(b.inputs[f"Xtest_k{k}"]))


def test_poly_weighted():
    """[C-04] ``reweighted w**2``: 2016 scaled rows by w, which is v1 with sample_weights =
    w**2; v1 with w differs.
    """
    b = load_baseline("surrogates", "poly_weighted")
    X, w, Xt = b.inputs["X"], b.inputs["weights"], b.inputs["Xtest"]
    s, s2 = _fit(X, 2, w), _fit(X, 2, w**2)
    idx = b.inputs["indices"]
    assert_matches_baseline(
        b, "coef", _coef_in_2016_order(s, idx), relation_value=_coef_in_2016_order(s2, idx)
    )
    assert_matches_baseline(b, "pred", s.evaluate(Xt), relation_value=s2.evaluate(Xt))


def test_rbf_scale():
    """[C-18], [N-05] ``improves``: at scale 100 v1's relative error is below 2016's, and v1 is
    scale-equivariant.
    """
    b = load_baseline("surrogates", "rbf_scale")
    Y, Yt = b.inputs["Y"][:, None], b.inputs["Ytest"][:, None]
    errs = []
    for c in b.inputs["scales"]:
        s = asub.fit_surrogate(
            X=Y,
            f=c * np.sin(3 * Y[:, 0]),
            space=asub.Uniform.reference(1),
            subspace=None,
            kind="rbf",
            noise_variance=np.full(Y.shape[0], c**2 * 1e-2),
        )
        truth = c * np.sin(3 * Yt[:, 0])
        errs.append(np.linalg.norm(s.evaluate(Yt) - truth) / np.linalg.norm(truth))
    assert_matches_baseline(b, "rel_error_scale100", np.float64(errs[1]))
    assert abs(errs[1] - errs[0]) <= 1e-8 * errs[0]
