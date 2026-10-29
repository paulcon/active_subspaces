"""Monte Carlo checks of the known-answer problems (spec 12 §3.3-3.4, §8; V12.8, V12.9).

Construction 2 of spec 17 §3.3 (Bonferroni), fixed seeds [D-78].
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

import active_subspaces as asub
from active_subspaces import test_functions as tf

M = 10**5


def z_k(k: int, alpha: float = 1e-6) -> float:
    """Spec 17 §3.3: max(5, Phi^-1(1 - alpha / (2k))) rounded up to two decimals."""
    return max(5.0, math.ceil(norm.isf(alpha / (2 * k)) * 100) / 100)


def _monte_carlo(model, seed):
    """Spec 12 §3.4 estimates from M normalized samples: eigenpairs, delta-method se, mean,
    variance and their se.
    """
    space = model.space
    X = space.from_normalized(space.sample(M, rng=seed, normalized=True))
    f, G = asub.evaluate_with_grads(model, X=X)
    g = space.grads_to_normalized(G)
    evals, W = np.linalg.eigh(g.T @ g / M)
    evals, W = evals[::-1], W[:, ::-1]
    S = (g @ W) ** 2
    lam, lam_se = S.mean(axis=0), S.std(axis=0, ddof=1) / math.sqrt(M)
    mean, mean_se = f.mean(), f.std(ddof=1) / math.sqrt(M)
    d2 = (f - mean) ** 2
    var, var_se = f.var(ddof=1), d2.std(ddof=1) / math.sqrt(M)
    return evals, W, lam, lam_se, (mean, mean_se), (var, var_se)


def _within(est, se, ref, z):
    # an exact (zero-variance) estimate is compared to F12 instead of 0 * z
    return abs(est - ref) <= z * se + 1e-12 * max(abs(ref), 1e-300)


@pytest.mark.parametrize("kind", ["uniform", "gaussian"])
def test_quadratic_monte_carlo(kind):
    """T: eigenvalue, mean and variance estimates; construction 2, Bonferroni k = m + 2; guards
    D-122.
    """
    model = tf.builtin_models()[f"quadratic_{kind}_10"]
    rr = model.reference_results
    m = model.dim
    z = z_k(m + 2)
    _, _, lam, lam_se, (mean, mean_se), (var, var_se) = _monte_carlo(model, seed=11)
    for k in range(m):
        assert _within(lam[k], lam_se[k], rr.eigenvalues[k], z), (
            k,
            lam[k],
            rr.eigenvalues[k],
            lam_se[k],
        )
    assert _within(mean, mean_se, rr.mean, z)
    assert _within(var, var_se, rr.variance, z)


@pytest.mark.parametrize("profile", ["linear", "sin", "square"])
@pytest.mark.parametrize("kind", ["uniform", "gaussian"])
def test_ridge_monte_carlo(profile, kind):
    """T: lambda_1, mean and variance; construction 2, Bonferroni k = 3 (z = 5.11); exact rank
    one by F14/F12.
    """
    space = asub.Uniform.reference(10) if kind == "uniform" else asub.Gaussian.reference(10)
    model = tf.ridge(np.arange(1, 11), profile=profile, space=space)
    rr = model.reference_results
    z = z_k(3)
    evals, _, lam, lam_se, (mean, mean_se), (var, var_se) = _monte_carlo(model, seed=12)
    assert _within(lam[0], lam_se[0], rr.eigenvalues[0], z)
    assert _within(mean, mean_se, rr.mean, z)
    assert _within(var, var_se, rr.variance, z)
    # eigenvalues of C = g^T g / M from the singular values of g: the Gram product sums M rounded
    # terms and, for the constant gradients of "linear", leaves O(sqrt(M) eps) ~ 1e-13 noise
    space_ = model.space
    Xs = space_.from_normalized(space_.sample(M, rng=12, normalized=True))
    g = space_.grads_to_normalized(asub.evaluate_grads(model, X=Xs))
    _, sv, Vt = np.linalg.svd(g / math.sqrt(M), full_matrices=False)
    assert np.all(sv[1:] ** 2 <= 1e-14 * sv[0] ** 2)
    a = rr.eigenvectors[:, 0]
    w = Vt[0]
    assert abs(sv[0] ** 2 - evals[0]) <= 1e-12 * evals[0]
    assert np.linalg.norm(np.outer(w, w) - np.outer(a, a), 2) <= 1e-12
