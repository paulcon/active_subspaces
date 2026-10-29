"""Statistical acceptance regions with false-failure probability ``ALPHA`` (spec 17 §3.3, §4).
NumPy/SciPy only.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as _st

__all__ = (
    "ALPHA",
    "assert_gof",
    "assert_within_se",
    "binomial_bounds",
    "pot_quantile",
    "standardized_moment_bounds",
    "t_bound",
    "z_bound",
)

ALPHA: float = 1e-6


def _ceil2(x: float) -> float:
    return math.ceil(round(x * 100, 9)) / 100


def z_bound(k: int = 1) -> float:
    """Bonferroni ``z_k = max(5, Φ^{-1}(1 − α/(2k)))`` rounded up to two decimals (§3.3 item 2)."""
    return max(5.0, _ceil2(float(_st.norm.isf(ALPHA / (2 * k)))))


def t_bound(nu: int, k: int = 1) -> float:
    """Two-sided ``t_ν`` quantile at ``1 − α/(2k)``, rounded up; never below ``z_bound(k)``
    (§3.3 item 1).
    """
    return max(z_bound(k), _ceil2(float(_st.t.isf(ALPHA / (2 * k), nu))))


def assert_within_se(estimate, truth, se, *, k: int = 1, nu: int | None = None) -> None:
    """``|estimate − truth| <= z·se`` elementwise, ``z = z_bound(k)`` or ``t_bound(nu, k)``
    (constructions 1–2).
    """
    z = z_bound(k) if nu is None else t_bound(nu, k)
    est, tru, s = (np.asarray(v, dtype=np.float64) for v in (estimate, truth, se))
    dev = np.abs(est - tru)
    assert np.all(np.isfinite(s)), f"invalid standard error {s}"
    assert np.all(s >= 0), f"invalid standard error {s}"
    assert np.all(dev <= z * s), (
        f"|estimate - truth| = {dev} > {z} * se = {z * s} (construction "
        f"{'1' if k == 1 else '2'}, k={k}, nu={nu})"
    )


def assert_gof(pvalue: float) -> None:
    """Goodness of fit passes when ``p >= ALPHA`` (construction 3)."""
    assert pvalue >= ALPHA, (
        f"goodness-of-fit p-value {pvalue:.3e} < ALPHA = {ALPHA:g} (construction 3)"
    )


def binomial_bounds(R: int, p_lo: float, p_hi: float) -> tuple[int, int]:
    """Acceptance range ``[L, U]`` of a count over ``R`` runs (construction 4)."""
    L = int(_st.binom.ppf(ALPHA / 2, R, p_lo))
    U = int(_st.binom.ppf(1 - ALPHA / 2, R, p_hi))
    return L, U


def standardized_moment_bounds(R: int, nu: int | None = None) -> tuple[float, tuple[float, float]]:
    """``(mean bound, variance interval)`` for ``R`` standardized values (construction 5)."""
    target = 1.0 if nu is None else nu / (nu - 2)
    lo = float(_st.chi2.ppf(ALPHA / 4, R - 1)) / (R - 1) * target
    hi = float(_st.chi2.ppf(1 - ALPHA / 4, R - 1)) / (R - 1) * target
    return z_bound(2) / math.sqrt(R) * math.sqrt(target), (lo, hi)


def pot_quantile(
    samples: ArrayLike, *, alpha: float = ALPHA, tail: float = 0.05
) -> tuple[float, float]:
    """Peaks-over-threshold exponential-tail estimate ``(q̂(α), se)`` (construction 6, [D-160])."""
    T = np.asarray(samples, dtype=np.float64).ravel()
    u = float(np.quantile(T, 1 - tail))
    exc = T[T > u] - u
    assert exc.size > 0, "no exceedances over the threshold"
    beta = float(np.mean(exc))
    log_ratio = math.log(tail / alpha)
    return u + beta * log_ratio, beta * log_ratio / math.sqrt(exc.size)
