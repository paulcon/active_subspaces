"""Tolerance tiers and norm-wise comparisons (spec 17 §3.2, §4, §5.1). NumPy only."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from helpers.metrics import subspace_distance

__all__ = (
    "TIERS",
    "U",
    "assert_close",
    "assert_eigenvalues_close",
    "assert_subspace_close",
    "eig_tolerance",
    "lstsq_tolerance",
)

U: float = 2.0**-53
TIERS: Mapping[str, float] = MappingProxyType(
    {"F14": 1e-14, "F12": 1e-12, "F10": 1e-10, "F8": 1e-8, "F6": 1e-6}
)


def assert_close(
    actual: ArrayLike,
    desired: ArrayLike,
    *,
    rtol: float,
    atol: float = 0.0,
    scale: float | None = None,
    err_msg: str = "",
) -> None:
    """``max|a − d| <= atol + rtol * scale`` with ``scale = max|desired|`` by default; NaN
    positions must match.
    """
    a = np.asarray(actual)
    d = np.asarray(desired, dtype=np.float64)
    assert a.shape == d.shape, f"shape {a.shape} != {d.shape}. {err_msg}"
    if a.dtype.kind == "f":
        assert a.dtype == np.float64, f"dtype {a.dtype} is not float64. {err_msg}"
    a = a.astype(np.float64)
    na, nd = np.isnan(a), np.isnan(d)
    assert np.array_equal(na, nd), f"NaN positions differ. {err_msg}"
    ok = ~nd
    if not ok.any():
        return
    if scale is None:
        scale = float(np.max(np.abs(d[ok]))) if ok.any() else 0.0
    err = float(np.max(np.abs(a[ok] - d[ok])))
    bound = atol + rtol * scale
    assert err <= bound, (
        f"max|actual - desired| = {err:.3e} > {bound:.3e} (atol={atol}, rtol={rtol}, "
        f"scale={scale:.3e}). {err_msg}"
    )


def assert_eigenvalues_close(actual: ArrayLike, desired: ArrayLike, *, rtol: float) -> None:
    """Eigenvalues on the scale of the largest, ``|λ_1|`` (spec 17 §3.1)."""
    d = np.asarray(desired, dtype=np.float64)
    assert_close(actual, d, rtol=rtol, scale=float(np.max(np.abs(d))), err_msg="eigenvalues")


def assert_subspace_close(U_: ArrayLike, V: ArrayLike, *, tol: float) -> None:
    dist = subspace_distance(U_, V)
    assert dist <= tol, f"subspace distance {dist:.3e} > {tol:.3e}"


def eig_tolerance(C: ArrayLike, i: int, *, safety: float = 100.0) -> float:
    """``max(F12, safety·m·u·‖C‖_2/gap_i)`` for eigenvalue ``i`` (0-based, descending order) of
    symmetric ``C``.
    """
    C = np.asarray(C, dtype=np.float64)
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    others = np.delete(lam, i)
    gap = float(np.min(np.abs(lam[i] - others))) if others.size else np.inf
    assert gap > 0, f"eigenvalue {i} is repeated; compare the invariant subspace instead"
    return max(TIERS["F12"], safety * C.shape[0] * U * float(np.max(np.abs(lam))) / gap)


def lstsq_tolerance(B: ArrayLike, *, safety: float = 100.0, kappa_max: float = 1e8) -> float:
    """``max(F12, safety·u·κ(B))``; asserts ``κ(B) <= kappa_max`` so the bound cannot become
    vacuous.
    """
    kappa = float(np.linalg.cond(np.asarray(B, dtype=np.float64)))
    assert kappa <= kappa_max, f"condition number {kappa:.3e} exceeds the cap {kappa_max:.1e}"
    return max(TIERS["F12"], safety * U * kappa)
