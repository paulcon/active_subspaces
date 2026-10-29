"""Subspace metrics for the tests (spec 17 §3.1, §4).

NumPy only, never ``active_subspaces``, so a bug in the package cannot hide in
the measuring tool. Every function raises only ``AssertionError``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ("eigvec_distances", "orth", "principal_angles", "subspace_distance")

_U = 2.0**-53


def _as_matrix(A: ArrayLike) -> NDArray[np.float64]:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    why = f"expected a vector or an (m, k) matrix, got shape {A.shape}"
    assert A.ndim == 2, why
    assert A.shape[1] >= 1, why
    assert np.all(np.isfinite(A)), "non-finite entries in a subspace basis"
    return A


def orth(A: ArrayLike) -> NDArray[np.float64]:
    """Orthonormal basis ``(m, k)`` of ``range(A)``; fails if ``A`` is rank deficient (spec 17
    §3.1).
    """
    A = _as_matrix(A)
    m, k = A.shape
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    assert s[0] > 0, "zero matrix has no range"
    r = int(np.sum(s > max(m, k) * _U * s[0]))
    assert r == k, (
        f"rank-deficient subspace argument: numerical rank {r} < {k} columns (sigma={s})"
    )
    return U[:, :k]


def subspace_distance(U: ArrayLike, V: ArrayLike) -> float:
    """``‖P_U − P_V‖_2``: the sine of the largest principal angle; 1.0 when the dimensions differ.

    Computed as ``‖(I − U_o U_o^T) V_o‖_2``, accurate to O(u) for nearby subspaces.
    """
    Uo, Vo = orth(U), orth(V)
    assert Uo.shape[0] == Vo.shape[0], f"ambient dimensions differ: {Uo.shape[0]} vs {Vo.shape[0]}"
    if Uo.shape[1] != Vo.shape[1]:
        return 1.0
    R = Vo - Uo @ (Uo.T @ Vo)
    return float(min(1.0, np.linalg.norm(R, 2)))


def principal_angles(U: ArrayLike, V: ArrayLike) -> NDArray[np.float64]:
    """Principal angles ``(min(k, l),)`` in ascending order, radians; sine branch for small
    angles.
    """
    Uo, Vo = orth(U), orth(V)
    assert Uo.shape[0] == Vo.shape[0], f"ambient dimensions differ: {Uo.shape[0]} vs {Vo.shape[0]}"
    if Uo.shape[1] > Vo.shape[1]:
        Uo, Vo = Vo, Uo
    cos = np.clip(
        np.linalg.svd(Uo.T @ Vo, compute_uv=False), 0.0, 1.0
    )  # descending -> angles ascending
    sin = np.clip(
        np.linalg.svd(Uo - Vo @ (Vo.T @ Uo), compute_uv=False)[::-1], 0.0, 1.0
    )  # ascending
    return np.where(sin < 1 / np.sqrt(2), np.arcsin(sin), np.arccos(cos))


def eigvec_distances(W_hat: ArrayLike, W: ArrayLike) -> NDArray[np.float64]:
    """Sign-free per-column distance ``‖ŵ_i − (ŵ_i^T w_i) w_i‖_2`` of unit-normalized columns
    ``(k,)``.
    """
    A, B = _as_matrix(W_hat), _as_matrix(W)
    assert A.shape == B.shape, f"shapes differ: {A.shape} vs {B.shape}"
    na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
    assert np.all(na > 0), "zero column"
    assert np.all(nb > 0), "zero column"
    A, B = A / na, B / nb
    proj = np.sum(A * B, axis=0)
    return np.linalg.norm(A - proj * B, axis=0)
