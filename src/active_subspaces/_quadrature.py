"""Private Gauss rules (spec 10 §3.4, §4.5, §5.1.2) [D-106].

No compatibility promise. Every function returns new float64 arrays
``(nodes, weights)``; 1-D nodes are ascending.

    gauss_from_recurrence(alpha, beta, *, symmetric=False)   # Golub-Welsch
    gauss_hermite(k), gauss_legendre(k)                      # N(0, 1); U[-1, 1] with mass 1
    tensor_rule(rules)                                       # C order, product weights
    lanczos_gauss(t, p, k)                                   # Gauss rule of a discrete measure
    box_spline_gauss(w, k)                                   # Gauss rule of the density of w^T x,
                                                             # x ~ U[-1, 1]^m [D-113]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.linalg

from active_subspaces import _validation as _v
from active_subspaces.errors import InputError, NumericalError

__all__ = ()


def _bad(message: str, param: str) -> InputError:
    return InputError(message, code="input.parameter", details={"param": param})


def _symmetrize(t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``t_i <- (t_i - t_{k+1-i}) / 2``, ``v_i <- (v_i + v_{k+1-i}) / 2`` (spec 10 §3.4); ``t``
    ascending.
    """
    return 0.5 * (t - t[::-1]), 0.5 * (v + v[::-1])


def _christoffel(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray | None:
    """Squared first eigenvector components ``V[0, i]**2`` of the Jacobi matrix, from the
    recurrence.

    The eigenvector for node ``t_i`` is ``(p_0(t_i), ..., p_{k-1}(t_i))`` normalized, with
    ``p_j`` the orthonormal polynomials of the measure scaled to ``p_0 = 1``, so
    ``V[0, i]**2 = 1 / sum_j p_j(t_i)**2``. A sum of positive terms keeps relative
    accuracy for tiny weights, which LAPACK's eigenvector components do not.
    Returns ``None`` if the recurrence overflows.
    """
    k = a.shape[0]
    sb = np.sqrt(b)
    with np.errstate(all="ignore"):
        p_prev = np.zeros_like(t)
        p = np.ones_like(t)
        total = np.ones_like(t)
        for j in range(k - 1):
            p_next = ((t - a[j]) * p - (sb[j] if j > 0 else 0.0) * p_prev) / sb[j + 1]
            p_prev, p = p, p_next
            total += p * p
        v = 1.0 / total
    if not np.all(np.isfinite(v)) or not np.all(v > 0):
        return None
    return v


def gauss_from_recurrence(
    alpha: Any, beta: Any, *, symmetric: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Golub-Welsch: the ``k``-point Gauss rule of a measure from its recurrence coefficients.

    Parameters
    ----------
    alpha : array_like (k,)
        ``alpha_0, ..., alpha_{k-1}``.
    beta : array_like (k,)
        ``beta_0`` (the mass) and ``beta_1, ..., beta_{k-1}``, all positive.
    symmetric : bool
        Symmetrize the rule about 0 (for a symmetric measure, ``alpha = 0``).

    Returns
    -------
    nodes : ndarray (k,), ascending
    weights : ndarray (k,), summing to ``beta[0]``
    """
    a = _v.vector(alpha, "alpha")
    b = _v.vector(beta, "beta", size=a.shape[0])
    k = a.shape[0]
    if k < 1:
        raise _bad("alpha: expected at least one coefficient", "alpha")
    if np.any(b <= 0):
        raise _bad("beta: recurrence coefficients must be positive", "beta")
    if k == 1:
        t, v = a.copy(), b[:1].copy()
    else:
        t, V = scipy.linalg.eigh_tridiagonal(a, np.sqrt(b[1:]))
        vc = _christoffel(a, b, t)
        v = V[0, :] ** 2 if vc is None else vc
        v = b[0] * (v / v.sum())
    if symmetric:
        t, v = _symmetrize(t, v)
    return np.ascontiguousarray(t, dtype=np.float64), np.ascontiguousarray(v, dtype=np.float64)


def gauss_hermite(k: int) -> tuple[np.ndarray, np.ndarray]:
    """``k``-point probabilists' Gauss-Hermite rule for ``N(0, 1)``; weights sum to 1."""
    k = _v.integer(k, "k", lo=1)
    beta = np.arange(k, dtype=np.float64)
    beta[0] = 1.0
    return gauss_from_recurrence(np.zeros(k), beta, symmetric=True)


def gauss_legendre(k: int) -> tuple[np.ndarray, np.ndarray]:
    """``k``-point Gauss-Legendre rule for ``U[-1, 1]`` with mass 1."""
    k = _v.integer(k, "k", lo=1)
    j = np.arange(k, dtype=np.float64)
    beta = j**2 / (4.0 * j**2 - 1.0)
    beta[0] = 1.0
    return gauss_from_recurrence(np.zeros(k), beta, symmetric=True)


def tensor_rule(rules: Sequence[tuple[Any, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Tensor product of 1-D rules ``[(nodes_i, weights_i), ...]``.

    Returns ``nodes (prod k_i, d)`` in C order (last coordinate fastest) and
    ``weights (prod k_i,)``, the products of the 1-D weights.
    """
    rules = list(rules)
    if not rules:
        raise _bad("rules: expected at least one 1-D rule", "rules")
    nodes_1d, weights_1d = [], []
    for i, rule in enumerate(rules):
        try:
            t, w = rule
        except (TypeError, ValueError) as exc:
            raise _bad(f"rules[{i}]: expected a (nodes, weights) pair", "rules") from exc
        t = _v.vector(t, f"rules[{i}] nodes")
        w = _v.vector(w, f"rules[{i}] weights", size=t.shape[0])
        if t.shape[0] < 1:
            raise _bad(f"rules[{i}]: empty rule", "rules")
        nodes_1d.append(t)
        weights_1d.append(w)
    grids = np.meshgrid(*nodes_1d, indexing="ij")
    nodes = np.stack([g.ravel() for g in grids], axis=1)
    weights = weights_1d[0]
    for w in weights_1d[1:]:
        weights = np.multiply.outer(weights, w).ravel()
    return np.ascontiguousarray(nodes, dtype=np.float64), np.ascontiguousarray(
        weights, dtype=np.float64
    )


def lanczos_gauss(t: Any, p: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``k``-point Gauss rule of the discrete measure ``sum_i p_i delta_{t_i}``.

    Lanczos on ``diag(t)`` from ``sqrt(p / sum p)`` with full
    reorthogonalization gives the recurrence coefficients, then
    :func:`gauss_from_recurrence`. Inner products use NumPy reductions, not
    BLAS, whose rounding can depend on memory alignment (Accelerate), so the
    rule is bitwise reproducible. Weights sum to ``sum(p)``. Raises
    ``numerical.singular`` if the measure has fewer than ``k`` support points.
    """
    t = _v.vector(t, "t")
    p = _v.vector(p, "p", size=t.shape[0])
    k = _v.integer(k, "k", lo=1, hi=max(1, t.shape[0]))
    if t.shape[0] < 1:
        raise _bad("t: expected at least one atom", "t")
    if np.any(p < 0) or not p.sum() > 0:
        raise _bad("p: weights must be nonnegative with a positive sum", "p")
    mass = float(p.sum())
    N = t.shape[0]
    Q = np.empty((N, k))
    alpha = np.empty(k)
    beta = np.empty(k)
    beta[0] = mass
    Q[:, 0] = np.sqrt(p / mass)
    Q[:, 0] /= np.sqrt(np.sum(Q[:, 0] ** 2))
    scale = float(np.abs(t).max())
    for j in range(k):
        q = Q[:, j]
        r = t * q
        alpha[j] = np.sum(q * r)
        if j + 1 == k:
            break
        r = r - alpha[j] * q
        if j > 0:
            r = r - np.sqrt(beta[j]) * Q[:, j - 1]
        Qj = Q[:, : j + 1]
        for _ in range(2):
            r = r - np.sum(Qj * np.sum(Qj * r[:, None], axis=0)[None, :], axis=1)
        nrm = float(np.sqrt(np.sum(r * r)))
        if not nrm > 1e-13 * max(scale, 1e-300):
            raise NumericalError(
                f"lanczos_gauss: the measure has fewer than k = {k} support points (breakdown at "
                f"step {j + 1})",
                code="numerical.singular",
                details={"k": int(k), "step": int(j + 1)},
            )
        beta[j + 1] = nrm**2
        Q[:, j + 1] = r / nrm
    return gauss_from_recurrence(alpha, beta)


def box_spline_gauss(w: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``k``-point Gauss rule of the density of ``y = w^T x``, ``x ~ U[-1, 1]^m`` [D-53, D-113].

    Moment-preserving convolution (spec 10 §5.1.2): start from the scaled
    Gauss-Legendre rule of the first nonzero ``w_j``; for each further nonzero
    ``w_j``, convolve with the scaled Legendre rule and compress the ``k**2``
    atoms back to ``k`` with :func:`lanczos_gauss`. The final rule is
    symmetrized about 0. Exact for polynomials of degree ``<= 2k - 1``.

    Returns
    -------
    nodes : ndarray (k,), ascending, in ``(-||w||_1, ||w||_1)``
    weights : ndarray (k,), positive, summing to 1
    """
    w = _v.vector(w, "w")
    k = _v.integer(k, "k", lo=1)
    J = np.flatnonzero(w != 0)
    if J.size == 0:
        raise _bad("w: expected at least one nonzero entry", "w")
    s, v = gauss_legendre(k)
    t, p = w[J[0]] * s, v.copy()
    for j in J[1:]:
        T = (t[:, None] + w[j] * s[None, :]).ravel()
        P = (p[:, None] * v[None, :]).ravel()
        t, p = lanczos_gauss(T, P, k)
    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order] / p.sum()
    t, p = _symmetrize(t, p)
    return np.ascontiguousarray(t), np.ascontiguousarray(p)
