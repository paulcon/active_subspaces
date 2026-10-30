r"""Polynomial and radial-basis surrogates on the active variables.

A surrogate approximates a quantity of interest by a function of fewer
variables, :math:`f(x) \approx g(W_1^\top x)`, where :math:`x` are the
normalized inputs and :math:`y = W_1^\top x` the active variables of a
:class:`~active_subspaces.SubspaceResult`. With ``subspace=None`` the surrogate
is :math:`g(x)` on all normalized inputs instead.

:func:`fit_surrogate` fits a surrogate to data, and
:func:`fit_surrogate_from_model` to conditional means of a model over the
inactive variables. Both return frozen results (:class:`PolynomialSurrogate`,
:class:`RBFSurrogate`) that satisfy the evaluation-only :class:`Surrogate`
protocol. :func:`predict` and :func:`predict_gradient` evaluate any such
surrogate at physical inputs; the methods ``evaluate`` and ``gradient`` work in
the surrogate's own (normalized) input coordinates.
"""

# Private cores (no compatibility promise; spec 09 §5.5, reused by spec 10's lift):
# ``_poly_fit``, ``_quadratic_terms``, ``_fit_active``.

from __future__ import annotations

import concurrent.futures
import itertools
import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.stats
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist

from active_subspaces import _random
from active_subspaces import _validation as _v
from active_subspaces.designs import (
    ActiveDesign,
    ConditionalSamples,
    _conditional_samples,
)
from active_subspaces.errors import AccuracyWarning, InputError, NumericalError, _warn
from active_subspaces.geometry import ActiveDomain
from active_subspaces.models import Model, ValueFn
from active_subspaces.results import (
    Provenance,
    QoIResult,
    _invariant,
    _make_provenance,
    _register_migration,
    _register_result,
)
from active_subspaces.spaces import InputSpace, _poincare_constant
from active_subspaces.subspaces import SubspaceResult

__all__ = (
    "IntervalResult",
    "PolynomialSurrogate",
    "RBFSurrogate",
    "Surrogate",
    "fit_surrogate",
    "fit_surrogate_from_model",
    "predict",
    "predict_gradient",
    "predict_interval",
    "quadratic_terms",
)

_EPS = float(np.finfo(np.float64).eps)
_NUGGET_FLOOR = 1e-6
_LEVERAGE_TOL = 1e-8
_RIPPA_TOL = 1e-12
_GRID_STEP = 0.25
_DEFAULT_BOUNDS = (-10.0, 1.0)
_FLAT_RTOL = 1e-8  # J within this of its minimum, relative, counts as flat [D-192]
_FLAT_DECADES = 4.0
_CHUNK = 1 << 22  # entries of a (rows, p, k) basis-gradient block


# ---------------------------------------------------------------------------
# Protocol (§4.1)
# ---------------------------------------------------------------------------


@runtime_checkable
class Surrogate(Protocol):
    r"""Evaluation-only protocol that every surrogate satisfies.

    Any object with these attributes and methods can be passed to
    :func:`predict`, :func:`predict_gradient` and the plotting functions; the
    built-in :class:`PolynomialSurrogate`, :class:`RBFSurrogate` and
    :class:`~active_subspaces.ReducedModel` all satisfy it. The check
    ``isinstance(obj, Surrogate)`` tests only that the members exist.

    The members are read-only: ``space``, ``subspace``, ``qoi`` and
    ``active_dim``, and the methods ``evaluate`` and ``gradient``.

    See Also
    --------
    predict, predict_gradient : Evaluate a surrogate at physical inputs.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> X = space.sample(20, rng=0)
    >>> f = X[:, 0] + X[:, 1] ** 2
    >>> s = ac.fit_surrogate(X=X, f=f, space=space, subspace=None, kind="poly")
    >>> isinstance(s, ac.Surrogate)
    True
    >>> s.active_dim
    2
    """

    # numpydoc ignore=PR01

    @property
    def space(self) -> InputSpace:
        """The input space whose normalized coordinates the surrogate is built on.

        Returns
        -------
        InputSpace
            The space that maps physical inputs to normalized ones.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
        >>> X = space.sample(20, rng=0)
        >>> s = ac.fit_surrogate(X=X, f=X[:, 0], space=space, subspace=None, kind="poly")
        >>> s.space.dim
        2
        """

    @property
    def subspace(self) -> SubspaceResult | None:
        r"""The subspace whose active variables are the input coordinates, or ``None``.

        Returns
        -------
        SubspaceResult or None
            The subspace, whose active variables :math:`y = W_1^\top x` are the
            surrogate's input coordinates, or ``None`` when they are all
            normalized inputs :math:`x`.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
        >>> X = space.sample(20, rng=0)
        >>> s = ac.fit_surrogate(X=X, f=X[:, 0], space=space, subspace=None, kind="poly")
        >>> s.subspace is None
        True
        """

    @property
    def qoi(self) -> str:
        """Name of the quantity of interest.

        Returns
        -------
        str
            The name.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
        >>> X = space.sample(20, rng=0)
        >>> s = ac.fit_surrogate(X=X, f=X[:, 0], space=space, subspace=None, kind="poly")
        >>> s.qoi
        'f'
        """

    @property
    def active_dim(self) -> int:
        """Number of input coordinates :math:`k` of the surrogate.

        :math:`k = n` (the subspace dimension) on a subspace, :math:`k = m`
        on all inputs.

        Returns
        -------
        int
            The number :math:`k`.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
        >>> X = space.sample(20, rng=0)
        >>> s = ac.fit_surrogate(X=X, f=X[:, 0], space=space, subspace=None, kind="poly")
        >>> s.active_dim
        2
        """

    def evaluate(self, Y: ArrayLike) -> NDArray[np.float64]:
        r"""Evaluate the surrogate at points in its input coordinates.

        Parameters
        ----------
        Y : array_like, shape (K, k)
            Points in the surrogate's input coordinates: normalized active
            variables :math:`y = W_1^\top x` on a subspace, normalized inputs
            :math:`x` when ``subspace`` is ``None``. A 1-D array is read as
            ``(K, 1)`` when :math:`k = 1`.

        Returns
        -------
        ndarray of float64, shape (K,)
            The values :math:`g(y)`.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([-1.0, -1.0], [1.0, 1.0])
        >>> X = space.sample(20, rng=0)
        >>> s = ac.fit_surrogate(X=X, f=1 + X[:, 0], space=space, subspace=None, kind="poly")
        >>> print(np.round(s.evaluate([[0.5, 0.0]]), 3))
        [1.5]
        """

    def gradient(self, Y: ArrayLike) -> NDArray[np.float64]:
        r"""Evaluate the gradient of the surrogate in its input coordinates.

        Parameters
        ----------
        Y : array_like, shape (K, k)
            Points in the surrogate's input coordinates, as for
            :meth:`evaluate`.

        Returns
        -------
        ndarray of float64, shape (K, k)
            The gradients :math:`\nabla_y g(y)` with respect to the input
            coordinates (not physical gradients; see :func:`predict_gradient`).

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([-1.0, -1.0], [1.0, 1.0])
        >>> X = space.sample(20, rng=0)
        >>> s = ac.fit_surrogate(X=X, f=1 + X[:, 0], space=space, subspace=None, kind="poly")
        >>> print(np.round(s.gradient([[0.5, 0.0]]), 3) + 0.0)
        [[1. 0.]]
        """


# ---------------------------------------------------------------------------
# Polynomial basis (§3.1)
# ---------------------------------------------------------------------------


def _indices(k: int, d: int) -> np.ndarray:
    """Multi-indices with |alpha| <= d: by total degree, then lexicographically descending."""
    rows: list[tuple[int, ...]] = []
    for t in range(d + 1):
        level = []
        for combo in itertools.combinations_with_replacement(range(k), t):
            a = [0] * k
            for j in combo:
                a[j] += 1
            level.append(tuple(a))
        level.sort(reverse=True)
        rows.extend(level)
    return np.array(rows, dtype=np.int64).reshape(len(rows), k)


def _power_table(Y: np.ndarray, d: int) -> np.ndarray:
    """``P[i, j, e] = Y[i, j] ** e`` for ``e = 0..d`` (repeated products, no loop over points)."""
    P = np.empty((*Y.shape, d + 1))
    P[..., 0] = 1.0
    for e in range(1, d + 1):
        P[..., e] = P[..., e - 1] * Y
    return P


def _basis(Y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Design matrix ``B[i, a] = prod_j Y[i, j] ** indices[a, j]``."""
    k = Y.shape[1]
    d = int(indices.max()) if indices.size else 0
    P = _power_table(Y, d)
    cols = np.arange(k)
    B = np.ones((Y.shape[0], indices.shape[0]))
    for j in cols:
        B *= P[:, j, indices[:, j]]
    return B


def _basis_gradient_times(Y: np.ndarray, indices: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """``sum_a coef[a] * d b_a(y) / d y_j`` for each row of ``Y``: ``(K, k)``."""
    K, k = Y.shape
    p = indices.shape[0]
    d = int(indices.max()) if indices.size else 0
    out = np.zeros((K, k))
    if d == 0:
        return out
    rows = max(1, _CHUNK // max(1, p * k))
    for s in range(0, K, rows):
        Ys = Y[s : s + rows]
        P = _power_table(Ys, d)
        Dp = np.zeros_like(P)
        Dp[..., 1:] = P[..., :-1] * np.arange(1, d + 1)
        Pa = np.stack([P[:, j, indices[:, j]] for j in range(k)], axis=2)  # (r, p, k)
        for j in range(k):
            others = np.prod(np.delete(Pa, j, axis=2), axis=2) if k > 1 else np.ones(Pa.shape[:2])
            out[s : s + rows, j] = (Dp[:, j, indices[:, j]] * others) @ coef
    return out


def _too_few(required: int, got: int, what: str) -> InputError:
    return InputError(
        f"{what} needs at least {required} points (the number of basis functions), got {got}",
        code="input.too_few_samples",
        details={"required": int(required), "got": int(got)},
    )


def _scaled_svd(
    B: np.ndarray, sqrt_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD of ``S^{1/2} B D`` (unit column norms).

    ``numerical.singular`` if rank-deficient (§5.1).
    """
    if not np.isfinite(B).all():
        raise NumericalError(
            "the polynomial basis overflowed; lower the degree",
            code="numerical.nonfinite",
            details={"what": "basis"},
        )
    M, p = B.shape
    Bs = B * sqrt_s[:, None]
    norms = np.linalg.norm(Bs, axis=0)
    D = np.where(norms > 0, 1.0 / np.where(norms > 0, norms, 1.0), 1.0)
    U, sig, Vt = np.linalg.svd(Bs * D, full_matrices=False)
    tol = max(M, p) * _EPS * (sig[0] if sig.size else 0.0)
    if sig.size == 0 or sig[-1] <= tol:
        rank = int(np.sum(sig > tol))
        raise NumericalError(
            f"the polynomial basis is rank-deficient on these points (rank {rank} < {p})",
            code="numerical.singular",
            details={"rank": rank, "p": int(p)},
        )
    return U, sig, Vt, D


def _poly_core(
    Y: np.ndarray, f: np.ndarray, degree: int, s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted polynomial least squares; returns ``(indices, B, coef, leverages)``."""
    k = Y.shape[1]
    indices = _indices(k, degree)
    p = indices.shape[0]
    n_pos = int(np.sum(s > 0))
    if n_pos < p:
        raise _too_few(p, n_pos, f"a degree-{degree} polynomial in {k} variables")
    B = _basis(Y, indices)
    sqrt_s = np.sqrt(s)
    U, sig, Vt, D = _scaled_svd(B, sqrt_s)
    coef = D * (Vt.T @ ((U.T @ (sqrt_s * f)) / sig))
    if not np.isfinite(coef).all():
        raise NumericalError(
            "non-finite polynomial coefficients",
            code="numerical.nonfinite",
            details={"what": "coefficients"},
        )
    h = np.sum(U * U, axis=1)
    return indices, B, coef, h


def _poly_fit(
    Y: Any, f: Any, degree: int, sample_weights: Any = None
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Private core (spec 09 §5.5).

    ``(indices (p, k), coefficients (p,))`` of the weighted fit on normalized ``Y``.
    """
    Y = np.array(Y, dtype=np.float64, ndmin=2, copy=True)
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    s = (
        np.ones(Y.shape[0])
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    indices, _, coef, _ = _poly_core(Y, f, int(degree), s)
    return indices, coef


def _quadratic_terms(
    indices: Any, coefficients: Any
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Private core (spec 09 §5.5).

    ``(c, a, H)`` with ``g = c + a^T y + y^T H y / 2 + ...`` (§3.2).
    """
    idx = np.asarray(indices, dtype=np.int64)
    coef = np.asarray(coefficients, dtype=np.float64)
    k = idx.shape[1]
    deg = idx.sum(axis=1)
    zero = np.flatnonzero(deg == 0)
    c = float(coef[zero[0]]) if zero.size else 0.0
    a = np.zeros(k)
    r1 = np.flatnonzero(deg == 1)
    a[np.argmax(idx[r1], axis=1)] = coef[r1]
    H = np.zeros((k, k))
    r2 = np.flatnonzero(deg == 2)
    if r2.size:
        I2 = idx[r2]
        j = np.argmax(I2 > 0, axis=1)
        l_ = k - 1 - np.argmax(I2[:, ::-1] > 0, axis=1)
        sq = j == l_
        H[j[sq], j[sq]] = 2.0 * coef[r2[sq]]
        H[j[~sq], l_[~sq]] = coef[r2[~sq]]
        H[l_[~sq], j[~sq]] = coef[r2[~sq]]
    return c, a, H


# ---------------------------------------------------------------------------
# Fit quality (§3.4)
# ---------------------------------------------------------------------------


def _loo_r_squared(f: np.ndarray, e: np.ndarray | None, s: np.ndarray) -> float | None:
    if e is None:
        return None
    pos = s > 0
    if np.ptp(f[pos]) == 0:
        return None
    fbar = np.sum(s * f) / np.sum(s)
    den = float(np.sum(s * (f - fbar) ** 2))
    if not den > 0:
        return None
    return float(1.0 - np.sum(s * e * e) / den)


def _warn_loo(kind: str, details: dict[str, Any]) -> None:
    _warn(
        AccuracyWarning,
        f"leave-one-out residuals of the {kind} surrogate are undefined: a point fully "
        "determines its own fit "
        "(e.g. as many points as basis functions); loo_residuals and loo_r_squared are None",
        code="accuracy.loo_undefined",
        details={"kind": kind, **details},
    )


# ---------------------------------------------------------------------------
# RBF pieces (§3.3, §5.2)
# ---------------------------------------------------------------------------


def _spectrum(subspace: SubspaceResult | None, k: int) -> tuple[np.ndarray, float, bool]:
    """``(ell_base, frac, has_eigs)``.

    ``ell = eta * ell_base`` and the default nugget ``eta * frac``.
    """
    if subspace is None or subspace.estimator == "basis":
        return np.ones(k), 0.0, False
    lam = np.abs(np.asarray(subspace.eigenvalues, dtype=np.float64))
    total = float(np.sum(lam))
    lam1 = float(lam[0])
    if lam1 == 0.0:
        ell_base = np.ones(k)
    else:
        ell_base = total / np.maximum(lam[:k], _EPS * lam1)
    frac = float(np.sum(lam[subspace.n :]) / total) if total > 0 else 0.0
    return ell_base, frac, True


class _RBFState:
    """Quantities of one trial eta (§5.2 step 4)."""

    __slots__ = ("J", "L", "Qr", "alpha", "beta", "delta", "ell", "rt", "sigma2")

    J: float
    L: np.ndarray
    Qr: np.ndarray
    alpha: np.ndarray | None
    beta: np.ndarray
    delta: np.ndarray
    ell: np.ndarray
    rt: np.ndarray
    sigma2: float

    def __init__(self, **kw: Any) -> None:
        for key, val in kw.items():
            setattr(self, key, val)


def _rbf_trial(
    Y: np.ndarray,
    f: np.ndarray,
    B: np.ndarray,
    eta: float,
    ell_base: np.ndarray,
    frac: float,
    v: np.ndarray | None,
    sf2: float,
) -> _RBFState | None:
    """Objective ``J(eta)`` and the GLS fit; ``None`` if the Cholesky factorization fails."""
    M = Y.shape[0]
    ell = eta * ell_base
    if v is not None:
        delta = np.maximum(v / sf2, _NUGGET_FLOOR)
    else:
        delta = np.full(M, max(eta * frac, _NUGGET_FLOOR))
    if not (np.isfinite(ell).all() and np.all(ell > 0)):
        return None
    Ys = Y / np.sqrt(ell)
    K = np.exp(-cdist(Ys, Ys, "sqeuclidean"))
    K[np.diag_indices(M)] += delta
    try:
        L = scipy.linalg.cholesky(K, lower=True, check_finite=False)
    except (np.linalg.LinAlgError, ValueError):
        return None
    diag = np.diagonal(L)
    if not (np.isfinite(L).all() and np.all(diag > 0)):
        return None
    Bt = scipy.linalg.solve_triangular(L, B, lower=True, check_finite=False)
    ft = scipy.linalg.solve_triangular(L, f, lower=True, check_finite=False)
    Qr, Rr = np.linalg.qr(Bt, mode="reduced")
    qf = Qr.T @ ft
    beta = scipy.linalg.solve_triangular(Rr, qf, lower=False, check_finite=False)
    rt = ft - Qr @ qf
    # Scalar reductions by math.fsum: BLAS dot products may differ in the last bit with the
    # memory alignment of their operands, which would break bitwise reproducibility.
    sigma2 = max(math.fsum((rt * rt).tolist()) / M, 5.0 * _EPS * sf2)
    J = math.fsum(np.log(diag).tolist()) + 0.5 * M * math.log(sigma2) if sigma2 > 0 else math.inf
    if not math.isfinite(J) or not np.isfinite(beta).all():
        return None
    return _RBFState(
        J=J, L=L, Qr=Qr, alpha=None, beta=beta, delta=delta, ell=ell, sigma2=sigma2, rt=rt
    )


def _rbf_search(trial: Any, bounds: tuple[float, float]) -> float:
    """Deterministic grid (step <= 0.25) plus bounded refinement of log10(eta) [D-101]."""
    lo, hi = bounds
    G = max(1, math.ceil(round((hi - lo) / _GRID_STEP, 9)))
    grid = np.linspace(lo, hi, G + 1)
    Js = np.array([_J(trial(u)) for u in grid])
    if not np.isfinite(Js).any():
        raise NumericalError(
            "every trial RBF kernel matrix failed its Cholesky factorization",
            code="numerical.singular",
            details={"log10_eta_bounds": [lo, hi], "grid_points": int(G + 1)},
        )
    g = int(np.argmin(Js))
    _warn_flat(grid, Js, g)
    a, b = grid[max(g - 1, 0)], grid[min(g + 1, G)]
    big = np.finfo(np.float64).max

    def obj(u: float) -> float:
        val = _J(trial(float(u)))
        return val if math.isfinite(val) else big

    res = scipy.optimize.minimize_scalar(
        obj, bounds=(float(a), float(b)), method="bounded", options={"xatol": 1e-3}
    )
    u_ref = float(res.x)
    J_ref = _J(trial(u_ref))
    return u_ref if J_ref <= Js[g] else float(grid[g])


def _warn_flat(grid: np.ndarray, Js: np.ndarray, g: int) -> None:
    """Warn ``accuracy.flat_objective`` when the objective is flat around the selected point.

    Flat: J stays within ``1e-8 |J*|`` of its minimum over at least four decades of consecutive
    grid points containing the selected one [D-192]. A run that reaches the lower end of the
    grid does not warn: there the kernel part vanishes and the RBF reduces to its polynomial
    tail, so the choice does not change the fit [D-212].
    """
    thr = Js[g] + _FLAT_RTOL * abs(Js[g])
    a = g
    while a > 0 and Js[a - 1] <= thr:
        a -= 1
    b = g
    while b < Js.size - 1 and Js[b + 1] <= thr:
        b += 1
    span = float(grid[b] - grid[a])
    if span >= _FLAT_DECADES and a > 0:
        _warn(
            AccuracyWarning,
            f"the RBF likelihood is flat over {span:g} decades of log10(eta) around the selected "
            "value, so the value is arbitrary; the published bounds (f_variance=) or narrower "
            "log10_eta_bounds constrain it",
            code="accuracy.flat_objective",
            details={"span": span, "interval": [float(grid[a]), float(grid[b])]},
        )


def _search_interval(
    bounds: tuple[float, float] | None,
    log10_eta: float | None,
    f_variance: float | None,
    subspace: SubspaceResult | None,
    space: InputSpace,
    f: np.ndarray,
) -> tuple[tuple[float, float], str, float | None]:
    """Resolve the search interval of §3.3 [D-101, D-192].

    Returns ``(interval, source, log10_eta)``; ``source`` is ``"given"``, ``"f_variance"`` or
    ``"default"``. An empty published interval collapses to its upper end, which becomes a
    fixed ``log10_eta``, and warns ``accuracy.empty_bounds``.
    """
    if bounds is not None or log10_eta is not None:
        return (bounds if bounds is not None else _DEFAULT_BOUNDS), "given", log10_eta
    if f_variance is None or subspace is None or subspace.estimator == "basis":
        return _DEFAULT_BOUNDS, "default", None
    total = math.fsum(np.abs(np.asarray(subspace.eigenvalues, dtype=np.float64)).tolist())
    if not total > 0:
        return _DEFAULT_BOUNDS, "default", None
    c1 = _poincare_constant(space)
    lo = math.log10(2.0 * f_variance / total)
    hi = math.log10(2.0 * c1)
    if lo < hi:
        return (lo, hi), "f_variance", None
    if np.ptp(f) == 0:  # a constant f runs no search and keeps log10_eta = 0 (§5.2 step 2)
        return (min(lo, hi), hi), "f_variance", None
    if lo == hi:  # a single point: nothing to search, nothing empty
        return (hi, hi), "f_variance", hi
    _warn(
        AccuracyWarning,
        f"the published bounds on log10(eta) are empty ({lo:.4g} > {hi:.4g}): f_variance "
        "exceeds C1 times the eigenvalue sum; the upper end is used",
        code="accuracy.empty_bounds",
        details={
            "lower": lo,
            "upper": hi,
            "f_variance": f_variance,
            "eigenvalue_sum": total,
            "C1": c1,
        },
    )
    return (hi, hi), "f_variance", hi


def _J(state: _RBFState | None) -> float:
    return math.inf if state is None else state.J


def _rbf_kernel(Y: np.ndarray, centers: np.ndarray, ell: np.ndarray) -> np.ndarray:
    s = np.sqrt(ell)
    K: np.ndarray = np.exp(-cdist(Y / s, centers / s, "sqeuclidean"))
    return K


# ---------------------------------------------------------------------------
# Result types (§4.4)
# ---------------------------------------------------------------------------


def _coords(Y: Any, k: int) -> np.ndarray:
    """Validate input coordinates ``(K, k)`` (1-D read as ``(K, 1)`` when ``k == 1``) [D-64]."""
    arr = _v._as_2d(_v.coerce(Y, "Y"), "Y", k, allow_1d=True)
    _v.check_finite(arr, "Y")
    return np.array(arr, dtype=np.float64, order="C", copy=True)


def _check_common(r: Any, k_from_fields: int) -> int:
    space, sub = r.space, r.subspace
    if sub is not None:
        _invariant(
            type(sub.space) is type(space) and sub.space == space,
            "subspace.space must equal space",
        )
        _invariant(r.qoi == sub.qoi, "qoi must equal subspace.qoi")
        k = sub.n
    else:
        k = space.dim
    _invariant(k_from_fields == k, "Y must have k = subspace.n (or space.dim) columns")
    M: int = r.Y.shape[0]
    _invariant(r.Y.ndim == 2 and M >= 1, "Y must be (M, k)")
    _invariant(r.degree >= 0, "degree must be >= 0")
    idx = _indices(k, r.degree)
    _invariant(
        r.indices.shape == idx.shape and np.array_equal(r.indices, idx),
        "indices must be the index set of §3.1",
    )
    for name in ("Y", "f", "residuals"):
        _invariant(bool(np.isfinite(getattr(r, name)).all()), f"{name} must be finite")
    _invariant(r.f.shape == (M,) and r.residuals.shape == (M,), "f and residuals must be (M,)")
    if r.loo_residuals is None:
        _invariant(
            r.loo_r_squared is None, "loo_r_squared must be None when loo_residuals is None"
        )
    else:
        _invariant(
            r.loo_residuals.shape == (M,) and bool(np.isfinite(r.loo_residuals).all()),
            "loo_residuals must be finite (M,)",
        )
    if r.samples is not None:
        smp = r.samples
        _invariant(smp.domain.subspace == sub, "samples.domain.subspace must equal subspace")
        _invariant(
            smp.Y.shape == r.Y.shape and np.array_equal(smp.Y, r.Y), "Y must equal samples.Y"
        )
        _invariant(np.array_equal(smp.means, r.f), "f must equal samples.means")
    return M


@_register_result
class PolynomialSurrogate(QoIResult):
    r"""Polynomial surrogate :math:`g(y) = b(y)^\top \beta` on the surrogate's input coordinates.

    Built by :func:`fit_surrogate` or :func:`fit_surrogate_from_model` with
    ``kind="poly"``; the constructor is used when a saved result is loaded.
    The input coordinates :math:`y` are the normalized active variables
    :math:`W_1^\top x` on a subspace, or the normalized inputs :math:`x` when
    ``subspace`` is ``None``; :math:`k` denotes their number.

    Parameters
    ----------
    space : InputSpace
        The input space.
    subspace : SubspaceResult or None
        The subspace whose active variables are the input coordinates, or
        ``None`` for all normalized inputs.
    degree : int
        Total degree :math:`d \ge 0`.
    indices : ndarray of int64, shape (p, k)
        Multi-indices :math:`\alpha` of the monomials :math:`y^\alpha`, ordered
        by total degree, then lexicographically descending;
        :math:`p = \binom{k+d}{k}`.
    coefficients : ndarray of float64, shape (p,)
        The coefficients :math:`\hat\beta`, in the order of ``indices``.
    Y : ndarray of float64, shape (M, k)
        Training points in the input coordinates (normalized).
    f : ndarray of float64, shape (M,)
        Training values, unweighted.
    sample_weights : ndarray of float64, shape (M,)
        Weights :math:`s_i \ge 0` of the least-squares fit (all ones if none
        were given).
    residuals : ndarray of float64, shape (M,)
        In-sample residuals :math:`f_i - g(y_i)`.
    loo_residuals : ndarray of float64, shape (M,), or None
        Leave-one-out residuals :math:`e_i`, or ``None`` when undefined.
    loo_r_squared : float or None
        Leave-one-out coefficient of determination :math:`R^2_{\mathrm{LOO}}`,
        or ``None`` when undefined.
    samples : ConditionalSamples or None
        The model evaluations behind ``Y`` and ``f`` on the model path
        (:func:`fit_surrogate_from_model`), else ``None``.
    provenance : Provenance
        How the result was computed.
    qoi : str, default "f"
        Name of the quantity of interest.

    See Also
    --------
    fit_surrogate : Fit a surrogate to data.
    quadratic_terms : Constant, linear and quadratic terms.
    RBFSurrogate : The radial-basis surrogate.

    Notes
    -----
    The basis is the monomials :math:`y^\alpha = \prod_{j=1}^k y_j^{\alpha_j}`
    with :math:`\lvert\alpha\rvert \le d`, and the coefficients solve the
    weighted least-squares problem

    .. math::

       \hat\beta = \arg\min_\beta \sum_{i=1}^M s_i \bigl(f_i - b(y_i)^\top\beta\bigr)^2 .

    The leave-one-out residuals are :math:`e_i = r_i / (1 - h_{ii})`, with
    :math:`r = f - B\hat\beta` and :math:`h_{ii}` the leverages of the
    weighted fit, and

    .. math::

       R^2_{\mathrm{LOO}} = 1 - \frac{\sum_i s_i e_i^2}{\sum_i s_i (f_i - \bar f_s)^2},
       \qquad \bar f_s = \frac{\sum_i s_i f_i}{\sum_i s_i} .

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([-1.0, -1.0], [1.0, 1.0])
    >>> X = space.sample(20, rng=0)
    >>> s = ac.fit_surrogate(X=X, f=X[:, 0] ** 2, space=space, subspace=None, kind="poly")
    >>> s.indices.tolist()
    [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]
    >>> print(np.round(s.coefficients, 3) + 0.0)
    [0. 0. 0. 1. 0. 0.]
    """

    TYPE = "PolynomialSurrogate"
    SCHEMA_VERSION = 1

    space: InputSpace
    subspace: SubspaceResult | None
    degree: int
    indices: NDArray[np.int64]
    coefficients: NDArray[np.float64]
    Y: NDArray[np.float64]
    f: NDArray[np.float64]
    sample_weights: NDArray[np.float64]
    residuals: NDArray[np.float64]
    loo_residuals: NDArray[np.float64] | None
    loo_r_squared: float | None
    samples: ConditionalSamples | None

    @property
    def active_dim(self) -> int:
        """Number of input coordinates :math:`k`.

        :math:`k = n` (the subspace dimension) on a subspace, :math:`k = m`
        on all inputs.

        Returns
        -------
        int
            The number :math:`k`.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([-1.0, -1.0], [1.0, 1.0])
        >>> X = space.sample(20, rng=0)
        >>> s = ac.fit_surrogate(X=X, f=X[:, 0], space=space, subspace=None, kind="poly")
        >>> s.active_dim
        2
        """
        return self.subspace.n if self.subspace is not None else self.space.dim

    @property
    def n_samples(self) -> int:
        """Number of training points :math:`M`.

        Returns
        -------
        int
            The number of rows of ``Y``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([-1.0, -1.0], [1.0, 1.0])
        >>> X = space.sample(20, rng=0)
        >>> s = ac.fit_surrogate(X=X, f=X[:, 0], space=space, subspace=None, kind="poly")
        >>> s.n_samples
        20
        """
        return int(self.Y.shape[0])

    def evaluate(self, Y: ArrayLike) -> NDArray[np.float64]:
        r"""Evaluate :math:`g(y) = b(y)^\top\beta` at points in the input coordinates.

        Parameters
        ----------
        Y : array_like, shape (K, k)
            Points in the surrogate's input coordinates: normalized active
            variables on a subspace, normalized inputs when ``subspace`` is
            ``None``. A 1-D array is read as ``(K, 1)`` when :math:`k = 1`.

        Returns
        -------
        ndarray of float64, shape (K,)
            The values :math:`g(y)`.

        Raises
        ------
        InputError
            ``input.shape``, ``input.dtype`` or ``input.nonfinite`` if `Y` is
            malformed.

        See Also
        --------
        predict : Evaluate at physical inputs.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([-1.0, -1.0], [1.0, 1.0])
        >>> X = space.sample(20, rng=0)
        >>> s = ac.fit_surrogate(X=X, f=X[:, 0] ** 2, space=space, subspace=None, kind="poly")
        >>> print(np.round(s.evaluate([[0.5, 0.0], [-1.0, 1.0]]), 3))
        [0.25 1.  ]
        """
        return _poly_evaluate(self, Y)

    def gradient(self, Y: ArrayLike) -> NDArray[np.float64]:
        r"""Evaluate :math:`\nabla_y g(y)` at points in the input coordinates.

        Parameters
        ----------
        Y : array_like, shape (K, k)
            Points in the surrogate's input coordinates, as for
            :meth:`evaluate`.

        Returns
        -------
        ndarray of float64, shape (K, k)
            Gradients with respect to the input coordinates.

        Raises
        ------
        InputError
            ``input.shape``, ``input.dtype`` or ``input.nonfinite`` if `Y` is
            malformed.

        See Also
        --------
        predict_gradient : Physical gradients at physical inputs.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([-1.0, -1.0], [1.0, 1.0])
        >>> X = space.sample(20, rng=0)
        >>> s = ac.fit_surrogate(X=X, f=X[:, 0] ** 2, space=space, subspace=None, kind="poly")
        >>> print(np.round(s.gradient([[0.5, 0.0]]), 3) + 0.0)
        [[1. 0.]]
        """
        return _poly_gradient(self, Y)

    def interval(
        self,
        Y: ArrayLike,
        *,
        level: float = 0.95,
        kind: Literal["prediction", "confidence"] = "prediction",
        side: Literal["two", "lower", "upper"] = "two",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Return prediction or confidence limits at active coordinates.

        Parameters
        ----------
        Y : array_like, shape (K, k), float
            Points in the surrogate's input coordinates; 1-D is read as
            ``(K, 1)`` when ``k == 1``.
        level : float, default 0.95, keyword-only
            Confidence level in :math:`(0, 1)`.
        kind : {"prediction", "confidence"}, default "prediction", keyword-only
            Limits for a new observation (``c = 1``) or for the mean (``c = 0``).
        side : {"two", "lower", "upper"}, default "two", keyword-only
            Two-sided limits, or one-sided with the other end :math:`\mp\infty`.

        Returns
        -------
        lower : numpy.ndarray, shape (K,), float64
            Lower limits; ``-inf`` for ``side="upper"``.
        upper : numpy.ndarray, shape (K,), float64
            Upper limits; ``+inf`` for ``side="lower"``.

        Raises
        ------
        InputError
            ``input.parameter`` for a bad option, or for no residual degrees of
            freedom (``details["reason"] == "no_dof"``); the spec 02 codes for a
            bad ``Y``.

        See Also
        --------
        predict_interval : The same limits at physical inputs.

        Notes
        -----
        Under :math:`f_i = b(y_i)^\top\beta + \varepsilon_i`,
        :math:`\varepsilon_i \sim N(0, \sigma^2/s_i)`, with
        :math:`\nu = \#\lbrace i: s_i > 0\rbrace - p` and
        :math:`s^2 = \sum_i s_i r_i^2/\nu`, a new point having weight 1, the
        limits are :math:`g(y) \pm t\, s\sqrt{c + b(y)^\top(B^\top SB)^{-1}b(y)}`
        with :math:`t = t_{\nu, 1-\alpha/2}` (two-sided) or
        :math:`t_{\nu, \mathrm{level}}` (one-sided) [1]_ [D-200]. They are
        conditional on :math:`W_1` and do not include its estimation error.

        References
        ----------
        .. [1] D. C. Montgomery, E. A. Peck, G. G. Vining. *Introduction to Linear
           Regression Analysis*, 5th ed. Wiley, 2012, §3.4-3.5.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform.reference(1)
        >>> X = np.linspace(-1.0, 1.0, 21)[:, None]
        >>> f = 1.0 + X[:, 0] + 0.01 * np.sin(40.0 * X[:, 0])
        >>> s = ac.fit_surrogate(X=X, f=f, space=space, subspace=None, kind="poly", degree=1)
        >>> lo, hi = s.interval([[0.0]], kind="confidence")
        >>> bool(lo[0] < 1.0 < hi[0])
        True
        """
        _, lower, upper, *_ = _poly_interval(self, Y, level, kind, side)
        return lower, upper

    def _check(self) -> None:
        M = _check_common(self, self.Y.shape[1] if self.Y.ndim == 2 else -1)
        p = self.indices.shape[0]
        _invariant(
            self.coefficients.shape == (p,) and bool(np.isfinite(self.coefficients).all()),
            "coefficients must be finite (p,)",
        )
        w = self.sample_weights
        _invariant(
            w.shape == (M,) and bool(np.isfinite(w).all()), "sample_weights must be finite (M,)"
        )
        _invariant(
            bool(np.all(w >= 0)) and bool(np.any(w > 0)),
            "sample_weights must be >= 0 with a positive entry",
        )


@_register_result
class RBFSurrogate(QoIResult):
    r"""Radial-basis surrogate with a polynomial tail on the surrogate's input coordinates.

    Built by :func:`fit_surrogate` or :func:`fit_surrogate_from_model` with
    ``kind="rbf"``; the constructor is used when a saved result is loaded.
    The input coordinates :math:`y` are the normalized active variables
    :math:`W_1^\top x` on a subspace, or the normalized inputs :math:`x` when
    ``subspace`` is ``None``; :math:`k` denotes their number.

    Parameters
    ----------
    space : InputSpace
        The input space.
    subspace : SubspaceResult or None
        The subspace whose active variables are the input coordinates, or
        ``None`` for all normalized inputs.
    degree : int
        Total degree :math:`d \ge 0` of the polynomial tail.
    indices : ndarray of int64, shape (p, k)
        Multi-indices of the tail's monomials, ordered as in
        :class:`PolynomialSurrogate`.
    Y : ndarray of float64, shape (M, k)
        Training points (the centres) in the input coordinates (normalized).
    f : ndarray of float64, shape (M,)
        Training values.
    radial_coefficients : ndarray of float64, shape (M,)
        The coefficients :math:`\alpha` of the kernel terms.
    tail_coefficients : ndarray of float64, shape (p,)
        The coefficients :math:`\beta` of the polynomial tail.
    length_scales : ndarray of float64, shape (k,)
        The length scales :math:`\ell_j > 0`.
    log10_eta : float
        The selected (or given) hyperparameter :math:`\log_{10}\eta`.
    nugget : ndarray of float64, shape (M,)
        The dimensionless nugget :math:`\delta_i \ge 10^{-6}`, relative to the
        amplitude.
    noise_variance : ndarray of float64, shape (M,), or None
        Noise variances :math:`v_i` in units of :math:`f^2`, as given or from
        the model path, else ``None``.
    training_variance : float
        :math:`s_f^2`, the sample variance of ``f`` (ddof 0) that converts
        ``noise_variance`` into the nugget.
    amplitude : float
        The amplitude estimate :math:`\hat\sigma^2_\eta` at the selected
        :math:`\eta`; it does not affect :math:`g`.
    objective : float or None
        :math:`J(\eta)` at the selected :math:`\eta`; ``None`` for constant
        ``f``.
    residuals : ndarray of float64, shape (M,)
        In-sample residuals :math:`f_i - g(y_i) = \delta_i \alpha_i`.
    loo_residuals : ndarray of float64, shape (M,), or None
        Leave-one-out residuals, or ``None`` when undefined.
    loo_r_squared : float or None
        Leave-one-out coefficient of determination, or ``None`` when
        undefined.
    samples : ConditionalSamples or None
        The model evaluations behind ``Y`` and ``f`` on the model path, else
        ``None``.
    provenance : Provenance
        How the result was computed.
    qoi : str, default "f"
        Name of the quantity of interest.

    See Also
    --------
    fit_surrogate : Fit a surrogate to data.
    PolynomialSurrogate : The polynomial surrogate.

    Notes
    -----
    The surrogate is

    .. math::

       g(y) = \sum_{i=1}^M \alpha_i\,\kappa(y, y_i) + b(y)^\top\beta,
       \qquad \kappa(y, y') = \exp\Bigl(-\sum_{j=1}^k \frac{(y_j - y'_j)^2}{\ell_j}\Bigr),

    with the coefficients solving
    :math:`K_\eta \alpha + B\beta = f`, :math:`B^\top\alpha = 0`, where
    :math:`K_\eta = R_\eta + \mathrm{diag}(\delta)` and :math:`R_\eta` is the
    kernel matrix of the centres. See :func:`fit_surrogate` for the length
    scales, the nugget and the choice of :math:`\eta`.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([-1.0], [1.0])
    >>> X = np.linspace(-1.0, 1.0, 15)[:, None]
    >>> s = ac.fit_surrogate(X=X, f=np.sin(3 * X[:, 0]), space=space, subspace=None)
    >>> s.radial_coefficients.shape, s.tail_coefficients.shape
    ((15,), (3,))
    >>> print(np.round(s.evaluate([[0.1]]), 2), np.round(np.sin(0.3), 2))
    [0.3] 0.3
    """

    TYPE = "RBFSurrogate"
    SCHEMA_VERSION = 2  # 2: f_variance renamed training_variance [D-206]

    space: InputSpace
    subspace: SubspaceResult | None
    degree: int
    indices: NDArray[np.int64]
    Y: NDArray[np.float64]
    f: NDArray[np.float64]
    radial_coefficients: NDArray[np.float64]
    tail_coefficients: NDArray[np.float64]
    length_scales: NDArray[np.float64]
    log10_eta: float
    nugget: NDArray[np.float64]
    noise_variance: NDArray[np.float64] | None
    training_variance: float
    amplitude: float
    objective: float | None
    residuals: NDArray[np.float64]
    loo_residuals: NDArray[np.float64] | None
    loo_r_squared: float | None
    samples: ConditionalSamples | None

    @property
    def active_dim(self) -> int:
        """Number of input coordinates :math:`k`.

        :math:`k = n` (the subspace dimension) on a subspace, :math:`k = m`
        on all inputs.

        Returns
        -------
        int
            The number :math:`k`.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([-1.0], [1.0])
        >>> X = np.linspace(-1.0, 1.0, 15)[:, None]
        >>> s = ac.fit_surrogate(X=X, f=np.sin(3 * X[:, 0]), space=space, subspace=None)
        >>> s.active_dim
        1
        """
        return self.subspace.n if self.subspace is not None else self.space.dim

    @property
    def n_samples(self) -> int:
        """Number of training points :math:`M`.

        Returns
        -------
        int
            The number of rows of ``Y``.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([-1.0], [1.0])
        >>> X = np.linspace(-1.0, 1.0, 15)[:, None]
        >>> s = ac.fit_surrogate(X=X, f=np.sin(3 * X[:, 0]), space=space, subspace=None)
        >>> s.n_samples
        15
        """
        return int(self.Y.shape[0])

    def evaluate(self, Y: ArrayLike) -> NDArray[np.float64]:
        r"""Evaluate :math:`g(y)` at points in the input coordinates.

        Parameters
        ----------
        Y : array_like, shape (K, k)
            Points in the surrogate's input coordinates: normalized active
            variables on a subspace, normalized inputs when ``subspace`` is
            ``None``. A 1-D array is read as ``(K, 1)`` when :math:`k = 1`.

        Returns
        -------
        ndarray of float64, shape (K,)
            The values
            :math:`g(y) = \sum_i \alpha_i \kappa(y, y_i) + b(y)^\top\beta`.

        Raises
        ------
        InputError
            ``input.shape``, ``input.dtype`` or ``input.nonfinite`` if `Y` is
            malformed.

        See Also
        --------
        predict : Evaluate at physical inputs.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([-1.0], [1.0])
        >>> X = np.linspace(-1.0, 1.0, 15)[:, None]
        >>> s = ac.fit_surrogate(X=X, f=np.sin(3 * X[:, 0]), space=space, subspace=None)
        >>> print(np.round(s.evaluate([0.1, 0.5]), 2))
        [0.3 1. ]
        """
        return _rbf_evaluate(self, Y)

    def gradient(self, Y: ArrayLike) -> NDArray[np.float64]:
        r"""Evaluate :math:`\nabla_y g(y)` at points in the input coordinates.

        Parameters
        ----------
        Y : array_like, shape (K, k)
            Points in the surrogate's input coordinates, as for
            :meth:`evaluate`.

        Returns
        -------
        ndarray of float64, shape (K, k)
            Gradients with respect to the input coordinates,
            :math:`\partial g / \partial y_j = -\sum_i \alpha_i\, 2 (y_j - y_{ij}) / \ell_j\,
            \kappa(y, y_i) + \partial b(y)^\top\beta / \partial y_j`.

        Raises
        ------
        InputError
            ``input.shape``, ``input.dtype`` or ``input.nonfinite`` if `Y` is
            malformed.

        See Also
        --------
        predict_gradient : Physical gradients at physical inputs.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([-1.0], [1.0])
        >>> X = np.linspace(-1.0, 1.0, 15)[:, None]
        >>> s = ac.fit_surrogate(X=X, f=np.sin(3 * X[:, 0]), space=space, subspace=None)
        >>> print(np.round(s.gradient([[0.0]]), 1), 3 * np.cos(0.0))
        [[3.]] 3.0
        """
        return _rbf_gradient(self, Y)

    def _check(self) -> None:
        k = self.Y.shape[1] if self.Y.ndim == 2 else -1
        M = _check_common(self, k)
        p = self.indices.shape[0]
        _invariant(self.radial_coefficients.shape == (M,), "radial_coefficients must be (M,)")
        _invariant(self.tail_coefficients.shape == (p,), "tail_coefficients must be (p,)")
        _invariant(self.length_scales.shape == (k,), "length_scales must be (k,)")
        _invariant(self.nugget.shape == (M,), "nugget must be (M,)")
        for name in ("radial_coefficients", "tail_coefficients", "length_scales", "nugget"):
            _invariant(bool(np.isfinite(getattr(self, name)).all()), f"{name} must be finite")
        _invariant(bool(np.all(self.length_scales > 0)), "length_scales must be > 0")
        _invariant(bool(np.all(self.nugget >= _NUGGET_FLOOR)), "nugget must be >= 1e-6")
        _invariant(math.isfinite(self.log10_eta), "log10_eta must be finite")
        _invariant(
            math.isfinite(self.training_variance) and self.training_variance >= 0,
            "training_variance must be >= 0",
        )
        _invariant(math.isfinite(self.amplitude) and self.amplitude >= 0, "amplitude must be >= 0")
        if self.objective is not None:
            _invariant(math.isfinite(self.objective), "objective must be finite")
        nv = self.noise_variance
        if nv is not None:
            _invariant(
                nv.shape == (M,) and bool(np.isfinite(nv).all()) and bool(np.all(nv >= 0)),
                "noise_variance must be finite, >= 0, (M,)",
            )
        if self.samples is not None:
            _invariant(
                nv is not None and np.array_equal(nv, self.samples.noise_variance),
                "noise_variance must equal samples.noise_variance",
            )


# ---------------------------------------------------------------------------
# Evaluation (private module functions called by the methods [D-22])
# ---------------------------------------------------------------------------


@_register_migration("RBFSurrogate", 1)
def _rbf_v1_to_v2(fields: dict[str, Any]) -> dict[str, Any]:
    """Version 1 called the training-value variance ``f_variance`` [D-206]."""
    out = dict(fields)
    out["training_variance"] = out.pop("f_variance")
    return out


def _poly_evaluate(s: PolynomialSurrogate, Y: Any) -> np.ndarray:
    Yv = _coords(Y, s.active_dim)
    return _basis(Yv, s.indices) @ s.coefficients


def _poly_gradient(s: PolynomialSurrogate, Y: Any) -> np.ndarray:
    Yv = _coords(Y, s.active_dim)
    return _basis_gradient_times(Yv, s.indices, s.coefficients)


def _rbf_evaluate(s: RBFSurrogate, Y: Any) -> np.ndarray:
    Yv = _coords(Y, s.active_dim)
    Kmat = _rbf_kernel(Yv, s.Y, s.length_scales)
    return Kmat @ s.radial_coefficients + _basis(Yv, s.indices) @ s.tail_coefficients


def _rbf_gradient(s: RBFSurrogate, Y: Any) -> np.ndarray:
    Yv = _coords(Y, s.active_dim)
    Kmat = _rbf_kernel(Yv, s.Y, s.length_scales)
    Ka = Kmat * s.radial_coefficients
    radial = -2.0 / s.length_scales * (Yv * Ka.sum(axis=1)[:, None] - Ka @ s.Y)
    out: np.ndarray = radial + _basis_gradient_times(Yv, s.indices, s.tail_coefficients)
    return out


# ---------------------------------------------------------------------------
# Fitting core (§5.1 steps 3-5, §5.2)
# ---------------------------------------------------------------------------


def _fit_poly(
    Y: np.ndarray, f: np.ndarray, degree: int, s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, float | None]:
    indices, B, coef, h = _poly_core(Y, f, degree, s)
    r = f - B @ coef
    one_minus_h = 1.0 - h
    if np.any(one_minus_h <= _LEVERAGE_TOL):
        _warn_loo("poly", {"min_one_minus_leverage": float(np.min(one_minus_h))})
        return indices, coef, r, None, None
    e = r / one_minus_h
    return indices, coef, r, e, _loo_r_squared(f, e, s)


def _fit_rbf(
    Y: np.ndarray,
    f: np.ndarray,
    degree: int,
    subspace: SubspaceResult | None,
    v: np.ndarray | None,
    log10_eta: float | None,
    bounds: tuple[float, float],
) -> dict[str, Any]:
    M, k = Y.shape
    indices = _indices(k, degree)
    p = indices.shape[0]
    if M < p:
        raise _too_few(p, M, f"an RBF with a degree-{degree} tail in {k} variables")
    B = _basis(Y, indices)
    ell_base, frac, _ = _spectrum(subspace, k)
    common = {"indices": indices, "noise_variance": v}
    if np.ptp(f) == 0:  # constant f: the tail fits exactly (§5.2 step 2)
        u = 0.0 if log10_eta is None else float(log10_eta)
        beta = np.linalg.lstsq(B, f, rcond=None)[0]
        return {
            **common,
            "radial_coefficients": np.zeros(M),
            "tail_coefficients": beta,
            "length_scales": (10.0**u) * ell_base,
            "log10_eta": u,
            "nugget": np.full(M, _NUGGET_FLOOR),
            "training_variance": 0.0,
            "amplitude": 0.0,
            "objective": None,
            "residuals": f - B @ beta,
            "loo_residuals": np.zeros(M),
            "loo_r_squared": None,
        }
    _scaled_svd(B, np.ones(M))  # full column rank (§5.2 step 3)
    fbar = math.fsum(f.tolist()) / M
    sf2 = math.fsum(((f - fbar) ** 2).tolist()) / M

    def trial(u: float) -> _RBFState | None:
        return _rbf_trial(Y, f, B, 10.0**u, ell_base, frac, v, sf2)

    u = _rbf_search(trial, bounds) if log10_eta is None else float(log10_eta)
    st = trial(u)
    if st is None:
        raise NumericalError(
            f"the RBF kernel matrix at log10_eta = {u} failed its Cholesky factorization",
            code="numerical.singular",
            details={"log10_eta": u},
        )
    alpha = scipy.linalg.solve_triangular(st.L, st.rt, lower=True, trans="T", check_finite=False)
    if not np.isfinite(alpha).all():
        raise NumericalError(
            "non-finite RBF coefficients",
            code="numerical.nonfinite",
            details={"what": "radial_coefficients"},
        )
    # LOO (§5.2 step 7): diag of (S^{-1})_11 = L^{-T} (I - Qr Qr^T) L^{-1}
    Linv = scipy.linalg.solve_triangular(st.L, np.eye(M), lower=True, check_finite=False)
    Z = Linv - st.Qr @ (st.Qr.T @ Linv)
    dS = np.sum(Z * Z, axis=0)
    # Scale: max diag(K^{-1}) >= max diag((S^{-1})_11). At M = p every (S^{-1})_ii is 0 up to
    # rounding, so the spec's max_l (S^{-1})_ll would be rounding noise too.
    ref = float(np.max(np.sum(Linv * Linv, axis=0)))
    if np.any(dS <= _RIPPA_TOL * ref):
        _warn_loo("rbf", {"min_relative_diagonal": float(dS.min() / ref)})
        e, r2 = None, None
    else:
        e = alpha / dS
        r2 = _loo_r_squared(f, e, np.ones(M))
    return {
        **common,
        "radial_coefficients": alpha,
        "tail_coefficients": st.beta,
        "length_scales": st.ell,
        "log10_eta": float(u),
        "nugget": st.delta,
        "training_variance": sf2,
        "amplitude": st.sigma2,
        "objective": st.J,
        "residuals": st.delta * alpha,
        "loo_residuals": e,
        "loo_r_squared": r2,
    }


def _fit_active(
    Y: np.ndarray,
    f: np.ndarray,
    *,
    space: InputSpace,
    subspace: SubspaceResult | None,
    kind: str,
    degree: int,
    sample_weights: np.ndarray | None,
    noise_variance: np.ndarray | None,
    log10_eta: float | None,
    log10_eta_bounds: tuple[float, float],
    qoi: str,
    samples: ConditionalSamples | None,
    provenance: Provenance | None = None,
) -> PolynomialSurrogate | RBFSurrogate:
    """Private core (spec 09 §5.5): steps 3-5 of §5.1 on given normalized input coordinates ``Y``.

    ``provenance`` (an addition to the spec's signature) is the caller's;
    ``None`` builds a minimal one for ``fit_surrogate``.
    """
    if provenance is None:
        provenance = _make_provenance(
            "fit_surrogate",
            {"kind": kind, "degree": degree, "qoi": qoi},
            space=space,
            digests={"Y": _v.digest(Y), "f": _v.digest(f)},
        )
    if kind == "poly":
        s = np.ones(Y.shape[0]) if sample_weights is None else sample_weights
        indices, coef, r, e, r2 = _fit_poly(Y, f, degree, s)
        return PolynomialSurrogate(
            space=space,
            subspace=subspace,
            degree=degree,
            indices=indices,
            coefficients=coef,
            Y=Y,
            f=f,
            sample_weights=s,
            residuals=r,
            loo_residuals=e,
            loo_r_squared=r2,
            samples=samples,
            qoi=qoi,
            provenance=provenance,
        )
    fields = _fit_rbf(Y, f, degree, subspace, noise_variance, log10_eta, log10_eta_bounds)
    return RBFSurrogate(
        space=space,
        subspace=subspace,
        degree=degree,
        Y=Y,
        f=f,
        samples=samples,
        qoi=qoi,
        provenance=provenance,
        **fields,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _param(message: str, param: str, **details: Any) -> InputError:
    return InputError(message, code="input.parameter", details={"param": param, **details})


def _check_subspace(subspace: Any) -> SubspaceResult | None:
    if subspace is not None and not isinstance(subspace, SubspaceResult):
        raise _param(
            f"subspace: expected a SubspaceResult or None, got {type(subspace).__name__}",
            "subspace",
        )
    return subspace


def _resolve_qoi(qoi: Any, bound: str | None) -> str:
    if qoi is None:
        return "f" if bound is None else bound
    if not isinstance(qoi, str) or not qoi:
        raise _param("qoi: expected a non-empty string", "qoi")
    if bound is not None and qoi != bound:
        raise _param(
            f"qoi: {qoi!r} differs from the subspace's {bound!r}", "qoi", reason="qoi_mismatch"
        )
    return qoi


def _check_kind_degree(kind: Any, degree: Any) -> tuple[Literal["rbf", "poly"], int]:
    kind = _v.choice(kind, "kind", ("rbf", "poly"))
    degree = _v.integer(degree, "degree", lo=0)
    return cast(Literal["rbf", "poly"], kind), degree


def _check_bounds(bounds: Any, log10_eta: Any) -> tuple[tuple[float, float] | None, float | None]:
    """Validate explicit bounds (or ``None``) and a fixed ``log10_eta``.

    A fixed ``log10_eta`` must lie inside the given bounds, or inside the default interval when
    none are given.
    """
    if bounds is None:
        lo, hi = _DEFAULT_BOUNDS
        eta = None if log10_eta is None else _v.real(log10_eta, "log10_eta", lo=lo, hi=hi)
        return None, eta
    try:
        lo_raw, hi_raw = bounds
    except (TypeError, ValueError) as exc:
        raise _param(
            "log10_eta_bounds: expected a pair (lower, upper)", "log10_eta_bounds"
        ) from exc
    lo = _v.real(lo_raw, "log10_eta_bounds")
    hi = _v.real(hi_raw, "log10_eta_bounds")
    if not lo < hi:
        raise _param(
            f"log10_eta_bounds: expected lower < upper, got ({lo}, {hi})",
            "log10_eta_bounds",
            got=[lo, hi],
        )
    eta = None if log10_eta is None else _v.real(log10_eta, "log10_eta", lo=lo, hi=hi)
    return (lo, hi), eta


def _not_for_kind(name: str, kind: str) -> InputError:
    return _param(f"{name}: not used by kind={kind!r}", name, reason="not_for_kind", kind=kind)


def _check_rbf_args(
    kind: str, log10_eta: Any, bounds: Any, f_variance: Any = None
) -> tuple[tuple[float, float] | None, float | None, float | None]:
    """Validate the RBF-only arguments; returns ``(bounds or None, log10_eta, f_variance)``."""
    if kind != "rbf":
        if log10_eta is not None:
            raise _not_for_kind("log10_eta", kind)
        if f_variance is not None:
            raise _not_for_kind("f_variance", kind)
        if bounds is not None:
            try:  # the old default (-10, 1) is accepted, as before
                default = tuple(bounds) == _DEFAULT_BOUNDS
            except TypeError:
                default = False
            if not default:
                raise _not_for_kind("log10_eta_bounds", kind)
        return None, None, None
    fv = None
    if f_variance is not None:
        fv = _v.real(f_variance, "f_variance", lo=0.0, lo_open=True)
    b, eta = _check_bounds(bounds, log10_eta)
    return b, eta, fv


def _noise_variance(v: Any, M: int) -> np.ndarray:
    arr = _v.coerce(v, "noise_variance")
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise _v._shape_error("noise_variance", f"({M},)", arr.shape, [M])
    if arr.shape[0] != M:
        raise _v._size_error("noise_variance", M, arr.shape[0])
    _v.check_finite(arr, "noise_variance")
    if np.any(arr < 0):
        i = int(np.argmax(arr < 0))
        raise _param(
            f"noise_variance: entry {i} is negative ({arr[i]!r})",
            "noise_variance",
            reason="negative",
            row=i,
        )
    return np.array(arr, dtype=np.float64, copy=True)


# ---------------------------------------------------------------------------
# Public functions (§4.2)
# ---------------------------------------------------------------------------


def fit_surrogate(
    *,
    X: ArrayLike,
    f: ArrayLike,
    space: InputSpace | None = None,
    subspace: SubspaceResult | None,
    kind: Literal["rbf", "poly"] = "rbf",
    degree: int = 2,
    sample_weights: ArrayLike | None = None,
    noise_variance: ArrayLike | None = None,
    log10_eta: float | None = None,
    log10_eta_bounds: tuple[float, float] | None = None,
    f_variance: float | None = None,
    qoi: str | None = None,
) -> PolynomialSurrogate | RBFSurrogate:
    r"""Fit a polynomial or radial-basis surrogate to data on the active variables.

    The physical points `X` are normalized with the input space and projected
    onto the active variables, :math:`y_i = W_1^\top x_i`, and the surrogate
    :math:`g` is fitted to :math:`(y_i, f_i)`. With ``subspace=None`` it is
    fitted on all normalized inputs, :math:`y_i = x_i`. The fit is
    deterministic.

    Parameters
    ----------
    X : array_like, shape (M, m)
        Input points in **physical** coordinates, inside the support of the
        space.
    f : array_like, shape (M,) or (M, 1)
        Values of the quantity of interest at `X`.
    space : InputSpace, optional
        The input space. Defaults to ``subspace.space``; a given space must
        equal it. Required when `subspace` is ``None``.
    subspace : SubspaceResult or None
        Required keyword. The subspace whose active variables are the
        surrogate's input coordinates, or ``None`` to fit on all normalized
        inputs.
    kind : {"rbf", "poly"}, default "rbf"
        Radial-basis surrogate with a polynomial tail, or a polynomial.
    degree : int, default 2
        Total degree of the polynomial, or of the RBF's polynomial tail.
    sample_weights : array_like, shape (M,), optional
        ``kind="poly"`` only. Weights :math:`s_i \ge 0` of the least-squares
        fit, with at least one positive; integer weights act as repeated rows.
    noise_variance : array_like, shape (M,), optional
        ``kind="rbf"`` only. Noise variances :math:`v_i \ge 0` of `f`, in
        units of :math:`f^2`.
    log10_eta : float, optional
        ``kind="rbf"`` only. Fixes :math:`\log_{10}\eta` (inside
        `log10_eta_bounds`, or inside :math:`[-10, 1]` when none are given)
        instead of searching for it.
    log10_eta_bounds : tuple of float, optional
        ``kind="rbf"`` only. The search interval for :math:`\log_{10}\eta`,
        lower < upper. Default: the published bounds when `f_variance` is
        given, else :math:`[-10, 1]` (see Notes).
    f_variance : float, optional
        ``kind="rbf"`` only. :math:`\hat\sigma^2 > 0`, the biased sample
        variance of :math:`f` at the gradient samples behind `subspace`
        (CDW (4.11)), not of the training values. Sets the search interval to
        the published bounds [D-192]; ignored (but recorded) when
        `log10_eta_bounds` or `log10_eta` is given.
    qoi : str, optional
        Name of the quantity of interest. Defaults to ``subspace.qoi`` (a
        given value must equal it), else ``"f"``.

    Returns
    -------
    PolynomialSurrogate or RBFSurrogate
        The fitted surrogate, with its training data, in-sample and
        leave-one-out residuals and :math:`R^2_{\mathrm{LOO}}`.

    Raises
    ------
    InputError
        ``input.shape``, ``input.dtype``, ``input.nonfinite``,
        ``input.size_mismatch``, ``input.vector_qoi`` or
        ``input.out_of_space`` if `X`, `f`, `noise_variance` are malformed or
        `X` leaves the support; ``input.weights`` for invalid
        `sample_weights`; ``input.parameter`` for a bad `kind`, `degree`,
        bounds or `log10_eta`, a missing `space`, a `space` or `qoi` that does
        not match the subspace, or an argument given with the other `kind`;
        ``input.too_few_samples`` if there are fewer (positively weighted)
        points than basis functions :math:`p`.
    NumericalError
        ``numerical.singular`` if the polynomial basis is rank-deficient on
        the points or every trial RBF kernel matrix fails its Cholesky
        factorization; ``numerical.nonfinite`` if the basis or the
        coefficients overflow (lower the degree).

    Warns
    -----
    AccuracyWarning
        ``accuracy.loo_undefined`` if a point fully determines its own fit
        (for example :math:`M = p`); ``loo_residuals`` and ``loo_r_squared``
        are then ``None``.

    See Also
    --------
    fit_surrogate_from_model : Fit to conditional means of a model.
    predict, predict_gradient : Evaluate at physical inputs.

    Notes
    -----
    **Polynomial.** The basis is the :math:`p = \binom{k+d}{k}` monomials
    :math:`y^\alpha` of total degree at most :math:`d`, and

    .. math::

       \hat\beta = \arg\min_\beta \bigl\lVert S^{1/2}(f - B\beta)\bigr\rVert_2^2,
       \qquad S = \mathrm{diag}(s), \quad B_{ia} = y_i^{\alpha_a},

    solved by an SVD of the column-scaled matrix :math:`S^{1/2}B`. The rows
    are scaled by :math:`\sqrt{s_i}`; the 2016 package scaled them by
    :math:`s_i`. Degrees above about 6 are ill-conditioned in the monomial
    basis.

    **RBF.** The model is :math:`f = B\beta + \varepsilon` with
    :math:`\varepsilon \sim N\bigl(0, \sigma^2 (R_\eta + \mathrm{diag}(\delta))\bigr)`
    and the squared-exponential correlation
    :math:`(R_\eta)_{il} = \exp(-\sum_j (y_{ij} - y_{lj})^2 / \ell_j)`. On a
    subspace with eigenvalues :math:`\lambda_1, \dots, \lambda_m`, the length
    scales are

    .. math::

       \ell_j = \eta\, \frac{\sum_{q=1}^m \lvert\lambda_q\rvert}
       {\max\bigl(\lvert\lambda_j\rvert,\ \epsilon_{\mathrm{mach}}\lvert\lambda_1\rvert\bigr)},
       \qquad j = 1, \dots, n,

    the rule of [1]_ §4.2-4.3, eq. (4.16): there
    :math:`\ell_i^2 = \alpha\sum_q\lambda_q/\lambda_i` in
    :math:`\exp(-\sum_i\Delta_i^2/2\ell_i^2)`, which is this kernel with
    :math:`\eta = 2\alpha`; otherwise :math:`\ell_j = \eta`. The nugget
    :math:`\delta` is dimensionless, relative to the amplitude :math:`\sigma^2`:
    :math:`\delta_i = v_i / s_f^2`
    with :math:`s_f^2` the sample variance of `f` when noise variances are
    given; else
    :math:`\delta_i = \eta \sum_{q>n}\lvert\lambda_q\rvert / \sum_q \lvert\lambda_q\rvert`
    on a subspace with eigenvalues; else 0; and always at least
    :math:`10^{-6}`. The fit is therefore equivariant under
    :math:`f \mapsto af + b`. Profiling out :math:`\beta` and
    :math:`\sigma^2` gives the generalized least-squares estimates
    :math:`\hat\beta_\eta`, :math:`r_\eta = f - B\hat\beta_\eta` and
    :math:`\hat\sigma^2_\eta = r_\eta^\top K_\eta^{-1} r_\eta / M`, and
    :math:`\eta` minimizes the concentrated negative log-likelihood

    .. math::

       J(\eta) = \tfrac12 \log\det K_\eta + \tfrac M2 \log\hat\sigma^2_\eta,
       \qquad K_\eta = R_\eta + \mathrm{diag}(\delta),

    over a grid of :math:`\log_{10}\eta` with step at most 0.25, refined by a
    bounded scalar minimization around the best grid point. The 2016 package
    weighted the second term by :math:`M` instead of :math:`M/2` and added
    the nugget in units of :math:`f^2` to a unit-amplitude kernel; both are
    changed here.

    **Search interval** [D-101], [D-192]. By default
    :math:`\log_{10}\eta \in [-10, 1]`. With `f_variance`
    :math:`= \hat\sigma^2` and a subspace with eigenvalues, it is the
    published interval [1]_ (4.13), :math:`\hat\sigma^2/\sum\lambda \le
    \alpha \le C_1` with :math:`\eta = 2\alpha`:

    .. math::

       \log_{10}\eta \in \Bigl[\log_{10}\frac{2\hat\sigma^2}{\sum_q\lvert\lambda_q\rvert},
       \ \log_{10} 2C_1\Bigr],

    with the Poincaré constant :math:`C_1 = 1` on a Gaussian space and
    :math:`4/\pi^2` on a uniform space. [1]_ (4.10) prints
    :math:`C_1 \le 2\sqrt m/\pi` for the uniform case, citing Bebendorf,
    whose diameter bound gives :math:`4m/\pi^2`; the sharp constant is
    :math:`4/\pi^2` in every dimension (Payne-Weinberger on :math:`[-1, 1]`
    and tensorization) [D-204]. An empty interval collapses to its upper end
    and warns ``accuracy.empty_bounds``. The bounds are derived for the
    eigenvalues of :math:`E[\nabla f\nabla f^\top]` (estimators ``"as"``
    and ``"opg"``); with another estimator they are a heuristic.

    **Departures from** [1]_, flagged [D-194]: (1) the amplitude is profiled,
    :math:`\hat\sigma^2_\eta`, where [1]_ ties it to the hyperparameter,
    :math:`\sigma^2 = \alpha\sum_q\lambda_q` (4.10); (2) the relative
    nugget is :math:`\eta\sum_{q>n}\lvert\lambda_q\rvert/\sum_q\lvert\lambda_q\rvert
    = 2\alpha\sum_{q>n}\lvert\lambda_q\rvert/\sum_q\lvert\lambda_q\rvert`,
    where [1]_ (4.14) gives :math:`\sum_{q>n}\lambda_q/\sum_q\lambda_q`,
    independent of :math:`\alpha`.

    **Fit quality.** The leave-one-out residual :math:`e_i` refits without
    point :math:`i` with the hyperparameters fixed; both kinds have closed
    forms (the leverages of the weighted fit for the polynomial, the inverse
    of the augmented kernel system for the RBF). The summary is

    .. math::

       R^2_{\mathrm{LOO}} = 1 - \frac{\sum_i s_i e_i^2}{\sum_i s_i (f_i - \bar f_s)^2},

    with :math:`s_i \equiv 1` for the RBF. It replaces the in-sample
    :math:`R^2` of the 2016 package.

    References
    ----------
    .. [1] P. G. Constantine, E. Dow, Q. Wang. Active subspace methods in
       theory and practice: applications to kriging surfaces. *SIAM J. Sci.
       Comput.* 36(4):A1500-A1524, 2014.
    .. [2] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    Fit a cubic on the one-dimensional active subspace of a ridge function.

    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.ridge([1.0, 0.5, 0.0], space=space)
    >>> X = space.sample(50, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, n=1)
    >>> s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=3)
    >>> type(s).__name__, s.coefficients.shape
    ('PolynomialSurrogate', (4,))
    >>> print(f"{s.loo_r_squared:.3f}")
    1.000

    The default is a radial-basis surrogate with a quadratic tail.

    >>> r = ac.fit_surrogate(X=X, f=f, subspace=sub)
    >>> type(r).__name__, r.length_scales.shape
    ('RBFSurrogate', (1,))
    """
    subspace = _check_subspace(subspace)
    if subspace is not None:
        if space is not None:
            _v.check_space(space)
            if not (type(space) is type(subspace.space) and space == subspace.space):
                raise _param(
                    "space: does not equal subspace.space", "space", reason="space_mismatch"
                )
        space = subspace.space
    elif space is None:
        raise _param("space: required when subspace is None", "space")
    space = _v.check_space(space)
    qoi = _resolve_qoi(qoi, None if subspace is None else subspace.qoi)
    kind, degree = _check_kind_degree(kind, degree)
    if kind != "poly" and sample_weights is not None:
        raise _not_for_kind("sample_weights", kind)
    if kind != "rbf" and noise_variance is not None:
        raise _not_for_kind("noise_variance", kind)
    bounds, eta, fv = _check_rbf_args(kind, log10_eta, log10_eta_bounds, f_variance)
    d = _v.data(
        space,
        X=X,
        f=f,
        sample_weights=sample_weights,
        require=("X", "f"),
        use_sample_weights=(kind == "poly"),
    )
    assert d.X is not None  # required above
    assert d.f is not None
    digests = dict(d.digests)
    v = None
    if noise_variance is not None:
        v = _noise_variance(noise_variance, d.M)
        digests["noise_variance"] = _v.digest(v)
    Y = d.X @ subspace.W1 if subspace is not None else np.array(d.X, copy=True)
    Y = np.ascontiguousarray(Y)
    interval, source, eta_fit = _DEFAULT_BOUNDS, None, eta
    if kind == "rbf":
        interval, source, eta_fit = _search_interval(bounds, eta, fv, subspace, space, d.f)
    params = {
        "subspace": subspace,
        "kind": kind,
        "degree": degree,
        "log10_eta": eta,
        "log10_eta_bounds": list(interval) if kind == "rbf" else None,
        "log10_eta_bounds_source": source,
        "f_variance": fv,
        "qoi": qoi,
    }
    prov = _make_provenance("fit_surrogate", params, space=space, digests=digests)
    return _fit_active(
        Y,
        d.f,
        space=space,
        subspace=subspace,
        kind=kind,
        degree=degree,
        sample_weights=d.sample_weights,
        noise_variance=v,
        log10_eta=eta_fit,
        log10_eta_bounds=interval,
        qoi=qoi,
        samples=None,
        provenance=prov,
    )


def fit_surrogate_from_model(
    *,
    domain: ActiveDomain,
    model: Model | ValueFn,
    kind: Literal["rbf", "poly"] = "rbf",
    degree: int = 2,
    n_design: int | None = None,
    n_mc: int = 50,  # [D-171]
    design: ActiveDesign | ArrayLike | None = None,
    include_vertices: bool = True,
    n_starts: int = 3,
    log10_eta: float | None = None,
    log10_eta_bounds: tuple[float, float] | None = None,
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
    qoi: str | None = None,
    rng: _random.RngLike = None,
) -> PolynomialSurrogate | RBFSurrogate:
    r"""Fit a surrogate to conditional means of a model over the inactive variables.

    For each point :math:`y_k` of a design on the active domain, the model is
    evaluated at `n_mc` draws of the inactive variables, and the surrogate
    is fitted to the conditional means :math:`\bar f_k`, the RBF with the
    variance of each mean as its noise variance. On a bounded domain the
    vertices of the domain are added to the design. The model evaluations
    are done by :func:`~active_subspaces.conditional_samples` and kept in the
    result's ``samples``.

    Parameters
    ----------
    domain : ActiveDomain
        The active domain of the subspace (from
        :func:`~active_subspaces.active_domain`).
    model : Model or callable
        A :class:`~active_subspaces.Model` on ``domain.subspace.space``, or a
        bare callable mapping physical points ``(L, m)`` to values ``(L,)``.
    kind : {"rbf", "poly"}, default "rbf"
        Radial-basis surrogate with a polynomial tail, or a polynomial.
    degree : int, default 2
        Total degree of the polynomial, or of the RBF's polynomial tail.
    n_design : int, optional
        Number of design points when `design` is not given; default
        :math:`5^n`. Do not give both.
    n_mc : int, default 50
        Draws of the inactive variables per design point, at least 2.
    design : ActiveDesign or array_like, shape (N_q, n), optional
        The design, in normalized active coordinates, inside the domain.
        Default: :func:`~active_subspaces.active_design` with `n_design`
        points.
    include_vertices : bool, default True
        Add the vertices of a bounded domain to the design.
    n_starts : int, default 3
        Starts of the maximin design, when it is built.
    log10_eta : float, optional
        ``kind="rbf"`` only; fixes :math:`\log_{10}\eta`.
    log10_eta_bounds : tuple of float, optional
        ``kind="rbf"`` only; the search interval for :math:`\log_{10}\eta`,
        default :math:`[-10, 1]`. This path takes no ``f_variance`` [D-207]:
        for the published bounds pass
        ``(log10(2 * f_variance / sum(abs(eigenvalues))), log10(2 * C1))``,
        with :math:`C_1 = 1` (Gaussian) or :math:`4/\pi^2` (uniform).
    executor : concurrent.futures.Executor, optional
        Runs the model evaluations; results do not depend on it.
    batch_size : int, optional
        Points per model call.
    qoi : str, optional
        Name of the quantity of interest; defaults to
        ``domain.subspace.qoi`` (a given value must equal it).
    rng : None, int, SeedSequence or Generator
        Seed for the design and the draws. The fit itself is deterministic.

    Returns
    -------
    PolynomialSurrogate or RBFSurrogate
        The fitted surrogate on the active variables; ``samples`` holds the
        :class:`~active_subspaces.ConditionalSamples`, and ``Y`` and ``f``
        are its design points and means.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad argument, a model on a different space,
        a `qoi` that does not match, or both `design` and `n_design`;
        ``input.shape``, ``input.dtype`` or ``input.nonfinite`` for a
        malformed `design`; ``input.out_of_domain`` if a design point lies
        outside the domain; ``input.too_few_samples`` if the design has fewer
        points than basis functions.
    ModelError
        ``model.*`` if the model raises or returns malformed or non-finite
        values.
    NumericalError
        ``numerical.singular`` or ``numerical.nonfinite`` as in
        :func:`fit_surrogate`.
    SolverError
        ``solver.failed`` if no maximin start is usable.
    UnsupportedError
        ``unsupported.space`` if the domain kind has no design.

    Warns
    -----
    AccuracyWarning
        ``accuracy.loo_undefined`` as in :func:`fit_surrogate`.
    ConvergenceWarning
        ``convergence.failed_starts`` if some maximin starts are unusable.

    See Also
    --------
    fit_surrogate : Fit a surrogate to data.
    active_subspaces.conditional_samples : The model evaluations alone.
    active_subspaces.active_design : Designs on the active domain.

    Notes
    -----
    With draws :math:`z_{kj} \sim \pi(z \mid y_k)` and
    :math:`f_{kj} = f(W_1 y_k + W_2 z_{kj})` (in normalized coordinates),

    .. math::

       \bar f_k = \frac1{N_{MC}} \sum_{j=1}^{N_{MC}} f_{kj},\qquad
       \hat V_k = \frac1{N_{MC}-1} \sum_{j=1}^{N_{MC}} (f_{kj} - \bar f_k)^2,\qquad
       v_k = \hat V_k / N_{MC},

    and the RBF uses :math:`v_k` as noise variance; the polynomial is fitted
    unweighted. The 2016 package passed :math:`\hat V_k` with ddof 0 where
    the variance of the mean belongs. A vertex whose preimage in the cube is
    a face (some inputs do not affect :math:`y`) is sampled over that face; a
    vertex with a unique preimage is evaluated once. :math:`v_k` treats the
    draws as independent.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.ridge([1.0, 0.5, 0.0], space=space)
    >>> X = space.sample(50, rng=0)
    >>> G = ac.evaluate_grads(model, X=X)
    >>> dom = ac.active_domain(ac.active_subspace(grads=G, space=space, n=1))
    >>> s = ac.fit_surrogate_from_model(
    ...     domain=dom, model=model, kind="poly", degree=3, n_design=7, n_mc=5, rng=0
    ... )
    >>> s.samples.Y.shape, s.samples.n_samples
    ((9, 1), 45)
    """
    if not isinstance(domain, ActiveDomain):
        raise _param(f"domain: expected an ActiveDomain, got {type(domain).__name__}", "domain")
    subspace = domain.subspace
    kind, degree = _check_kind_degree(kind, degree)
    bounds, eta, _ = _check_rbf_args(kind, log10_eta, log10_eta_bounds)
    interval = _DEFAULT_BOUNDS if bounds is None else bounds  # no f_variance here [D-207]
    qoi = _resolve_qoi(qoi, subspace.qoi)
    gen, rng_desc = _random.generator(rng)
    samples, _model_obj, params = _conditional_samples(
        domain=domain,
        model=model,
        design=design,
        n_design=n_design,
        n_mc=n_mc,
        include_vertices=include_vertices,
        n_starts=n_starts,
        executor=executor,
        batch_size=batch_size,
        qoi=qoi,
        gen=gen,
        rng_desc=rng_desc,
    )
    params = {
        **params,
        "kind": kind,
        "degree": degree,
        "log10_eta": eta,
        "log10_eta_bounds": list(interval) if kind == "rbf" else None,
    }
    prov = _make_provenance(
        "fit_surrogate_from_model", params, space=subspace.space, rng_state=rng_desc
    )
    Y = np.array(samples.Y, dtype=np.float64, copy=True)
    fbar = np.array(samples.means, dtype=np.float64, copy=True)
    return _fit_active(
        Y,
        fbar,
        space=subspace.space,
        subspace=subspace,
        kind=kind,
        degree=degree,
        sample_weights=None,
        noise_variance=np.array(samples.noise_variance, copy=True) if kind == "rbf" else None,
        log10_eta=eta,
        log10_eta_bounds=interval,
        qoi=qoi,
        samples=samples,
        provenance=prov,
    )


def _check_surrogate(surrogate: Any) -> Any:
    if not isinstance(surrogate, Surrogate):
        raise _param(
            "surrogate: expected an object satisfying the Surrogate protocol, got "
            f"{type(surrogate).__name__}",
            "surrogate",
        )
    return surrogate


def _project(surrogate: Any, x: np.ndarray) -> np.ndarray:
    sub = surrogate.subspace
    return x @ np.asarray(sub.W1) if sub is not None else x


def predict(surrogate: Surrogate, X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate a surrogate at physical input points.

    Parameters
    ----------
    surrogate : Surrogate
        Any object satisfying the :class:`Surrogate` protocol, including a
        :class:`~active_subspaces.ReducedModel`.
    X : array_like, shape (M, m)
        Input points in **physical** coordinates, inside the support of
        ``surrogate.space``.

    Returns
    -------
    ndarray of float64, shape (M,)
        The values :math:`g(W_1^\top x)` at the normalized points :math:`x`
        of `X` (:math:`g(x)` when ``surrogate.subspace`` is ``None``).

    Raises
    ------
    InputError
        ``input.parameter`` if `surrogate` does not satisfy the protocol;
        ``input.shape``, ``input.dtype``, ``input.nonfinite`` or
        ``input.out_of_space`` if `X` is malformed or leaves the support.

    See Also
    --------
    predict_gradient : Physical gradients at physical inputs.
    Surrogate.evaluate : Values in the surrogate's input coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.ridge([1.0, 0.5, 0.0], space=space)
    >>> X = space.sample(50, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, n=1)
    >>> s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=3)
    >>> Xt = space.sample(3, rng=1)
    >>> print(np.round(ac.predict(s, Xt), 3))
    [0.411 0.593 0.484]
    >>> print(np.round(ac.evaluate(model, X=Xt), 3))
    [0.411 0.593 0.484]
    """
    s = _check_surrogate(surrogate)
    x = _v.points(X, s.space)
    return np.asarray(s.evaluate(_project(s, x)), dtype=np.float64)


def predict_gradient(surrogate: Surrogate, X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate the physical gradient of a surrogate at physical input points.

    Parameters
    ----------
    surrogate : Surrogate
        Any object satisfying the :class:`Surrogate` protocol.
    X : array_like, shape (M, m)
        Input points in **physical** coordinates, inside the support of
        ``surrogate.space``.

    Returns
    -------
    ndarray of float64, shape (M, m)
        Gradients with respect to the **physical** inputs :math:`\xi`, one row
        per point.

    Raises
    ------
    InputError
        ``input.parameter`` if `surrogate` does not satisfy the protocol;
        ``input.shape``, ``input.dtype``, ``input.nonfinite`` or
        ``input.out_of_space`` if `X` is malformed or leaves the support.

    See Also
    --------
    predict : Values at physical inputs.
    Surrogate.gradient : Gradients in the surrogate's input coordinates.

    Notes
    -----
    With the physical inputs :math:`\xi = A x + b` of the space,

    .. math::

       \nabla_\xi g(W_1^\top x) = A^{-\top} W_1 \nabla_y g(y), \qquad y = W_1^\top x,

    and :math:`A^{-\top}\nabla_x g(x)` when ``surrogate.subspace`` is
    ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.ridge([1.0, 0.5, 0.0], space=space)
    >>> X = space.sample(50, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, n=1)
    >>> s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=3)
    >>> dg = ac.predict_gradient(s, X[:5])
    >>> dg.shape
    (5, 3)
    >>> bool(np.allclose(dg, G[:5], rtol=0.1, atol=0.05))
    True
    """
    s = _check_surrogate(surrogate)
    x = _v.points(X, s.space)
    Gy = np.asarray(s.gradient(_project(s, x)), dtype=np.float64)
    Gx = Gy @ np.asarray(s.subspace.W1).T if s.subspace is not None else Gy
    out: NDArray[np.float64] = s.space.grads_to_physical(Gx)
    return out


def quadratic_terms(
    surrogate: PolynomialSurrogate,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    r"""Return the constant, linear and quadratic terms of a polynomial surrogate.

    Parameters
    ----------
    surrogate : PolynomialSurrogate
        A polynomial surrogate of degree at least 1.

    Returns
    -------
    c : float
        The constant term.
    a : ndarray of float64, shape (k,)
        The linear terms.
    H : ndarray of float64, shape (k, k)
        The symmetric matrix of quadratic terms; zero for degree 1.

    Raises
    ------
    InputError
        ``input.parameter`` if `surrogate` is not a
        :class:`PolynomialSurrogate` or has degree 0.

    Notes
    -----
    In the surrogate's input coordinates :math:`y` (normalized),

    .. math::

       g(y) = c + a^\top y + \tfrac12 y^\top H y + \text{(terms of degree} > 2),

    with :math:`c = \hat\beta_0`, :math:`a_j = \hat\beta_{e_j}`,
    :math:`H_{jj} = 2\hat\beta_{2e_j}` and
    :math:`H_{jl} = H_{lj} = \hat\beta_{e_j + e_l}` for :math:`j \ne l`. These
    are the Taylor coefficients of :math:`g` at :math:`y = 0`.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([-1.0, -1.0], [1.0, 1.0])
    >>> X = space.sample(20, rng=0)
    >>> f = 1 + X[:, 0] + X[:, 0] * X[:, 1] + 1.5 * X[:, 1] ** 2
    >>> s = ac.fit_surrogate(X=X, f=f, space=space, subspace=None, kind="poly")
    >>> c, a, H = ac.quadratic_terms(s)
    >>> print(round(c, 6), np.round(a, 6) + 0.0)
    1.0 [1. 0.]
    >>> print(np.round(H, 6) + 0.0)
    [[0. 1.]
     [1. 3.]]
    """
    if not isinstance(surrogate, PolynomialSurrogate):
        raise _param(
            f"surrogate: expected a PolynomialSurrogate, got {type(surrogate).__name__}",
            "surrogate",
        )
    if surrogate.degree < 1:
        raise _param(
            "quadratic_terms: needs degree >= 1",
            "surrogate",
            reason="degree",
            degree=surrogate.degree,
        )
    return _quadratic_terms(surrogate.indices, surrogate.coefficients)


# ---------------------------------------------------------------------------
# Intervals for polynomial surrogates [D-200]
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class IntervalResult:
    r"""Prediction or confidence limits of a polynomial surrogate at new points [D-200].

    A value type returned by :func:`predict_interval`, like the arrays of
    :func:`predict`: it holds no provenance and is not serialized.

    Parameters
    ----------
    estimate : numpy.ndarray, shape (K,), float64
        The surrogate at the points.
    lower : numpy.ndarray, shape (K,), float64
        Lower limits; ``-inf`` for ``side="upper"``.
    upper : numpy.ndarray, shape (K,), float64
        Upper limits; ``+inf`` for ``side="lower"``.
    level : float
        Confidence level.
    kind : {"prediction", "confidence"}
        Limits for a new observation or for the mean.
    side : {"two", "lower", "upper"}
        Two-sided or one-sided.
    dof : int
        Residual degrees of freedom :math:`\nu`.
    residual_scale : float
        :math:`s`.
    """

    estimate: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    level: float
    kind: Literal["prediction", "confidence"]
    side: Literal["two", "lower", "upper"]
    dof: int
    residual_scale: float


def _readonly(a: np.ndarray) -> NDArray[np.float64]:
    out = np.array(a, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


def _interval_options(
    s: PolynomialSurrogate, level: Any, kind: Any, side: Any
) -> tuple[float, Literal["prediction", "confidence"], Literal["two", "lower", "upper"], int]:
    """Validate the interval options and the residual degrees of freedom (§5.5 steps 2-3)."""
    level = _v.real(level, "level", lo=0.0, hi=1.0, lo_open=True, hi_open=True)
    kind = cast(
        Literal["prediction", "confidence"], _v.choice(kind, "kind", ("prediction", "confidence"))
    )
    side = cast(
        Literal["two", "lower", "upper"], _v.choice(side, "side", ("two", "lower", "upper"))
    )
    w = np.asarray(s.sample_weights, dtype=np.float64)
    p = s.indices.shape[0]
    nu = int(np.sum(w > 0)) - p
    if nu <= 0:
        raise _param(
            f"no residual degrees of freedom ({int(np.sum(w > 0))} weighted points, {p} basis "
            "functions)",
            "surrogate",
            reason="no_dof",
            dof=nu,
        )
    return level, kind, side, nu


def _poly_interval(
    s: PolynomialSurrogate, Y: Any, level: Any, kind: Any, side: Any
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    Literal["prediction", "confidence"],
    Literal["two", "lower", "upper"],
    int,
    float,
]:
    """``(estimate, lower, upper, level, kind, side, dof, s)`` of §3.2 at input coordinates."""
    level, kind, side, nu = _interval_options(s, level, kind, side)
    Yn = _coords(Y, s.active_dim)
    w = np.asarray(s.sample_weights, dtype=np.float64)
    est = _poly_evaluate(s, Yn)
    r = np.asarray(s.residuals, dtype=np.float64)
    scale = math.sqrt(math.fsum((w * r * r).tolist()) / nu)
    _, sig, Vt, D = _scaled_svd(_basis(np.asarray(s.Y), s.indices), np.sqrt(w))
    Z = (_basis(Yn, s.indices) * D) @ Vt.T / sig
    q = np.sum(Z * Z, axis=1)
    c = 1.0 if kind == "prediction" else 0.0
    half_width = np.sqrt(c + q) * scale
    if side == "two":
        tq = float(scipy.stats.t.ppf(1.0 - (1.0 - level) / 2.0, nu))
        lower, upper = est - tq * half_width, est + tq * half_width
    else:
        tq = float(scipy.stats.t.ppf(level, nu))
        if side == "lower":
            lower, upper = est - tq * half_width, np.full(est.shape, np.inf)
        else:
            lower, upper = np.full(est.shape, -np.inf), est + tq * half_width
    return est, lower, upper, level, kind, side, nu, scale


def predict_interval(
    surrogate: PolynomialSurrogate,
    X: ArrayLike,
    *,
    level: float = 0.95,
    kind: Literal["prediction", "confidence"] = "prediction",
    side: Literal["two", "lower", "upper"] = "two",
) -> IntervalResult:
    r"""Return prediction or confidence limits of a polynomial surrogate at physical inputs.

    Parameters
    ----------
    surrogate : PolynomialSurrogate
        A polynomial surrogate (an RBF or other surrogate is not accepted;
        for a ``ReducedModel`` pass ``rm.surrogate``).
    X : array_like, shape (K, m), float
        Points in physical coordinates, inside the support of
        ``surrogate.space``.
    level : float, default 0.95, keyword-only
        Confidence level in :math:`(0, 1)`.
    kind : {"prediction", "confidence"}, default "prediction", keyword-only
        Limits for a new observation, or for the mean of :math:`f` given :math:`y`.
    side : {"two", "lower", "upper"}, default "two", keyword-only
        Two-sided, or one-sided with the one-sided quantile :math:`t_{\nu,\mathrm{level}}`.

    Returns
    -------
    IntervalResult
        ``estimate``, ``lower``, ``upper`` (each ``(K,)``), ``level``,
        ``kind``, ``side``, ``dof`` and ``residual_scale``.

    Raises
    ------
    InputError
        ``input.parameter`` for a surrogate that is not polynomial
        (``details["reason"] == "not_polynomial"``), no residual degrees of
        freedom (``"no_dof"``) or a bad option; the spec 02 codes for a bad
        ``X``.

    See Also
    --------
    PolynomialSurrogate.interval : The same limits at active coordinates.
    predict : The point prediction.

    Notes
    -----
    The Gaussian-error least-squares limits
    :math:`g(y) \pm t\, s\sqrt{c + b(y)^\top(B^\top SB)^{-1} b(y)}` with
    :math:`c = 1` (prediction) or 0 (confidence) [1]_ [D-200], from the
    factorization behind the leave-one-out leverages. With sample weights
    :math:`s_i`, :math:`s^2 = \sum_i s_i r_i^2/\nu`,
    :math:`\nu = \#\lbrace i: s_i > 0\rbrace - p`, and a new point has
    weight 1. The limits treat :math:`W_1` as fixed; on the model path the
    errors are the scatter of the conditional means about the polynomial.

    The quantile is Student's :math:`t`, exact under Gaussian errors. The
    HyShotII safe box of [2]_ (§5.3) uses the normal quantile
    :math:`z_{0.99} = 2.33` instead; multiplying the one-sided half-width by
    :math:`z_{0.99}/t_{\nu,0.99}` reproduces its limit [D-211].

    References
    ----------
    .. [1] D. C. Montgomery, E. A. Peck, G. G. Vining. *Introduction to Linear
       Regression Analysis*, 5th ed. Wiley, 2012, §3.4-3.5.
    .. [2] P. G. Constantine, M. Emory, J. Larsson, G. Iaccarino. Exploiting
       active subspaces to quantify uncertainty in the numerical simulation of
       the HyShot II scramjet. *J. Comput. Phys.* 302:1-20, 2015.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> X = space.sample(40, rng=0)
    >>> f = X[:, 0] + 0.5 * X[:, 1] + 0.01 * np.sin(30.0 * X[:, 0])
    >>> s = ac.fit_surrogate(X=X, f=f, space=space, subspace=None, kind="poly", degree=1)
    >>> res = ac.predict_interval(s, [[0.5, 1.0]], level=0.9)
    >>> res.dof, bool(res.lower[0] < res.estimate[0] < res.upper[0])
    (37, True)
    """
    if not isinstance(surrogate, PolynomialSurrogate):
        raise _param(
            f"surrogate: intervals need a PolynomialSurrogate, got {type(surrogate).__name__}",
            "surrogate",
            reason="not_polynomial",
        )
    s = surrogate
    _interval_options(s, level, kind, side)  # spec 09 §5.5 order: options and dof before X
    x = _v.points(X, s.space)
    Y = _project(s, x)
    est, lower, upper, level_, kind_, side_, nu, scale = _poly_interval(s, Y, level, kind, side)
    return IntervalResult(
        estimate=_readonly(est),
        lower=_readonly(lower),
        upper=_readonly(upper),
        level=level_,
        kind=kind_,
        side=side_,
        dof=nu,
        residual_scale=scale,
    )
