"""Active subspaces and sufficient dimension reduction from samples.

One entry point, :func:`active_subspace`, runs every estimator of the closed
registry :data:`ESTIMATORS` and every dimension rule of
:data:`DIMENSION_RULES`, optionally with a bootstrap whose replicates run on
independent random streams. The result is a :class:`SubspaceResult` stored in
normalized coordinates; :func:`with_dimension` picks a new dimension from it
without recomputation.

Estimators: ``as``, ``nas``, ``asx``, ``nasx`` (from gradients) and ``ols``,
``qphd``, ``sir``, ``phd``, ``save``, ``ss``, ``opg`` (from values). Dimension
rules: ``evg`` (eigenvalue gap), ``rs`` (response-surface bound) and ``li``
(the ladle of Luo and Li).
"""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, NamedTuple, Protocol, cast

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from active_subspaces import _random
from active_subspaces import _validation as _v
from active_subspaces.errors import (
    AccuracyWarning,
    InputError,
    NumericalError,
    ValidityWarning,
    _warn,
)
from active_subspaces.gradients import (
    GradientResult,
    _distinct_locations,
    _neighbor_sets,
)
from active_subspaces.results import (
    QoIResult,
    _invariant,
    _make_provenance,
    _register_migration,
    _register_result,
)
from active_subspaces.spaces import InputSpace, _orient_columns

__all__ = (  # noqa: RUF022 -- plain sorted() order (spec 15 §4.3)
    "AS",
    "ASX",
    "DIMENSION_RULES",
    "DimensionRule",
    "ESTIMATORS",
    "EVG",
    "Estimator",
    "LI",
    "NAS",
    "NASX",
    "OLS",
    "OPG",
    "PHD",
    "QPHD",
    "RS",
    "SAVE",
    "SIR",
    "SS",
    "SubspaceResult",
    "active_subspace",
    "activity_scores",
    "bootstrap_interval",
    "dimension_criterion",
    "mean_distance",
    "orient",
    "subspace_from_basis",
    "subspace_from_matrix",
    "with_dimension",
)

# Re-exported at the top level; the classes, protocols and registries stay module-qualified
# [D-148].
_toplevel = (
    "SubspaceResult",
    "active_subspace",
    "activity_scores",
    "bootstrap_interval",
    "dimension_criterion",
    "mean_distance",
    "orient",
    "subspace_from_basis",
    "subspace_from_matrix",
    "with_dimension",
)

_log = logging.getLogger(__name__)

_EPS = float(np.finfo(np.float64).eps)
_SQRT_EPS = math.sqrt(_EPS)
_ORTHO_TOL = 1e-10
_BASIS_RANK_TOL = 1e-12
_BLOCK_ELEMENTS = 1 << 21  # elements of a (block, M, m) work array in SS
_ALIGN_MIN = 0.5  # |w_j^T w_j^(b)| below this warns accuracy.unaligned_replicates [D-196]

_F64 = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Data passed to the estimators (normalized coordinates)
# ---------------------------------------------------------------------------


class _Arrays(NamedTuple):
    """Normalized arrays of one data set (or one bootstrap replicate)."""

    X: _F64 | None  # (M, m)
    f: _F64 | None  # (M,)
    G: _F64 | None  # (M, m)
    w: _F64  # (M,), quadrature weights summing to 1
    Sigma: _F64  # (m, m), declared reference covariance


def _inv_sqrt(S: _F64) -> _F64:
    """Symmetric inverse square root of an SPD matrix."""
    lam, V = np.linalg.eigh(S)
    return (V / np.sqrt(lam)) @ V.T


def _nonfinite(what: str) -> NumericalError:
    return NumericalError(
        f"{what}: the estimator's matrix is not finite",
        code="numerical.nonfinite",
        details={"estimator": what},
    )


def _singular(what: str, exc: Exception) -> NumericalError:
    return NumericalError(
        f"{what}: the eigen- or singular-value solver failed ({exc})",
        code="numerical.singular",
        details={"estimator": what},
    )


def _from_rows(R: _F64, m: int, what: str, notes: dict[str, Any]) -> tuple[_F64, _F64]:
    """Eigenpairs of ``R^T R`` from the singular values and right vectors of ``R`` (§3.3).

    ``lambda = sigma^2``, ``W = V``, signs per §3.2. When ``R`` has more rows than columns, the
    SVD is taken of the ``(m, m)`` triangular factor of a Householder QR of ``R``, computed in
    place, so neither ``Q`` nor the ``(M, m)`` left singular vectors are formed and the peak
    memory stays at the rows themselves [D-202]. ``R`` may be overwritten.
    """
    if not np.all(np.isfinite(R)):
        raise _nonfinite(what)
    if not R.any():
        notes["degenerate"] = True
        return np.zeros(m), np.eye(m)
    try:
        if R.shape[0] > m:
            _, s, Vt = np.linalg.svd(_qr_factor(R), full_matrices=False)
        else:
            _, s, Vt = np.linalg.svd(R, full_matrices=R.shape[0] < m)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as exc:
        raise _singular(what, exc) from exc
    lam = np.zeros(m)
    lam[: s.size] = s * s
    if not np.all(np.isfinite(lam)):
        raise _nonfinite(what)
    return lam, _orient_columns(Vt.T)


def _qr_factor(R: _F64) -> _F64:
    """Return the ``(m, m)`` upper-triangular ``T`` with ``R^T R = T^T T`` (``R`` is ``(M, m)``).

    Blocked (tall-skinny) Householder QR: each row block is stacked under the current ``T`` and
    reduced to its triangular factor, so only block-sized arrays are allocated, whatever the
    memory order of ``R``. It is backward stable like a single QR, and ``R`` and ``T`` have the
    same singular values and right singular vectors [D-202].
    """
    T: _F64 | None = None
    for i in range(0, R.shape[0], _ROW_BLOCK):
        block = R[i : i + _ROW_BLOCK]
        A = block if T is None else np.vstack([T, block])
        T = np.linalg.qr(A, mode="r")
    assert T is not None  # M > m rows were stacked
    return T


def _from_matrix(C: _F64, what: str, notes: dict[str, Any]) -> tuple[_F64, _F64]:
    """Eigenpairs of symmetric ``C``, ranked by ``|lambda|`` (stable), reported signed.

    See [D-31]; eigenvector signs per [D-81].
    """
    if not np.all(np.isfinite(C)):
        raise _nonfinite(what)
    m = C.shape[0]
    C = 0.5 * (C + C.T)
    if not C.any():
        notes["degenerate"] = True
        return np.zeros(m), np.eye(m)
    try:
        lam, V = np.linalg.eigh(C)
    except np.linalg.LinAlgError as exc:
        raise _singular(what, exc) from exc
    order = np.argsort(-np.abs(lam), kind="stable")
    return lam[order], _orient_columns(V[:, order])


_ROW_BLOCK = 1 << 16


def _scale_rows(G: _F64, w: _F64, out: _F64) -> _F64:
    """``out[i] = sqrt(w_i) G[i]`` in row blocks (``out`` may be ``G``); returns ``out``."""
    for i in range(0, G.shape[0], _ROW_BLOCK):
        j = i + _ROW_BLOCK
        np.multiply(np.sqrt(w[i:j])[:, None], G[i:j], out=out[i:j])
    return out


def _back_transform(H: _F64, Sigma: _F64) -> _F64:
    """Li's directions for a non-scalar ``Sigma`` [D-205].

    ``Sigma^{-1/2} H = Q R`` with ``R_jj > 0``, so ``span(Q[:, :k]) = Sigma^{-1/2} span(H[:, :k])``
    for every ``k``; then the sign rule. A multiple of the identity returns ``H`` unchanged.
    """
    if np.array_equal(Sigma, Sigma[0, 0] * np.eye(Sigma.shape[0])):
        return H
    Q, R = np.linalg.qr(_inv_sqrt(Sigma) @ H)
    signs = np.sign(np.diagonal(R))
    signs[signs == 0] = 1.0
    return _orient_columns(Q * signs)


def _unit_rows(A: _F64) -> tuple[_F64, int]:
    """Rows scaled to unit norm; rows with norm <= sqrt(eps) * max norm become 0 [D-87].

    Returns (rows, #zeroed).
    """
    scale = np.max(np.abs(A), axis=1)
    safe = np.where(scale > 0, scale, 1.0)
    norms = scale * np.sqrt(np.sum((A / safe[:, None]) ** 2, axis=1))
    keep = norms > _SQRT_EPS * norms.max()
    out = np.zeros_like(A)
    out[keep] = A[keep] / norms[keep, None]
    return out, int(A.shape[0] - keep.sum())


def _lstsq(A: _F64, b: _F64, what: str) -> tuple[_F64, bool]:
    """Least squares; returns (coefficients, rank deficient?).

    With at least as many rows as columns, a Householder QR and a triangular solve, used when
    every ``|R_jj| > max(M, p) eps max_l |R_ll|``; otherwise, and when ``M < p``, the
    minimum-norm solution by ``gelsd``, whose rank decides the warning (V1-11, D-203).
    """
    M, p = A.shape
    if M >= p and p > 0:
        try:
            Q, R = np.linalg.qr(A, mode="reduced")
        except np.linalg.LinAlgError:  # pragma: no cover - LAPACK failure; gelsd below decides
            pass
        else:
            d = np.abs(np.diagonal(R))
            if np.all(np.isfinite(d)) and d.min() > max(M, p) * _EPS * d.max():
                coef = scipy.linalg.solve_triangular(R, Q.T @ b, check_finite=False)
                return coef, False
    try:
        coef, _, rank, _ = scipy.linalg.lstsq(A, b, lapack_driver="gelsd", check_finite=False)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise NumericalError(
            f"{what}: least-squares solve failed ({exc})",
            code="numerical.singular",
            details={"estimator": what},
        ) from exc
    return coef, int(rank) < A.shape[1]


# ---------------------------------------------------------------------------
# Slicing for SIR and SAVE [D-44]
# ---------------------------------------------------------------------------


def _slices(f: _F64, S: int) -> list[NDArray[np.intp]]:
    """Row indices of each slice.

    Equal-count slices of sorted f, ties together, no singletons (§3.4).
    """
    M = f.shape[0]
    perm = np.argsort(f, kind="stable")
    fs = f[perm]
    bounds = [0]
    for h in range(1, S):
        b = (h * M) // S
        if 0 < b < M and fs[b - 1] == fs[b]:
            b = int(np.searchsorted(fs, fs[b - 1], side="right"))
        bounds.append(max(b, bounds[-1]))
    bounds.append(M)
    spans = [(a, b) for a, b in itertools.pairwise(bounds) if b > a]
    merged: list[tuple[int, int]] = []
    pending: int | None = None
    for a, b in spans:
        if pending is not None:
            a, pending = pending, None
        if b - a == 1:
            if merged:
                merged[-1] = (merged[-1][0], b)
            else:
                pending = a
            continue
        merged.append((a, b))
    if pending is not None:
        merged.append((pending, M))
    _log.debug("slicing: %d slices of sizes %s", len(merged), [b - a for a, b in merged])
    return [perm[a:b] for a, b in merged]


def _whiten(d: _Arrays) -> _F64:
    """Whitened inputs with the declared covariance and the weighted mean [D-39]."""
    assert d.X is not None  # required by the estimator
    xbar = d.w @ d.X
    return (d.X - xbar) @ _inv_sqrt(d.Sigma)


# ---------------------------------------------------------------------------
# Local linear gradients for OPG [D-32], [D-82] and its amendment
# ---------------------------------------------------------------------------


def _default_p(m: int, Md: int) -> int:
    return min(max(m + 1, math.floor(1.7 * m)), Md)


def _local_linear(X: _F64, f: _F64, p: int | None) -> tuple[_F64, int]:
    """Local linear gradient at every row (§3.8.1).

    Returns (grads (M, m), #rank-deficient distinct centers).

    Uses the neighbor rule of ``gradients.local_linear_gradients`` (``p``
    distinct locations, ties by first row index, duplicates as multiplicity
    weights) and the same least-squares cutoff, so the gradients are the ones
    that function returns. The fit depends only on the center's location, so it
    is solved once per distinct location.
    """
    _, m = X.shape
    loc, label = _distinct_locations(X)
    Md = loc.shape[0]
    p = _default_p(m, Md) if p is None else min(p, Md)
    first = np.unique(label, return_index=True)[
        1
    ]  # first row of each location (labels are 0..Md-1)
    G_loc = np.empty((Md, m))
    deficient = 0
    for j, (i, nbrs) in enumerate(zip(first, _neighbor_sets(X, first, p), strict=False)):
        A = np.empty((nbrs.shape[0], m + 1))
        A[:, 0] = 1.0
        A[:, 1:] = X[nbrs] - X[i]
        try:
            coef, _, rank, _ = scipy.linalg.lstsq(
                A, f[nbrs], cond=_EPS * max(A.shape), lapack_driver="gelsd", check_finite=False
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise NumericalError(
                f"opg: local least-squares solve failed ({exc})",
                code="numerical.singular",
                details={"estimator": "opg"},
            ) from exc
        deficient += int(rank) < m + 1
        G_loc[j] = coef[1:]
    return G_loc[label], deficient


# ---------------------------------------------------------------------------
# Estimators (closed set, P10)
# ---------------------------------------------------------------------------


class Estimator(Protocol):
    """Protocol of the built-in estimators.

    The set of estimators is closed: only the classes in :data:`ESTIMATORS`
    implement it, and :func:`active_subspace` rejects other objects. Pass a
    name (``"sir"``) or an instance (``SIR(n_slices=5)``).

    Parameters
    ----------
    *args, **kwargs
        A protocol is not instantiated; use one of the estimator classes.

    Attributes
    ----------
    name : str
        Registry name, for example ``"as"``.
    requires : frozenset of str
        Data the estimator needs, from ``"X"``, ``"f"`` and ``"grads"``.
    needs_gradients : bool
        Whether it needs gradients.
    psd : bool
        Whether its matrix is positive semidefinite; otherwise eigenpairs
        are ranked by absolute value and reported signed.

    See Also
    --------
    ESTIMATORS : The registry.
    active_subspace : Runs an estimator.

    Examples
    --------
    >>> from active_subspaces.subspaces import ESTIMATORS
    >>> sir = ESTIMATORS["sir"](n_slices=5)
    >>> sir.name, sorted(sir.requires), sir.psd
    ('sir', ['X', 'f'], True)
    """

    name: ClassVar[str]
    requires: ClassVar[frozenset[str]]
    needs_gradients: ClassVar[bool]
    psd: ClassVar[bool]

    def options(self) -> dict[str, int | float | str | None]:
        """Return the estimator's options as a JSON-compatible dict.

        Returns
        -------
        dict
            Option name to value; ``{}`` for an estimator without options.

        See Also
        --------
        SubspaceResult : Records them as ``estimator_options``.

        Examples
        --------
        >>> from active_subspaces.subspaces import AS, SIR
        >>> AS().options(), SIR().options()
        ({}, {'n_slices': 10})
        """


class _EstimatorBase:
    __slots__ = ()
    name: ClassVar[str]
    requires: ClassVar[frozenset[str]]
    needs_gradients: ClassVar[bool]
    psd: ClassVar[bool]
    _svd: ClassVar[bool] = False
    _whitened: ClassVar[bool] = False  # eigenvectors are Li's eta, directions Sigma^{-1/2} eta

    def options(self) -> dict[str, int | float | str | None]:
        """Return the estimator's options, e.g. ``{"n_slices": 10}``; ``{}`` if none."""
        return {}

    def _min_samples(self, m: int) -> int:
        return 1

    def _check_data(self, d: _Arrays) -> None:
        M, m = d.w.shape[0], d.Sigma.shape[0]
        _v.min_samples(M, self._min_samples(m), self.name)

    def _rows(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        raise NotImplementedError

    def _matrix(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        """Return the symmetric matrix C of §3.3-3.4 (normalized coordinates)."""
        R = self._rows(d, notes)
        return R.T @ R

    def _decompose(self, d: _Arrays, notes: dict[str, Any]) -> tuple[_F64, _F64]:
        m = d.Sigma.shape[0]
        with np.errstate(
            over="ignore", invalid="ignore", divide="ignore", under="ignore"
        ):  # non-finite C raises below
            if self._svd:
                return _from_rows(self._rows(d, notes), m, self.name, notes)
            lam, W = _from_matrix(self._matrix(d, notes), self.name, notes)
        if self._whitened and not notes.get("degenerate"):
            W = _back_transform(W, d.Sigma)
        return lam, W


@dataclass(frozen=True, slots=True)
class AS(_EstimatorBase):
    r"""The active subspace estimator, from gradients (name ``"as"``).

    Needs ``grads``. Positive semidefinite; valid for any input space.

    See Also
    --------
    active_subspace : Run the estimator on data.
    ESTIMATORS : The registry of estimators.

    Notes
    -----
    The matrix is

    .. math::

        C = \sum_{i=1}^M \omega_i \, g_i g_i^\top,

    with :math:`g_i = \nabla_x f(x_i)` in normalized coordinates and
    quadrature weights :math:`\omega_i` (default :math:`1/M`). It is
    computed from the singular values and right singular vectors of the
    matrix :math:`G_\omega` with rows :math:`\sqrt{\omega_i} g_i^\top`:
    :math:`\lambda_k = \sigma_k^2` and :math:`W = V` [1]_, which is more
    accurate for small eigenvalues than forming :math:`C`. When
    :math:`M > m` the SVD is taken of the triangular factor of a QR of
    :math:`G_\omega`, computed block by block, so no other :math:`(M, m)` array is formed:
    without a bootstrap the peak memory is about :math:`1 + 2/m` times the gradient array.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> from active_subspaces.subspaces import AS
    >>> space = ac.Gaussian.reference(4)
    >>> X = space.sample(400, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)  # f is a ridge along w
    >>> G = np.outer(np.cos(X @ w), w)
    >>> sub = ac.active_subspace(grads=G, space=space, estimator=AS(), n=1)
    >>> print(f"{abs(w @ sub.W1[:, 0]):.2f}")
    1.00
    """

    name: ClassVar[str] = "as"
    requires: ClassVar[frozenset[str]] = frozenset({"grads"})
    needs_gradients: ClassVar[bool] = True
    psd: ClassVar[bool] = True
    _svd: ClassVar[bool] = True

    def _rows(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        assert d.G is not None  # required by the estimator
        # Scaled in row blocks, so no (M,) temporary; in place on the library's own normalized
        # copy when nothing reads it later (no bootstrap) [D-202].
        out = d.G if notes.pop("_overwrite_G", False) else np.empty_like(d.G)
        return _scale_rows(d.G, d.w, out)


@dataclass(frozen=True, slots=True)
class NAS(_EstimatorBase):
    r"""The normalized active subspace estimator, from gradients (name ``"nas"``).

    Needs ``grads``. Positive semidefinite; valid for any input space. NAS has
    no published reference; it is original to this library.

    See Also
    --------
    active_subspace : Run the estimator on data.
    ESTIMATORS : The registry of estimators.

    Notes
    -----
    As :class:`AS` with unit gradients, so each sample's direction counts
    equally whatever :math:`\lVert \nabla f \rVert`:

    .. math::

        C = \sum_{i=1}^M \omega_i \, \hat g_i \hat g_i^\top, \qquad
        \hat g_i = g_i / \lVert g_i \rVert,

    with :math:`\hat g_i = 0` when
    :math:`\lVert g_i \rVert \le \sqrt{\epsilon} \max_k \lVert g_k \rVert`.
    The weights are not renormalized, so :math:`\operatorname{tr} C \le 1`.
    For an exact ridge :math:`f = h(W_1^\top x)`, :math:`C` has its range in
    :math:`\operatorname{span}(W_1)` for any input density.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> from active_subspaces.subspaces import NAS
    >>> space = ac.Gaussian.reference(4)
    >>> X = space.sample(400, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)  # f is a ridge along w
    >>> G = np.outer(np.cos(X @ w), w)
    >>> sub = ac.active_subspace(grads=G, space=space, estimator=NAS(), n=1)
    >>> print(f"{abs(w @ sub.W1[:, 0]):.2f}")
    1.00
    """

    name: ClassVar[str] = "nas"
    requires: ClassVar[frozenset[str]] = frozenset({"grads"})
    needs_gradients: ClassVar[bool] = True
    psd: ClassVar[bool] = True
    _svd: ClassVar[bool] = True

    def _rows(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        assert d.G is not None  # required by the estimator
        Gh, nz = _unit_rows(d.G)
        _log.debug("nas: %d zeroed gradient rows", nz)
        return np.sqrt(d.w)[:, None] * Gh


@dataclass(frozen=True, slots=True)
class ASX(_EstimatorBase):
    r"""The symmetrized gradient-input estimator, from gradients and points (name ``"asx"``).

    Needs ``X`` and ``grads``. Indefinite: eigenpairs are ranked by
    :math:`|\lambda|` and reported signed. Valid for Gaussian inputs; on a
    uniform space it issues ``ValidityWarning``. ASX has no published
    reference; it is original to this library.

    See Also
    --------
    active_subspace : Run the estimator on data.
    ESTIMATORS : The registry of estimators.

    Notes
    -----
    .. math::

        C = \tfrac12 (A + A^\top), \qquad A = \sum_{i=1}^M \omega_i \, g_i x_i^\top.

    For a Gaussian input density, Stein's lemma gives
    :math:`\mathbb{E}[\partial_j f \, x_k] = \mathbb{E}[\partial_j \partial_k f]`,
    so :math:`C` estimates :math:`\mathbb{E}[\nabla^2 f]`, the pHd matrix
    computed from gradients [1]_, [2]_.

    References
    ----------
    .. [1] C. M. Stein. Estimation of the mean of a multivariate normal
       distribution. *Ann. Statist.* 9(6):1135-1151, 1981.
    .. [2] K.-C. Li. On principal Hessian directions for data visualization and
       dimension reduction: another application of Stein's lemma. *J. Amer.
       Statist. Assoc.* 87(420):1025-1039, 1992.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> from active_subspaces.subspaces import ASX
    >>> space = ac.Gaussian.reference(4)
    >>> X = space.sample(400, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)  # f is a ridge along w
    >>> G = np.outer(2.0 * (X @ w), w)
    >>> sub = ac.active_subspace(X=X, grads=G, space=space, estimator=ASX(), n=1)
    >>> print(f"{abs(w @ sub.W1[:, 0]):.2f}")
    1.00
    """

    name: ClassVar[str] = "asx"
    requires: ClassVar[frozenset[str]] = frozenset({"X", "grads"})
    needs_gradients: ClassVar[bool] = True
    psd: ClassVar[bool] = False

    def _matrix(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        assert d.X is not None  # required by the estimator
        assert d.G is not None  # required by the estimator
        A = d.G.T @ (d.w[:, None] * d.X)
        return 0.5 * (A + A.T)


@dataclass(frozen=True, slots=True)
class NASX(_EstimatorBase):
    r"""ASX with unit gradients and unit points (name ``"nasx"``).

    Needs ``X`` and ``grads``. Indefinite: eigenpairs are ranked by
    :math:`|\lambda|` and reported signed. Valid for Gaussian inputs; on a
    uniform space it issues ``ValidityWarning``. NASX has no published
    reference; it is original to this library.

    See Also
    --------
    active_subspace : Run the estimator on data.
    ESTIMATORS : The registry of estimators.

    Notes
    -----
    As :class:`ASX` with :math:`g_i \to g_i / \lVert g_i \rVert` and
    :math:`x_i \to x_i / \lVert x_i \rVert`, each set to 0 when its norm is
    at most :math:`\sqrt{\epsilon}` times the largest norm:

    .. math::

        C = \tfrac12 (A + A^\top), \qquad
        A = \sum_{i=1}^M \omega_i \, \hat g_i \hat x_i^\top.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> from active_subspaces.subspaces import NASX
    >>> space = ac.Gaussian.reference(4)
    >>> X = space.sample(400, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)  # f is a ridge along w
    >>> G = np.outer(2.0 * (X @ w), w)
    >>> sub = ac.active_subspace(X=X, grads=G, space=space, estimator=NASX(), n=1)
    >>> print(f"{abs(w @ sub.W1[:, 0]):.2f}")
    1.00
    """

    name: ClassVar[str] = "nasx"
    requires: ClassVar[frozenset[str]] = frozenset({"X", "grads"})
    needs_gradients: ClassVar[bool] = True
    psd: ClassVar[bool] = False

    def _matrix(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        assert d.X is not None  # required by the estimator
        assert d.G is not None  # required by the estimator
        Gh, nzg = _unit_rows(d.G)
        Xh, nzx = _unit_rows(d.X)
        _log.debug("nasx: %d zeroed gradient rows, %d zeroed point rows", nzg, nzx)
        A = Gh.T @ (d.w[:, None] * Xh)
        return 0.5 * (A + A.T)


@dataclass(frozen=True, slots=True)
class OLS(_EstimatorBase):
    r"""The linear least-squares estimator, from values (name ``"ols"``).

    Needs ``X`` and ``f``, and at least :math:`m + 1` samples. Positive
    semidefinite with one nonzero eigenvalue. Valid for Gaussian inputs; on a
    uniform space it issues ``ValidityWarning``.

    See Also
    --------
    active_subspace : Run the estimator on data.
    ESTIMATORS : The registry of estimators.

    Notes
    -----
    Weighted least squares on :math:`[1 \ x^\top]`,

    .. math::

        (\hat u_0, \hat u) = \arg\min_{u_0, u} \sum_i \omega_i
        (f_i - u_0 - u^\top x_i)^2, \qquad C = \hat u \hat u^\top,

    so :math:`\lambda_1 = \lVert \hat u \rVert^2` and the other eigenvalues
    are 0. The first column is the signed direction
    :math:`w_1 = \hat u / \lVert \hat u \rVert`, so :math:`f` increases along
    :math:`w_1`, as published ([3]_ Algorithm 1.3, eq. (1.6); [4]_ Algorithms
    1-2, eq. (22)); the largest-magnitude sign rule is not applied to it
    [D-195]. The other columns are an arbitrary orthonormal complement. For Gaussian
    inputs :math:`\hat u \to \mathbb{E}[\nabla f]` by Stein's lemma [1]_; on a
    uniform space the linearity condition of [2]_ holds only in special
    cases. A rank-deficient fit uses the minimum-norm solution and warns.

    References
    ----------
    .. [1] C. M. Stein. Estimation of the mean of a multivariate normal
       distribution. *Ann. Statist.* 9(6):1135-1151, 1981.
    .. [2] K.-C. Li. Sliced inverse regression for dimension reduction. *J. Amer.
       Statist. Assoc.* 86(414):316-327, 1991.
    .. [3] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.
    .. [4] P. G. Constantine, M. Emory, J. Larsson, G. Iaccarino. Exploiting active
       subspaces to quantify uncertainty in the numerical simulation of the HyShot II
       scramjet. *J. Comput. Phys.* 302:1-20, 2015.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> from active_subspaces.subspaces import OLS
    >>> space = ac.Gaussian.reference(4)
    >>> X = space.sample(400, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)  # f is a ridge along w
    >>> f = np.sin(X @ w)
    >>> sub = ac.active_subspace(X=X, f=f, space=space, estimator=OLS(), n=1)
    >>> print(f"{abs(w @ sub.W1[:, 0]):.2f}")
    1.00
    """

    name: ClassVar[str] = "ols"
    requires: ClassVar[frozenset[str]] = frozenset({"X", "f"})
    needs_gradients: ClassVar[bool] = False
    psd: ClassVar[bool] = True
    _svd: ClassVar[bool] = True

    def _min_samples(self, m: int) -> int:
        return m + 1

    def _rows(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        assert d.X is not None  # required by the estimator
        assert d.f is not None  # required by the estimator
        M, m = d.X.shape
        sw = np.sqrt(d.w)
        A = np.empty((M, m + 1))
        A[:, 0] = sw
        A[:, 1:] = sw[:, None] * d.X
        coef, bad = _lstsq(A, sw * d.f, self.name)
        if bad:
            notes["rank_deficient"] = True
        notes["_u"] = coef[1:]
        return coef[None, 1:]

    def _decompose(self, d: _Arrays, notes: dict[str, Any]) -> tuple[_F64, _F64]:
        lam, W = _EstimatorBase._decompose(self, d, notes)
        u = notes.pop("_u")
        if float(W[:, 0] @ u) < 0.0:  # the published signed direction u / ||u|| [D-195]
            W[:, 0] = -W[:, 0]
        return lam, W


def _quadratic_design(X: _F64) -> _F64:
    """Columns 1, x_j, x_j^2, x_j x_k (j < k, row-major)."""
    M, m = X.shape
    ju, ku = np.triu_indices(m, k=1)
    return np.hstack([np.ones((M, 1)), X, X * X, X[:, ju] * X[:, ku]])


@dataclass(frozen=True, slots=True)
class QPHD(_EstimatorBase):
    r"""The global quadratic estimator, from values (name ``"qphd"``).

    Needs ``X`` and ``f``, and at least :math:`(m + 1)(m + 2)/2` samples.
    Positive semidefinite. Valid for Gaussian inputs; on a uniform space it
    issues ``ValidityWarning``.

    See Also
    --------
    active_subspace : Run the estimator on data.
    ESTIMATORS : The registry of estimators.

    Notes
    -----
    Fit :math:`q(x) = c + g^\top x + \tfrac12 x^\top H x` by weighted least
    squares on the monomials :math:`1, x_j, x_j^2, x_j x_k` (:math:`j < k`) in
    normalized coordinates, then

    .. math::

        C = \mathbb{E}_\rho[\nabla q \, \nabla q^\top] = g g^\top + H \Sigma H,

    with :math:`\Sigma` the covariance of the reference density
    (``space.reference_covariance``). This follows Corollary 3.1 of [1]_ with
    :math:`q = 2`. It averages the outer product of the gradient of the
    fitted quadratic, where pHd uses its Hessian alone. The 2016 code used
    :math:`g g^\top + \gamma H H^\top` with :math:`\gamma` guessed from the
    data range; the two agree when that guess equals the declared
    :math:`\Sigma = \gamma I`.

    References
    ----------
    .. [1] K.-C. Li. On principal Hessian directions for data visualization and
       dimension reduction: another application of Stein's lemma. *J. Amer.
       Statist. Assoc.* 87(420):1025-1039, 1992.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> from active_subspaces.subspaces import QPHD
    >>> space = ac.Gaussian.reference(4)
    >>> X = space.sample(400, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)  # f is a ridge along w
    >>> f = np.sin(X @ w)
    >>> sub = ac.active_subspace(X=X, f=f, space=space, estimator=QPHD(), n=1)
    >>> print(f"{abs(w @ sub.W1[:, 0]):.2f}")
    1.00
    """

    name: ClassVar[str] = "qphd"
    requires: ClassVar[frozenset[str]] = frozenset({"X", "f"})
    needs_gradients: ClassVar[bool] = False
    psd: ClassVar[bool] = True

    def _min_samples(self, m: int) -> int:
        return (m + 1) * (m + 2) // 2

    def _matrix(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        assert d.X is not None  # required by the estimator
        assert d.f is not None  # required by the estimator
        m = d.X.shape[1]
        sw = np.sqrt(d.w)
        coef, bad = _lstsq(sw[:, None] * _quadratic_design(d.X), sw * d.f, self.name)
        if bad:
            notes["rank_deficient"] = True
        g = coef[1 : m + 1]
        H = np.diag(2.0 * coef[m + 1 : 2 * m + 1])
        ju, ku = np.triu_indices(m, k=1)
        H[ju, ku] = coef[2 * m + 1 :]
        H[ku, ju] = coef[2 * m + 1 :]
        return np.outer(g, g) + H @ d.Sigma @ H


def _check_slices(n_slices: Any) -> int:
    return _v.integer(n_slices, "n_slices", lo=2)


@dataclass(frozen=True, slots=True)
class SIR(_EstimatorBase):
    r"""Sliced inverse regression, from values (name ``"sir"``).

    Needs ``X`` and ``f``, and at least ``2 * n_slices`` samples. Positive
    semidefinite. Valid for Gaussian inputs; on a uniform space it issues
    ``ValidityWarning``.

    Parameters
    ----------
    n_slices : int, default 10
        Nominal number of slices :math:`S`, at least 2.

    See Also
    --------
    active_subspace : Run the estimator on data.
    ESTIMATORS : The registry of estimators.

    Notes
    -----
    The samples are sorted by :math:`f` and cut into :math:`S` slices of
    nearly equal count; equal values of :math:`f` stay in one slice, and a
    one-sample slice is merged into its neighbor. With slice weights
    :math:`\hat p_h = \sum_{i \in h} \omega_i`,

    .. math::

        \tilde x_i = \Sigma^{-1/2} (x_i - \bar x_\omega), \qquad
        \hat m_h = \frac{1}{\hat p_h} \sum_{i \in h} \omega_i \tilde x_i, \qquad
        C = \sum_h \hat p_h \, \hat m_h \hat m_h^\top,

    as in [1]_.

    **Departures from the published method.** The inputs are whitened with
    the declared covariance :math:`\Sigma` of the reference density and the
    weighted mean :math:`\bar x_\omega = \sum_i \omega_i x_i`, not the sample
    covariance; the slice probabilities and means are weighted by the
    quadrature weights; and the slices are equal-count with ties kept
    together. For both input spaces :math:`\Sigma = \gamma I`, so the
    eigenvectors are used as directions in :math:`x` directly. On a uniform
    space the linearity condition fails in general [2]_.

    References
    ----------
    .. [1] K.-C. Li. Sliced inverse regression for dimension reduction. *J. Amer.
       Statist. Assoc.* 86(414):316-327, 1991.
    .. [2] A. Glaws, P. G. Constantine, R. D. Cook. Inverse regression for ridge
       recovery: a data-driven approach for parameter reduction in computer
       experiments. *Statistics and Computing* 30:237-253, 2020 (details to
       verify).

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> from active_subspaces.subspaces import SIR
    >>> space = ac.Gaussian.reference(4)
    >>> X = space.sample(400, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)  # f is a ridge along w
    >>> f = np.sin(X @ w)
    >>> sub = ac.active_subspace(X=X, f=f, space=space, estimator=SIR(n_slices=5), n=1)
    >>> print(f"{abs(w @ sub.W1[:, 0]):.2f}")
    1.00
    """

    n_slices: int = 10
    name: ClassVar[str] = "sir"
    requires: ClassVar[frozenset[str]] = frozenset({"X", "f"})
    needs_gradients: ClassVar[bool] = False
    psd: ClassVar[bool] = True
    _whitened: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the options."""
        object.__setattr__(self, "n_slices", _check_slices(self.n_slices))

    def options(self) -> dict[str, int | float | str | None]:
        """Return the estimator's options as a JSON-compatible dict.

        Returns
        -------
        dict
            ``{"n_slices": ...}``.

        See Also
        --------
        Estimator.options : The protocol method.

        Examples
        --------
        >>> from active_subspaces.subspaces import SIR
        >>> SIR(n_slices=5).options()
        {'n_slices': 5}
        """
        return {"n_slices": self.n_slices}

    def _min_samples(self, m: int) -> int:
        return 2 * self.n_slices

    def _matrix(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        assert d.X is not None  # required by the estimator
        assert d.f is not None  # required by the estimator
        m = d.X.shape[1]
        Z = _whiten(d)
        C = np.zeros((m, m))
        for idx in _slices(d.f, self.n_slices):
            w = d.w[idx]
            p = w.sum()
            if p == 0:
                continue
            mh = (w @ Z[idx]) / p
            C += p * np.outer(mh, mh)
        return C


@dataclass(frozen=True, slots=True)
class PHD(_EstimatorBase):
    r"""Principal Hessian directions, from values (name ``"phd"``).

    Needs ``X`` and ``f``. Indefinite: eigenpairs are ranked by
    :math:`|\lambda|` and reported signed. Valid for Gaussian inputs (Stein's
    lemma); on a uniform space it issues ``ValidityWarning``.

    See Also
    --------
    active_subspace : Run the estimator on data.
    ESTIMATORS : The registry of estimators.

    Notes
    -----
    .. math::

        \Sigma_{fxx} = \sum_i \omega_i (f_i - \bar f_\omega)
        (x_i - \bar x_\omega)(x_i - \bar x_\omega)^\top, \qquad
        C = \Sigma^{-1/2} \Sigma_{fxx} \Sigma^{-1/2},

    ranked by :math:`|\lambda|` as in [1]_; see also [2]_.

    **Departures from the published method.** The whitening uses the
    declared covariance :math:`\Sigma` of the reference density, not the
    sample covariance, and the centering uses the weighted means
    :math:`\bar f_\omega`, :math:`\bar x_\omega` (identical to the sample
    means for the default weights :math:`1/M`).

    References
    ----------
    .. [1] K.-C. Li. On principal Hessian directions for data visualization and
       dimension reduction: another application of Stein's lemma. *J. Amer.
       Statist. Assoc.* 87(420):1025-1039, 1992.
    .. [2] C. M. Stein. Estimation of the mean of a multivariate normal
       distribution. *Ann. Statist.* 9(6):1135-1151, 1981.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> from active_subspaces.subspaces import PHD
    >>> space = ac.Gaussian.reference(4)
    >>> X = space.sample(400, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)  # f is a ridge along w
    >>> f = (X @ w) ** 2
    >>> sub = ac.active_subspace(X=X, f=f, space=space, estimator=PHD(), n=1)
    >>> print(f"{abs(w @ sub.W1[:, 0]):.2f}")
    1.00
    """

    name: ClassVar[str] = "phd"
    requires: ClassVar[frozenset[str]] = frozenset({"X", "f"})
    needs_gradients: ClassVar[bool] = False
    psd: ClassVar[bool] = False
    _whitened: ClassVar[bool] = True

    def _min_samples(self, m: int) -> int:
        return 2

    def _matrix(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        assert d.X is not None  # required by the estimator
        assert d.f is not None  # required by the estimator
        Xc = d.X - d.w @ d.X
        r = d.w * (d.f - d.w @ d.f)
        Sfxx = (Xc * r[:, None]).T @ Xc
        S = _inv_sqrt(d.Sigma)
        return S @ Sfxx @ S


@dataclass(frozen=True, slots=True)
class SAVE(_EstimatorBase):
    r"""Sliced average variance estimation, from values (name ``"save"``).

    Needs ``X`` and ``f``, and at least ``2 * n_slices`` samples. Positive
    semidefinite. Valid for Gaussian inputs; on a uniform space it issues
    ``ValidityWarning``.

    Parameters
    ----------
    n_slices : int, default 10
        Nominal number of slices :math:`S`, at least 2.

    See Also
    --------
    active_subspace : Run the estimator on data.
    ESTIMATORS : The registry of estimators.

    Notes
    -----
    With the slices, :math:`\tilde x_i`, :math:`\hat p_h` and :math:`\hat m_h`
    of :class:`SIR` and :math:`s_i = \omega_i / \hat p_h` for :math:`i \in h`,

    .. math::

        \widehat{\mathrm{Cov}}_h = \frac{\sum_{i \in h} s_i (\tilde x_i - \hat m_h)
        (\tilde x_i - \hat m_h)^\top}{1 - \sum_{i \in h} s_i^2}, \qquad
        V_h = I - \widehat{\mathrm{Cov}}_h, \qquad
        C = \sum_h \hat p_h V_h^2,

    with the factor :math:`\hat p_h` of [1]_. The factor
    :math:`1/(1 - \sum s_i^2)` equals :math:`n_h/(n_h - 1)` for equal weights.
    A slice whose weight sits on one sample takes :math:`V_h = I`.

    **Departures from the published method.** As for :class:`SIR`: whitening
    with the declared :math:`\Sigma` and the weighted mean, weighted slice
    probabilities and means, and equal-count slices with ties kept together.

    References
    ----------
    .. [1] R. D. Cook, S. Weisberg. Discussion of "Sliced inverse regression for
       dimension reduction". *J. Amer. Statist. Assoc.* 86(414):328-332, 1991.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> from active_subspaces.subspaces import SAVE
    >>> space = ac.Gaussian.reference(4)
    >>> X = space.sample(400, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)  # f is a ridge along w
    >>> f = np.sin(X @ w)
    >>> sub = ac.active_subspace(X=X, f=f, space=space, estimator=SAVE(), n=1)
    >>> print(f"{abs(w @ sub.W1[:, 0]):.2f}")
    1.00
    """

    n_slices: int = 10
    name: ClassVar[str] = "save"
    requires: ClassVar[frozenset[str]] = frozenset({"X", "f"})
    needs_gradients: ClassVar[bool] = False
    psd: ClassVar[bool] = True
    _whitened: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the options."""
        object.__setattr__(self, "n_slices", _check_slices(self.n_slices))

    def options(self) -> dict[str, int | float | str | None]:
        """Return the estimator's options as a JSON-compatible dict.

        Returns
        -------
        dict
            ``{"n_slices": ...}``.

        See Also
        --------
        Estimator.options : The protocol method.

        Examples
        --------
        >>> from active_subspaces.subspaces import SAVE
        >>> SAVE(n_slices=5).options()
        {'n_slices': 5}
        """
        return {"n_slices": self.n_slices}

    def _min_samples(self, m: int) -> int:
        return 2 * self.n_slices

    def _matrix(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        assert d.X is not None  # required by the estimator
        assert d.f is not None  # required by the estimator
        m = d.X.shape[1]
        Z = _whiten(d)
        eye = np.eye(m)
        C = np.zeros((m, m))
        for idx in _slices(d.f, self.n_slices):
            w = d.w[idx]
            p = w.sum()
            if p == 0:
                continue
            s = w / p
            mh = s @ Z[idx]
            if np.count_nonzero(s) <= 1:
                V = eye
            else:
                Zc = Z[idx] - mh
                V = eye - ((Zc * s[:, None]).T @ Zc) / (1.0 - np.sum(s * s))
            C += p * (V @ V)
        return C


@dataclass(frozen=True, slots=True)
class SS(_EstimatorBase):
    r"""The swarm subspace, from values (name ``"ss"``).

    Needs ``X`` and ``f``. Positive semidefinite. SS is a heuristic with no
    published reference, original to this library; it is biased on the
    uniform cube, so it issues ``ValidityWarning`` for every input space.

    See Also
    --------
    active_subspace : Run the estimator on data.
    ESTIMATORS : The registry of estimators.

    Notes
    -----
    With :math:`u_{ij}` the unit vector from :math:`x_i` to :math:`x_j`,

    .. math::

        C = \sum_i \sum_{j : x_j \ne x_i} \omega_i \omega_j (f_i - f_j)^2 \,
        u_{ij} u_{ij}^\top,

    the weighted average of squared difference quotients times
    :math:`\lVert x_i - x_j \rVert^2`. Exact duplicate points are skipped;
    both orders of each pair are counted. The cost is
    :math:`O(M^2 m^2)`.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> from active_subspaces.subspaces import SS
    >>> space = ac.Gaussian.reference(4)
    >>> X = space.sample(400, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)  # f is a ridge along w
    >>> f = np.sin(X @ w)
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", ac.ValidityWarning)
    ...     sub = ac.active_subspace(X=X, f=f, space=space, estimator=SS(), n=1)
    >>> print(f"{abs(w @ sub.W1[:, 0]):.2f}")
    1.00
    """

    name: ClassVar[str] = "ss"
    requires: ClassVar[frozenset[str]] = frozenset({"X", "f"})
    needs_gradients: ClassVar[bool] = False
    psd: ClassVar[bool] = True

    def _min_samples(self, m: int) -> int:
        return 2

    def _matrix(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        assert d.X is not None  # required by the estimator
        assert d.f is not None  # required by the estimator
        X, f, w = d.X, d.f, d.w
        M, m = X.shape
        C = np.zeros((m, m))
        block = max(1, min(M, _BLOCK_ELEMENTS // max(1, M * m)))
        for i0 in range(0, M, block):
            i1 = min(M, i0 + block)
            D = X[None, :, :] - X[i0:i1, None, :]
            scale = np.max(np.abs(D), axis=2)
            dup = scale == 0
            scale[dup] = 1.0
            U = D / scale[:, :, None]
            nrm = np.sqrt(np.sum(U * U, axis=2))
            nrm[dup] = 1.0
            U /= nrm[:, :, None]
            a = (w[i0:i1, None] * w[None, :]) * (f[i0:i1, None] - f[None, :]) ** 2
            a[dup] = 0.0
            U2 = U.reshape(-1, m)
            C += (U2 * a.reshape(-1)[:, None]).T @ U2
        return C


@dataclass(frozen=True, slots=True)
class OPG(_EstimatorBase):
    r"""Outer product of local linear gradients, from values (name ``"opg"``).

    Needs ``X`` and ``f``, with at least :math:`m + 1` distinct points.
    Positive semidefinite; valid for any input space.

    Parameters
    ----------
    p : int, optional
        Number of distinct locations per local fit, at least 2 here and
        :math:`m + 1 \le p \le M_d` on the data. Default
        :math:`\min(\max(m + 1, \lfloor 1.7 m \rfloor), M_d)`.

    See Also
    --------
    active_subspace : Run the estimator on data.
    ESTIMATORS : The registry of estimators.

    Notes
    -----
    The local linear gradient :math:`\hat g_i` of
    :func:`~active_subspaces.local_linear_gradients` is computed at every
    sample, then

    .. math::

        C = \sum_i \omega_i \, \hat g_i \hat g_i^\top,

    computed as for :class:`AS`. A bootstrap replicate with fewer distinct
    points than ``p`` uses all of them.

    **Departure from the published method.** Published OPG [1]_, [2]_ weights
    each local fit with a fixed-bandwidth kernel :math:`K_h(x_j - x_i)`. This
    estimator, like the 2016 code, uses the uniform kernel on the ``p``
    nearest distinct locations, that is, an adaptive bandwidth.

    References
    ----------
    .. [1] M. Hristache, A. Juditsky, J. Polzehl, V. Spokoiny. Structure adaptive
       approach for dimension reduction. *Ann. Statist.* 29(6):1537-1566, 2001.
    .. [2] Y. Xia, H. Tong, W. K. Li, L.-X. Zhu. An adaptive estimation of dimension
       reduction space. *J. R. Statist. Soc. B* 64(3):363-410, 2002.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> from active_subspaces.subspaces import OPG
    >>> space = ac.Gaussian.reference(4)
    >>> X = space.sample(400, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)  # f is a ridge along w
    >>> f = np.sin(X @ w)
    >>> sub = ac.active_subspace(X=X, f=f, space=space, estimator=OPG(), n=1)
    >>> print(f"{abs(w @ sub.W1[:, 0]):.2f}")
    1.00
    """

    p: int | None = None
    name: ClassVar[str] = "opg"
    requires: ClassVar[frozenset[str]] = frozenset({"X", "f"})
    needs_gradients: ClassVar[bool] = False
    psd: ClassVar[bool] = True
    _svd: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the options."""
        if self.p is not None:
            object.__setattr__(self, "p", _v.integer(self.p, "p", lo=2))

    def options(self) -> dict[str, int | float | str | None]:
        """Return the estimator's options as a JSON-compatible dict.

        Returns
        -------
        dict
            ``{"p": ...}``.

        See Also
        --------
        Estimator.options : The protocol method.

        Examples
        --------
        >>> from active_subspaces.subspaces import OPG
        >>> OPG(p=6).options()
        {'p': 6}
        """
        return {"p": self.p}

    def _min_samples(self, m: int) -> int:
        return m + 1

    def _check_data(self, d: _Arrays) -> None:
        assert d.X is not None  # required by the estimator
        M, m = d.X.shape
        _v.min_samples(M, m + 1, self.name)
        Md = _distinct_locations(d.X)[0].shape[0]
        _v.min_samples(Md, m + 1, "opg (distinct locations)")
        if self.p is not None and not (m + 1 <= self.p <= Md):
            raise InputError(
                f"p: expected an integer in [{m + 1}, {Md}] (m + 1 to the number of distinct "
                f"points), got {self.p}",
                code="input.parameter",
                details={"param": "p", "expected": [m + 1, Md], "got": self.p},
            )

    def _rows(self, d: _Arrays, notes: dict[str, Any]) -> _F64:
        assert d.X is not None  # required by the estimator
        assert d.f is not None  # required by the estimator
        G, deficient = _local_linear(d.X, d.f, self.p)
        if deficient:
            notes["rank_deficient"] = True
            notes["centers"] = notes.get("centers", 0) + deficient
        return np.sqrt(d.w)[:, None] * G


ESTIMATORS: Mapping[str, type[Estimator]] = MappingProxyType(
    {cls.name: cls for cls in (AS, NAS, ASX, NASX, OLS, QPHD, SIR, PHD, SAVE, SS, OPG)}
)
"""Read-only registry of the estimators: name to class (closed set)."""


# ---------------------------------------------------------------------------
# Dimension rules (closed set, P10)
# ---------------------------------------------------------------------------


class DimensionRule(Protocol):
    """Protocol of the built-in dimension rules.

    The set is closed: only :class:`EVG`, :class:`RS` and :class:`LI`
    implement it. Pass a name (``"evg"``) or an instance (``EVG(gap="log")``).

    Parameters
    ----------
    *args, **kwargs
        A protocol is not instantiated; use one of the rule classes.

    Attributes
    ----------
    name : str
        Registry name.
    needs_bootstrap : bool
        Whether the rule needs bootstrap replicates (``nboot > 0``).

    See Also
    --------
    DIMENSION_RULES : The registry.
    with_dimension : Apply a rule to an existing result.

    Examples
    --------
    >>> from active_subspaces.subspaces import DIMENSION_RULES
    >>> [(name, cls.needs_bootstrap) for name, cls in sorted(DIMENSION_RULES.items())]
    [('evg', False), ('li', True), ('rs', True)]
    """

    name: ClassVar[str]
    needs_bootstrap: ClassVar[bool]

    def options(self) -> dict[str, str]:
        """Return the rule's options as a JSON-compatible dict.

        Returns
        -------
        dict
            Option name to value; ``{}`` for a rule without options.

        See Also
        --------
        SubspaceResult : Records them as ``dimension_options``.

        Examples
        --------
        >>> from active_subspaces.subspaces import EVG, LI
        >>> EVG(gap="log").options(), LI().options()
        ({'gap': 'log'}, {})
        """


def _first_extreme(c: _F64, m: int, scale: float, *, largest: bool) -> int:
    """0-based index of the first entry within rounding of the min (or max) of ``c``.

    Rounding means ``8 m eps scale``.

    The rules use "min argmin"/"min argmax". Criteria that tie in exact
    arithmetic (an exact ridge, where every replicate is identical, gives
    ``g(k) = 0`` and ``eps_k = 0`` for all ``k >= 1``) differ only by rounding,
    so values within this tolerance of the criterion's natural ``scale`` count as ties.
    """
    tol = 8.0 * m * _EPS * scale
    if largest:
        return int(np.flatnonzero(c >= np.max(c) - tol)[0])
    return int(np.flatnonzero(c <= np.min(c) + tol)[0])


def _ladle_kn(m: int) -> int:
    return m - 1 if m <= 10 else math.floor(m / math.log(m))


@dataclass(frozen=True, slots=True)
class EVG:
    r"""The eigenvalue-gap dimension rule (name ``"evg"``), the default.

    Parameters
    ----------
    gap : {"linear", "log"}, default "linear"
        Gap between consecutive eigenvalue magnitudes, or between their
        logarithms.

    Raises
    ------
    InputError
        ``input.parameter`` for an unknown ``gap``.

    See Also
    --------
    RS : The response-surface rule.
    LI : The ladle.

    Notes
    -----
    With :math:`\mu_k = |\lambda_k|`, the linear gap is
    :math:`c_k = \mu_k - \mu_{k+1}`, and the log gap is
    :math:`c_k = \log \tilde\mu_k - \log \tilde\mu_{k+1}` with
    :math:`\tilde\mu_k = \max(\mu_k, m \epsilon \mu_1)`. The rule chooses
    :math:`n = \min \arg\max_{1 \le k \le m-1} c_k`, where values within
    :math:`8 m \epsilon` times the scale of :math:`c` count as ties. It is a
    heuristic, valid with every estimator, and needs no bootstrap.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(4)
    >>> X = space.sample(200, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)
    >>> G = np.outer(np.cos(X @ w), w)  # gradients of sin(w^T x)
    >>> sub = ac.active_subspace(grads=G, space=space, dimension_rule=ac.subspaces.EVG(gap="log"))
    >>> sub.n, sub.dimension_options
    (1, (('gap', 'log'),))
    """

    gap: Literal["linear", "log"] = "linear"
    name: ClassVar[str] = "evg"
    needs_bootstrap: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """Validate the options."""
        _v.choice(self.gap, "gap", ("linear", "log"))

    def options(self) -> dict[str, str]:
        """Return the rule's options as a JSON-compatible dict.

        Returns
        -------
        dict
            ``{"gap": "linear"}`` or ``{"gap": "log"}``.

        See Also
        --------
        DimensionRule.options : The protocol method.

        Examples
        --------
        >>> from active_subspaces.subspaces import EVG
        >>> EVG().options()
        {'gap': 'linear'}
        """
        return {"gap": self.gap}

    def _evaluate(self, mu: _F64, distances: Any, dets: Any) -> tuple[NDArray[Any], _F64, int]:
        m = mu.size
        if self.gap == "linear":
            c, scale = mu[:-1] - mu[1:], float(mu[0])
        elif mu[0] == 0:
            c, scale = np.zeros(m - 1), 0.0
        else:
            lg = np.log(np.maximum(mu, m * _EPS * mu[0]))
            c = lg[:-1] - lg[1:]
            scale = float(np.max(np.abs(c)))
        return np.arange(1, m, dtype=np.int64), c, _first_extreme(c, m, scale, largest=True) + 1


@dataclass(frozen=True, slots=True)
class RS:
    r"""The response-surface dimension rule (name ``"rs"``).

    Needs bootstrap replicates. Valid with the ``"as"`` and ``"opg"``
    estimators, which estimate :math:`\mathbb{E}[\nabla f \nabla f^\top]`; with
    the others it issues ``ValidityWarning``.

    See Also
    --------
    EVG : The eigenvalue-gap rule.
    mean_distance : The :math:`\hat\delta_k` used here.

    Notes
    -----
    With :math:`\mu_k = |\lambda_k|` and :math:`\hat\delta_k` the bootstrap
    mean subspace distance,

    .. math::

        \varepsilon_k = \Bigl(\sum_{j \le k} \mu_j\Bigr)^{1/2} \hat\delta_k
        + \Bigl(\sum_{j > k} \mu_j\Bigr)^{1/2}, \qquad
        n = \min \arg\min_{1 \le k \le m-1} \varepsilon_k,

    the bound on the mean squared error of a response surface on the active
    variables [1]_ with the Poincaré constant dropped, so
    :math:`\varepsilon_k` ranks dimensions but is not an error estimate.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(4)
    >>> X = space.sample(200, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)
    >>> G = np.outer(np.cos(X @ w), w)  # gradients of sin(w^T x)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=20, rng=0, dimension_rule="rs")
    >>> sub.n, sub.criterion.shape
    (1, (3,))
    """

    name: ClassVar[str] = "rs"
    needs_bootstrap: ClassVar[bool] = True

    def options(self) -> dict[str, str]:
        """Return the rule's options, ``{}``.

        Returns
        -------
        dict
            The empty dict.

        See Also
        --------
        DimensionRule.options : The protocol method.

        Examples
        --------
        >>> from active_subspaces.subspaces import RS
        >>> RS().options()
        {}
        """
        return {}

    def _evaluate(self, mu: _F64, distances: _F64, dets: Any) -> tuple[NDArray[Any], _F64, int]:
        m = mu.size
        delta = distances.mean(axis=0)
        head = np.cumsum(mu)[:-1]
        tail = np.cumsum(mu[::-1])[::-1][1:]
        eps = np.sqrt(head) * delta + np.sqrt(tail)
        return (
            np.arange(1, m, dtype=np.int64),
            eps,
            _first_extreme(eps, m, math.sqrt(float(mu.sum())), largest=False) + 1,
        )


@dataclass(frozen=True, slots=True)
class LI:
    r"""The ladle dimension rule of Luo and Li (name ``"li"``).

    Needs bootstrap replicates. Valid with the positive semidefinite
    estimators; with ``"asx"``, ``"nasx"`` and ``"phd"`` it issues
    ``ValidityWarning`` and is applied to :math:`|\lambda|`.

    See Also
    --------
    EVG : The eigenvalue-gap rule.
    dimension_criterion : The criterion :math:`g(k)` for plots.

    Notes
    -----
    Let :math:`k_n = m - 1` if :math:`m \le 10` and
    :math:`\lfloor m / \ln m \rfloor` otherwise, and
    :math:`d_k^{(b)} = |\det(W_{1:k}^\top W^{(b)}_{1:k})|` for replicate
    :math:`b`. With :math:`\mu_k = |\lambda_k|`,

    .. math::

        f^0(0) = 0, \qquad f^0(k) = \frac1B \sum_{b=1}^B (1 - d_k^{(b)}), \qquad
        g(k) = \frac{f^0(k)}{1 + \sum_{i=0}^{k_n} f^0(i)}
        + \frac{\mu_{k+1}}{1 + \sum_{i=0}^{k_n} \mu_{i+1}},

    and :math:`n = \hat k = \min \arg\min_{0 \le k \le k_n} g(k)` [1]_. If
    :math:`\hat k = 0`, then :math:`n = 1` and ``AccuracyWarning``
    (``accuracy.zero_dimension``) is issued.

    **Departure from the published method.** The paper averages :math:`f^0`
    over :math:`n` bootstrap replicates, :math:`n` being the sample size;
    here :math:`B` is ``nboot``, chosen by the user, as in the 2016 code. The
    paper's theory takes :math:`B = M`.

    References
    ----------
    .. [1] W. Luo, B. Li. Combining eigenvalues and variation of eigenvectors for
       order determination. *Biometrika* 103(4):875-887, 2016 (details to verify).

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(4)
    >>> X = space.sample(200, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)
    >>> G = np.outer(np.cos(X @ w), w)  # gradients of sin(w^T x)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=20, rng=0, dimension_rule="li")
    >>> sub.n, sub.criterion.shape
    (1, (4,))
    """

    name: ClassVar[str] = "li"
    needs_bootstrap: ClassVar[bool] = True

    def options(self) -> dict[str, str]:
        """Return the rule's options, ``{}``.

        Returns
        -------
        dict
            The empty dict.

        See Also
        --------
        DimensionRule.options : The protocol method.

        Examples
        --------
        >>> from active_subspaces.subspaces import LI
        >>> LI().options()
        {}
        """
        return {}

    def _evaluate(self, mu: _F64, distances: Any, dets: _F64) -> tuple[NDArray[Any], _F64, int]:
        kn = _ladle_kn(mu.size)
        f0 = np.concatenate([[0.0], (1.0 - dets[:, :kn]).mean(axis=0)])
        phi = mu[: kn + 1]
        g = f0 / (1.0 + f0.sum()) + phi / (1.0 + phi.sum())
        return (
            np.arange(0, kn + 1, dtype=np.int64),
            g,
            _first_extreme(g, mu.size, 1.0, largest=False),
        )


DIMENSION_RULES: Mapping[str, type[DimensionRule]] = MappingProxyType(
    {cls.name: cls for cls in (EVG, RS, LI)}
)
"""Read-only registry of the dimension rules: name to class (closed set)."""


# ---------------------------------------------------------------------------
# Validity table [D-41] (private data; the docs render it)
# ---------------------------------------------------------------------------

# The "product" column takes the stricter entry of the other two [D-198].
_VALIDITY_SPACE: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "as": MappingProxyType({"uniform": "ok", "gaussian": "ok", "product": "ok"}),
        "nas": MappingProxyType({"uniform": "ok", "gaussian": "ok", "product": "ok"}),
        "asx": MappingProxyType({"uniform": "warn", "gaussian": "ok", "product": "warn"}),
        "nasx": MappingProxyType({"uniform": "warn", "gaussian": "ok", "product": "warn"}),
        "ols": MappingProxyType({"uniform": "warn", "gaussian": "ok", "product": "warn"}),
        "qphd": MappingProxyType({"uniform": "warn", "gaussian": "ok", "product": "warn"}),
        "sir": MappingProxyType({"uniform": "warn", "gaussian": "ok", "product": "warn"}),
        "phd": MappingProxyType({"uniform": "warn", "gaussian": "ok", "product": "warn"}),
        "save": MappingProxyType({"uniform": "warn", "gaussian": "ok", "product": "warn"}),
        "ss": MappingProxyType({"uniform": "warn", "gaussian": "warn", "product": "warn"}),
        "opg": MappingProxyType({"uniform": "ok", "gaussian": "ok", "product": "ok"}),
    }
)
# Estimator tags of results that no registered estimator produced (§4.5).
_NON_ESTIMATORS = frozenset({"basis", "matrix"})

_VALIDITY_SPACE_BASIS: Mapping[str, str] = MappingProxyType(
    {
        "asx": "the symmetrized E[grad f x^T] is ridge-exact only for Gaussian inputs (Stein's "
        "lemma)",
        "nasx": "ridge-exactness needs the z -> -z symmetry of a Gaussian density",
        "ols": "needs the linearity condition, which the cube satisfies only in special cases "
        "[Li1991]",
        "qphd": "the least-squares quadratic gives E[grad f] and E[Hessian] only for Gaussian "
        "inputs",
        "sir": "needs the linearity condition [Li1991], [GlawsConstantineCook2020]",
        "phd": "Stein's lemma needs Gaussian inputs [Li1992]",
        "save": "needs the linearity and constant-variance conditions [CookWeisberg1991]",
        "ss": "a heuristic with no reference; biased on the cube",
    }
)
_VALIDITY_RULE: Mapping[str, frozenset[str]] = MappingProxyType(
    {  # rule -> estimators it warns with
        "evg": frozenset(),
        "rs": frozenset(ESTIMATORS) - {"as", "opg"},
        "li": frozenset({"asx", "nasx", "phd"}),
    }
)
_VALIDITY_RULE_BASIS: Mapping[str, str] = MappingProxyType(
    {
        "rs": "the response-surface bound holds for C = E[grad f grad f^T] only [Constantine2015]",
        "li": "the ladle assumes a PSD candidate matrix [LuoLi2016]; applied to |lambda|",
    }
)


def _rule_cell(rule: str, estimator: str) -> dict[str, Any] | None:
    if estimator in _VALIDITY_RULE.get(rule, ()):
        return {"rule": rule, "estimator": estimator, "basis": _VALIDITY_RULE_BASIS[rule]}
    return None


def _warn_validity(cells: list[dict[str, Any]]) -> None:
    if not cells:
        return
    parts = [
        (
            f"estimator {c['estimator']!r} on a {c['space']} space"
            if "space" in c
            else f"rule {c['rule']!r} with estimator {c['estimator']!r}"
        )
        + f": {c['basis']}"
        for c in cells
    ]
    _warn(
        ValidityWarning,
        "outside the method's assumptions; " + "; ".join(parts),
        code="validity.assumption",
        details={"cells": cells},
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


def _criterion_length(rule: str, m: int) -> int | None:
    if rule == "fixed" or m == 1:
        return None
    return _ladle_kn(m) + 1 if rule == "li" else m - 1


@_register_result
class SubspaceResult(QoIResult):
    r"""An estimated subspace: signed eigenvalues, eigenvectors and the chosen dimension.

    Produced by :func:`active_subspace`, :func:`with_dimension`,
    :func:`orient`, :func:`subspace_from_basis` and
    :func:`subspace_from_matrix`; the constructor is used by deserialization.
    Everything is in normalized coordinates; :meth:`physical_directions` gives
    the physical view.

    Parameters
    ----------
    space : Uniform or Gaussian
        The input space of the data.
    estimator : str
        Registered estimator name, ``"basis"`` for :func:`subspace_from_basis`,
        or ``"matrix"`` for :func:`subspace_from_matrix`.
    estimator_options : tuple of (str, value) pairs
        The estimator's options, sorted by name.
    eigenvalues : numpy.ndarray, shape (m,), float64
        Signed eigenvalues ordered by :math:`|\lambda|`, descending; NaN for
        ``"basis"``.
    eigenvectors : numpy.ndarray, shape (m, m), float64
        Orthonormal eigenvectors as columns, normalized coordinates. By
        default each column's largest-magnitude entry (the first of those
        within a relative sqrt(eps) of it) is positive; for ``"ols"`` the
        first column is the signed :math:`\hat u / \lVert \hat u \rVert`;
        :func:`orient` sets other signs.
    n : int
        Active dimension, :math:`1 \le n \le m`.
    dimension_rule : {"fixed", "evg", "rs", "li"}
        How ``n`` was chosen.
    dimension_options : tuple of (str, str) pairs
        The rule's options, for example ``(("gap", "linear"),)``.
    criterion : numpy.ndarray, float64, or None
        The rule's criterion: EVG gaps or RS bounds for :math:`k = 1, \dots,
        m-1`, or the ladle :math:`g(0), \dots, g(k_n)`; ``None`` for a fixed
        ``n`` or :math:`m = 1`.
    n_samples : int
        Number of samples :math:`M` (0 for ``"basis"`` and ``"matrix"``).
    nboot : int
        Number of bootstrap replicates :math:`B` (0 if none).
    bootstrap_eigenvalues : numpy.ndarray, shape (B, m), float64, or None
        Signed eigenvalues of each replicate.
    bootstrap_distances : numpy.ndarray, shape (B, m - 1), float64, or None
        Subspace distances :math:`\lVert W_{1:k}^\top W^{(b)}_{k+1:m} \rVert_2`
        for :math:`k = 1, \dots, m-1`.
    bootstrap_dets : numpy.ndarray, shape (B, m - 1), float64, or None
        Ladle statistics :math:`|\det(W_{1:k}^\top W^{(b)}_{1:k})|`.
    bootstrap_eigenvectors : numpy.ndarray, shape (B, m, k), float64, or None
        The first ``k`` eigenvectors of each replicate, each multiplied by
        the sign of its inner product with the full-sample vector, so
        :math:`w_j^\top v_j^{(b)} \ge 0`; ``None`` unless requested with
        ``bootstrap_vectors`` [D-196].
    provenance : Provenance, keyword-only
        How the result was produced.
    qoi : str, keyword-only, default "f"
        Name of the quantity of interest.

    See Also
    --------
    active_subspace : Estimate a subspace.
    with_dimension : Change ``n``.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(4)
    >>> X = space.sample(200, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)
    >>> G = np.outer(np.cos(X @ w), w)  # gradients of sin(w^T x)
    >>> sub = ac.active_subspace(grads=G, space=space)
    >>> sub.n, sub.m, sub.W1.shape, sub.W2.shape
    (1, 4, (4, 1), (4, 3))
    >>> print(np.round(sub.eigenvalues, 3))
    [0.764 0.    0.    0.   ]
    """

    TYPE = "SubspaceResult"
    SCHEMA_VERSION = 2  # 2: bootstrap_eigenvectors [D-196]

    space: InputSpace
    estimator: str
    estimator_options: tuple[tuple[str, int | float | str | None], ...]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    n: int
    dimension_rule: str
    dimension_options: tuple[tuple[str, str], ...]
    criterion: NDArray[np.float64] | None
    n_samples: int
    nboot: int
    bootstrap_eigenvalues: NDArray[np.float64] | None
    bootstrap_distances: NDArray[np.float64] | None
    bootstrap_dets: NDArray[np.float64] | None
    bootstrap_eigenvectors: NDArray[np.float64] | None

    @property
    def m(self) -> int:
        """Number of inputs ``m``.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> ac.active_subspace(grads=np.eye(3), space=ac.Uniform.reference(3)).m
        3
        """
        return self.space.dim

    @property
    def W(self) -> NDArray[np.float64]:
        """All eigenvectors ``(m, m)``, normalized coordinates; the same array as ``eigenvectors``.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> sub = ac.active_subspace(grads=np.eye(3), space=ac.Uniform.reference(3))
        >>> sub.W is sub.eigenvectors
        True
        """
        return self.eigenvectors

    @property
    def W1(self) -> NDArray[np.float64]:
        """Active directions ``(m, n)``, normalized coordinates (a read-only view).

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> G = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> W1 = ac.active_subspace(grads=G, space=ac.Uniform.reference(3), n=1).W1
        >>> W1.ravel() + 0.0  # + 0.0 turns -0.0 into 0.0
        array([1., 0., 0.])
        """
        return self.eigenvectors[:, : self.n]

    @property
    def W2(self) -> NDArray[np.float64]:
        """Inactive directions ``(m, m - n)``, normalized coordinates (a read-only view).

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> G = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> ac.active_subspace(grads=G, space=ac.Uniform.reference(3), n=1).W2.shape
        (3, 2)
        """
        return self.eigenvectors[:, self.n :]

    def physical_directions(self, k: int | None = None) -> NDArray[np.float64]:
        r"""Return an orthonormal basis of the physical view of the first ``k`` directions.

        Parameters
        ----------
        k : int, optional
            Number of directions, :math:`1 \le k \le m`; default ``n``.

        Returns
        -------
        numpy.ndarray, shape (m, k), float64
            Orthonormal basis of :math:`\operatorname{span}(A^{-\top} W_{1:k})`
            in physical coordinates; column :math:`j` keeps the sign of
            :math:`w_j` [D-197]. Eigenvalues have no physical counterpart.

        Raises
        ------
        InputError
            ``input.parameter`` for a bad ``k``.

        See Also
        --------
        active_subspaces.Uniform.directions_to_physical : The map used.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [1.0, 10.0])
        >>> sub = ac.active_subspace(grads=[[1.0, 0.1], [2.0, 0.2]], space=space)
        >>> sub.W1.round(4).ravel(), sub.physical_directions().round(4).ravel()
        (array([0.7071, 0.7071]), array([0.995 , 0.0995]))
        """
        k = self.n if k is None else _v.integer(k, "k", lo=1, hi=self.m)
        return self.space.directions_to_physical(self.eigenvectors[:, :k])

    def _check(self) -> None:
        m = self.space.dim
        basis = self.estimator == "basis"
        _invariant(
            self.estimator in ESTIMATORS or self.estimator in _NON_ESTIMATORS,
            f"unknown estimator {self.estimator!r}",
        )
        _invariant(
            self.dimension_rule in DIMENSION_RULES or self.dimension_rule == "fixed",
            "unknown dimension rule",
        )
        _invariant(self.eigenvalues.shape == (m,), "eigenvalues must be (m,)")
        _invariant(self.eigenvectors.shape == (m, m), "eigenvectors must be (m, m)")
        _invariant(1 <= self.n <= m, "n must satisfy 1 <= n <= m")
        W = self.eigenvectors
        _invariant(
            bool(np.all(np.isfinite(W))) and np.max(np.abs(W.T @ W - np.eye(m))) <= _ORTHO_TOL,
            "eigenvectors must be orthonormal",
        )
        if basis:
            _invariant(
                bool(np.all(np.isnan(self.eigenvalues))), "a 'basis' result has NaN eigenvalues"
            )
            _invariant(
                self.dimension_rule == "fixed" and self.nboot == 0,
                "a 'basis' result has a fixed n and no bootstrap",
            )
        else:
            _invariant(bool(np.all(np.isfinite(self.eigenvalues))), "eigenvalues must be finite")
        if self.estimator in _NON_ESTIMATORS:
            _invariant(
                self.n_samples == 0 and self.nboot == 0,
                f"a {self.estimator!r} result has no samples and no bootstrap",
            )
        _invariant(self.n_samples >= 0 and self.nboot >= 0, "n_samples and nboot must be >= 0")
        boots = (self.bootstrap_eigenvalues, self.bootstrap_distances, self.bootstrap_dets)
        if self.nboot == 0:
            _invariant(
                all(b is None for b in boots), "bootstrap arrays must be None when nboot == 0"
            )
        else:
            B = self.nboot
            _invariant(
                all(b is not None for b in boots), "bootstrap arrays required when nboot > 0"
            )
            assert self.bootstrap_eigenvalues is not None  # checked just above
            assert self.bootstrap_distances is not None
            assert self.bootstrap_dets is not None
            _invariant(
                self.bootstrap_eigenvalues.shape == (B, m), "bootstrap_eigenvalues must be (B, m)"
            )
            _invariant(
                self.bootstrap_distances.shape == (B, m - 1),
                "bootstrap_distances must be (B, m-1)",
            )
            _invariant(self.bootstrap_dets.shape == (B, m - 1), "bootstrap_dets must be (B, m-1)")
        V = self.bootstrap_eigenvectors
        if V is not None:
            _invariant(self.nboot > 0, "bootstrap_eigenvectors needs bootstrap replicates")
            _invariant(
                V.ndim == 3 and V.shape[:2] == (self.nboot, m) and 1 <= V.shape[2] <= m,
                "bootstrap_eigenvectors must be (B, m, k) with 1 <= k <= m",
            )
            k = V.shape[2]
            _invariant(bool(np.all(np.isfinite(V))), "bootstrap_eigenvectors must be finite")
            inner = np.einsum("ij,bij->bj", W[:, :k], V)
            _invariant(
                bool(np.all(inner >= -1e-12)), "bootstrap_eigenvectors must be sign-aligned"
            )
        if self.dimension_rule in ("rs", "li"):
            _invariant(self.nboot > 0, f"rule {self.dimension_rule!r} needs bootstrap replicates")
        length = _criterion_length(self.dimension_rule, m)
        if length is None:
            _invariant(self.criterion is None, "criterion must be None for a fixed n or m = 1")
        else:
            _invariant(
                self.criterion is not None and self.criterion.shape == (length,),
                "criterion has the wrong shape",
            )
        _invariant(
            self.dimension_rule != "fixed" or self.dimension_options == (),
            "a fixed n has no rule options",
        )


@_register_migration("SubspaceResult", 1)
def _subspace_v1_to_v2(fields: dict[str, Any]) -> dict[str, Any]:
    """Version 1 stored no eigenvector replicates [D-196]."""
    out = dict(fields)
    out["bootstrap_eigenvectors"] = None
    return out


# ---------------------------------------------------------------------------
# Resolution of estimators and rules
# ---------------------------------------------------------------------------


def _resolve(value: Any, registry: Mapping[str, type], param: str) -> Any:
    if isinstance(value, str):
        cls = registry.get(value.lower())
        if cls is None:
            raise InputError(
                f"{param}: unknown name {value!r}; expected one of {sorted(registry)}",
                code="input.parameter",
                details={"param": param, "expected": sorted(registry), "got": value},
            )
        return cls()
    if type(value) in registry.values():
        return value
    raise InputError(
        f"{param}: expected a registered name or a built-in instance "
        f"({', '.join(c.__name__ for c in registry.values())}), "
        f"got {type(value).__name__}; the set is closed",
        code="input.parameter",
        details={"param": param, "got": type(value).__name__},
    )


def _no_bootstrap(rule: str) -> InputError:
    return InputError(
        f"dimension rule {rule!r} needs bootstrap replicates; pass nboot > 0",
        code="input.no_bootstrap",
        details={"rule": rule},
    )


def _apply_rule(rule: Any, lam: _F64, distances: Any, dets: Any) -> tuple[int, _F64 | None, int]:
    """Return (n, criterion or None, raw k_hat)."""
    m = lam.size
    if m == 1:
        return 1, None, 1
    _, values, k_hat = rule._evaluate(np.abs(lam), distances, dets)
    return max(k_hat, 1), values, k_hat


def _warn_zero_dimension(rule: str, k_hat: int) -> None:
    if rule == "li" and k_hat == 0:
        _warn(
            AccuracyWarning,
            "the ladle chose k = 0; n is set to 1",
            code="accuracy.zero_dimension",
            details={"k_hat": 0},
        )


# ---------------------------------------------------------------------------
# Bootstrap [D-30], [D-80]
# ---------------------------------------------------------------------------


def _resample(d: _Arrays, idx: NDArray[Any]) -> _Arrays:
    w = d.w[idx]
    total = w.sum()
    if total == 0:
        raise NumericalError(
            "a bootstrap replicate drew only rows with zero quadrature weight",
            code="numerical.singular",
            details={"reason": "zero_weight_replicate"},
        )

    def pick(a: _F64 | None) -> _F64 | None:
        return None if a is None else a[idx]

    return _Arrays(pick(d.X), pick(d.f), pick(d.G), w / total, d.Sigma)


def _replicate(
    est: Any, d: _Arrays, W: _F64, gen: np.random.Generator, k: int = 0
) -> tuple[_F64, _F64, _F64, dict[str, Any], _F64 | None]:
    """One bootstrap replicate.

    Draw indices from ``gen`` only, recompute, compare with W (§3.7); return the first ``k``
    replicate eigenvectors as computed (aligned later, in replicate order).
    """
    M = d.w.shape[0]
    idx = np.asarray(gen.integers(0, M, size=M))  # asarray: an ndarray for the type checker
    # _resample makes fresh arrays, so AS may scale the replicate's gradients in place [D-202].
    notes: dict[str, Any] = {"_overwrite_G": True}
    lam, Wb = est._decompose(_resample(d, idx), notes)
    notes.pop("_overwrite_G", None)
    m = W.shape[0]
    P = W.T @ Wb
    dist = np.empty(m - 1)
    dets = np.empty(m - 1)
    for j in range(1, m):
        dist[j - 1], dets[j - 1] = _distance_and_det(P, j)
    return lam, dist, dets, notes, (Wb[:, :k].copy() if k else None)


def _distance_and_det(P: _F64, k: int) -> tuple[float, float]:
    """``(||P[:k, k:]||_2, |det P[:k, :k]|)`` for an orthogonal ``P``, on the smaller side.

    The spectral norm is the square root of the largest eigenvalue of the smaller Gram matrix of
    ``P[:k, k:]``, which is accurate to rounding relative to the norm itself. Because ``P`` is
    orthogonal, ``|det P[:k, :k]| = |det P[k:, k:]|`` (Jacobi's complementary-minor identity), so
    the smaller block is used. Both are exact rewrites of the per-k definitions; they cut the
    ``O(m^4)`` cost of a replicate by about 2.5 at ``m = 1000`` [D-178].
    """
    m = P.shape[0]
    B = P[:k, k:]
    G = B @ B.T if k <= m - k else B.T @ B
    top = G.shape[0] - 1
    lam_max = scipy.linalg.eigvalsh(G, subset_by_index=[top, top], check_finite=False)[0]
    A = P[:k, :k] if k <= m - k else P[k:, k:]
    return math.sqrt(max(float(lam_max), 0.0)), abs(float(np.linalg.det(A)))


def _bootstrap(
    est: Any, d: _Arrays, W: _F64, gen: np.random.Generator, nboot: int, executor: Any, k: int = 0
) -> tuple[_F64, _F64, _F64, list[dict[str, Any]], _F64 | None]:
    children = _random.spawn(gen, nboot)
    if executor is None:
        outs = [_replicate(est, d, W, c, k) for c in children]
    else:
        futures = [executor.submit(_replicate, est, d, W, c, k) for c in children]
        outs = [fu.result() for fu in futures]
    lam = np.array([o[0] for o in outs])
    dist = np.array([o[1] for o in outs]).reshape(nboot, W.shape[0] - 1)
    dets = np.array([o[2] for o in outs]).reshape(nboot, W.shape[0] - 1)
    vecs = _align_replicates(W, [o[4] for o in outs]) if k else None
    return lam, dist, dets, [o[3] for o in outs], vecs


def _warn_unaligned(W: _F64, V: _F64 | None) -> None:
    """Warn ``accuracy.unaligned_replicates`` for poorly aligned stored replicates [D-196].

    A column warns when its smallest ``|w_j^T w_j^(b)|`` is below 0.5. The warning is issued
    last, after the other warnings of the call (07 §6).
    """
    if V is None:
        return
    minima = np.abs(np.einsum("ij,bij->bj", W[:, : V.shape[2]], V)).min(axis=0)
    bad = [j for j in range(V.shape[2]) if minima[j] < _ALIGN_MIN]
    if bad:
        _warn(
            AccuracyWarning,
            "bootstrap eigenvector replicates are poorly aligned with the full-sample vectors "
            f"(columns {bad}); their components are ill-defined, e.g. near repeated eigenvalues",
            code="accuracy.unaligned_replicates",
            details={"columns": bad, "minima": [float(minima[j]) for j in bad]},
        )


def _align_replicates(W: _F64, reps: list[Any]) -> _F64:
    """Stack replicate eigenvectors ``(B, m, k)``, aligned with the full-sample vectors.

    Column ``j`` of each replicate is multiplied by ``sign(w_j^T w_j^(b))`` (a zero product keeps
    the replicate's sign) [D-196].
    """
    V = np.stack(reps)  # (B, m, k)
    k = V.shape[2]
    inner = np.einsum("ij,bij->bj", W[:, :k], V)
    V *= np.where(inner < 0.0, -1.0, 1.0)[:, None, :]
    return V


def _warn_accuracy(name: str, main: dict[str, Any], reps: list[dict[str, Any]]) -> None:
    n_rank = sum(1 for r in reps if r.get("rank_deficient"))
    if main.get("rank_deficient") or n_rank:
        details: dict[str, Any] = {
            "estimator": name,
            "data": bool(main.get("rank_deficient")),
            "replicates": n_rank,
        }
        if "centers" in main:
            details["centers"] = int(main["centers"])
        _warn(
            AccuracyWarning,
            f"{name}: rank-deficient least-squares fit; minimum-norm solution used",
            code="accuracy.rank_deficient",
            details=details,
        )
    n_deg = sum(1 for r in reps if r.get("degenerate"))
    if main.get("degenerate") or n_deg:
        _warn(
            AccuracyWarning,
            f"{name}: the estimator's matrix is exactly zero; eigenvectors are arbitrary (W = I)",
            code="accuracy.degenerate_matrix",
            details={"estimator": name, "data": bool(main.get("degenerate")), "replicates": n_deg},
        )


# ---------------------------------------------------------------------------
# active_subspace
# ---------------------------------------------------------------------------


def _space_mismatch() -> InputError:
    return InputError(
        "space: differs from the space of the GradientResult given as grads",
        code="input.parameter",
        details={"param": "space", "reason": "space_mismatch"},
    )


def _prepare(
    space: Any, X: Any, f: Any, grads: Any, quad_weights: Any, qoi: Any
) -> tuple[Any, _Arrays, dict[str, Any], str]:
    """Validate the data arrays (spec 02) and return (space, normalized arrays, digests, qoi)."""
    if isinstance(grads, GradientResult):
        gr = grads
        if space is not None:
            _v.check_space(space)
            if space != gr.space:
                raise _space_mismatch()
        space = gr.space
        G: _F64 | None = np.array(gr.grads, dtype=np.float64)
        M = gr.grads.shape[0]
        digests: dict[str, Any] = {}
        Xn = Fn = None
        if X is not None or f is not None:
            dd = _v.data(space, X=X, f=f, quad_weights=quad_weights, use_quad_weights=True)
            if dd.M != M:
                raise InputError(
                    f"grads: the GradientResult has {M} rows, the other arrays {dd.M}",
                    code="input.size_mismatch",
                    details={"param": "grads", "expected": dd.M, "got": M},
                )
            Xn, Fn, w, digests = dd.X, dd.f, dd.quad_weights, dict(dd.digests)
        else:
            w = _v.quad_weights(quad_weights, M)
            if quad_weights is not None:
                digests["quad_weights"] = _v.digest(_v.coerce(quad_weights, "quad_weights"))
        if Xn is None:
            Xn = np.array(gr.points, dtype=np.float64)
        digests["grads"] = _v.digest(gr.grads_physical)
        qoi = gr.qoi if qoi is None else qoi
    else:
        if space is None:
            raise InputError(
                "space: required unless grads is a GradientResult",
                code="input.parameter",
                details={"param": "space"},
            )
        dd = _v.data(
            space, X=X, f=f, grads=grads, quad_weights=quad_weights, use_quad_weights=True
        )
        Xn, Fn, G, w, digests = dd.X, dd.f, dd.grads, dd.quad_weights, dict(dd.digests)
        qoi = "f" if qoi is None else qoi
    if not isinstance(qoi, str) or not qoi:
        raise InputError(
            "qoi: expected a non-empty string", code="input.parameter", details={"param": "qoi"}
        )
    Sigma = np.array(space.reference_covariance, dtype=np.float64)
    assert w is not None  # use_quad_weights=True: never None
    return space, _Arrays(Xn, Fn, G, w, Sigma), digests, qoi


def active_subspace(
    *,
    space: InputSpace | None = None,
    X: Any = None,
    f: Any = None,
    grads: Any = None,
    quad_weights: Any = None,
    estimator: str | Estimator = "as",
    n: int | None = None,
    dimension_rule: str | DimensionRule | None = None,
    nboot: int = 0,
    bootstrap_vectors: int = 0,
    rng: _random.RngLike = None,
    executor: concurrent.futures.Executor | None = None,
    qoi: str | None = None,
) -> SubspaceResult:
    r"""Estimate an active subspace, or a sufficient dimension reduction subspace, from samples.

    All arguments are keyword-only. Data are given in physical coordinates
    and mapped to normalized coordinates on entry.

    Parameters
    ----------
    space : Uniform or Gaussian, optional
        The input space. Required unless ``grads`` is a
        :class:`~active_subspaces.GradientResult`; then, if given, it must
        equal that result's space.
    X : array_like, shape (M, m), float, optional
        Sample points in physical coordinates, inside the support of
        ``space``.
    f : array_like, shape (M,) or (M, 1), float, optional
        Values of the scalar quantity of interest at ``X``.
    grads : array_like, shape (M, m), float, or GradientResult, optional
        Gradients with respect to the physical inputs, or a
        ``GradientResult`` (its normalized points and gradients are used).
    quad_weights : array_like, shape (M,), float, optional
        Quadrature weights :math:`\omega_i \ge 0` summing to 1 (within
        :math:`10^{-10}`); default :math:`1/M`.
    estimator : str or estimator instance, default "as"
        A name in :data:`ESTIMATORS` (case-insensitive) or an instance such
        as ``SIR(n_slices=5)``. Each estimator needs some of ``X``, ``f`` and
        ``grads``.
    n : int, optional
        A fixed dimension, :math:`1 \le n \le m`. Give at most one of ``n``
        and ``dimension_rule``.
    dimension_rule : str or rule instance, optional
        A name in :data:`DIMENSION_RULES` or an instance; default
        ``EVG(gap="linear")`` when ``n`` is not given. ``"rs"`` and ``"li"``
        need ``nboot > 0``.
    nboot : int, default 0
        Number of bootstrap replicates :math:`B`.
    bootstrap_vectors : int, default 0
        Number :math:`k` of leading eigenvectors whose replicates are stored,
        sign-aligned, in ``bootstrap_eigenvectors`` (:math:`8Bmk` bytes);
        :math:`0 \le k \le m`, :math:`k \le n` for a fixed ``n``, and
        :math:`k \ge 1` needs ``nboot > 0``. With a dimension rule the
        :math:`k` columns are kept whatever ``n`` the rule selects [D-196].
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator, optional
        Source of randomness for the bootstrap only; one child stream per
        replicate, so the result does not depend on ``executor``.
    executor : concurrent.futures.Executor, optional
        Runs the bootstrap replicates; never changes the result.
    qoi : str, optional
        Name of the quantity of interest; default the ``GradientResult``'s,
        else ``"f"``.

    Returns
    -------
    SubspaceResult
        Signed eigenvalues, eigenvectors (normalized coordinates), the
        dimension ``n``, the rule's criterion and the bootstrap replicates.

    Raises
    ------
    InputError
        ``input.dtype``, ``input.shape``, ``input.vector_qoi``,
        ``input.size_mismatch``, ``input.nonfinite``, ``input.out_of_space``
        or ``input.weights`` for bad data; ``input.parameter`` for a bad
        option, a missing array the estimator needs, both ``n`` and
        ``dimension_rule``, or a space that differs from the
        ``GradientResult``'s (``details["reason"] == "space_mismatch"``);
        ``input.too_few_samples`` below the estimator's minimum;
        ``input.no_bootstrap`` for ``"rs"`` or ``"li"`` without a bootstrap;
        ``input.parameter`` for a bad ``bootstrap_vectors``.
    NumericalError
        ``numerical.nonfinite`` if the estimator's matrix is not finite;
        ``numerical.singular`` if a factorization fails or a bootstrap
        replicate draws only zero-weight rows.

    Warns
    -----
    ValidityWarning
        ``validity.assumption`` for an estimator or rule used outside its
        assumptions (for example SIR on a uniform space);
        ``validity.bootstrap_weights`` for a bootstrap with non-uniform
        quadrature weights.
    AccuracyWarning
        ``accuracy.rank_deficient`` for a rank-deficient least-squares fit
        (OLS, QPHD, OPG); ``accuracy.degenerate_matrix`` if the matrix is
        exactly zero (``W = I`` is returned); ``accuracy.zero_dimension`` if
        the ladle chooses 0 (``n = 1`` is used); ``accuracy.unaligned_replicates``
        if a stored eigenvector column has a replicate with
        :math:`|w_j^\top w_j^{(b)}| < 0.5`.

    See Also
    --------
    with_dimension : Choose another dimension without recomputation.
    bootstrap_interval : Intervals from the replicates.
    local_linear_gradients : Gradients from data, accepted as ``grads``.

    Notes
    -----
    Each estimator gives a symmetric matrix :math:`C` with eigendecomposition

    .. math::

        C = W \Lambda W^\top, \qquad |\lambda_1| \ge \dots \ge |\lambda_m|,

    with eigenvalues reported signed (they are non-negative for the positive
    semidefinite estimators), ties kept in the order the factorization
    returns, and each eigenvector signed so that its largest-magnitude entry
    is positive (for ``"ols"``, :math:`w_1 = \hat u/\lVert \hat u \rVert`
    instead). :math:`W = [W_1 \ W_2]` with :math:`W_1` the first ``n``
    columns. For ``"as"``,
    :math:`C = \sum_i \omega_i \nabla_x f(x_i) \nabla_x f(x_i)^\top` [1]_.

    The bootstrap resamples the rows :math:`B` times, renormalizes the
    weights, recomputes the eigenpairs and stores the eigenvalues, the
    subspace distances
    :math:`\lVert W_{1:k}^\top W^{(b)}_{k+1:m} \rVert_2` and the statistics
    :math:`|\det(W_{1:k}^\top W^{(b)}_{1:k})|` [2]_. The eigenvalue bootstrap
    is inconsistent at repeated or zero eigenvalues [3]_, [4]_, and it
    assumes i.i.d. rows.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.
    .. [2] P. G. Constantine, D. F. Gleich. Computing active subspaces with Monte
       Carlo. arXiv:1408.0545, 2014 (details to verify).
    .. [3] R. Beran, M. S. Srivastava. Bootstrap tests and confidence regions for
       functions of a covariance matrix. *Ann. Statist.* 13(1):95-115, 1985
       (details to verify).
    .. [4] M. L. Eaton, D. E. Tyler. On Wielandt's inequality and its application to
       the asymptotic distribution of the eigenvalues of a random symmetric
       matrix. *Ann. Statist.* 19(1):260-271, 1991 (details to verify).

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(4)
    >>> X = space.sample(200, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)
    >>> G = np.outer(np.cos(X @ w), w)  # gradients of sin(w^T x)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=50, rng=0)
    >>> sub.n, sub.dimension_rule
    (1, 'evg')
    >>> print(f"{abs(w @ sub.W1[:, 0]):.3f}")
    1.000

    Values instead of gradients, with sliced inverse regression on a Gaussian
    space:

    >>> gspace = ac.Gaussian.reference(4)
    >>> Xg = gspace.sample(500, rng=1)
    >>> sir = ac.active_subspace(X=Xg, f=np.sin(Xg @ w), space=gspace, estimator="sir", n=1)
    >>> print(f"{abs(w @ sir.W1[:, 0]):.2f}")
    1.00
    """
    space, d, digests, qoi = _prepare(space, X, f, grads, quad_weights, qoi)
    m = space.dim
    nboot = _v.integer(nboot, "nboot", lo=0)
    kvec = _v.integer(bootstrap_vectors, "bootstrap_vectors", lo=0, hi=m)
    if kvec > 0 and nboot == 0:
        raise InputError(
            "bootstrap_vectors: needs bootstrap replicates; pass nboot > 0",
            code="input.parameter",
            details={"param": "bootstrap_vectors", "reason": "no_bootstrap"},
        )
    est = _resolve(estimator, ESTIMATORS, "estimator")
    if n is not None and dimension_rule is not None:
        raise InputError(
            "give at most one of n and dimension_rule",
            code="input.parameter",
            details={"param": "dimension_rule"},
        )
    rule = None
    if n is not None:
        n = _v.integer(n, "n", lo=1, hi=m)
        if kvec > n:
            raise InputError(
                f"bootstrap_vectors: {kvec} exceeds n = {n}",
                code="input.parameter",
                details={"param": "bootstrap_vectors", "expected": [0, n], "got": kvec},
            )
    else:
        rule = (
            EVG()
            if dimension_rule is None
            else _resolve(dimension_rule, DIMENSION_RULES, "dimension_rule")
        )
        if rule.needs_bootstrap and nboot == 0:
            raise _no_bootstrap(rule.name)
    if executor is not None and not callable(getattr(executor, "submit", None)):
        raise InputError(
            "executor: expected a concurrent.futures.Executor",
            code="input.parameter",
            details={"param": "executor"},
        )
    present = {"X": d.X is not None, "f": d.f is not None, "grads": d.G is not None}
    missing = sorted(k for k in est.requires if not present[k])
    if missing:
        raise InputError(
            f"estimator {est.name!r} needs {', '.join(missing)}",
            code="input.parameter",
            details={"param": missing[0], "estimator": est.name, "missing": missing},
        )
    est._check_data(d)
    gen, rng_desc = _random.generator(rng)

    cells = []
    if _VALIDITY_SPACE[est.name][space.kind] == "warn":
        cells.append(
            {"estimator": est.name, "space": space.kind, "basis": _VALIDITY_SPACE_BASIS[est.name]}
        )
    if rule is not None and (cell := _rule_cell(rule.name, est.name)) is not None:
        cells.append(cell)
    _warn_validity(cells)
    if nboot > 0 and not np.all(d.w == d.w[0]):
        _warn(
            ValidityWarning,
            "bootstrap with non-uniform quadrature weights: i.i.d. resampling of the rows does "
            "not hold",
            code="validity.bootstrap_weights",
            details={"nboot": nboot},
        )

    # Without a bootstrap the normalized gradients are read once, so AS may scale them in place.
    notes: dict[str, Any] = {"_overwrite_G": nboot == 0}
    lam, W = est._decompose(d, notes)
    notes.pop("_overwrite_G", None)
    boot_lam = boot_dist = boot_dets = boot_vecs = None
    rep_notes: list[dict[str, Any]] = []
    if nboot > 0:
        boot_lam, boot_dist, boot_dets, rep_notes, boot_vecs = _bootstrap(
            est, d, W, gen, nboot, executor, kvec
        )
    if rule is not None:
        n, criterion, k_hat = _apply_rule(rule, lam, boot_dist, boot_dets)
    else:
        assert n is not None  # a fixed n, validated above
        criterion, k_hat = None, n
    _warn_accuracy(est.name, notes, rep_notes)
    if rule is not None:
        _warn_zero_dimension(rule.name, k_hat)
    _warn_unaligned(W, boot_vecs)

    params: dict[str, Any] = {
        "estimator": est.name,
        "estimator_options": est.options(),
        "n": None if rule is not None else n,
        "dimension_rule": "fixed" if rule is None else rule.name,
        "dimension_options": {} if rule is None else rule.options(),
        "nboot": nboot,
        "bootstrap_vectors": kvec,
        "qoi": qoi,
    }
    prov = _make_provenance(
        "active_subspace",
        params,
        space=space,
        rng_state=rng_desc if nboot > 0 else None,
        digests=digests,
    )
    return SubspaceResult(
        space=space,
        estimator=est.name,
        estimator_options=tuple(sorted(est.options().items())),
        eigenvalues=lam,
        eigenvectors=W,
        n=int(n),
        dimension_rule="fixed" if rule is None else rule.name,
        dimension_options=() if rule is None else tuple(sorted(rule.options().items())),
        criterion=criterion,
        n_samples=int(d.w.shape[0]),
        nboot=nboot,
        bootstrap_eigenvalues=boot_lam,
        bootstrap_distances=boot_dist,
        bootstrap_dets=boot_dets,
        bootstrap_eigenvectors=boot_vecs,
        qoi=qoi,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Helpers (§4.5)
# ---------------------------------------------------------------------------


def _check_result(result: Any) -> SubspaceResult:
    if not isinstance(result, SubspaceResult):
        raise InputError(
            f"result: expected a SubspaceResult, got {type(result).__name__}",
            code="input.parameter",
            details={"param": "result"},
        )
    return result


def _rule_for(result: SubspaceResult, rule: Any, param: str) -> Any:
    r = _resolve(rule, DIMENSION_RULES, param)
    if result.estimator == "basis":
        raise InputError(
            "a 'basis' result has no eigenvalues; only a fixed n applies",
            code="input.parameter",
            details={"param": param, "estimator": "basis"},
        )
    if r.needs_bootstrap and result.nboot == 0:
        raise _no_bootstrap(r.name)
    return r


def with_dimension(
    result: SubspaceResult,
    n: int | None = None,
    *,
    dimension_rule: str | DimensionRule | None = None,
) -> SubspaceResult:
    r"""Return a subspace result with a new dimension, without recomputation.

    Parameters
    ----------
    result : SubspaceResult
        The result to change.
    n : int, optional
        A fixed dimension, :math:`1 \le n \le m`.
    dimension_rule : str or rule instance, optional, keyword-only
        A rule of :data:`DIMENSION_RULES`. Give exactly one of ``n`` and
        ``dimension_rule``.

    Returns
    -------
    SubspaceResult
        A new result in which only ``n``, ``dimension_rule``,
        ``dimension_options``, ``criterion`` and the provenance differ; the
        eigenvector signs and any eigenvector replicates are kept.

    Raises
    ------
    InputError
        ``input.parameter`` unless exactly one of ``n`` and
        ``dimension_rule`` is given, for a bad value, or for a rule applied to
        a ``"basis"`` result; ``input.no_bootstrap`` for ``"rs"`` or ``"li"``
        on a result without replicates.

    Warns
    -----
    ValidityWarning
        ``validity.assumption`` for a rule used outside its assumptions.
    AccuracyWarning
        ``accuracy.zero_dimension`` if the ladle chooses 0.

    See Also
    --------
    active_subspace : Estimate a subspace.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(4)
    >>> X = space.sample(200, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)
    >>> G = np.outer(np.cos(X @ w), w)  # gradients of sin(w^T x)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=20, rng=0)
    >>> ac.with_dimension(sub, 2).n, ac.with_dimension(sub, dimension_rule="rs").n
    (2, 1)
    """
    result = _check_result(result)
    if (n is None) == (dimension_rule is None):
        raise InputError(
            "give exactly one of n and dimension_rule",
            code="input.parameter",
            details={"param": "dimension_rule"},
        )
    src = result.provenance.to_dict()
    src["input_space"] = None  # the space is recorded at the top level
    if n is not None:
        n = _v.integer(n, "n", lo=1, hi=result.m)
        rule_name, opts, criterion = "fixed", (), None
        params: dict[str, Any] = {"n": n, "source": src}
    else:
        rule = _rule_for(result, dimension_rule, "dimension_rule")
        _warn_validity([c for c in [_rule_cell(rule.name, result.estimator)] if c is not None])
        n, criterion, k_hat = _apply_rule(
            rule, result.eigenvalues, result.bootstrap_distances, result.bootstrap_dets
        )
        _warn_zero_dimension(rule.name, k_hat)
        rule_name, opts = rule.name, tuple(sorted(rule.options().items()))
        params = {"dimension_rule": rule.name, "dimension_options": rule.options(), "source": src}
    prov = _make_provenance("with_dimension", params, space=result.space)
    return SubspaceResult(
        space=result.space,
        estimator=result.estimator,
        estimator_options=result.estimator_options,
        eigenvalues=result.eigenvalues,
        eigenvectors=result.eigenvectors,
        n=int(n),
        dimension_rule=rule_name,
        dimension_options=opts,
        criterion=criterion,
        n_samples=result.n_samples,
        nboot=result.nboot,
        bootstrap_eigenvalues=result.bootstrap_eigenvalues,
        bootstrap_distances=result.bootstrap_distances,
        bootstrap_dets=result.bootstrap_dets,
        bootstrap_eigenvectors=result.bootstrap_eigenvectors,
        qoi=result.qoi,
        provenance=prov,
    )


def dimension_criterion(
    result: SubspaceResult, *, rule: str | DimensionRule
) -> tuple[NDArray[np.int64], NDArray[np.float64], int]:
    """Evaluate a dimension rule on a result without changing its dimension.

    Parameters
    ----------
    result : SubspaceResult
        A result with eigenvalues (not ``"basis"``), and with bootstrap
        replicates for ``"rs"`` and ``"li"``.
    rule : str or rule instance, keyword-only
        A rule of :data:`DIMENSION_RULES`.

    Returns
    -------
    ks : numpy.ndarray, int64
        Candidate dimensions: 1 to m-1 for ``"evg"`` and ``"rs"``, 0 to
        :math:`k_n` for ``"li"``; empty when :math:`m = 1`.
    values : numpy.ndarray, float64
        The criterion at each candidate.
    k_hat : int
        The rule's raw choice; 0 is possible for ``"li"``.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad result or rule;
        ``input.no_bootstrap`` for a rule that needs replicates.

    See Also
    --------
    with_dimension : Apply the rule's choice.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(4)
    >>> X = space.sample(200, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)
    >>> G = np.outer(np.cos(X @ w), w)  # gradients of sin(w^T x)
    >>> sub = ac.active_subspace(grads=G, space=space)
    >>> ks, values, k_hat = ac.dimension_criterion(sub, rule="evg")
    >>> ks, k_hat
    (array([1, 2, 3]), 1)
    """
    result = _check_result(result)
    r = _rule_for(result, rule, "rule")
    if result.m == 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0), 1
    ks, values, k_hat = r._evaluate(
        np.abs(result.eigenvalues), result.bootstrap_distances, result.bootstrap_dets
    )
    return ks, np.asarray(values, dtype=np.float64), int(k_hat)


def bootstrap_interval(
    replicates: Any, *, kind: Literal["percentile", "minmax"] = "percentile", level: float = 0.95
) -> NDArray[np.float64]:
    r"""Return intervals from bootstrap replicates.

    Parameters
    ----------
    replicates : array_like, shape (B, k), float
        One replicate per row, for example ``result.bootstrap_eigenvalues``;
        a 1-D array is read as ``(B, 1)``. Finite.
    kind : {"percentile", "minmax"}, default "percentile", keyword-only
        Percentile intervals, or the column minimum and maximum.
    level : float, default 0.95, keyword-only
        Confidence level in :math:`(0, 1)`, for ``"percentile"``.

    Returns
    -------
    numpy.ndarray, shape (2, k), float64
        Row 0 the lower ends, row 1 the upper ends.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad ``kind`` or ``level`` or an empty
        array; ``input.shape``, ``input.dtype`` or ``input.nonfinite`` for
        bad replicates.

    See Also
    --------
    active_subspace : Computes the replicates with ``nboot > 0``.

    Notes
    -----
    ``"percentile"`` uses the :math:`\alpha/2` and :math:`1 - \alpha/2`
    empirical quantiles (NumPy's ``linear`` method) with
    :math:`\alpha = 1 - \mathrm{level}`. ``"minmax"`` reproduces the 2016
    intervals; they widen as :math:`B` grows. The eigenvalue bootstrap is
    inconsistent at repeated or zero eigenvalues.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> reps = np.arange(1.0, 11.0).reshape(10, 1)
    >>> ac.bootstrap_interval(reps, level=0.8)
    array([[1.9],
           [9.1]])
    >>> ac.bootstrap_interval(reps, kind="minmax")
    array([[ 1.],
           [10.]])
    """
    kind = cast(Literal["percentile", "minmax"], _v.choice(kind, "kind", ("percentile", "minmax")))
    level = _v.real(level, "level", lo=0.0, hi=1.0, lo_open=True, hi_open=True)
    arr = _v.coerce(replicates, "replicates")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise InputError(
            f"replicates: expected shape (B, k), got {arr.shape}",
            code="input.shape",
            details={"param": "replicates", "got": list(arr.shape)},
        )
    if arr.size == 0:
        raise InputError(
            "replicates: empty array",
            code="input.parameter",
            details={"param": "replicates", "got": list(arr.shape)},
        )
    _v.check_finite(arr, "replicates")
    if kind == "minmax":
        return np.vstack([arr.min(axis=0), arr.max(axis=0)])
    alpha = 1.0 - level
    return np.quantile(arr, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0, method="linear")


def mean_distance(result: SubspaceResult) -> NDArray[np.float64]:
    r"""Return the bootstrap mean subspace distance for each dimension.

    Parameters
    ----------
    result : SubspaceResult
        A result with bootstrap replicates.

    Returns
    -------
    numpy.ndarray, shape (m - 1,), float64
        :math:`\hat\delta_k = \frac1B \sum_b \lVert W_{1:k}^\top W^{(b)}_{k+1:m} \rVert_2`
        for :math:`k = 1, \dots, m-1`, the sine of the largest principal
        angle between the estimated and replicate subspaces, averaged.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad result; ``input.no_bootstrap`` if it
        has no replicates.

    See Also
    --------
    RS : The rule that uses it.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(4)
    >>> X = space.sample(200, rng=0)
    >>> w = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5.0)
    >>> G = np.outer(np.cos(X @ w), w)  # gradients of sin(w^T x)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=20, rng=0)
    >>> ac.mean_distance(sub).shape
    (3,)
    """
    result = _check_result(result)
    if result.bootstrap_distances is None:
        raise InputError(
            "the result has no bootstrap replicates; pass nboot > 0",
            code="input.no_bootstrap",
            details={},
        )
    return result.bootstrap_distances.mean(axis=0)


def subspace_from_basis(basis: Any, *, space: InputSpace) -> SubspaceResult:
    r"""Wrap a basis of physical active directions as a subspace result.

    Use it to apply geometry, surrogates and the other tools to a subspace
    found elsewhere.

    Parameters
    ----------
    basis : array_like, shape (m, n), float
        Linearly independent physical directions as columns,
        :math:`1 \le n \le m`, read as covectors: a direction :math:`a` means
        :math:`f(\xi) \approx h(a^\top \xi)`.
    space : Uniform or Gaussian, keyword-only
        The input space.

    Returns
    -------
    SubspaceResult
        ``estimator="basis"``, NaN eigenvalues, ``n`` the number of columns,
        ``dimension_rule="fixed"`` and no bootstrap.

    Raises
    ------
    InputError
        ``input.shape`` for a bad shape; ``input.parameter`` for a bad space
        or linearly dependent columns (``details["reason"] ==
        "rank_deficient"``); ``input.dtype`` or ``input.nonfinite`` for bad
        values.

    See Also
    --------
    SubspaceResult.physical_directions : The inverse view.

    Notes
    -----
    The normalized directions span :math:`A^\top B` for the basis :math:`B`;
    a thin QR factorization gives :math:`W_1`, completed to an orthonormal
    :math:`W` and sign-oriented.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 1.0, 1.0])
    >>> sub = ac.subspace_from_basis([[1.0], [1.0], [0.0], [0.0]], space=space)
    >>> sub.estimator, sub.n, sub.W1.round(4).ravel()
    ('basis', 1, array([0.4472, 0.8944, 0.    , 0.    ]))
    """
    space = _v.check_space(space)
    m = space.dim
    B = _v.matrix(basis, "basis", rows=m)
    k = B.shape[1]
    if not 1 <= k <= m:
        raise InputError(
            f"basis: expected shape ({m}, n) with 1 <= n <= {m}, got {B.shape}",
            code="input.shape",
            details={"param": "basis", "expected": [m, "n"], "got": list(B.shape)},
        )
    A = np.array(space.scale, dtype=np.float64)
    Q1, R = np.linalg.qr(A.T @ B, mode="reduced")
    diag = np.abs(np.diagonal(R))
    if not diag.max() > 0 or diag.min() <= _BASIS_RANK_TOL * diag.max():
        raise InputError(
            "basis: columns are linearly dependent (rank-deficient)",
            code="input.parameter",
            details={"param": "basis", "reason": "rank_deficient"},
        )
    Qf, _ = np.linalg.qr(Q1, mode="complete")
    W = _orient_columns(Qf)
    prov = _make_provenance(
        "subspace_from_basis", {}, space=space, digests={"basis": _v.digest(B)}
    )
    return SubspaceResult(
        space=space,
        estimator="basis",
        estimator_options=(),
        eigenvalues=np.full(m, np.nan),
        eigenvectors=W,
        n=k,
        dimension_rule="fixed",
        dimension_options=(),
        criterion=None,
        n_samples=0,
        nboot=0,
        bootstrap_eigenvalues=None,
        bootstrap_distances=None,
        bootstrap_dets=None,
        bootstrap_eigenvectors=None,
        provenance=prov,
    )


def subspace_from_matrix(
    C: Any,
    *,
    space: InputSpace,
    n: int | None = None,
    dimension_rule: str | DimensionRule | None = None,
    qoi: str | None = None,
) -> SubspaceResult:
    r"""Wrap a precomputed symmetric matrix as a subspace result with eigenvalues.

    Use it for a matrix accumulated by hand, for example
    :math:`\sum_i \omega_i g_i g_i^\top` over a tensor quadrature rule too
    large to hold as one gradient array.

    Parameters
    ----------
    C : array_like, shape (m, m), float
        A symmetric matrix in **normalized** coordinates (for a matrix
        :math:`C_\xi` of physical gradients, pass :math:`A^\top C_\xi A` with
        ``A = space.scale``), symmetric to a relative :math:`\sqrt\epsilon`.
    space : Uniform or Gaussian, keyword-only
        The input space.
    n : int, optional, keyword-only
        A fixed dimension, :math:`1 \le n \le m`.
    dimension_rule : str or rule instance, optional, keyword-only
        A rule of :data:`DIMENSION_RULES`; default ``EVG(gap="linear")``.
        ``"rs"`` and ``"li"`` need replicates, which this result has not.
    qoi : str, optional, keyword-only
        Name of the quantity of interest; default ``"f"``.

    Returns
    -------
    SubspaceResult
        ``estimator="matrix"``, signed eigenvalues ranked by :math:`|\lambda|`,
        eigenvectors with the default sign rule, ``n_samples=0`` and no
        bootstrap.

    Raises
    ------
    InputError
        ``input.shape``, ``input.dtype`` or ``input.nonfinite`` for a bad
        ``C``; ``input.parameter`` for a non-symmetric ``C``
        (``details["reason"] == "not_symmetric"``), both ``n`` and
        ``dimension_rule``, or a bad option; ``input.no_bootstrap`` for
        ``"rs"`` or ``"li"``.
    NumericalError
        ``numerical.singular`` if the eigensolver fails.

    Warns
    -----
    AccuracyWarning
        ``accuracy.degenerate_matrix`` if ``C`` is exactly zero (``W = I``).

    See Also
    --------
    active_subspace : Estimate the matrix from samples.
    subspace_from_basis : Wrap a basis without eigenvalues.

    Notes
    -----
    The eigenpairs are those of :math:`\tfrac12(C + C^\top)` by
    ``numpy.linalg.eigh``, ranked by :math:`|\lambda|` and reported signed,
    with the sign rule of :func:`active_subspace` [D-202].

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> C = np.diag([4.0, 1.0, 0.01])
    >>> sub = ac.subspace_from_matrix(C, space=ac.Uniform.reference(3))
    >>> sub.estimator, sub.n, sub.eigenvalues
    ('matrix', 1, array([4.  , 1.  , 0.01]))
    """
    space = _v.check_space(space)
    m = space.dim
    A = _v.matrix(C, "C", rows=m, cols=m)
    scale = float(np.max(np.abs(A)))
    if float(np.max(np.abs(A - A.T))) > _SQRT_EPS * scale:
        raise InputError(
            "C: not symmetric to a relative sqrt(eps)",
            code="input.parameter",
            details={"param": "C", "reason": "not_symmetric"},
        )
    if n is not None and dimension_rule is not None:
        raise InputError(
            "give at most one of n and dimension_rule",
            code="input.parameter",
            details={"param": "dimension_rule"},
        )
    qoi = "f" if qoi is None else qoi
    if not isinstance(qoi, str) or not qoi:
        raise InputError(
            "qoi: expected a non-empty string", code="input.parameter", details={"param": "qoi"}
        )
    rule = None
    if n is not None:
        n = _v.integer(n, "n", lo=1, hi=m)
    else:
        rule = (
            EVG()
            if dimension_rule is None
            else _resolve(dimension_rule, DIMENSION_RULES, "dimension_rule")
        )
        if rule.needs_bootstrap:
            raise _no_bootstrap(rule.name)
    notes: dict[str, Any] = {}
    lam, W = _from_matrix(A, "matrix", notes)
    _warn_accuracy("matrix", notes, [])
    if rule is not None:
        n, criterion, _ = _apply_rule(rule, lam, None, None)
    else:
        assert n is not None  # a fixed n, validated above
        criterion = None
    params: dict[str, Any] = {
        "n": None if rule is not None else n,
        "dimension_rule": "fixed" if rule is None else rule.name,
        "dimension_options": {} if rule is None else rule.options(),
        "qoi": qoi,
    }
    prov = _make_provenance(
        "subspace_from_matrix", params, space=space, digests={"C": _v.digest(A)}
    )
    return SubspaceResult(
        space=space,
        estimator="matrix",
        estimator_options=(),
        eigenvalues=lam,
        eigenvectors=W,
        n=int(n),
        dimension_rule="fixed" if rule is None else rule.name,
        dimension_options=() if rule is None else tuple(sorted(rule.options().items())),
        criterion=criterion,
        n_samples=0,
        nboot=0,
        bootstrap_eigenvalues=None,
        bootstrap_distances=None,
        bootstrap_dets=None,
        bootstrap_eigenvectors=None,
        qoi=qoi,
        provenance=prov,
    )


def activity_scores(subspace: SubspaceResult, n: int | None = None) -> NDArray[np.float64]:
    r"""Return the activity scores of the inputs, in normalized coordinates.

    Parameters
    ----------
    subspace : SubspaceResult
        A result with eigenvalues (not ``"basis"``).
    n : int, optional
        Number of eigenpairs, :math:`1 \le n \le m`; default ``subspace.n``.

    Returns
    -------
    numpy.ndarray, shape (m,), float64
        :math:`\alpha_i(n) = \sum_{j=1}^n |\lambda_j| \, w_{ij}^2`, a new array.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad ``n`` or result, and for a ``"basis"``
        result (``details["reason"] == "no_eigenvalues"``).

    See Also
    --------
    active_subspace : Estimate the eigenpairs.

    Notes
    -----
    Activity scores are global sensitivity metrics [1]_. For the positive
    semidefinite estimators :math:`|\lambda_j| = \lambda_j`; for the indefinite
    ones the magnitude matches their ranking [D-201]. The scores depend on
    the normalization: inputs scaled to :math:`[0, 1]` instead of
    :math:`[-1, 1]` multiply every eigenvalue by 4.

    References
    ----------
    .. [1] P. G. Constantine, P. Diaz. Global sensitivity metrics from active
       subspaces. *Reliab. Eng. Syst. Saf.* 162:1-13, 2017.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> sub = ac.subspace_from_matrix(np.diag([4.0, 1.0, 0.0]), space=ac.Uniform.reference(3))
    >>> ac.activity_scores(sub), ac.activity_scores(sub, n=2)
    (array([4., 0., 0.]), array([4., 1., 0.]))
    """
    sub = _check_result(subspace)
    if sub.estimator == "basis":
        raise InputError(
            "a 'basis' result has no eigenvalues, so no activity scores",
            code="input.parameter",
            details={"param": "subspace", "reason": "no_eigenvalues"},
        )
    k = sub.n if n is None else _v.integer(n, "n", lo=1, hi=sub.m)
    W = sub.eigenvectors[:, :k]
    scores: _F64 = (np.abs(sub.eigenvalues[:k]) * W * W).sum(axis=1)
    return scores


_ORIENT_ARGS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"reference": ("reference",), "covariance": ("X", "f"), "mean_gradient": ("grads",)}
)


def _criterion_signs(crit: _F64, scale: _F64) -> _F64:
    """+1 or -1 per column; a criterion within sqrt(eps) of its scale keeps the sign [D-197]."""
    s = np.where(crit < 0.0, -1.0, 1.0)
    s[np.abs(crit) <= _SQRT_EPS * scale] = 1.0
    return s


def orient(
    subspace: SubspaceResult,
    *,
    by: Literal["reference", "covariance", "mean_gradient"],
    reference: Any = None,
    X: Any = None,
    f: Any = None,
    grads: Any = None,
    columns: int | None = None,
) -> SubspaceResult:
    r"""Return a subspace result whose leading eigenvectors carry user-defined signs.

    Parameters
    ----------
    subspace : SubspaceResult
        The result to re-sign.
    by : {"reference", "covariance", "mean_gradient"}, keyword-only
        The criterion (see Notes).
    reference : array_like, shape (m, k), or SubspaceResult, optional, keyword-only
        For ``"reference"``: directions in normalized coordinates as
        columns, :math:`k \ge` ``columns``, or a result on an equal space
        (its ``W`` is used).
    X : array_like, shape (M, m), optional, keyword-only
        For ``"covariance"``: points in physical coordinates.
    f : array_like, shape (M,) or (M, 1), optional, keyword-only
        For ``"covariance"``: values at ``X``.
    grads : array_like, shape (M, m), or GradientResult, optional, keyword-only
        For ``"mean_gradient"``: gradients in physical coordinates.
    columns : int, optional, keyword-only
        Number :math:`c` of leading columns to re-sign,
        :math:`1 \le c \le m`; default ``subspace.n``.

    Returns
    -------
    SubspaceResult
        A new result in which only ``eigenvectors``, ``bootstrap_eigenvectors``
        (re-signed with the same signs, so they stay aligned) and the
        provenance differ.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad ``by`` or ``columns``, a missing
        argument, an argument the criterion does not use
        (``details["reason"] == "not_for_criterion"``), a reference with too
        few columns, or a reference result or ``GradientResult`` on another
        space (``details["reason"] == "space_mismatch"``); the spec 02 codes
        for bad arrays.

    See Also
    --------
    active_subspace : Its default sign rule.
    SubspaceResult.physical_directions : Follows the signs set here.

    Notes
    -----
    Column :math:`j \le c` is multiplied by :math:`s_j = \operatorname{sign}`
    of

    - ``"reference"``: :math:`w_j^\top r_j`;
    - ``"covariance"``: :math:`\sum_i \omega_i (y_{ij} - \bar y_j)(f_i - \bar f)`
      with :math:`y_i = W^\top x_i` for the normalized inputs and
      :math:`\omega_i = 1/M` (a result keeps only a digest of its quadrature
      weights);
    - ``"mean_gradient"``: :math:`\bar g^\top w_j` with :math:`\bar g` the
      mean normalized gradient.

    A criterion whose magnitude is at most :math:`\sqrt\epsilon` times its
    scale (:math:`\lVert r_j\rVert`; the product of the two standard
    deviations; :math:`\lVert \bar g\rVert`) keeps the current sign [D-197].
    Bootstrap distances and determinants are sign-invariant.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> sub = ac.subspace_from_matrix(np.diag([2.0, 1.0]), space=ac.Uniform.reference(2))
    >>> flipped = ac.orient(sub, by="reference", reference=[[-1.0], [0.0]])
    >>> flipped.W1.ravel() + 0.0, flipped.physical_directions().ravel() + 0.0
    (array([-1.,  0.]), array([-1.,  0.]))
    """
    sub = _check_result(subspace)
    crit_name = _v.choice(by, "by", ("reference", "covariance", "mean_gradient"))
    space = sub.space
    m = sub.m
    c = sub.n if columns is None else _v.integer(columns, "columns", lo=1, hi=m)
    used = _ORIENT_ARGS[crit_name]
    given = {"reference": reference, "X": X, "f": f, "grads": grads}
    for name, value in given.items():
        if value is not None and name not in used:
            raise InputError(
                f"{name}: not used by the {crit_name!r} criterion",
                code="input.parameter",
                details={"param": name, "reason": "not_for_criterion", "by": crit_name},
            )
    for name in used:
        if given[name] is None:
            raise InputError(
                f"{name}: required by the {crit_name!r} criterion",
                code="input.parameter",
                details={"param": name, "by": crit_name},
            )
    W = sub.eigenvectors
    digests: dict[str, Any] = {}
    if crit_name == "reference":
        if isinstance(reference, SubspaceResult):
            if reference.space != space:
                raise InputError(
                    "reference: a result on a different input space",
                    code="input.parameter",
                    details={"param": "reference", "reason": "space_mismatch"},
                )
            R = np.array(reference.eigenvectors, dtype=np.float64)
        else:
            R = _v.matrix(reference, "reference", rows=m)
        if R.shape[1] < c:
            raise InputError(
                f"reference: has {R.shape[1]} columns, fewer than columns = {c}",
                code="input.parameter",
                details={"param": "reference", "expected": c, "got": int(R.shape[1])},
            )
        digests["reference"] = _v.digest(R)
        crit = np.einsum("ij,ij->j", W[:, :c], R[:, :c])
        scale = np.linalg.norm(R[:, :c], axis=0)
    elif crit_name == "covariance":
        dd = _v.data(space, X=X, f=f, require=("X", "f"))
        assert dd.X is not None  # required above
        assert dd.f is not None
        digests.update(dd.digests)
        Y = dd.X @ W[:, :c]
        Yc = Y - Y.mean(axis=0)
        fc = dd.f - dd.f.mean()
        M = Y.shape[0]
        crit = (Yc * fc[:, None]).sum(axis=0) / M
        scale = np.sqrt((Yc * Yc).sum(axis=0) / M) * math.sqrt(float(fc @ fc) / M)
    else:
        if isinstance(grads, GradientResult):
            if grads.space != space:
                raise _space_mismatch()
            G = np.array(grads.grads, dtype=np.float64)
            digests["grads"] = _v.digest(grads.grads_physical)
        else:
            dd = _v.data(space, grads=grads, require=("grads",))
            assert dd.grads is not None  # required above
            G = dd.grads
            digests.update(dd.digests)
        gbar = G.mean(axis=0)
        crit = gbar @ W[:, :c]
        scale = np.full(c, float(np.linalg.norm(gbar)))
    s = _criterion_signs(np.asarray(crit, dtype=np.float64), np.asarray(scale, dtype=np.float64))
    Wn = np.array(W, dtype=np.float64, copy=True)
    Wn[:, :c] *= s
    V = sub.bootstrap_eigenvectors
    if V is not None:
        V = np.array(V, dtype=np.float64, copy=True)
        k = min(c, V.shape[2])
        V[:, :, :k] *= s[:k]
    src = sub.provenance.to_dict()
    src["input_space"] = None  # the space is recorded at the top level
    params: dict[str, Any] = {
        "by": crit_name,
        "columns": c,
        "signs": [int(v) for v in s],
        "source": src,
    }
    prov = _make_provenance("orient", params, space=space, digests=digests)
    return SubspaceResult(
        space=space,
        estimator=sub.estimator,
        estimator_options=sub.estimator_options,
        eigenvalues=sub.eigenvalues,
        eigenvectors=Wn,
        n=sub.n,
        dimension_rule=sub.dimension_rule,
        dimension_options=sub.dimension_options,
        criterion=sub.criterion,
        n_samples=sub.n_samples,
        nboot=sub.nboot,
        bootstrap_eigenvalues=sub.bootstrap_eigenvalues,
        bootstrap_distances=sub.bootstrap_distances,
        bootstrap_dets=sub.bootstrap_dets,
        bootstrap_eigenvectors=V,
        qoi=sub.qoi,
        provenance=prov,
    )
