"""Gradients from data: local linear fits and finite differences.

Both functions are deterministic and return a :class:`GradientResult`
holding normalized points and gradients, which
:func:`~active_subspaces.active_subspace` accepts as ``grads=``.
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

from active_subspaces import _validation as _v
from active_subspaces import models
from active_subspaces.errors import AccuracyWarning, InputError, NumericalError, _warn
from active_subspaces.results import (
    QoIResult,
    _invariant,
    _make_provenance,
    _register_result,
)
from active_subspaces.spaces import InputSpace, Uniform

__all__ = (
    "GradientResult",
    "finite_difference_gradients",
    "finite_difference_gradients_multi",
    "local_linear_gradients",
)

_toplevel = __all__

_log = logging.getLogger(__name__)

_METHODS = ("local_linear", "finite_difference")
#: Elements of the (centers x locations x m) difference block held at once.
_BLOCK_ELEMENTS = 1 << 22
_SCHEMES = ("forward", "central")


@_register_result
class GradientResult(QoIResult):
    r"""Estimated gradients at a set of points, stored in normalized coordinates.

    Produced by :func:`local_linear_gradients` and
    :func:`finite_difference_gradients`; the constructor is used by
    deserialization. Pass it to :func:`~active_subspaces.active_subspace` as
    ``grads=``.

    Parameters
    ----------
    space : Uniform, Gaussian or Product
        The input space; ``points`` and ``grads`` are normalized with it.
    method : {"local_linear", "finite_difference"}
        How the gradients were estimated.
    points : numpy.ndarray, shape (K, m), float64
        Normalized points where the gradients are estimated.
    grads : numpy.ndarray, shape (K, m), float64
        Normalized gradients :math:`\nabla_x f`, one per row.
    values : numpy.ndarray, shape (K,), float64, or None
        :math:`f` at the points.
    centers : numpy.ndarray, shape (K,), int64, or None
        Row indices into the ``X`` passed in (local linear only).
    p : int or None
        Neighbor-set size, in distinct locations (local linear only).
    scheme : {"forward", "central"} or None
        Difference scheme (finite differences only).
    steps : numpy.ndarray, shape (K, m), float64, or None
        Signed normalized steps :math:`s_{ij}` used (finite differences only).
    central_mask : numpy.ndarray, shape (K, m), bool, or None
        ``True`` where the central formula was used (finite differences only).
    n_evaluations : int
        Number of model evaluations; 0 for local linear.
    provenance : Provenance, keyword-only
        How the result was produced.
    qoi : str, keyword-only, default "f"
        Name of the quantity of interest.

    See Also
    --------
    local_linear_gradients : Gradients from data.
    finite_difference_gradients : Gradients from a model.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [2.0, 4.0])
    >>> model = ac.Model(space=space, value=lambda X: X[:, 0] ** 2 + X[:, 1])
    >>> res = ac.finite_difference_gradients(model, X=[[1.0, 1.0]], scheme="central")
    >>> res.grads.round(4)  # normalized: d f / d x = A^T grad_xi f
    array([[2., 2.]])
    >>> res.grads_physical.round(4)
    array([[2., 1.]])
    """

    TYPE = "GradientResult"
    SCHEMA_VERSION = 1

    space: InputSpace
    method: str
    points: NDArray[np.float64]
    grads: NDArray[np.float64]
    values: NDArray[np.float64] | None
    centers: NDArray[np.int64] | None
    p: int | None
    scheme: str | None
    steps: NDArray[np.float64] | None
    central_mask: NDArray[np.bool_] | None
    n_evaluations: int

    @property
    def m(self) -> int:
        """Number of inputs ``m``.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform.reference(2)
        >>> X = space.sample(10, rng=0)
        >>> ac.local_linear_gradients(X=X, f=X[:, 0], space=space).m
        2
        """
        return self.space.dim

    @property
    def points_physical(self) -> NDArray[np.float64]:
        """The points in physical coordinates, ``space.from_normalized(points)``, read-only.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [2.0, 4.0])
        >>> model = ac.Model(space=space, value=lambda X: X[:, 0] + X[:, 1])
        >>> ac.finite_difference_gradients(model, X=[[1.0, 3.0]]).points_physical
        array([[1., 3.]])
        """
        return _v.freeze_array(self.space.from_normalized(self.points))

    @property
    def grads_physical(self) -> NDArray[np.float64]:
        """The gradients in physical coordinates, ``space.grads_to_physical(grads)``, read-only.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [2.0, 4.0])
        >>> model = ac.Model(space=space, value=lambda X: 3.0 * X[:, 0] + X[:, 1])
        >>> ac.finite_difference_gradients(model, X=[[1.0, 3.0]]).grads_physical.round(4)
        array([[3., 1.]])
        """
        return _v.freeze_array(self.space.grads_to_physical(self.grads))

    def _check(self) -> None:
        m = self.space.dim
        _invariant(self.method in _METHODS, f"method must be one of {_METHODS}")
        _invariant(
            self.points.ndim == 2 and self.points.shape[1] == m and self.points.shape[0] >= 1,
            "points must be (K, m)",
        )
        K = self.points.shape[0]
        _invariant(self.grads.shape == (K, m), "grads must have the shape of points")
        _invariant(
            bool(np.isfinite(self.points).all() and np.isfinite(self.grads).all()),
            "points and grads must be finite",
        )
        _invariant(self.values is None or self.values.shape == (K,), "values must be (K,)")
        _invariant(self.n_evaluations >= 0, "n_evaluations must be >= 0")
        if self.method == "local_linear":
            _invariant(
                self.centers is not None and self.centers.shape == (K,),
                "local_linear needs centers (K,)",
            )
            assert self.centers is not None  # checked just above
            _invariant(bool((self.centers >= 0).all()), "centers must be non-negative")
            _invariant(self.p is not None and self.p >= m + 1, "local_linear needs p >= m + 1")
            _invariant(
                self.scheme is None and self.steps is None and self.central_mask is None,
                "scheme, steps and central_mask are for finite_difference only",
            )
            _invariant(self.n_evaluations == 0, "local_linear makes no model evaluations")
        else:
            _invariant(
                self.centers is None and self.p is None, "centers and p are for local_linear only"
            )
            _invariant(self.scheme in _SCHEMES, f"scheme must be one of {_SCHEMES}")
            _invariant(
                self.steps is not None and self.steps.shape == (K, m), "steps must be (K, m)"
            )
            _invariant(
                self.central_mask is not None and self.central_mask.shape == (K, m),
                "central_mask must be (K, m)",
            )
            assert self.steps is not None  # checked just above
            assert self.central_mask is not None
            _invariant(bool((self.steps != 0).all()), "steps must be nonzero")
            _invariant(
                self.scheme == "central" or not self.central_mask.any(),
                "central_mask must be all False for forward",
            )


# ---------------------------------------------------------------------------
# Local linear gradients (§3.8.1, §5.2)
# ---------------------------------------------------------------------------


def _check_finite_grads(grads: np.ndarray, function: str) -> None:
    """Raise ``NumericalError`` ``numerical.nonfinite`` for non-finite gradients (spec 06 §5)."""
    bad = ~np.isfinite(grads)
    if bad.any():
        i, j = (int(v) for v in np.argwhere(bad)[0])
        raise NumericalError(
            f"{function}: the gradient estimate overflowed to a non-finite value (row {i}, "
            f"coordinate {j}); rescale f",
            code="numerical.nonfinite",
            details={"what": "grads", "row": i, "coordinate": j, "count": int(bad.sum())},
        )


def _qoi(qoi: Any) -> str:
    if not isinstance(qoi, str) or not qoi:
        raise InputError(
            "qoi: expected a non-empty string",
            code="input.parameter",
            details={"param": "qoi", "got": repr(qoi)},
        )
    return qoi


def _centers(centers: Any, M: int) -> np.ndarray:
    """Validate center indices: 1-D integers in [0, M), K >= 1; duplicates and order kept."""
    if centers is None:
        return np.arange(M, dtype=np.int64)
    arr = np.asarray(centers)
    if arr.dtype.kind not in "iu" or arr.ndim != 1 or arr.shape[0] < 1:
        raise InputError(
            "centers: expected a non-empty 1-D array of integer row indices, got dtype "
            f"{arr.dtype} and shape {arr.shape}",
            code="input.parameter",
            details={"param": "centers", "got": [str(arr.dtype), list(arr.shape)]},
        )
    bad = (arr < 0) | (arr >= M)
    if bad.any():
        i = int(np.argmax(bad))
        raise InputError(
            f"centers: entry {i} = {int(arr[i])} is not a row index in [0, {M})",
            code="input.parameter",
            details={"param": "centers", "row": i, "expected": [0, M - 1], "got": int(arr[i])},
        )
    return arr.astype(np.int64)


def _distinct_locations(Xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the distinct rows of ``Xn`` in first-occurrence order and each row's location."""
    Z = Xn + 0.0  # -0.0 -> 0.0, so equal coordinates compare equal bytewise
    _, first, inverse = np.unique(Z, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.shape[0])
    return Z[first[order]], rank[inverse]


def _neighbor_sets(Xn: np.ndarray, centers: np.ndarray, p: int) -> list[np.ndarray]:
    """Rows of ``Xn`` at the ``p`` distinct locations nearest each center, in ascending order.

    See §3.8.1.

    Distances are Euclidean in normalized coordinates, computed by blocked brute
    force; ties are broken by the lower first row index of a location. The set is
    the first ``p`` of a stable argsort of the distances, found in linear time: every
    location closer than the ``p``-th smallest distance, then the lowest-indexed
    locations at exactly that distance.
    """
    loc, label = _distinct_locations(Xn)
    Md, m = loc.shape
    out: list[np.ndarray] = []
    block = max(1, _BLOCK_ELEMENTS // max(1, Md * m))
    for start in range(0, centers.shape[0], block):
        rows = centers[start : start + block]
        diff = loc[None, :, :] - Xn[rows][:, None, :]
        d2 = np.einsum("bjk,bjk->bj", diff, diff)
        if p >= Md or p <= 0:
            chosen = np.full(d2.shape, p > 0)
        else:
            kth = np.partition(d2, p - 1, axis=1)[:, p - 1 : p]
            below = d2 < kth
            tie = d2 == kth
            need = p - np.count_nonzero(below, axis=1, keepdims=True)
            chosen = below | (tie & (np.cumsum(tie, axis=1) <= need))
        out.extend(np.flatnonzero(member) for member in chosen[:, label])
    return out


def local_linear_gradients(
    *,
    X: ArrayLike,
    f: ArrayLike,
    space: InputSpace,
    p: int | None = None,
    centers: ArrayLike | None = None,
    qoi: str = "f",
) -> GradientResult:
    r"""Estimate gradients by local linear least squares on nearest neighbors.

    Parameters
    ----------
    X : array_like, shape (M, m), float, keyword-only
        Sample points in physical coordinates.
    f : array_like, shape (M,) or (M, 1), float, keyword-only
        Values of the quantity of interest at ``X``.
    space : Uniform, Gaussian or Product, keyword-only
        The input space of ``X``.
    p : int, optional, keyword-only
        Number of distinct locations per fit, :math:`m + 1 \le p \le M_d`,
        with :math:`M_d` the number of distinct rows of ``X``. Default
        :math:`\min(\max(m + 1, \lfloor 1.7 m \rfloor), M_d)`.
    centers : array_like of int, shape (K,), optional, keyword-only
        Row indices of ``X`` at which to estimate gradients (duplicates and
        order kept); default every row.
    qoi : str, default "f", keyword-only
        Name of the quantity of interest, recorded in the result.

    Returns
    -------
    GradientResult
        ``method="local_linear"``, with ``points``, ``grads`` (normalized) and
        ``values`` at the centers.

    Raises
    ------
    InputError
        ``input.*`` for malformed data, ``input.too_few_samples`` if there are
        fewer than ``m + 1`` distinct locations, ``input.parameter`` for a bad
        ``p``, ``centers`` or ``qoi``.

    Warns
    -----
    AccuracyWarning
        ``accuracy.rank_deficient`` once if any local system has rank below
        ``m + 1``; the minimum-norm solution is used.

    See Also
    --------
    finite_difference_gradients : Gradients from a model.
    active_subspaces.active_subspace : Accepts the result as ``grads``.

    Notes
    -----
    At each center :math:`x_i` (normalized coordinates), the gradient estimate
    is :math:`a` from the ordinary least-squares fit

    .. math::

        f_j \approx a_0 + a^\top (x_j - x_i)

    over every row :math:`j` at the ``p`` distinct sample locations nearest to
    :math:`x_i` in Euclidean distance, the center included. Ties in distance go
    to the location with the lower first row index; duplicate rows enter the
    fit with their multiplicity. The estimate is exact for linear :math:`f`
    [1]_, [2]_.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.
    .. [2] M. Hristache, A. Juditsky, J. Polzehl, V. Spokoiny. Structure
       adaptive approach for dimension reduction. *Ann. Statist.*
       29(6):1537-1566, 2001.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(2)
    >>> X = space.sample(30, rng=0)
    >>> res = ac.local_linear_gradients(X=X, f=2.0 * X[:, 0] - X[:, 1], space=space)
    >>> res.grads.shape, res.p, bool(np.allclose(res.grads, [2.0, -1.0]))
    ((30, 2), 3, True)
    """
    d = _v.data(space, X=X, f=f, require=("X", "f"))
    M, m = d.M, d.m
    Xn, fv = d.X, d.f
    assert Xn is not None  # required above
    assert fv is not None
    qoi = _qoi(qoi)
    Md = _distinct_locations(Xn)[0].shape[0]
    _v.min_samples(Md, m + 1, "local_linear_gradients (distinct locations)")
    if p is None:
        p_val = min(max(m + 1, int(np.floor(1.7 * m))), Md)
    else:
        p_val = _v.integer(p, "p", lo=m + 1, hi=Md)
    idx = _centers(centers, M)
    K = idx.shape[0]

    grads = np.empty((K, m))
    n_deficient = 0
    cond = np.finfo(np.float64).eps
    for k, (i, nbrs) in enumerate(zip(idx, _neighbor_sets(Xn, idx, p_val), strict=False)):
        A = np.empty((nbrs.shape[0], m + 1))
        A[:, 0] = 1.0
        A[:, 1:] = Xn[nbrs] - Xn[i]
        coef, _, rank, _ = scipy.linalg.lstsq(
            A, fv[nbrs], cond=cond * max(A.shape), lapack_driver="gelsd", check_finite=False
        )
        if rank < m + 1:
            n_deficient += 1
        grads[k] = coef[1:]
    _check_finite_grads(grads, "local_linear_gradients")
    if n_deficient:
        _warn(
            AccuracyWarning,
            f"local_linear_gradients: {n_deficient} of {K} local systems are rank-deficient; "
            "minimum-norm solutions used",
            code="accuracy.rank_deficient",
            details={"centers": n_deficient, "total": K},
        )

    params = {"p": p, "p_used": p_val, "centers": None if centers is None else idx, "qoi": qoi}
    prov = _make_provenance("local_linear_gradients", params, space=space, digests=d.digests)
    return GradientResult(
        space=space,
        method="local_linear",
        points=Xn[idx],
        grads=grads,
        values=fv[idx],
        centers=idx,
        p=p_val,
        scheme=None,
        steps=None,
        central_mask=None,
        n_evaluations=0,
        qoi=qoi,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Finite differences (§3.8.2, §5.2)
# ---------------------------------------------------------------------------


def _step(h: Any, m: int) -> np.ndarray:
    """Validate ``h``: a scalar or ``(m,)`` array with every entry in (0, 1/2]; return ``(m,)``."""
    if np.ndim(h) == 0:
        if isinstance(h, np.ndarray):
            h = h[()]  # a 0-d array is a scalar
        return np.full(m, _v.real(h, "h", lo=0.0, lo_open=True, hi=0.5))
    arr = _v.vector(h, "h", size=m)
    bad = ~((arr > 0.0) & (arr <= 0.5))
    if bad.any():
        j = int(np.argmax(bad))
        raise InputError(
            f"h: entry {j} = {arr[j]!r} is outside (0, 1/2]",
            code="input.parameter",
            details={"param": "h", "coordinate": j, "expected": [0.0, 0.5], "got": float(arr[j])},
        )
    return arr


class _Stencil:
    """The stencil of a finite-difference call: steps, stacked batch and the quotient arithmetic.

    Shared by :func:`finite_difference_gradients` and
    :func:`finite_difference_gradients_multi`, so both use the same steps, batch and arithmetic.
    """

    def __init__(
        self,
        mdl: models.Model,
        scheme: str,
        h_param: float | list[float],
        digests: Any,
        Xn: np.ndarray,
        S: np.ndarray,
        H: np.ndarray,
        central: np.ndarray,
        batch: np.ndarray,
    ) -> None:
        self.mdl = mdl
        self.sp = mdl.space
        self.scheme = scheme
        self.h_param = h_param
        self.digests = digests
        self.Xn = Xn
        self.S = S
        self.H = H
        self.central = central
        self.batch = batch
        self.n_eval = int(batch.shape[0])

    @classmethod
    def build(
        cls, model: Any, X: ArrayLike, space: InputSpace | None, h: Any, scheme: Any
    ) -> _Stencil:
        mdl = models.as_model(model, space=space)
        sp = mdl.space
        scheme = cast(Literal["forward", "central"], _v.choice(scheme, "scheme", _SCHEMES))
        hv = _step(h, sp.dim)
        d = _v.data(sp, X=X, require=("X",))
        M, m = d.M, d.m
        Xn = d.X
        assert Xn is not None  # required above
        Xphys = _v.points(X, sp, normalize=False)

        H = hv[None, :] * np.maximum(1.0, np.abs(Xn))
        # Uniform coordinates (a Uniform space, or the uniform blocks of a Product) have a
        # boundary; Gaussian ones do not [D-198]. For a Uniform or Gaussian space the masks are
        # all True or all False, so the steps are those of the homogeneous rule, bitwise.
        u: NDArray[np.bool_] = np.array(cast(Any, sp)._uniform_mask, dtype=np.bool_)
        S = np.where((Xn + H > 1.0) & u, -H, H)
        if scheme == "central":
            central = (np.abs(Xn) + H <= 1.0) | np.logical_not(u)
        else:
            central = np.zeros((M, m), dtype=bool)
        S = np.where(central, H, S)

        # Stacked batch: base points; x_i + s_ij e_j for j = 1..m; x_i - h_ij e_j where central,
        # for j = 1..m.
        plus = np.repeat(Xn[None, :, :], m, axis=0)  # (m, M, m)
        jj = np.arange(m)
        plus[jj, :, jj] += S.T
        minus_parts = []
        for j in range(m):
            rows = np.flatnonzero(central[:, j])
            if rows.size:
                pts = Xn[rows].copy()
                pts[:, j] -= H[rows, j]
                minus_parts.append(pts)
        pert = np.concatenate([plus.reshape(m * M, m), *minus_parts], axis=0)
        pert_phys = sp.from_normalized(pert)
        if u.any():  # remove rounding outside the bounds of the uniform coordinates
            lo, hi = (sp.lower, sp.upper) if isinstance(sp, Uniform) else cast(Any, sp)._bounds()
            np.clip(pert_phys, lo, hi, out=pert_phys)
        batch = np.concatenate([Xphys, pert_phys], axis=0)
        h_param = float(hv[0]) if np.ndim(h) == 0 else hv.tolist()
        return cls(mdl, scheme, h_param, d.digests, Xn, S, H, central, batch)

    def quotients(self, vals: np.ndarray, function: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(grads, f0)`` from the ``(L,)`` values at the rows of ``batch``."""
        M, m = self.Xn.shape
        central = self.central
        f0 = vals[:M]
        fplus = vals[M : M + m * M].reshape(m, M).T  # (M, m)
        fminus = np.zeros((M, m))  # zeros where unused: np.where evaluates both branches
        pos = M + m * M
        for j in range(m):
            rows = np.flatnonzero(central[:, j])
            fminus[rows, j] = vals[pos : pos + rows.size]
            pos += rows.size

        with np.errstate(over="ignore", invalid="ignore"):
            grads = np.where(
                central, (fplus - fminus) / (2.0 * self.H), (fplus - f0[:, None]) / self.S
            )
        _check_finite_grads(grads, function)
        if self.scheme == "central" and _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "%s: %d of %d entries fell back to one-sided differences",
                function,
                int((~central).sum()),
                M * m,
            )
        return grads, f0

    def result(
        self, grads: np.ndarray, f0: np.ndarray, *, mdl_qoi: str, provenance: Any
    ) -> GradientResult:
        return GradientResult(
            space=self.sp,
            method="finite_difference",
            points=self.Xn,
            grads=grads,
            values=f0,
            centers=None,
            p=None,
            scheme=self.scheme,
            steps=self.S,
            central_mask=self.central,
            n_evaluations=self.n_eval,
            qoi=mdl_qoi,
            provenance=provenance,
        )


def finite_difference_gradients(
    model: Any,
    *,
    X: ArrayLike,
    space: InputSpace | None = None,
    h: float | ArrayLike = 1e-6,
    scheme: Literal["forward", "central"] = "forward",
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
) -> GradientResult:
    r"""Estimate gradients by finite differences along the normalized axes.

    Parameters
    ----------
    model : Model or callable
        A :class:`~active_subspaces.Model`, or a batch value callable mapping
        physical points ``(k, m)`` to ``(k,)`` (then ``space`` is required).
    X : array_like, shape (M, m), float, keyword-only
        Points in physical coordinates, validated against the model's space.
    space : Uniform, Gaussian or Product, optional, keyword-only
        Required for a bare callable; for a ``Model`` it must equal
        ``model.space``.
    h : float or array_like, shape (m,), default 1e-6, keyword-only
        Relative step in normalized units, :math:`0 < h_j \le 1/2`.
    scheme : {"forward", "central"}, default "forward", keyword-only
        Difference scheme.
    executor : concurrent.futures.Executor, optional, keyword-only
        Runs the model batches; never changes the result.
    batch_size : int, optional, keyword-only
        Rows per model call, as for :func:`~active_subspaces.evaluate`.

    Returns
    -------
    GradientResult
        ``method="finite_difference"``, with the model values at ``X``, the signed
        ``steps``, ``central_mask`` and ``n_evaluations``.

    Raises
    ------
    InputError
        ``input.*`` for malformed ``X``; ``input.parameter`` for a bad ``h``,
        ``scheme``, ``executor``, ``batch_size``, a bare callable without
        ``space``, or a ``space`` that differs from ``model.space``
        (``details["reason"] == "space_mismatch"``).
    ModelError
        ``model.*`` if the model raises or returns non-finite or misshaped values.
    NumericalError
        ``numerical.nonfinite`` if a difference quotient overflows.

    See Also
    --------
    local_linear_gradients : Gradients from data.
    active_subspaces.evaluate_grads : Exact gradients from a model that has them.

    Notes
    -----
    In normalized coordinates, the step for row :math:`i` and coordinate
    :math:`j` is :math:`h_{ij} = h_j \max(1, |x_{ij}|)`, and

    .. math::

        \partial_j f(x_i) \approx \frac{f(x_i + s_{ij} e_j) - f(x_i)}{s_{ij}}
        \quad\text{or}\quad
        \frac{f(x_i + h_{ij} e_j) - f(x_i - h_{ij} e_j)}{2 h_{ij}}

    for the forward and central schemes, with :math:`s_{ij} = h_{ij}`. On a
    ``Uniform`` space, and on the uniform coordinates of a ``Product`` [D-198],
    a forward step that would leave :math:`[-1, 1]` is taken backward
    (:math:`s_{ij} = -h_{ij}`), and a central entry whose points would leave it
    falls back to that one-sided rule, so the model is never evaluated outside
    the support. Gaussian coordinates have no boundary and always take the
    forward or central step. The model is called on all :math:`M (m + 1)` points
    (plus the central ones) in one evaluation.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [2.0, 4.0])
    >>> model = ac.Model(space=space, value=lambda X: X[:, 0] ** 2 + X[:, 1])
    >>> res = ac.finite_difference_gradients(model, X=[[1.0, 1.0], [2.0, 3.0]], scheme="central")
    >>> res.grads_physical.round(4)
    array([[2., 1.],
           [4., 1.]])
    >>> res.central_mask  # x1 = 2 lies on the upper bound: one-sided there
    array([[ True,  True],
           [False,  True]])
    """
    st = _Stencil.build(model, X, space, h, scheme)
    vals = models.evaluate(st.mdl, X=st.batch, executor=executor, batch_size=batch_size)
    vals = np.asarray(vals, dtype=np.float64).reshape(-1)
    grads, f0 = st.quotients(vals, "finite_difference_gradients")

    params = {
        "model": st.mdl.name,
        "batch_size": batch_size,
        "h": st.h_param,
        "scheme": st.scheme,
        "qoi": st.mdl.qoi,
    }
    prov = _make_provenance("finite_difference_gradients", params, space=st.sp, digests=st.digests)
    return st.result(grads, f0, mdl_qoi=st.mdl.qoi, provenance=prov)


def finite_difference_gradients_multi(
    model: Any,
    *,
    X: ArrayLike,
    space: InputSpace | None = None,
    outputs: int,
    h: float | ArrayLike = 1e-6,
    scheme: Literal["forward", "central"] = "forward",
    qoi_names: Sequence[str] | None = None,
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
) -> tuple[GradientResult, ...]:
    r"""Estimate the gradients of every output of a model that returns several values.

    Each stencil point is evaluated once, whatever the number of outputs.

    Parameters
    ----------
    model : Model or callable
        A :class:`~active_subspaces.Model`, or a batch value callable mapping
        physical points ``(k, m)`` to ``(k, outputs)`` (then ``space`` is
        required). The model's own ``qoi`` is not used.
    X : array_like, shape (M, m), float, keyword-only
        Points in physical coordinates, validated against the model's space.
    space : Uniform, Gaussian or Product, optional, keyword-only
        As for :func:`finite_difference_gradients`.
    outputs : int, keyword-only
        Number of value columns :math:`P \ge 1`.
    h : float or array_like, shape (m,), default 1e-6, keyword-only
        Relative step in normalized units, :math:`0 < h_j \le 1/2`.
    scheme : {"forward", "central"}, default "forward", keyword-only
        Difference scheme.
    qoi_names : sequence of str, optional, keyword-only
        One unique, non-empty name per output; default ``("f0", "f1", ...)``.
    executor : concurrent.futures.Executor, optional, keyword-only
        Runs the model batches; never changes the result.
    batch_size : int, optional, keyword-only
        Rows per model call, as for :func:`~active_subspaces.evaluate`.

    Returns
    -------
    tuple of GradientResult
        One result per output, in output order. They share ``steps`` and
        ``central_mask`` and carry their own ``qoi``. Result :math:`p` equals,
        bitwise, :func:`finite_difference_gradients` on a model returning column
        :math:`p`; ``n_evaluations`` is the number of rows evaluated once for all
        outputs.

    Raises
    ------
    InputError
        As for :func:`finite_difference_gradients`; ``input.parameter`` for an
        ``outputs`` that is not an integer >= 1 or a ``qoi_names`` of the wrong
        length, with a duplicate or with an empty or non-string entry.
    ModelError
        ``model.*`` as for :func:`finite_difference_gradients`; ``model.shape``
        unless the model returns exactly ``(k, outputs)``.
    NumericalError
        ``numerical.nonfinite`` if a difference quotient overflows.

    See Also
    --------
    finite_difference_gradients : One output.
    active_subspaces.evaluate_multi : Evaluate a model with several outputs.

    Notes
    -----
    The steps, the stacked batch and the arithmetic are those of
    :func:`finite_difference_gradients`; see its Notes.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [2.0, 4.0])
    >>> def model(X):
    ...     return np.stack([X[:, 0] ** 2 + X[:, 1], 3.0 * X[:, 1]], axis=1)
    >>> res = ac.finite_difference_gradients_multi(
    ...     model, X=[[1.0, 1.0]], space=space, outputs=2, scheme="central"
    ... )
    >>> [r.qoi for r in res], res[0].n_evaluations == res[1].n_evaluations
    (['f0', 'f1'], True)
    >>> res[0].grads_physical.round(4), res[1].grads_physical.round(4)
    (array([[2., 1.]]), array([[0., 3.]]))
    """
    P = models._outputs(outputs)
    names = _qoi_names(qoi_names, P)
    st = _Stencil.build(model, X, space, h, scheme)
    vals = models.evaluate_multi(
        st.mdl, X=st.batch, outputs=P, executor=executor, batch_size=batch_size
    )
    out = []
    for p in range(P):
        try:
            grads, f0 = st.quotients(
                np.ascontiguousarray(vals[:, p]), "finite_difference_gradients_multi"
            )
        except NumericalError as exc:
            exc.details["output"] = p
            raise
        params = {
            "model": st.mdl.name,
            "batch_size": batch_size,
            "h": st.h_param,
            "scheme": st.scheme,
            "qoi": names[p],
            "outputs": P,
            "output": p,
            "qoi_names": list(names),
        }
        prov = _make_provenance(
            "finite_difference_gradients_multi", params, space=st.sp, digests=st.digests
        )
        out.append(st.result(grads, f0, mdl_qoi=names[p], provenance=prov))
    return tuple(out)


def _qoi_names(qoi_names: Sequence[str] | None, P: int) -> tuple[str, ...]:
    """Validate ``qoi_names``: ``P`` unique non-empty strings; default ``f0, f1, ...``."""
    if qoi_names is None:
        return tuple(f"f{p}" for p in range(P))
    if isinstance(qoi_names, str) or not isinstance(qoi_names, Sequence):
        raise InputError(
            f"qoi_names: expected a sequence of strings, got {type(qoi_names).__name__}",
            code="input.parameter",
            details={"param": "qoi_names", "got": type(qoi_names).__name__},
        )
    names = tuple(qoi_names)
    if len(names) != P:
        raise InputError(
            f"qoi_names: expected {P} names, got {len(names)}",
            code="input.parameter",
            details={"param": "qoi_names", "expected": P, "got": len(names)},
        )
    for i, q in enumerate(names):
        if not isinstance(q, str) or not q:
            raise InputError(
                f"qoi_names: entry {i} is not a non-empty string",
                code="input.parameter",
                details={"param": "qoi_names", "index": i, "got": repr(q)},
            )
    if len(set(names)) != len(names):
        raise InputError(
            "qoi_names: names must be unique",
            code="input.parameter",
            details={"param": "qoi_names", "reason": "duplicate"},
        )
    return names
