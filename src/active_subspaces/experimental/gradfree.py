r"""Active subspaces from gradient sketches: :math:`k < m` directional derivatives per sample.

Instead of full gradients, each sample :math:`x_i` gives :math:`k`
directional derivatives :math:`d_i = E_i^\top \nabla f(x_i)` along the
columns of a sketch matrix :math:`E_i \in \mathbb{R}^{m \times k}`: with
forward differences, :math:`k + 1` model evaluations per point instead of
:math:`m + 1`. Two published estimators of the sketched gradient matrix
:math:`G \in \mathbb{R}^{m \times M}` (normalized coordinates) are offered:
the projection estimator of Qi and Hughes [1]_ and the alternating
minimization estimator of Constantine, Eftekhari and Wakin [2]_. The result
holds the thin SVD :math:`G = W \operatorname{diag}(M s) V^\top`.

:math:`s = \sigma / M` is **not** an eigenvalue estimate. For the Gaussian
and orthogonal designs, :math:`\sigma_j^2 / M` estimates
:math:`\alpha \lambda_j + \beta \operatorname{tr}(C)` with

.. math::

    \alpha = \frac{k (k m + m - 2)}{m (m - 1)(m + 2)}, \qquad
    \beta = \frac{k (m - k)}{m (m - 1)(m + 2)}.

A de-biased estimate would be
:math:`\hat\lambda_j = (\sigma_j^2 / M - \beta\, \widehat{\operatorname{tr}}) / \alpha`
with :math:`\widehat{\operatorname{tr}} = (m / k) \sum_j \sigma_j^2 / M`; it
is not applied. Do not compare :math:`s` with active-subspace eigenvalues.

The alternating estimator stacks an :math:`(M k) \times (m r)` least-squares
matrix, :math:`8 M k m r` bytes (40 MB for :math:`M = 1000`, :math:`k = 10`,
:math:`m = 100`, :math:`r = 5`).

This module is experimental: its API and results may change in any release.

References
----------
.. [1] H. Qi, S. M. Hughes. Invariance of principal components under
   low-dimensional random projection of the data. *Proc. IEEE ICIP*, 2012
   (details to verify).
.. [2] P. G. Constantine, A. Eftekhari, M. B. Wakin. Computing active
   subspaces efficiently with gradient sketching. *Proc. IEEE CAMSAP*, 2015
   (details to verify).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent.futures import Executor
from typing import Any, Literal, cast

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

from active_subspaces import _random
from active_subspaces import _validation as _v
from active_subspaces.errors import (
    ConvergenceWarning,
    InputError,
    NumericalError,
    _warn,
)
from active_subspaces.models import Model, ValueFn, as_model, evaluate
from active_subspaces.results import (
    QoIResult,
    _invariant,
    _make_provenance,
    _register_result,
)
from active_subspaces.spaces import InputSpace, _sign_pivots

__all__ = ("SketchedSubspace", "sketched_active_subspace", "subspace_from_sketches")

_LOG = logging.getLogger(__name__)

_METHODS = ("projection", "alternating")
_DESIGNS = ("gaussian", "bernoulli", "orthogonal")
_ORTHO_TOL = 1e-10


@_register_result
class SketchedSubspace(QoIResult):
    r"""Thin SVD of a sketched gradient matrix; unpacks as ``W, V, s = res``.

    Returned by :func:`sketched_active_subspace` and
    :func:`subspace_from_sketches`; the constructor is used by
    deserialization. This is not a ``SubspaceResult``: geometry, surrogate
    and reduced-model functions do not accept it. Physical directions are
    ``res.space.directions_to_physical(res.W[:, :n])``.

    Parameters
    ----------
    W : ndarray, shape (m, p)
        Left singular vectors of :math:`G`, normalized coordinates,
        :math:`p = \min(m, M)`, orthonormal columns. Each column's
        largest-magnitude entry (the first within a relative sqrt(eps)) is
        positive.
    V : ndarray, shape (M, p)
        Right singular vectors, with the same column sign flips as `W`, so
        :math:`G = W \operatorname{diag}(M s) V^\top`.
    s : ndarray, shape (p,)
        Singular values of :math:`G` divided by :math:`M`, nonincreasing,
        nonnegative. **Not** eigenvalue estimates (see Notes).
    method : {"projection", "alternating"}
        The estimator.
    design : {"gaussian", "bernoulli", "orthogonal", "user"}
        The sketch design; ``"user"`` for :func:`subspace_from_sketches`.
    k : int
        Measurements per sample, :math:`1 \le k \le m`.
    n_samples : int
        Number of samples :math:`M`.
    rank : int or None
        Rank :math:`r` of the alternating method; ``None`` for projection.
    h : float or None
        Finite-difference step in normalized coordinates; ``None`` for
        :func:`subspace_from_sketches`.
    n_evaluations : int
        Model evaluations, :math:`M (k + 1)`, or 0 for
        :func:`subspace_from_sketches`.
    residual : float or None
        Final relative residual of the alternating method; ``None`` for
        projection.
    iterations : int or None
        Iterations of the alternating method; ``None`` for projection.
    stop_reason : {"tolerance", "stalled", "max_iter"} or None
        Why the alternating method stopped; ``None`` for projection.
    space : InputSpace
        The input space.
    provenance : Provenance
        How the result was computed (keyword-only).
    qoi : str, default "f"
        Name of the quantity of interest (keyword-only).

    See Also
    --------
    sketched_active_subspace : Estimate from a model.
    subspace_from_sketches : Estimate from given directional derivatives.

    Notes
    -----
    :math:`\tfrac1M G G^\top` estimates :math:`\mathbb{E}[P C P]`, with
    :math:`P` the orthogonal projector onto the range of a sketch matrix and
    :math:`C = \int \nabla f \nabla f^\top \rho \, dx`. For the Gaussian and
    orthogonal designs
    :math:`\mathbb{E}[P C P] = \alpha C + \beta \operatorname{tr}(C) I`
    with :math:`\alpha`, :math:`\beta` as in the module description, so the
    eigenvectors of :math:`C` are preserved in order, but
    :math:`s_j = \sigma_j / M` is not on the scale of :math:`\lambda_j`.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", ac.ExperimentalWarning)
    ...     from active_subspaces import experimental
    >>> model = ac.test_functions.ridge([1.0, 2.0, 0.0, 0.0])
    >>> res = experimental.sketched_active_subspace(model, k=2, M=50, rng=0)
    >>> W, V, s = res
    >>> W.shape, V.shape, s.shape
    ((4, 4), (50, 4), (4,))
    >>> res.n_evaluations
    150
    """

    TYPE = "SketchedSubspace"
    SCHEMA_VERSION = 1
    W: NDArray[np.float64]
    V: NDArray[np.float64]
    s: NDArray[np.float64]
    method: Literal["projection", "alternating"]
    design: Literal["gaussian", "bernoulli", "orthogonal", "user"]
    k: int
    n_samples: int
    rank: int | None
    h: float | None
    n_evaluations: int
    residual: float | None
    iterations: int | None
    stop_reason: Literal["tolerance", "stalled", "max_iter"] | None
    space: InputSpace

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        """Unpack as ``(W, V, s)``."""
        return iter((self.W, self.V, self.s))

    def _check(self) -> None:
        m, M = self.space.dim, self.n_samples
        p = min(m, M)
        _invariant(M >= 1 and 1 <= self.k <= m, "need n_samples >= 1 and 1 <= k <= m")
        _invariant(
            self.W.shape == (m, p) and self.V.shape == (M, p) and self.s.shape == (p,),
            "W, V, s must be (m, p), (M, p), (p,)",
        )
        _invariant(
            bool(np.all(np.isfinite(self.s))) and bool(np.all(self.s >= 0)),
            "s must be finite and >= 0",
        )
        _invariant(bool(np.all(np.diff(self.s) <= 0)), "s must be nonincreasing")
        _invariant(
            np.max(np.abs(self.W.T @ self.W - np.eye(p))) <= _ORTHO_TOL,
            "W must have orthonormal columns",
        )
        _invariant(
            np.max(np.abs(self.V.T @ self.V - np.eye(p))) <= _ORTHO_TOL,
            "V must have orthonormal columns",
        )
        als = (self.rank, self.residual, self.iterations, self.stop_reason)
        if self.method == "alternating":
            _invariant(
                all(v is not None for v in als),
                "alternating needs rank, residual, iterations, stop_reason",
            )
            rank, iterations, residual = self.rank, self.iterations, self.residual
            _invariant(
                rank is not None
                and iterations is not None
                and residual is not None
                and 1 <= rank <= p
                and iterations >= 1
                and residual >= 0,
                "bad ALS fields",
            )
        else:
            _invariant(
                all(v is None for v in als),
                "projection has no rank, residual, iterations, stop_reason",
            )
        if self.design == "user":
            _invariant(
                self.h is None and self.n_evaluations == 0,
                "user sketches have h None and no evaluations",
            )
        else:
            _invariant(self.h is not None and self.h > 0, "h must be > 0")
            _invariant(
                self.n_evaluations == M * (self.k + 1), "n_evaluations must equal M (k + 1)"
            )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _lstsq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        x: np.ndarray = np.linalg.lstsq(A, b, rcond=None)[0]
        return x
    except np.linalg.LinAlgError as exc:
        raise NumericalError(
            f"least-squares solve failed ({exc})",
            code="numerical.singular",
            details={"step": "lstsq"},
        ) from exc


def _projection(D: np.ndarray, E: np.ndarray) -> np.ndarray:
    """G (m, M): g_i = (E_i^T)^+ d_i, the minimum-norm least-squares solution."""
    M, m = E.shape[0], E.shape[1]
    G = np.empty((m, M))
    for i in range(M):
        G[:, i] = _lstsq(E[i].T, D[i])
    return G


def _svd(G: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD with the spec 07 sign rule on W and the same flips on V [C-24, D-81]."""
    if not np.all(np.isfinite(G)):
        raise NumericalError(
            "the sketched gradient matrix is not finite",
            code="numerical.nonfinite",
            details={"step": "G"},
        )
    try:
        W, sig, Vt = np.linalg.svd(G, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise NumericalError(
            f"SVD failed ({exc})", code="numerical.singular", details={"step": "svd"}
        ) from exc
    idx = _sign_pivots(W)
    signs = np.sign(W[idx, np.arange(W.shape[1])])
    signs[signs == 0] = 1.0
    return W * signs, sig, Vt.T * signs


def _alternating(
    D: np.ndarray, E: np.ndarray, r: int, tol: float, max_iter: int, stall_tol: float
) -> tuple[np.ndarray, float, int, str]:
    """Alternating minimization of sum_i ||d_i - E_i^T A b_i||^2.

    Returns (G, residual, iterations, stop_reason).
    """
    M, m, k = E.shape
    _, _, V0 = _svd(_projection(D, E))
    B = V0[:, :r].copy()
    Et = E.transpose(0, 2, 1)  # (M, k, m): E_i^T
    rhs = D.reshape(-1)  # vec(D^T), column-major
    normD = float(np.linalg.norm(D))
    rhos: list[float] = []
    ell = 0
    while True:
        ell += 1
        T = (B[:, None, :, None] * Et[:, :, None, :]).reshape(
            M * k, r * m
        )  # rows b_i^T kron E_i^T
        A = _lstsq(T, rhs).reshape(r, m).T  # column-major unvec to (m, r)
        EA = Et @ A  # (M, k, r)
        for i in range(M):
            B[i] = _lstsq(EA[i], D[i])
        R = D - np.einsum("ikr,ir->ik", EA, B)
        rho = 0.0 if normD == 0.0 else float(np.linalg.norm(R)) / normD
        rhos.append(rho)
        _LOG.debug("sketched_active_subspace ALS iteration %d: residual %r", ell, rho)
        if rho < tol:
            stop = "tolerance"
        elif ell >= 5 and float(np.std(rhos[-5:])) < stall_tol:
            stop = "stalled"
        elif ell == max_iter:
            stop = "max_iter"
        else:
            continue
        break
    if stop == "max_iter":
        _warn(
            ConvergenceWarning,
            f"alternating minimization reached max_iter={max_iter} with relative residual {rho!r}",
            code="convergence.max_iter",
            details={"residual": rho, "iterations": ell},
        )
    return A @ B.T, rho, ell, stop


def _fit(
    D: np.ndarray,
    E: np.ndarray,
    method: str,
    rank: int | None,
    tol: float,
    max_iter: int,
    stall_tol: float,
) -> dict[str, Any]:
    M = D.shape[0]
    if method == "projection":
        G = _projection(D, E)
        als: dict[str, Any] = {
            "rank": None,
            "residual": None,
            "iterations": None,
            "stop_reason": None,
        }
    else:
        G, rho, it, stop = _alternating(D, E, rank, tol, max_iter, stall_tol)  # type: ignore[arg-type]
        als = {"rank": rank, "residual": rho, "iterations": it, "stop_reason": stop}
    W, sig, V = _svd(G)
    return {"W": W, "V": V, "s": sig / M, **als}


def _check_method(
    method: Any, rank: Any, tol: Any, max_iter: Any, stall_tol: Any, m: int, M: int
) -> tuple:
    method = _v.choice(method, "method", _METHODS)
    if method == "alternating":
        if rank is None:
            raise InputError(
                "rank: required for method='alternating'",
                code="input.parameter",
                details={"param": "rank", "reason": "missing"},
            )
        rank = _v.integer(rank, "rank", lo=1, hi=min(m, M))
    elif rank is not None:
        raise InputError(
            "rank: only allowed with method='alternating'",
            code="input.parameter",
            details={"param": "rank", "reason": "forbidden"},
        )
    tol = _v.real(tol, "tol", lo=0.0, lo_open=True)
    max_iter = _v.integer(max_iter, "max_iter", lo=1)
    stall_tol = _v.real(stall_tol, "stall_tol", lo=0.0, lo_open=True)
    return method, rank, tol, max_iter, stall_tol


def _check_qoi(qoi: Any) -> str:
    if not isinstance(qoi, str) or not qoi:
        raise InputError(
            "qoi: expected a non-empty string", code="input.parameter", details={"param": "qoi"}
        )
    return qoi


def _draw_design(g: np.random.Generator, design: str, M: int, m: int, k: int) -> np.ndarray:
    if design == "gaussian":
        return g.standard_normal((M, m, k))
    if design == "bernoulli":
        return 2.0 * g.integers(0, 2, size=(M, m, k)) - 1.0
    Z = g.standard_normal((M, m, k))
    Q, R = np.linalg.qr(Z, mode="reduced")
    d = np.sign(np.diagonal(R, axis1=1, axis2=2))
    d[d == 0] = 1.0
    return Q * d[:, None, :]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def sketched_active_subspace(
    model: Model | ValueFn,
    /,
    *,
    space: InputSpace | None = None,
    k: int,
    X: ArrayLike | None = None,
    M: int | None = None,
    method: Literal["projection", "alternating"] = "projection",
    rank: int | None = None,
    design: Literal["gaussian", "bernoulli", "orthogonal"] = "gaussian",
    h: float = 1e-6,
    tol: float = 1e-3,
    max_iter: int = 100,
    stall_tol: float = 1e-5,
    executor: Executor | None = None,
    batch_size: int | None = None,
    rng: _random.RngLike = None,
    qoi: str | None = None,
) -> SketchedSubspace:
    r"""Estimate an active subspace from :math:`k` directional derivatives per sample.

    The directional derivatives are forward differences in normalized
    coordinates along the columns of random sketch matrices, so the model is
    evaluated :math:`M (k + 1)` times instead of :math:`M (m + 1)`.

    Parameters
    ----------
    model : Model or callable
        A ``Model``, or a batch value callable mapping physical points
        ``(K, m)`` to values ``(K,)`` (then `space` is required).
    space : InputSpace, optional
        The input space; default ``model.space``. With a ``Model`` it must
        equal ``model.space``.
    k : int
        Measurements per sample, :math:`1 \le k \le m`.
    X : array_like, shape (M, m), optional
        Sample points in **physical** coordinates, inside the support of
        `space`. Give exactly one of `X` and `M`.
    M : int, optional
        Number of sample points to draw from `space` with `rng`, at least 1.
    method : {"projection", "alternating"}, default "projection"
        Projection estimator [1]_, or alternating minimization [2]_ of rank
        `rank`.
    rank : int, optional
        Rank :math:`r`, :math:`1 \le r \le \min(m, M)`; required for
        ``"alternating"`` and not allowed for ``"projection"``.
    design : {"gaussian", "bernoulli", "orthogonal"}, default "gaussian"
        Distribution of the sketch matrices :math:`E_i` (m, k): standard
        normal entries, random signs, or Haar-distributed orthonormal columns.
    h : float, default 1e-6
        Forward-difference step in **normalized** coordinates, > 0. For a
        uniform space, a point that would leave :math:`[-1, 1]^m` uses the
        backward difference instead.
    tol : float, default 1e-3
        ``"alternating"``: stop when the relative residual is below `tol`.
    max_iter : int, default 100
        ``"alternating"``: maximum number of iterations, at least 1.
    stall_tol : float, default 1e-5
        ``"alternating"``: stop when the standard deviation of the last five
        residuals is below `stall_tol`.
    executor : concurrent.futures.Executor, optional
        Executor for the model evaluations; default serial.
    batch_size : int, optional
        Rows per model call; default the model's own.
    rng : int, numpy.random.SeedSequence or numpy.random.Generator, optional
        Randomness. Child stream 0 draws `X` when `M` is given; child stream
        1 draws the sketch matrices.
    qoi : str, optional
        Name of the quantity of interest; default ``model.qoi``.

    Returns
    -------
    SketchedSubspace
        ``W, V, s = res``; `W` (m, p) in normalized coordinates and
        :math:`s = \sigma / M`, which is not an eigenvalue estimate.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad `k`, `M`, `rank`, `method`, `design`,
        `h`, `tol`, `max_iter`, `stall_tol` or `qoi`, if both or neither of
        `X` and `M` are given, or if `space` is missing or differs from
        ``model.space``; ``input.shape``, ``input.dtype``, ``input.nonfinite``
        or ``input.out_of_space`` for a bad `X`; ``input.out_of_space`` if
        both the forward and the backward difference point of a sample leave
        the support.
    ModelError
        If the model raises or returns values of the wrong shape, dtype or
        finiteness (``model.*`` codes).
    NumericalError
        ``numerical.singular`` if a least-squares solve or the SVD fails;
        ``numerical.nonfinite`` if the sketched gradient matrix is not finite.

    Warns
    -----
    ConvergenceWarning
        ``convergence.max_iter`` if the alternating method stops at
        `max_iter`; the result is returned.

    See Also
    --------
    subspace_from_sketches : The same estimators on given sketches.
    active_subspaces.active_subspace : The estimator from full gradients.

    Notes
    -----
    In normalized coordinates, :math:`d_{ij} = (f(x_i + h e_{ij}) - f(x_i)) / h`
    with :math:`e_{ij}` the :math:`j`-th column of :math:`E_i`, an
    approximation of :math:`d_i = E_i^\top \nabla f(x_i)`.

    The projection estimator [1]_ takes the minimum-norm least-squares
    solution :math:`g_i = (E_i^\top)^{+} d_i = P_i \nabla f(x_i)`, with
    :math:`P_i` the orthogonal projector onto the range of :math:`E_i`, and
    :math:`G = [g_1 \cdots g_M]`. The pseudo-inverse also covers
    rank-deficient Bernoulli sketches.

    The alternating method [2]_ fits :math:`G \approx A B^\top`,
    :math:`A \in \mathbb{R}^{m \times r}`, :math:`B \in \mathbb{R}^{M \times r}`
    with rows :math:`b_i^\top`, by alternating least squares on

    .. math::

        \min_{A, B} \sum_{i=1}^M \lVert d_i - E_i^\top A b_i \rVert_2^2 ,

    starting from the leading :math:`r` right singular vectors of the
    projection estimate. Iteration :math:`\ell` has relative residual
    :math:`\rho_\ell`; it stops with ``"tolerance"`` if
    :math:`\rho_\ell <` `tol`, ``"stalled"`` if :math:`\ell \ge 5` and the
    standard deviation of the last five residuals is below `stall_tol`, or
    ``"max_iter"``.

    The result is the thin SVD :math:`G = W \Sigma V^\top` with
    :math:`s = \sigma / M`. See :class:`SketchedSubspace` for why `s` is not
    an eigenvalue estimate.

    References
    ----------
    .. [1] H. Qi, S. M. Hughes. Invariance of principal components under
       low-dimensional random projection of the data. *Proc. IEEE ICIP*, 2012
       (details to verify).
    .. [2] P. G. Constantine, A. Eftekhari, M. B. Wakin. Computing active
       subspaces efficiently with gradient sketching. *Proc. IEEE CAMSAP*,
       2015 (details to verify).

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", ac.ExperimentalWarning)
    ...     from active_subspaces import experimental
    >>> model = ac.test_functions.ridge([1.0, 2.0, 0.0, 0.0])
    >>> a = np.array([1.0, 2.0, 0.0, 0.0]) / np.sqrt(5.0)
    >>> res = experimental.sketched_active_subspace(
    ...     model, k=2, M=50, method="alternating", rank=1, rng=0
    ... )
    >>> res.stop_reason
    'tolerance'
    >>> print(f"{abs(res.W[:, 0] @ a):.2f}")
    1.00
    """
    model = as_model(model, space=space)
    space = model.space
    m = space.dim
    qoi = model.qoi if qoi is None else _check_qoi(qoi)
    k = _v.integer(k, "k", lo=1, hi=m)
    if (X is None) == (M is None):
        raise InputError(
            "give exactly one of X and M",
            code="input.parameter",
            details={"param": "X", "reason": "x_or_m"},
        )
    if X is not None:
        Xn = _v.points(X, space, "X")
        M = Xn.shape[0]
        X_digest = _v.digest(_v.coerce(X, "X"))
    else:
        M = _v.integer(M, "M", lo=1)
        X_digest = None
    method, rank, tol, max_iter, stall_tol = _check_method(
        method, rank, tol, max_iter, stall_tol, m, M
    )
    design = cast(
        Literal["gaussian", "bernoulli", "orthogonal"], _v.choice(design, "design", _DESIGNS)
    )
    h = _v.real(h, "h", lo=0.0, lo_open=True)

    gen, rng_desc = _random.generator(rng)
    child = _random.spawn(gen, 2)
    if X is None:
        Xn = space.sample(M, rng=child[0], normalized=True)
    E = _draw_design(child[1], design, M, m, k)

    # Forward points x_i + h e_ij in row order i k + j; backward where a box would be left.
    step = np.ones((M, k))
    P = Xn[:, None, :] + h * E.transpose(0, 2, 1)
    if space.kind == "uniform":
        out = np.any(np.abs(P) > 1.0, axis=2)
        if out.any():
            back = Xn[:, None, :] - h * E.transpose(0, 2, 1)
            both = out & np.any(np.abs(back) > 1.0, axis=2)
            if both.any():
                i, j = (int(v) for v in np.argwhere(both)[0])
                raise InputError(
                    f"forward and backward difference points of sample {i}, direction {j} both "
                    "leave the support; reduce h",
                    code="input.out_of_space",
                    details={"param": "h", "sample": i, "direction": j},
                )
            P = np.where(out[:, :, None], back, P)
            step[out] = -1.0
    f0 = evaluate(model, X=space.from_normalized(Xn), executor=executor, batch_size=batch_size)
    fp = evaluate(
        model,
        X=space.from_normalized(P.reshape(M * k, m)),
        executor=executor,
        batch_size=batch_size,
    )
    D = step * (fp.reshape(M, k) - f0[:, None]) / h

    fit = _fit(D, E, method, rank, tol, max_iter, stall_tol)
    params = {
        "k": k,
        "M": M,
        "method": method,
        "rank": rank,
        "design": design,
        "h": h,
        "tol": tol,
        "max_iter": max_iter,
        "stall_tol": stall_tol,
        "model": model.name,
        "batch_size": batch_size,
        "qoi": qoi,
    }
    digests = {"X": X_digest} if X_digest is not None else None
    return SketchedSubspace(
        **fit,
        method=method,
        design=design,
        k=k,
        n_samples=M,
        h=h,
        n_evaluations=M * (k + 1),
        space=space,
        qoi=qoi,
        provenance=_make_provenance(
            "experimental.sketched_active_subspace",
            params,
            space=space,
            rng_state=rng_desc,
            digests=digests,
        ),
    )


def subspace_from_sketches(
    *,
    sketches: ArrayLike,
    directions: ArrayLike,
    space: InputSpace,
    method: Literal["projection", "alternating"] = "projection",
    rank: int | None = None,
    tol: float = 1e-3,
    max_iter: int = 100,
    stall_tol: float = 1e-5,
    qoi: str = "f",
) -> SketchedSubspace:
    r"""Estimate an active subspace from given directional derivatives.

    Use this when the directional derivatives come from elsewhere, for
    example an adjoint or a tangent-linear code. It is deterministic and
    evaluates no model.

    Parameters
    ----------
    sketches : array_like, shape (M, k)
        Directional derivatives :math:`d_{ij} = e_{ij}^\top \nabla_\xi f(\xi_i)`
        in **physical** coordinates, finite; :math:`1 \le k \le m`.
    directions : array_like, shape (M, m, k)
        The **physical** directions :math:`E_i` whose columns are the
        :math:`e_{ij}`, finite.
    space : InputSpace
        The input space; it maps the directions to normalized coordinates.
    method : {"projection", "alternating"}, default "projection"
        As in :func:`sketched_active_subspace`.
    rank : int, optional
        As in :func:`sketched_active_subspace`.
    tol : float, default 1e-3
        As in :func:`sketched_active_subspace`.
    max_iter : int, default 100
        As in :func:`sketched_active_subspace`.
    stall_tol : float, default 1e-5
        As in :func:`sketched_active_subspace`.
    qoi : str, default "f"
        Name of the quantity of interest.

    Returns
    -------
    SketchedSubspace
        With ``design="user"``, ``h=None`` and ``n_evaluations=0``.

    Raises
    ------
    InputError
        ``input.shape`` if `sketches` is not (M, k) with :math:`M \ge 1`,
        :math:`1 \le k \le m`, or `directions` is not (M', m, k);
        ``input.size_mismatch`` if the numbers of sketches and direction
        matrices differ; ``input.nonfinite`` or ``input.dtype`` for bad
        values; ``input.parameter`` for a bad `space`, `method`, `rank`,
        `tol`, `max_iter`, `stall_tol` or `qoi`.
    NumericalError
        ``numerical.singular`` if a least-squares solve or the SVD fails;
        ``numerical.nonfinite`` if the sketched gradient matrix is not finite.

    Warns
    -----
    ConvergenceWarning
        ``convergence.max_iter`` if the alternating method stops at
        `max_iter`.

    See Also
    --------
    sketched_active_subspace : Estimate from a model with forward differences.

    Notes
    -----
    With :math:`\xi = A x + b` the map from normalized to physical
    coordinates, :math:`e^\top \nabla_\xi f = (A^{-1} e)^\top \nabla_x f`, so
    the directions are mapped to :math:`\tilde E_i = A^{-1} E_i` and the
    sketches are used unchanged. The estimators are those of
    :func:`sketched_active_subspace`.

    References
    ----------
    .. [1] H. Qi, S. M. Hughes. Invariance of principal components under
       low-dimensional random projection of the data. *Proc. IEEE ICIP*, 2012
       (details to verify).
    .. [2] P. G. Constantine, A. Eftekhari, M. B. Wakin. Computing active
       subspaces efficiently with gradient sketching. *Proc. IEEE CAMSAP*,
       2015 (details to verify).

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", ac.ExperimentalWarning)
    ...     from active_subspaces import experimental
    >>> model = ac.test_functions.ridge([1.0, 2.0, 0.0, 0.0])
    >>> X = model.space.sample(30, rng=1)
    >>> grads = ac.evaluate_grads(model, X=X)
    >>> E = np.random.default_rng(0).standard_normal((30, 4, 2))
    >>> D = np.einsum("imk,im->ik", E, grads)
    >>> res = experimental.subspace_from_sketches(sketches=D, directions=E, space=model.space)
    >>> res.design, res.h, res.n_evaluations
    ('user', None, 0)
    >>> res.W.shape
    (4, 4)
    """
    _v.check_space(space)
    m = space.dim
    qoi = _check_qoi(qoi)
    D = _v.matrix(sketches, "sketches")
    M, k = D.shape
    if M < 1 or not (1 <= k <= m):
        raise InputError(
            f"sketches: expected shape (M, k) with M >= 1 and 1 <= k <= {m}, got {D.shape}",
            code="input.shape",
            details={"param": "sketches", "expected": ["M", "k"], "got": list(D.shape)},
        )
    Ephys = _v.coerce(directions, "directions")
    if Ephys.ndim != 3 or Ephys.shape[1:] != (m, k):
        raise InputError(
            f"directions: expected shape (M, {m}, {k}), got {Ephys.shape}",
            code="input.shape",
            details={"param": "directions", "expected": ["M", m, k], "got": list(Ephys.shape)},
        )
    if Ephys.shape[0] != M:
        raise InputError(
            f"directions: expected {M} matrices to match sketches, got {Ephys.shape[0]}",
            code="input.size_mismatch",
            details={"param": "directions", "expected": M, "got": int(Ephys.shape[0])},
        )
    _v.check_finite(Ephys, "directions")
    method, rank, tol, max_iter, stall_tol = _check_method(
        method, rank, tol, max_iter, stall_tol, m, M
    )

    A = np.asarray(space.scale)
    if np.count_nonzero(A - np.diag(np.diagonal(A))) == 0:
        E = Ephys / np.diagonal(A)[None, :, None]
    else:
        flat = Ephys.transpose(1, 0, 2).reshape(m, M * k)
        E = scipy.linalg.solve_triangular(A, flat, lower=True).reshape(m, M, k).transpose(1, 0, 2)
    E = np.ascontiguousarray(E)

    fit = _fit(D, E, method, rank, tol, max_iter, stall_tol)
    params = {
        "method": method,
        "rank": rank,
        "tol": tol,
        "max_iter": max_iter,
        "stall_tol": stall_tol,
        "qoi": qoi,
    }
    digests = {"sketches": _v.digest(D), "directions": _v.digest(Ephys)}
    return SketchedSubspace(
        **fit,
        method=method,
        design="user",
        k=k,
        n_samples=M,
        h=None,
        n_evaluations=0,
        space=space,
        qoi=qoi,
        provenance=_make_provenance(
            "experimental.subspace_from_sketches", params, space=space, digests=digests
        ),
    )
