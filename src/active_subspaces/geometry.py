r"""Active domains, the forward map and the inverse map.

All geometry is in normalized coordinates. :func:`active_domain` builds the
exact active domain of a subspace result: an interval (:math:`n = 1`) or a
zonotope (:math:`n \ge 2`) with its vertices and facet inequalities
:math:`A y \le b` for a uniform space, and :math:`\mathbb{R}^n` for a Gaussian
space. :func:`forward_map` maps physical inputs to active and inactive
variables; :func:`inverse_map` draws inputs uniformly (uniform space) or
normally (Gaussian space) on the fiber
:math:`\lbrace x : W_1^\top x = y \rbrace`.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from active_subspaces import _random, solvers
from active_subspaces import _validation as _v
from active_subspaces.errors import (
    AccuracyWarning,
    InputError,
    NumericalError,
    SolverError,
    UnsupportedError,
    _warn,
)
from active_subspaces.results import (
    Result,
    _invariant,
    _make_provenance,
    _register_result,
)
from active_subspaces.solvers import SolverBackend
from active_subspaces.subspaces import SubspaceResult

__all__ = (
    "ActiveCoordinates",
    "ActiveDomain",
    "InverseMapResult",
    "active_domain",
    "forward_map",
    "in_active_domain",
    "inverse_map",
    "zonotope_vertex_count",
)

_toplevel = __all__

_log = logging.getLogger(__name__)

_KINDS = ("uniform", "gaussian")
#: Normalized support tolerance of D-66, used by the feasibility check of draws.
_SUPPORT_TOL = 2e-12
#: A row of W2 (or entry of w2) at most this large does not bound a chord.
_W2_ZERO = 1e-12
#: Facet-inequality tolerance of ``ActiveDomain._check`` (relative to 1 + max b).
_VERTEX_TOL = 1e-12
#: Tolerance of ``W1^T x = y`` for draws (relative to 1 + max |y|).
_FIBER_TOL = 1e-10
#: (p-1)-subsets processed per batched SVD in the facet enumeration.
_SUBSET_BATCH = 4096
#: Hit-and-run chains run in lockstep per block (the output does not depend on it).
_HR_BLOCK = 64
#: Upper bound on the elements of one block of direction rows drawn at once.
_HR_CHUNK_ELEMENTS = 1 << 20
#: Rows of the direction stream drawn per chunk (None: chosen from _HR_CHUNK_ELEMENTS).
_HR_CHUNK: int | None = None
_MAX_WARN_ROWS = 20
#: Above this many (vertex, facet) pairs, ActiveDomain._check uses the equivalent linear-cost test.
_PAIRWISE_CHECK = 1 << 22

_F64 = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@_register_result
class ActiveDomain(Result):
    r"""The active domain of a subspace result, in normalized coordinates.

    Produced by :func:`active_domain`; the constructor is used by
    deserialization.

    Parameters
    ----------
    subspace : SubspaceResult
        The subspace the domain belongs to, with :math:`n \le m - 1`. Its QoI
        name is ``subspace.qoi``.
    kind : {"interval", "zonotope", "unbounded"}
        ``"unbounded"`` for a Gaussian space; for a uniform space
        ``"interval"`` when :math:`n = 1` and ``"zonotope"`` otherwise.
    vertices : numpy.ndarray, shape (N_v, n), float64, or None
        The vertices in active coordinates; ``None`` if unbounded.
    vertex_preimages : numpy.ndarray, shape (N_v, m), float64, or None
        Normalized cube points mapping onto the vertices, entries in
        :math:`\lbrace -1, 0, 1 \rbrace` (0 on inputs with a zero row of
        :math:`W_1`).
    facet_normals : numpy.ndarray, shape (N_f, n), float64, or None
        Unit outward facet normals :math:`\nu_k`.
    facet_offsets : numpy.ndarray, shape (N_f,), float64, or None
        Offsets :math:`b_k = \lVert W_1 \nu_k \rVert_1`; the domain is
        :math:`\lbrace y : \nu_k^\top y \le b_k \rbrace`.
    general_position : bool or None
        Whether the vertex count is that of general position and no row of
        :math:`W_1` is zero; ``None`` if unbounded.
    tol : float
        Tolerance used to merge parallel and drop zero rows, and the default
        tolerance of :func:`in_active_domain`.
    provenance : Provenance, keyword-only
        How the result was produced.

    See Also
    --------
    active_domain : Build it.
    in_active_domain : Test points against it.

    Notes
    -----
    For a uniform space the active domain is the zonotope
    :math:`Z(W_1) = W_1^\top [-1, 1]^m = \sum_j [-g_j, g_j]`, with
    :math:`g_j` the rows of :math:`W_1`; for :math:`n = 1` it is the interval
    :math:`[-\lVert w \rVert_1, \lVert w \rVert_1]` [1]_.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(3)
    >>> X = space.sample(200, rng=0)
    >>> a = np.array([1.0, 0.5, 0.2])
    >>> sub = ac.active_subspace(grads=np.outer(np.cos(X @ a), a), space=space, n=1)
    >>> dom = ac.active_domain(sub)
    >>> dom.kind, dom.n_vertices, dom.vertices.round(4).ravel()
    ('interval', 2, array([-1.4968,  1.4968]))
    """

    TYPE = "ActiveDomain"
    SCHEMA_VERSION = 1

    subspace: SubspaceResult
    kind: Literal["interval", "zonotope", "unbounded"]
    vertices: NDArray[np.float64] | None
    vertex_preimages: NDArray[np.float64] | None
    facet_normals: NDArray[np.float64] | None
    facet_offsets: NDArray[np.float64] | None
    general_position: bool | None
    tol: float

    @property
    def m(self) -> int:
        """Number of inputs ``m``.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> sub = ac.active_subspace(grads=np.eye(3), space=ac.Uniform.reference(3), n=1)
        >>> ac.active_domain(sub).m
        3
        """
        return self.subspace.m

    @property
    def n(self) -> int:
        """Active dimension ``n``.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> sub = ac.active_subspace(grads=np.eye(3), space=ac.Uniform.reference(3), n=2)
        >>> ac.active_domain(sub).n
        2
        """
        return self.subspace.n

    @property
    def n_vertices(self) -> int:
        """Number of vertices, 0 for an unbounded domain.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> sub = ac.active_subspace(grads=np.eye(3), space=ac.Uniform.reference(3), n=2)
        >>> ac.active_domain(sub).n_vertices
        4
        """
        return 0 if self.vertices is None else int(self.vertices.shape[0])

    @property
    def n_facets(self) -> int:
        """Number of facets, 0 for an unbounded domain.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> sub = ac.active_subspace(grads=np.eye(3), space=ac.Gaussian.reference(3), n=2)
        >>> ac.active_domain(sub).n_facets
        0
        """
        return 0 if self.facet_normals is None else int(self.facet_normals.shape[0])

    @property
    def vertex_preimages_physical(self) -> NDArray[np.float64] | None:
        """``vertex_preimages`` in physical coordinates, clipped to the box; ``None`` if unbounded.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> sub = ac.active_subspace(grads=np.eye(3), space=space, n=1)
        >>> ac.active_domain(sub).vertex_preimages_physical
        array([[0.5, 1. , 0. ],
               [0.5, 1. , 4. ]])
        """
        if self.vertex_preimages is None:
            return None
        return _to_physical(self.subspace.space, self.vertex_preimages)

    def _check(self) -> None:
        sub = self.subspace
        _invariant(isinstance(sub, SubspaceResult), "subspace must be a SubspaceResult")
        m, n = sub.m, sub.n
        _invariant(n <= m - 1, "an active domain needs n <= m - 1")
        _invariant(self.tol > 0 and math.isfinite(self.tol), "tol must be positive and finite")
        skind = sub.space.kind
        expected = "unbounded" if skind == "gaussian" else ("interval" if n == 1 else "zonotope")
        _invariant(
            skind in _KINDS and self.kind == expected,
            f"kind {self.kind!r} does not match a {skind} space with n={n}",
        )
        arrays = (self.vertices, self.vertex_preimages, self.facet_normals, self.facet_offsets)
        if self.kind == "unbounded":
            _invariant(
                all(a is None for a in arrays) and self.general_position is None,
                "an unbounded domain has no vertices or facets",
            )
            return
        _invariant(
            all(a is not None for a in arrays) and self.general_position is not None,
            "a bounded domain needs vertices and facets",
        )
        V, P, A, b = arrays
        assert V is not None  # checked just above
        assert P is not None
        assert A is not None
        assert b is not None
        nv, nf = V.shape[0], A.shape[0]
        _invariant(V.shape == (nv, n) and nv >= 2, "vertices must be (N_v, n)")
        _invariant(P.shape == (nv, m), "vertex_preimages must be (N_v, m)")
        _invariant(
            bool(np.all((P == -1) | (P == 0) | (P == 1))),
            "vertex_preimages entries must be -1, 0 or 1",
        )
        _invariant(
            A.shape == (nf, n) and nf >= 2 and b.shape == (nf,),
            "facet_normals (N_f, n) and facet_offsets (N_f,)",
        )
        _invariant(bool(np.all(b > 0)), "facet_offsets must be positive")
        tol = _VERTEX_TOL * (1.0 + float(np.max(b)))
        if nv * nf <= _PAIRWISE_CHECK:
            ok = bool(np.all(A @ V.T - b[:, None] <= tol))
        else:
            # Equivalent linear-cost check: nu^T (W1^T p) <= ||W1 nu||_1 for every p in the cube,
            # so offsets at least the support function and vertices = preimages @ W1 imply A V^T <=
            # b.
            W1 = sub.W1
            ok = bool(np.all(np.sum(np.abs(W1 @ A.T), axis=0) - b <= tol)) and bool(
                np.all(np.abs(P @ W1 - V) <= tol)
            )
        _invariant(ok, "every vertex must satisfy every facet inequality")


@_register_result
class InverseMapResult(Result):
    r"""Inputs drawn on the fibers of given active points, in normalized coordinates.

    Produced by :func:`inverse_map`; the constructor is used by
    deserialization.

    Parameters
    ----------
    subspace : SubspaceResult
        The subspace, with :math:`n \le m - 1`.
    Y : numpy.ndarray, shape (M, n), float64
        The active points given.
    X : numpy.ndarray, shape (M * n_draws, m), float64
        Normalized draws; row ``k * n_draws + j`` is draw ``j`` for ``Y[k]``,
        and ``X @ W1`` equals ``Y[index]``.
    index : numpy.ndarray, shape (M * n_draws,), int64
        ``np.repeat(np.arange(M), n_draws)``: the row of ``Y`` of each draw.
    n_draws : int
        Draws per active point.
    sampler : {"gaussian", "exact_interval", "hit_and_run"}
        How the draws were made.
    burn_in : int or None
        Hit-and-run steps discarded before the first draw; ``None`` for the
        other samplers.
    thin : int or None
        Hit-and-run steps between kept draws; ``None`` for the other samplers.
    degenerate : numpy.ndarray, shape (M,), bool
        Rows whose fiber has empty interior; their draws are copies of one
        point.
    provenance : Provenance, keyword-only
        How the result was produced.

    See Also
    --------
    inverse_map : Draw them.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(3)
    >>> X = space.sample(200, rng=0)
    >>> a = np.array([1.0, 0.5, 0.2])
    >>> sub = ac.active_subspace(grads=np.outer(np.cos(X @ a), a), space=space, n=1)
    >>> res = ac.inverse_map(sub, Y=[[0.5]], n_draws=4, rng=0)
    >>> res.X.shape, res.sampler, res.index.tolist()
    ((4, 3), 'hit_and_run', [0, 0, 0, 0])
    >>> bool(np.allclose(res.X @ sub.W1, 0.5))
    True
    """

    TYPE = "InverseMapResult"
    SCHEMA_VERSION = 1

    subspace: SubspaceResult
    Y: NDArray[np.float64]
    X: NDArray[np.float64]
    index: NDArray[np.int64]
    n_draws: int
    sampler: Literal["gaussian", "exact_interval", "hit_and_run"]
    burn_in: int | None
    thin: int | None
    degenerate: NDArray[np.bool_]

    @property
    def Z(self) -> NDArray[np.float64]:
        """Inactive variables ``X @ W2``, shape ``(M * n_draws, m - n)``.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> sub = ac.active_subspace(grads=np.eye(3), space=ac.Gaussian.reference(3), n=1)
        >>> ac.inverse_map(sub, Y=[[0.0]], n_draws=5, rng=0).Z.shape
        (5, 2)
        """
        return self.X @ self.subspace.W2

    @property
    def X_physical(self) -> NDArray[np.float64]:
        """``X`` in physical coordinates, clipped to the box for a uniform space.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [1.0, 10.0])
        >>> sub = ac.active_subspace(grads=np.array([[1.0, 0.0]]), space=space, n=1)
        >>> res = ac.inverse_map(sub, Y=[[1.0]], n_draws=3, rng=0)
        >>> bool(np.all(res.X_physical[:, 0] == 1.0))
        True
        """
        return _to_physical(self.subspace.space, self.X)

    def _check(self) -> None:
        sub = self.subspace
        _invariant(isinstance(sub, SubspaceResult), "subspace must be a SubspaceResult")
        m, n = sub.m, sub.n
        _invariant(n <= m - 1, "the inverse map needs n <= m - 1")
        M = self.Y.shape[0] if self.Y.ndim == 2 else -1
        _invariant(self.Y.shape == (M, n) and M >= 1, "Y must be (M, n)")
        _invariant(self.n_draws >= 1, "n_draws must be >= 1")
        N = M * self.n_draws
        _invariant(self.X.shape == (N, m), "X must be (M * n_draws, m)")
        _invariant(
            self.index.shape == (N,)
            and bool(np.array_equal(self.index, np.repeat(np.arange(M), self.n_draws))),
            "index must be repeat(arange(M), n_draws)",
        )
        _invariant(self.degenerate.shape == (M,), "degenerate must be (M,)")
        skind = sub.space.kind
        expected = (
            "gaussian"
            if skind == "gaussian"
            else ("exact_interval" if m - n == 1 else "hit_and_run")
        )
        _invariant(
            skind in _KINDS and self.sampler == expected,
            f"sampler {self.sampler!r} does not match a {skind} space with m-n={m - n}",
        )
        if self.sampler == "hit_and_run":
            _invariant(
                self.burn_in is not None
                and self.burn_in >= 0
                and self.thin is not None
                and self.thin >= 1,
                "hit-and-run needs burn_in >= 0 and thin >= 1",
            )
        else:
            _invariant(
                self.burn_in is None and self.thin is None,
                "burn_in and thin are recorded only for hit-and-run",
            )
        _invariant(bool(np.all(np.isfinite(self.X))), "X must be finite")
        err = np.abs(self.X @ sub.W1 - self.Y[self.index])
        scale = 1.0 + float(np.max(np.abs(self.Y)))
        _invariant(bool(np.all(err <= _FIBER_TOL * scale)), "X @ W1 must equal Y[index]")
        if skind == "uniform":
            _invariant(bool(np.all(np.abs(self.X) <= 1.0)), "uniform draws must lie in [-1, 1]^m")


class ActiveCoordinates(NamedTuple):
    """Active and inactive variables of a set of points, returned by :func:`forward_map`.

    Both arrays are new and writeable.

    Parameters
    ----------
    Y : numpy.ndarray, shape (M, n), float64
        Active variables.
    Z : numpy.ndarray, shape (M, m - n), float64
        Inactive variables.

    See Also
    --------
    forward_map : Compute them.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> sub = ac.active_subspace(grads=np.eye(3), space=ac.Uniform.reference(3), n=1)
    >>> Y, Z = ac.forward_map(sub, X=[[0.5, 0.0, 0.0]])
    >>> Y.shape, Z.shape
    ((1, 1), (1, 2))
    """

    Y: NDArray[np.float64]
    """(M, n) active variables."""
    Z: NDArray[np.float64]
    """(M, m - n) inactive variables."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_physical(space: Any, X: _F64) -> _F64:
    P: _F64 = space.from_normalized(X)
    if space.kind == "uniform":
        P = np.clip(P, space.lower, space.upper)
    return P


def _check_subspace(subspace: Any, *, need_inactive: bool) -> SubspaceResult:
    if not isinstance(subspace, SubspaceResult):
        raise InputError(
            f"subspace: expected a SubspaceResult, got {type(subspace).__name__}; wrap a basis "
            "with subspace_from_basis",
            code="input.parameter",
            details={"param": "subspace", "got": type(subspace).__name__},
        )
    if need_inactive and subspace.n >= subspace.m:
        raise InputError(
            f"subspace: n = m = {subspace.m} leaves no inactive variables; use with_dimension to "
            "choose n <= m - 1",
            code="input.parameter",
            details={
                "param": "subspace",
                "reason": "no_inactive_variables",
                "n": subspace.n,
                "m": subspace.m,
            },
        )
    return subspace


def _space_kind(subspace: SubspaceResult, function: str) -> str:
    kind = subspace.space.kind
    if kind not in _KINDS:
        raise UnsupportedError(
            f"{function} is defined only for uniform or Gaussian input spaces, got kind {kind!r}",
            code="unsupported.space",
            details={"function": function, "kind": kind},
        )
    return kind


def _active_points(Y: Any, n: int, name: str = "Y") -> np.ndarray:
    arr = _v.coerce(Y, name)
    if arr.ndim == 1 and n == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[1] != n or arr.shape[0] < 1:
        raise InputError(
            f"{name}: expected shape (M, {n}) with M >= 1, got {arr.shape}",
            code="input.shape",
            details={"param": name, "expected": ["M", n], "got": list(arr.shape)},
        )
    _v.check_finite(arr, name)
    return np.array(arr, dtype=np.float64, order="C", copy=True)


def _svd(A: np.ndarray, what: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        _, s, Vh = np.linalg.svd(A, full_matrices=True)
    except np.linalg.LinAlgError as exc:
        raise NumericalError(
            f"SVD failed in {what}: {exc}", code="numerical.singular", details={"step": what}
        ) from exc
    return s, Vh


# ---------------------------------------------------------------------------
# Zonotope vertices and facets [D-90]
# ---------------------------------------------------------------------------


def zonotope_vertex_count(m: int, n: int) -> int:
    r"""Return the number of vertices of a zonotope in general position.

    Parameters
    ----------
    m : int
        Number of generators (inputs), at least 2.
    n : int
        Dimension of the zonotope, :math:`1 \le n \le m - 1`.

    Returns
    -------
    int
        :math:`N_v = 2 \sum_{i=0}^{n-1} \binom{m-1}{i}`: the exact count in
        general position, an upper bound otherwise.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad ``m`` or ``n``.

    See Also
    --------
    active_domain : Builds the zonotope.

    Notes
    -----
    General position means no zero rows of :math:`W_1`, no two rows parallel
    and any :math:`n` rows linearly independent [1]_, [2]_.

    References
    ----------
    .. [1] P. McMullen. On zonotopes. *Trans. Amer. Math. Soc.* 159:91-109,
       1971 (details to verify).
    .. [2] G. M. Ziegler. *Lectures on Polytopes.* Graduate Texts in
       Mathematics 152, Springer, 1995, chapter 7.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> ac.zonotope_vertex_count(10, 2)
    20
    """
    m = _v.integer(m, "m", lo=2)
    n = _v.integer(n, "n", lo=1, hi=m - 1)
    return 2 * sum(math.comb(m - 1, i) for i in range(n))


class _Reduction(NamedTuple):
    zero: np.ndarray  # (m,) bool, zero rows
    cls: np.ndarray  # (m,) intp, class of each row (-1 on zero rows)
    alpha: np.ndarray  # (m,) float, g_j = alpha_j * hbar_c
    H: np.ndarray  # (m'', n) reduced generators h_c


def _reduce(W1: np.ndarray, tau: float) -> _Reduction:
    """Drop zero rows and merge parallel rows in row order (spec 08 §5.1 step 2)."""
    m, n = W1.shape
    norms = np.sqrt(np.sum(W1 * W1, axis=1))
    zero = norms <= tau * float(np.max(norms))
    cls = np.full(m, -1, dtype=np.intp)
    alpha = np.zeros(m)
    reps = np.empty((m, n))
    k = 0
    for j in np.flatnonzero(~zero):
        gh = W1[j] / norms[j]
        c = k
        if k:
            R = reps[:k]
            res = gh[None, :] - (R @ gh)[:, None] * R
            hit = np.flatnonzero(np.sqrt(np.sum(res * res, axis=1)) <= tau)
            if hit.size:
                c = int(hit[0])
        if c == k:
            reps[k] = gh
            k += 1
        cls[j] = c
        alpha[j] = float(W1[j] @ reps[c])
    weight = np.zeros(k)
    nz = ~zero
    np.add.at(weight, cls[nz], np.abs(alpha[nz]))
    return _Reduction(zero, cls, alpha, weight[:, None] * reps[:k])


def _polygon_signs(G: np.ndarray) -> np.ndarray:
    """Vertex sign vectors of a 2-D zonotope, counter-clockwise from ``-sum g`` (§5.1 step 4).

    Signs are relative to the generators as given (orientation flips undone).
    """
    k = G.shape[0]
    flip = (G[:, 1] < 0) | ((G[:, 1] == 0) & (G[:, 0] < 0))
    Go = np.where(flip[:, None], -G, G)
    theta = np.arctan2(Go[:, 1], Go[:, 0])
    order = np.argsort(theta, kind="stable")
    rank = np.empty(k, dtype=np.intp)
    rank[order] = np.arange(1, k + 1)
    steps = np.arange(2 * k)[:, None]
    first = (steps <= k) & (rank[None, :] <= steps)
    second = (steps > k) & (rank[None, :] > steps - k)
    sig = np.where(first | second, 1, -1).astype(np.int8)
    return np.where(flip[None, :], -sig, sig).astype(np.int8)


def _verts(G: np.ndarray, tau: float, facets: list[_F64] | None = None) -> np.ndarray:
    """Vertex sign vectors ``(N, k)`` of the zonotope of the rows of ``G`` (§5.1 step 5).

    With ``facets`` a list, run the facet enumeration even for a parallelotope
    and append the facet normals found (``+nu`` then ``-nu``).
    """
    k, p = G.shape
    if p == 1:
        s = np.sign(G[:, 0]).astype(np.int8)
        return np.stack([s, -s])
    if k == p and facets is None:  # before p == 2: the same set, found faster
        return np.array(list(itertools.product((-1, 1), repeat=p)), dtype=np.int8)
    if p == 2:
        return _polygon_signs(G)
    Gh = G / np.sqrt(np.sum(G * G, axis=1))[:, None]
    combos = np.array(list(itertools.product((-1, 1), repeat=p - 1)), dtype=np.int8)
    seen: set[bytes] = set()
    blocks: list[np.ndarray] = []
    subsets = itertools.combinations(range(k), p - 1)
    while True:
        chunk = list(itertools.islice(subsets, _SUBSET_BATCH))
        if not chunk:
            break
        S = np.array(chunk, dtype=np.intp)
        s, Vh = _svd(Gh[S], "facet enumeration")
        Omask = np.abs(Vh[:, -1, :] @ Gh.T) <= tau
        new: list[int] = []
        for i in np.flatnonzero(s[:, -1] > tau):
            key = np.packbits(Omask[i]).tobytes()
            if key not in seen:
                seen.add(key)
                new.append(int(i))
        if not new:
            continue
        sizes = Omask[new].sum(axis=1)
        nus = np.empty((len(new), p))
        # Facets with |O| = p - 1 are parallelotopes: frames by one batched SVD, all sign vectors.
        par = [i for i, c in zip(new, sizes, strict=False) if c == p - 1]
        if par:
            Opar = np.array([np.flatnonzero(Omask[i]) for i in par], dtype=np.intp)
            _, Vo = _svd(G[Opar], "facet frame")
            nu = _canonical(Vo[:, -1, :])
            sig = np.sign(nu @ G.T).astype(np.int8)
            full = np.repeat(sig[:, None, :], combos.shape[0], axis=1)
            full[
                np.arange(len(par))[:, None, None],
                np.arange(combos.shape[0])[None, :, None],
                Opar[:, None, :],
            ] = combos[None]
            full = full.reshape(-1, k)
            blocks += [full, -full]
            nus[[j for j, c in enumerate(sizes) if c == p - 1]] = nu
        for j, (i, c) in enumerate(zip(new, sizes, strict=False)):  # type: ignore[assignment]  # i: int index
            if c == p - 1:
                continue
            Oi = np.flatnonzero(Omask[i])
            out = np.flatnonzero(~Omask[i])
            _, Vo = _svd(G[Oi], "facet frame")
            nu = _canonical(Vo[-1][None, :])[0]
            sub = _verts(G[Oi] @ Vo[:-1].T, tau)
            full = np.empty((sub.shape[0], k), dtype=np.int8)
            full[:, Oi] = sub
            full[:, out] = np.sign(G[out] @ nu).astype(np.int8)
            blocks += [full, -full]
            nus[j] = nu
        if facets is not None:
            for nu in nus:
                facets.append(nu)
                facets.append(-nu)
    return np.unique(np.concatenate(blocks), axis=0)


def _canonical(nu: np.ndarray) -> np.ndarray:
    """Give each row its canonical sign: largest-magnitude entry positive (first on a tie)."""
    idx = np.argmax(np.abs(nu), axis=1)
    flip = nu[np.arange(nu.shape[0]), idx] < 0
    return np.where(flip[:, None], -nu, nu)


def _preimages(signs: np.ndarray, red: _Reduction) -> np.ndarray:
    """Cube preimages ``x_j = sign(alpha_j) sigma_{c(j)}``, 0 on zero rows (§5.1 step 6)."""
    P = np.zeros((signs.shape[0], red.cls.shape[0]))
    nz = ~red.zero
    P[:, nz] = np.sign(red.alpha[nz])[None, :] * signs[:, red.cls[nz]]
    return P


def active_domain(subspace: SubspaceResult, *, tol: float = 1e-10) -> ActiveDomain:
    r"""Build the active domain of a subspace.

    Parameters
    ----------
    subspace : SubspaceResult
        A subspace with :math:`n \le m - 1`.
    tol : float, default 1e-10, keyword-only
        Relative tolerance in :math:`(0, 1)` for zero and parallel rows of
        :math:`W_1` and for facet hyperplanes; also the default tolerance of
        :func:`in_active_domain`.

    Returns
    -------
    ActiveDomain
        ``"unbounded"`` for a Gaussian space; ``"interval"`` (``n = 1``) or
        ``"zonotope"`` (``n >= 2``) for a uniform space.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad ``subspace`` or ``tol``, or ``n == m``.
    UnsupportedError
        ``unsupported.space`` for another kind of space.
    NumericalError
        ``numerical.singular`` if an SVD of the facet enumeration fails.

    See Also
    --------
    in_active_domain : Test points against the domain.
    inverse_map : Draw inputs for active points in the domain.
    zonotope_vertex_count : The vertex count in general position.

    Notes
    -----
    The computation is exact and deterministic. With the rows
    :math:`g_j` of :math:`W_1` as generators, zero rows are dropped and
    parallel rows merged; the zonotope :math:`Z(W_1) = \sum_j [-g_j, g_j]`
    then has the facet description

    .. math::

        Z(W_1) = \lbrace y : \nu_k^\top y \le \lVert W_1 \nu_k \rVert_1,
        \ k = 1, \dots, N_f \rbrace,

    where each unit normal :math:`\nu_k` is orthogonal to a rank
    :math:`n - 1` subset of generators. The vertices are the sums
    :math:`\sum_c \sigma_c h_c` of the merged generators over the sign vectors
    of the facets' vertices [1]_, [2]_, [3]_. For :math:`n = 2` the vertices
    are ordered counter-clockwise; for :math:`n \ge 3` lexicographically by
    their cube preimages. Run time grows like the number of facets,
    :math:`2 \binom{m}{n-1}`.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.
    .. [2] P. McMullen. On zonotopes. *Trans. Amer. Math. Soc.* 159:91-109,
       1971 (details to verify).
    .. [3] G. M. Ziegler. *Lectures on Polytopes.* Graduate Texts in
       Mathematics 152, Springer, 1995, chapter 7.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(3)
    >>> X = space.sample(200, rng=0)
    >>> a = np.array([1.0, 0.5, 0.2])
    >>> sub = ac.active_subspace(grads=np.outer(np.cos(X @ a), a), space=space, n=1)
    >>> dom = ac.active_domain(ac.with_dimension(sub, 2))
    >>> dom.kind, dom.n_vertices, dom.n_facets, dom.general_position
    ('zonotope', 6, 6, True)
    """
    subspace = _check_subspace(subspace, need_inactive=True)
    tol = _v.real(tol, "tol", lo=0.0, hi=1.0, lo_open=True, hi_open=True)
    kind = _space_kind(subspace, "active_domain")
    prov = _make_provenance("active_domain", {"tol": tol}, space=subspace.space)
    m, n = subspace.m, subspace.n
    if kind == "gaussian":
        return ActiveDomain(
            subspace=subspace,
            kind="unbounded",
            vertices=None,
            vertex_preimages=None,
            facet_normals=None,
            facet_offsets=None,
            general_position=None,
            tol=tol,
            provenance=prov,
        )
    W1 = np.array(subspace.W1, dtype=np.float64, order="C")
    red = _reduce(W1, tol)
    n_zero = int(np.count_nonzero(red.zero))
    if n == 1:
        w = W1[:, 0]
        h = float(np.sum(np.abs(w)))
        sgn = np.where(red.zero, 0.0, np.sign(w))
        P = np.stack([-sgn, sgn])
        V = np.array([[-h], [h]])
        A = np.array([[-1.0], [1.0]])
        general = n_zero == 0
    elif n == 2:
        signs = _polygon_signs(red.H)
        P = _preimages(signs, red)
        V = P @ W1
        E = np.roll(V, -1, axis=0) - V
        A = np.stack([E[:, 1], -E[:, 0]], axis=1)
        A /= np.sqrt(np.sum(A * A, axis=1))[:, None]
        general = n_zero == 0 and V.shape[0] == zonotope_vertex_count(m, n)
    else:
        facets: list[np.ndarray] = []
        signs = _verts(red.H, tol, facets)
        P = _preimages(signs, red)
        P = P[np.lexsort(P.T[::-1])]
        V = P @ W1
        A = np.array(facets)
        general = n_zero == 0 and V.shape[0] == zonotope_vertex_count(m, n)
    b = np.sum(np.abs(W1 @ A.T), axis=0)
    _log.debug(
        "active_domain: m=%d n=%d zero rows=%d classes=%d N_v=%d N_f=%d",
        m,
        n,
        n_zero,
        red.H.shape[0],
        V.shape[0],
        A.shape[0],
    )
    return ActiveDomain(
        subspace=subspace,
        kind="interval" if n == 1 else "zonotope",
        vertices=V,
        vertex_preimages=P,
        facet_normals=A,
        facet_offsets=b,
        general_position=bool(general),
        tol=tol,
        provenance=prov,
    )


def in_active_domain(
    domain: ActiveDomain, *, Y: ArrayLike, tol: float | None = None
) -> NDArray[np.bool_]:
    r"""Return whether each active point lies in the active domain.

    Parameters
    ----------
    domain : ActiveDomain
        The domain.
    Y : array_like, shape (M, n), float, keyword-only
        Active points; ``(M,)`` is accepted when :math:`n = 1`.
    tol : float, optional, keyword-only
        Relative tolerance, non-negative; default ``domain.tol``.

    Returns
    -------
    numpy.ndarray, shape (M,), bool
        ``True`` where :math:`\nu_k^\top y \le b_k (1 + \mathrm{tol})` for every
        facet :math:`k`. An unbounded domain contains every point.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad ``domain`` or ``tol``; ``input.shape``,
        ``input.dtype`` or ``input.nonfinite`` for a bad ``Y``.

    See Also
    --------
    active_domain : Build the domain.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(3)
    >>> X = space.sample(200, rng=0)
    >>> a = np.array([1.0, 0.5, 0.2])
    >>> sub = ac.active_subspace(grads=np.outer(np.cos(X @ a), a), space=space, n=1)
    >>> dom = ac.active_domain(sub)
    >>> ac.in_active_domain(dom, Y=[[0.0], [2.0]])
    array([ True, False])
    """
    if not isinstance(domain, ActiveDomain):
        raise InputError(
            f"domain: expected an ActiveDomain, got {type(domain).__name__}",
            code="input.parameter",
            details={"param": "domain", "got": type(domain).__name__},
        )
    tau = domain.tol if tol is None else _v.real(tol, "tol", lo=0.0)
    Yv = _active_points(Y, domain.n)
    if domain.kind == "unbounded":
        return np.ones(Yv.shape[0], dtype=bool)
    assert domain.facet_normals is not None  # bounded domain
    assert domain.facet_offsets is not None
    lhs = Yv @ domain.facet_normals.T
    return np.all(lhs <= domain.facet_offsets[None, :] * (1.0 + tau), axis=1)


# ---------------------------------------------------------------------------
# Forward map
# ---------------------------------------------------------------------------


def forward_map(subspace: SubspaceResult, *, X: ArrayLike) -> ActiveCoordinates:
    r"""Map physical inputs to active and inactive variables.

    Parameters
    ----------
    subspace : SubspaceResult
        The subspace; any :math:`1 \le n \le m`.
    X : array_like, shape (M, m), float, keyword-only
        Points in physical coordinates, normalized with ``subspace.space``
        and checked against its support.

    Returns
    -------
    ActiveCoordinates
        ``Y`` of shape ``(M, n)`` and ``Z`` of shape ``(M, m - n)`` (``(M, 0)``
        when :math:`n = m`).

    Raises
    ------
    InputError
        ``input.parameter`` for a bad ``subspace``; ``input.dtype``,
        ``input.shape``, ``input.nonfinite`` or ``input.out_of_space`` for a
        bad ``X``.

    See Also
    --------
    inverse_map : The reverse direction, by sampling.

    Notes
    -----
    With :math:`x` the normalized points, :math:`y = W_1^\top x` and
    :math:`z = W_2^\top x`; row-wise :math:`Y = X W_1`, :math:`Z = X W_2`.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(3)
    >>> X = space.sample(200, rng=0)
    >>> a = np.array([1.0, 0.5, 0.2])
    >>> sub = ac.active_subspace(grads=np.outer(np.cos(X @ a), a), space=space, n=1)
    >>> coords = ac.forward_map(sub, X=[[0.5, 0.5, 0.5]])
    >>> coords.Y.round(4), coords.Z.shape
    (array([[0.7484]]), (1, 2))
    """
    subspace = _check_subspace(subspace, need_inactive=False)
    x = _v.points(X, subspace.space)
    return ActiveCoordinates(x @ subspace.W1, x @ subspace.W2)


# ---------------------------------------------------------------------------
# Inverse map
# ---------------------------------------------------------------------------


def _rowdot(A: _F64, Bt: _F64) -> _F64:
    """Return ``A @ Bt.T`` with each entry a contiguous last-axis sum.

    The result does not depend on the number of rows.
    """
    out: _F64 = np.sum(A[:, None, :] * Bt[None, :, :], axis=-1)
    return out


def _feasibility(x: _F64, y: _F64, W1: _F64, row: int) -> _F64:
    """D-46 feasibility check of the draws ``x (N, m)`` for ``y``.

    Return them clipped to the cube.
    """
    viol = float(np.max(np.abs(x))) - 1.0
    resid = float(np.max(np.abs(x @ W1 - y[None, :])))
    if viol > _SUPPORT_TOL or resid > _FIBER_TOL * (1.0 + float(np.max(np.abs(y)))):
        raise NumericalError(
            f"inverse_map: a draw for row {row} violates its constraints (box excess {viol:.3g}, "
            f"W1^T x - y = {resid:.3g})",
            code="numerical.infeasible_draw",
            details={"row": row, "box_excess": viol, "residual": resid},
        )
    return np.clip(x, -1.0, 1.0)


def _out_of_domain(row: int, reason: str) -> InputError:
    return InputError(
        f"Y: row {row} lies outside the active domain ({reason})",
        code="input.out_of_domain",
        details={"param": "Y", "row": row},
    )


def _exact_interval(
    Yv: _F64,
    W1: _F64,
    w2: _F64,
    children: list[np.random.Generator],
    n_draws: int,
    btol: float,
) -> tuple[_F64, NDArray[np.bool_]]:
    M = Yv.shape[0]
    S = Yv @ W1.T
    bound = np.abs(w2) > _W2_ZERO
    wb = w2[bound]
    X = np.empty((M * n_draws, W1.shape[0]))
    degenerate = np.zeros(M, dtype=bool)
    for k in range(M):
        s = S[k]
        if np.any(np.abs(s[~bound]) > 1.0 + btol):
            raise _out_of_domain(k, "a coordinate fixed by y lies outside [-1, 1]")
        a = (-1.0 - s[bound]) / wb
        c = (1.0 - s[bound]) / wb
        t_lo = float(np.max(np.minimum(a, c)))
        t_hi = float(np.min(np.maximum(a, c)))
        if t_lo > t_hi + 2.0 * btol:
            raise _out_of_domain(k, "the fiber is empty")
        if (t_hi - t_lo) / 2.0 <= btol:
            degenerate[k] = True
            t = np.full(n_draws, 0.5 * (t_lo + t_hi))
        else:
            v = children[k].random(n_draws)
            t = t_lo + v * (t_hi - t_lo)
        x = s[None, :] + t[:, None] * w2[None, :]
        X[k * n_draws : (k + 1) * n_draws] = _feasibility(x, Yv[k], W1, k)
    return X, degenerate


def _chebyshev_start(
    y: _F64, W1: _F64, kappa: _F64, backend: SolverBackend, row: int
) -> tuple[_F64, float]:
    m, n = W1.shape
    c = np.zeros(m + 1)
    c[-1] = -1.0
    eye = np.eye(m)
    A_ub = np.block([[eye, kappa[:, None]], [-eye, kappa[:, None]]])
    b_ub = np.ones(2 * m)
    A_eq = np.hstack([W1.T, np.zeros((n, 1))])
    bounds = np.vstack([np.tile([-1.0, 1.0], (m, 1)), [[0.0, np.inf]]])
    try:
        out = solvers.solve_lp(c, A_ub, b_ub, A_eq, y, bounds, backend=backend)
    except SolverError as exc:
        if exc.code == "solver.infeasible":
            raise _out_of_domain(row, "the Chebyshev-center LP is infeasible") from exc
        raise
    _log.debug("inverse_map: Chebyshev LP for row %d: %s", row, out.status)
    assert out.x is not None  # solve_lp returns only optimal solutions
    x = np.array(out.x[:m], dtype=np.float64)
    # row sums, not BLAS gemv: the chain amplifies alignment-dependent rounding [D-15]
    x = np.clip(x + _rowdot((y - _rowdot(x[None, :], W1.T)[0])[None, :], W1)[0], -1.0, 1.0)
    pos = kappa > _W2_ZERO
    r_hat = float(np.min((1.0 - np.abs(x[pos])) / kappa[pos])) if np.any(pos) else 0.0
    return x, r_hat


def _walk_block(
    X0: _F64,
    Yb: _F64,
    W1: _F64,
    streams: list[tuple[np.random.Generator, ...]],
    burn_in: int,
    thin: int,
    n_draws: int,
) -> _F64:
    """Run hit-and-run chains in lockstep from ``X0 (B, m)``; return draws ``(B, n_draws, m)``."""
    B, m = X0.shape
    W1t = np.ascontiguousarray(W1.T)
    W1c = np.ascontiguousarray(W1)
    T = burn_in + n_draws * thin
    chunk = (
        _HR_CHUNK if _HR_CHUNK is not None else max(1, min(T, _HR_CHUNK_ELEMENTS // max(1, B * m)))
    )
    X = X0.copy()
    out = np.empty((B, n_draws, m))
    t = 0
    while t < T:
        L = min(chunk, T - t)
        U = np.stack([dirs.standard_normal((L, m)) for dirs, _ in streams])
        V = np.stack([steps.random(L) for _, steps in streams])
        for i in range(L):
            u = U[:, i, :]
            delta = u - _rowdot(_rowdot(u, W1t), W1c)
            nz = delta != 0
            safe = np.where(nz, delta, 1.0)
            a = np.maximum(1.0 - X, 0.0) / safe  # (1 - x) / delta
            b = -np.maximum(1.0 + X, 0.0) / safe  # (-1 - x) / delta
            hi = np.where(nz, np.maximum(a, b), np.inf).min(axis=1)
            lo = np.where(nz, np.minimum(a, b), -np.inf).max(axis=1)
            ok = np.isfinite(hi) & np.isfinite(lo)
            lam = np.where(ok, lo + V[:, i] * np.where(ok, hi - lo, 0.0), 0.0)
            X = X + lam[:, None] * delta
            X = X + _rowdot(Yb - _rowdot(X, W1t), W1c)
            t += 1
            if t > burn_in and (t - burn_in) % thin == 0:
                out[:, (t - burn_in) // thin - 1, :] = X
    return out


def _hit_and_run(
    Yv: _F64,
    W1: _F64,
    W2: _F64,
    children: list[np.random.Generator],
    n_draws: int,
    burn_in: int,
    thin: int,
    btol: float,
    backend: SolverBackend,
) -> tuple[_F64, NDArray[np.bool_]]:
    M = Yv.shape[0]
    m = W1.shape[0]
    kappa = np.sqrt(np.sum(W2 * W2, axis=1))
    X = np.empty((M * n_draws, m))
    degenerate = np.zeros(M, dtype=bool)
    starts = np.empty((M, m))
    for k in range(M):
        starts[k], r_hat = _chebyshev_start(Yv[k], W1, kappa, backend, k)
        degenerate[k] = r_hat <= btol
    live = np.flatnonzero(~degenerate)
    streams = {int(k): tuple(children[k].spawn(2)) for k in live}
    for b0 in range(0, live.size, _HR_BLOCK):
        rows = live[b0 : b0 + _HR_BLOCK]
        draws = _walk_block(
            starts[rows], Yv[rows], W1, [streams[int(k)] for k in rows], burn_in, thin, n_draws
        )
        for j, k in enumerate(rows):
            X[k * n_draws : (k + 1) * n_draws] = _feasibility(draws[j], Yv[k], W1, int(k))
    for k in np.flatnonzero(degenerate):  # type: ignore[assignment]  # k: an index, as above
        x = np.repeat(starts[k][None, :], n_draws, axis=0)
        X[k * n_draws : (k + 1) * n_draws] = _feasibility(x, Yv[k], W1, int(k))
    return X, degenerate


def inverse_map(
    subspace: SubspaceResult,
    *,
    Y: ArrayLike,
    n_draws: int = 1,
    burn_in: int | None = None,
    thin: int | None = None,
    boundary_tol: float = 1e-6,
    backend: SolverBackend | str | None = None,
    rng: _random.RngLike = None,
) -> InverseMapResult:
    r"""Draw inputs on the fiber of each active point.

    Parameters
    ----------
    subspace : SubspaceResult
        A subspace with :math:`n \le m - 1`.
    Y : array_like, shape (M, n), float, keyword-only
        Active points (normalized coordinates); ``(M,)`` is accepted when
        :math:`n = 1`.
    n_draws : int, default 1, keyword-only
        Draws per point.
    burn_in : int, optional, keyword-only
        Hit-and-run steps discarded before the first draw; default
        :math:`10 (m - n)`. Used only by hit-and-run.
    thin : int, optional, keyword-only
        Hit-and-run steps between kept draws, at least 1; default
        :math:`m - n`. Used only by hit-and-run.
    boundary_tol : float, default 1e-6, keyword-only
        Chebyshev radius (or half chord) at or below which a fiber counts as
        degenerate.
    backend : SolverBackend, "scipy" or None, keyword-only
        LP backend for the Chebyshev center (hit-and-run only).
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator, optional
        Source of randomness, keyword-only. Each row of ``Y`` gets its own
        child stream, so its draws do not depend on the other rows.

    Returns
    -------
    InverseMapResult
        ``M * n_draws`` normalized draws with ``X @ W1 == Y[index]``.

    Raises
    ------
    InputError
        ``input.parameter`` for bad arguments or ``n == m``; ``input.out_of_domain``
        for a row of ``Y`` outside the active domain; ``input.shape``, ``input.dtype`` or
        ``input.nonfinite`` for a bad ``Y``.
    UnsupportedError
        ``unsupported.space`` for another kind of space.
    NumericalError
        ``numerical.infeasible_draw`` if a draw fails the feasibility check.
    SolverError
        ``solver.failed`` or ``solver.unbounded`` from the Chebyshev LP.

    Warns
    -----
    AccuracyWarning
        ``accuracy.correlated_draws`` for hit-and-run with ``thin == 1`` and
        ``n_draws > 1``; ``accuracy.degenerate_fiber`` once per call when a
        fiber is degenerate (a point on the boundary of the active domain).

    See Also
    --------
    forward_map : The forward direction.
    active_domain : The set of valid active points.

    Notes
    -----
    With :math:`x = W_1 y + W_2 z`, the map draws :math:`z` from its
    conditional distribution given :math:`y` under the reference density.

    - Gaussian space: :math:`z \mid y \sim N(0, I_{m-n})`.
    - Uniform space: :math:`x` is uniform on the fiber
      :math:`F(y) = \lbrace x \in [-1, 1]^m : W_1^\top x = y \rbrace`.
      When :math:`m - n = 1` the fiber is a segment
      :math:`W_1 y + t w_2`, :math:`t \in [t_{lo}, t_{hi}]`, sampled exactly.
      Otherwise hit-and-run [2]_ runs on the fiber from its Chebyshev center:
      from :math:`x`, draw :math:`u \sim N(0, I_m)`, set
      :math:`\delta = (I - W_1 W_1^\top) u`, and move to
      :math:`x + \lambda \delta` with :math:`\lambda` uniform on the exact
      chord within the cube. The Chebyshev center solves

      .. math::

          \max_{x, r} r \quad \text{s.t.} \quad W_1^\top x = y, \quad
          \pm x_i + \kappa_i r \le 1, \quad r \ge 0,

      with :math:`\kappa_i` the norm of row :math:`i` of :math:`W_2` [1]_.

    Draws from one chain are correlated; ``thin`` controls how strongly.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.
    .. [2] R. L. Smith. Efficient Monte Carlo procedures for generating points
       uniformly distributed over bounded regions. *Oper. Res.*
       32(6):1296-1308, 1984.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(3)
    >>> X = space.sample(200, rng=0)
    >>> a = np.array([1.0, 0.5, 0.2])
    >>> sub = ac.active_subspace(grads=np.outer(np.cos(X @ a), a), space=space, n=1)
    >>> res = ac.inverse_map(sub, Y=[[0.5], [-0.5]], n_draws=3, rng=0)
    >>> res.X.shape, res.sampler, res.burn_in, res.thin
    ((6, 3), 'hit_and_run', 20, 2)
    >>> bool(np.allclose(res.X @ sub.W1, res.Y[res.index]))
    True
    """
    subspace = _check_subspace(subspace, need_inactive=True)
    be = solvers.get_backend(backend)
    m, n = subspace.m, subspace.n
    d = m - n
    Yv = _active_points(Y, n)
    n_draws = _v.integer(n_draws, "n_draws", lo=1)
    if burn_in is not None:
        burn_in = _v.integer(burn_in, "burn_in", lo=0)
    if thin is not None:
        thin = _v.integer(thin, "thin", lo=1)
    boundary_tol = _v.real(boundary_tol, "boundary_tol", lo=0.0, lo_open=True)
    kind = _space_kind(subspace, "inverse_map")
    gen, rng_desc = _random.generator(rng)
    M = Yv.shape[0]
    children = _random.spawn(gen, M)

    W1 = np.array(subspace.W1, dtype=np.float64, order="C")
    W2 = np.array(subspace.W2, dtype=np.float64, order="C")
    sampler: Literal["gaussian", "exact_interval", "hit_and_run"]
    if kind == "gaussian":
        sampler = "gaussian"
        burn_in = thin = None
        X = np.empty((M * n_draws, m))
        for k in range(M):
            Zk = children[k].standard_normal((n_draws, d))
            X[k * n_draws : (k + 1) * n_draws] = (W1 @ Yv[k])[None, :] + Zk @ W2.T
        degenerate = np.zeros(M, dtype=bool)
    elif d == 1:
        sampler = "exact_interval"
        burn_in = thin = None
        X, degenerate = _exact_interval(Yv, W1, W2[:, 0], children, n_draws, boundary_tol)
    else:
        sampler = "hit_and_run"
        burn_in = 10 * d if burn_in is None else burn_in
        thin = d if thin is None else thin
        if thin == 1 and n_draws > 1:
            _warn(
                AccuracyWarning,
                "inverse_map: hit-and-run draws with thin=1 are correlated",
                code="accuracy.correlated_draws",
                details={"thin": thin, "n_draws": n_draws},
            )
        X, degenerate = _hit_and_run(
            Yv, W1, W2, children, n_draws, burn_in, thin, boundary_tol, be
        )

    if degenerate.any():
        rows = [int(k) for k in np.flatnonzero(degenerate)[:_MAX_WARN_ROWS]]
        _warn(
            AccuracyWarning,
            f"inverse_map: {int(degenerate.sum())} point(s) of Y lie on the boundary of the "
            "active domain; "
            "their fibers have empty interior and every draw is the same point",
            code="accuracy.degenerate_fiber",
            details={"rows": rows},
        )
    params = {
        "n_draws": n_draws,
        "burn_in": burn_in,
        "thin": thin,
        "boundary_tol": boundary_tol,
        "backend": str(getattr(be, "name", type(be).__name__)),
    }
    prov = _make_provenance(
        "inverse_map",
        params,
        space=subspace.space,
        rng_state=rng_desc,
        digests={"Y": _v.digest(Yv)},
    )
    return InverseMapResult(
        subspace=subspace,
        Y=Yv,
        X=X,
        index=np.repeat(np.arange(M, dtype=np.int64), n_draws),
        n_draws=n_draws,
        sampler=sampler,
        burn_in=burn_in,
        thin=thin,
        degenerate=degenerate,
        provenance=prov,
    )
