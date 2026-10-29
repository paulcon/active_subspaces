r"""Designs on the active domain and conditional sampling of a model.

:func:`active_design` builds a point set on an
:class:`~active_subspaces.ActiveDomain`, in normalized active coordinates:
equispaced interior points on an interval, a maximin design on a zonotope, and
tensor Gauss-Hermite nodes on an unbounded (Gaussian) domain.
:func:`conditional_samples` evaluates a model at draws of the inactive
variables for each design point, and at the domain's vertices; it is the
model-evaluation half of :func:`~active_subspaces.fit_surrogate_from_model`.
"""

from __future__ import annotations

import concurrent.futures
import math
from typing import Any, Literal, cast

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

from active_subspaces import _quadrature, _random
from active_subspaces import _validation as _v
from active_subspaces.errors import (
    ConvergenceWarning,
    InputError,
    SolverError,
    UnsupportedError,
    _warn,
)
from active_subspaces.geometry import ActiveDomain, in_active_domain, inverse_map
from active_subspaces.models import Model, ValueFn, as_model, evaluate
from active_subspaces.results import (
    QoIResult,
    Result,
    _invariant,
    _make_provenance,
    _register_result,
)
from active_subspaces.spaces import Uniform

__all__ = ("ActiveDesign", "ConditionalSamples", "active_design", "conditional_samples")

_KINDS = ("interval", "zonotope", "unbounded")
_FEAS_TOL = 1e-9  # facet slack of a usable maximin start, scaled units (§5.3 step 5)
_REJECT_TOL = 1e-9  # interior margin of a rejection-sampled start, times rho (§5.3 step 3)
_MAX_BATCHES = 10_000
_MAX_ROUNDS = 20
# Epigraph SLSQP while its constraint count N(N-1)/2 + N N_f is at most this; greedy
# farthest-point above [D-170, D-177].
_SLSQP_MAX_CONSTRAINTS = 2000
_GREEDY_CANDIDATES = 50  # candidates per design point in the greedy design [D-170]
_SLSQP_OPTIONS = {"maxiter": 200, "ftol": 1e-9}


# ---------------------------------------------------------------------------
# Result types (§4.4)
# ---------------------------------------------------------------------------


@_register_result
class ActiveDesign(Result):
    r"""A point set on an active domain, in normalized active coordinates.

    Built by :func:`active_design`, or by :func:`conditional_samples` for a
    given design; the constructor is used when a saved result is loaded.

    Parameters
    ----------
    domain : ActiveDomain
        The active domain the points lie in.
    points : ndarray of float64, shape (N, n)
        The design points in normalized active coordinates
        :math:`y = W_1^\top x`. :math:`N` may be less than the number asked
        for (Gauss-Hermite designs).
    method : {"interval", "maximin_slsqp", "maximin_greedy", "gauss_hermite", "given"}
        How the points were built: equispaced on an interval; the maximin
        design by SLSQP, or its greedy farthest-point version for large
        designs; tensor Gauss-Hermite nodes; or given by the user.
    min_distance : float or None
        Maximin designs only: the smallest distance among the points and
        from the points to the vertices of the domain; else ``None``.
    provenance : Provenance
        How the result was computed.

    See Also
    --------
    active_design : Build a design.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(3)
    >>> sub = ac.subspace_from_basis([[1.0], [0.5], [0.0]], space=space)
    >>> design = ac.active_design(domain=ac.active_domain(sub), n_points=3)
    >>> design.method
    'interval'
    >>> print(np.round(design.points, 3) + 0.0)
    [[-0.671]
     [ 0.   ]
     [ 0.671]]
    """

    TYPE = "ActiveDesign"
    SCHEMA_VERSION = 1

    domain: ActiveDomain
    points: NDArray[np.float64]
    method: Literal["interval", "maximin_slsqp", "maximin_greedy", "gauss_hermite", "given"]
    min_distance: float | None

    def _check(self) -> None:
        n = self.domain.subspace.n
        P = self.points
        _invariant(
            P.ndim == 2 and P.shape[0] >= 1 and P.shape[1] == n,
            "points must be (N, n) with N >= 1",
        )
        _invariant(bool(np.isfinite(P).all()), "points must be finite")
        if self.method in ("maximin_slsqp", "maximin_greedy"):
            _invariant(
                self.min_distance is not None
                and math.isfinite(self.min_distance)
                and self.min_distance >= 0,
                "maximin needs min_distance >= 0",
            )
        else:
            _invariant(
                self.min_distance is None,
                "min_distance is None unless the design is a maximin design",
            )
        if self.domain.kind != "unbounded":
            b = cast(NDArray[np.float64], self.domain.facet_offsets)
            nu = cast(NDArray[np.float64], self.domain.facet_normals)
            slack = P @ nu.T - b
            _invariant(
                bool(np.all(slack <= _FEAS_TOL * (1.0 + float(np.max(b))))),
                "points must lie in the active domain",
            )
        else:
            _invariant(
                self.method in ("gauss_hermite", "given"),
                "an unbounded domain has a gauss_hermite or given design",
            )


@_register_result
class ConditionalSamples(QoIResult):
    r"""Model values at draws of the inactive variables for each design point and vertex.

    Built by :func:`conditional_samples` (and kept by
    :func:`~active_subspaces.fit_surrogate_from_model` in its ``samples``);
    the constructor is used when a saved result is loaded. Rows of `Y` are
    the :math:`K` points (design points, then vertices); rows of `X` are the
    :math:`L` evaluated inputs, grouped by point.

    Parameters
    ----------
    domain : ActiveDomain
        The active domain.
    Y : ndarray of float64, shape (K, n)
        Design points, then vertices, in normalized active coordinates.
    is_vertex : ndarray of bool, shape (K,)
        Whether each row of `Y` is a vertex of the domain.
    X : ndarray of float64, shape (L, m)
        The evaluated inputs in **normalized** coordinates, point-major.
    f_values : ndarray of float64, shape (L,)
        Model values at `X`.
    index : ndarray of int64, shape (L,)
        The row of `Y` that each row of `X` belongs to (nondecreasing).
    counts : ndarray of int64, shape (K,)
        Evaluations per point: `n_mc`, or 1 at a vertex with a unique
        preimage.
    means : ndarray of float64, shape (K,)
        Conditional means :math:`\bar f_k`.
    variances : ndarray of float64, shape (K,)
        Sample variances :math:`\hat V_k` (ddof 1; 0 where ``counts == 1``).
    n_mc : int
        Draws per design point, at least 2.
    provenance : Provenance
        How the result was computed.
    qoi : str, default "f"
        Name of the quantity of interest.

    See Also
    --------
    conditional_samples : Build the samples.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(3)
    >>> model = ac.test_functions.ridge([1.0, 0.5, 1.0], space=space)
    >>> sub = ac.subspace_from_basis([[1.0], [0.5], [0.0]], space=space)
    >>> cs = ac.conditional_samples(
    ...     domain=ac.active_domain(sub), model=model, n_design=3, n_mc=4, rng=0
    ... )
    >>> cs.Y.shape, cs.X.shape
    ((5, 1), (20, 3))
    >>> cs.is_vertex.tolist()
    [False, False, False, True, True]
    """

    TYPE = "ConditionalSamples"
    SCHEMA_VERSION = 1

    domain: ActiveDomain
    Y: NDArray[np.float64]
    is_vertex: NDArray[np.bool_]
    X: NDArray[np.float64]
    f_values: NDArray[np.float64]
    index: NDArray[np.int64]
    counts: NDArray[np.int64]
    means: NDArray[np.float64]
    variances: NDArray[np.float64]
    n_mc: int

    @property
    def noise_variance(self) -> NDArray[np.float64]:
        r"""Variance of each conditional mean, :math:`\hat V_k / \text{counts}_k`.

        Returns
        -------
        ndarray of float64, shape (K,)
            ``variances / counts``, a new array; in units of :math:`f^2`.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform.reference(3)
        >>> model = ac.test_functions.ridge([1.0, 0.5, 1.0], space=space)
        >>> sub = ac.subspace_from_basis([[1.0], [0.5], [0.0]], space=space)
        >>> cs = ac.conditional_samples(
        ...     domain=ac.active_domain(sub), model=model, n_design=3, n_mc=4, rng=0
        ... )
        >>> bool(np.allclose(cs.noise_variance, cs.variances / 4))
        True
        """
        return self.variances / self.counts

    @property
    def n_samples(self) -> int:
        """Number of model evaluations :math:`L`.

        Returns
        -------
        int
            The number of rows of ``X``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform.reference(3)
        >>> model = ac.test_functions.ridge([1.0, 0.5, 1.0], space=space)
        >>> sub = ac.subspace_from_basis([[1.0], [0.5], [0.0]], space=space)
        >>> cs = ac.conditional_samples(
        ...     domain=ac.active_domain(sub), model=model, n_design=3, n_mc=4, rng=0
        ... )
        >>> cs.n_samples
        20
        """
        return int(self.X.shape[0])

    def _check(self) -> None:
        sub = self.domain.subspace
        n, m = sub.n, sub.space.dim
        K = self.Y.shape[0] if self.Y.ndim == 2 else -1
        L = self.X.shape[0] if self.X.ndim == 2 else -1
        _invariant(self.Y.shape == (K, n) and K >= 1, "Y must be (K, n)")
        _invariant(self.X.shape == (L, m) and L >= 1, "X must be (L, m)")
        for name, shape in (
            ("is_vertex", (K,)),
            ("counts", (K,)),
            ("means", (K,)),
            ("variances", (K,)),
            ("f_values", (L,)),
            ("index", (L,)),
        ):
            _invariant(getattr(self, name).shape == shape, f"{name} must have shape {shape}")
        for name in ("Y", "X", "f_values", "means", "variances"):
            _invariant(bool(np.isfinite(getattr(self, name)).all()), f"{name} must be finite")
        _invariant(self.n_mc >= 2, "n_mc must be >= 2")
        iv = self.is_vertex
        _invariant(
            bool(np.all(np.diff(iv.astype(np.int64)) >= 0)),
            "vertices must follow the design points",
        )
        c = self.counts
        _invariant(bool(np.all(c[~iv] == self.n_mc)), "design points have n_mc draws")
        _invariant(
            bool(np.all((c[iv] == self.n_mc) | (c[iv] == 1))), "vertices have n_mc draws or one"
        )
        idx = self.index
        _invariant(bool(np.all(idx >= 0)) and bool(np.all(idx < K)), "index must lie in [0, K)")
        _invariant(bool(np.all(np.diff(idx) >= 0)), "index must be nondecreasing")
        _invariant(
            np.array_equal(np.bincount(idx, minlength=K), c),
            "index must take each value counts[k] times",
        )
        means, var = _reduce(self.f_values, idx, c)
        scale = max(1.0, float(np.max(np.abs(self.f_values))))
        _invariant(
            bool(np.allclose(self.means, means, rtol=1e-12, atol=1e-12 * scale)),
            "means must agree with f_values",
        )
        _invariant(
            bool(np.allclose(self.variances, var, rtol=1e-12, atol=1e-12 * scale * scale)),
            "variances must agree with f_values",
        )
        _invariant(
            bool(np.all(self.variances >= 0)) and bool(np.all(self.variances[c == 1] == 0)),
            "variances must be >= 0 and 0 where counts == 1",
        )


def _reduce(f: np.ndarray, index: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point means and ddof-1 variances (0 where a point has one value), in row order."""
    K = counts.shape[0]
    means = np.bincount(index, weights=f, minlength=K) / counts
    dev = f - means[index]
    ss = np.bincount(index, weights=dev * dev, minlength=K)
    var = np.where(counts > 1, ss / np.maximum(counts - 1, 1), 0.0)
    return means, var


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _param(message: str, param: str, **details: Any) -> InputError:
    return InputError(message, code="input.parameter", details={"param": param, **details})


def _check_domain(domain: Any) -> ActiveDomain:
    if not isinstance(domain, ActiveDomain):
        raise _param(f"domain: expected an ActiveDomain, got {type(domain).__name__}", "domain")
    return domain


# ---------------------------------------------------------------------------
# Designs (§3.6, §5.3)
# ---------------------------------------------------------------------------


def _interval(domain: ActiveDomain, N: int) -> np.ndarray:
    a, b = sorted(float(v) for v in np.asarray(domain.vertices)[:, 0])
    k = np.arange(1, N + 1, dtype=np.float64)
    return (a + k * ((b - a) / (N + 1))).reshape(N, 1)


def _int_root(N: int, n: int) -> int:
    """Largest ``q`` with ``q**n <= N`` (exact integer arithmetic)."""
    q: int = max(1, round(N ** (1.0 / n)))
    while q**n > N:
        q -= 1
    while (q + 1) ** n <= N:
        q += 1
    return q


def _gauss_hermite(n: int, N: int) -> np.ndarray:
    q = _int_root(N, n)
    nodes, _ = _quadrature.tensor_rule([_quadrature.gauss_hermite(q)] * n)
    return nodes


def _sq_dists(Y: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Squared distances of all point pairs ``i < j`` and point-vertex pairs.

    Pairs in ``triu_indices`` order; point-vertex pairs point-major.
    """
    N = Y.shape[0]
    iu, ju = np.triu_indices(N, 1)
    dpp = np.sum((Y[iu] - Y[ju]) ** 2, axis=1)
    dpv = np.sum((Y[:, None, :] - V[None, :, :]) ** 2, axis=2).ravel()
    return dpp, dpv


def _min_sq(Y: np.ndarray, V: np.ndarray) -> float:
    dpp, dpv = _sq_dists(Y, V)
    return float(np.min(np.concatenate([dpp, dpv])))


def _slsqp(x0: np.ndarray, cons: Any, cons_jac: Any, bounds: Any = None) -> np.ndarray:
    """One SLSQP solve of ``max t`` (objective ``-t``) under ``cons(x) >= 0``.

    Returns the final iterate.
    """
    nvar = x0.shape[0]
    grad = np.zeros(nvar)
    grad[-1] = -1.0

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        return -float(x[-1]), grad

    res = scipy.optimize.minimize(
        fun,
        x0,
        jac=True,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
        bounds=bounds,
        options=dict(_SLSQP_OPTIONS),
    )
    return np.asarray(res.x, dtype=np.float64)


def _maximin_local(Y0: np.ndarray, V: np.ndarray, nu: np.ndarray, bt: np.ndarray) -> np.ndarray:
    """Epigraph maximin from ``Y0`` with distance constraints generated in rounds (§5.3 step 4).

    Scaled units.
    """
    N, n = Y0.shape
    Nv, Nf = V.shape[0], nu.shape[0]
    iu, ju = np.triu_indices(N, 1)
    pi, pv = np.repeat(np.arange(N), Nv), np.tile(np.arange(Nv), N)
    nvar = N * n + 1
    Jf = np.zeros((N * Nf, nvar))
    for i in range(N):
        Jf[i * Nf : (i + 1) * Nf, i * n : (i + 1) * n] = -nu
    bf = np.tile(bt, N)
    # Box bounds on y implied by the facets (the bounding box of the scaled zonotope); t is free.
    # They change no solution, but keep SLSQP's iterates from diverging. Bounding t as well made
    # some linearized subproblems incompatible.
    box = np.max(np.abs(V), axis=0)
    bounds = scipy.optimize.Bounds(
        np.concatenate([np.tile(-box, N), [-np.inf]]), np.concatenate([np.tile(box, N), [np.inf]])
    )
    dpp, dpv = _sq_dists(Y0, V)
    t = float(np.min(np.concatenate([dpp, dpv])))
    # First-round pairs: within 4 max(t0, t_est), with t_est = (vol(box) / N)^(2/n) a packing
    # estimate of the optimal t (an overestimate, so it errs towards more pairs). With t0 alone a
    # random start's tiny t0 leaves the first solve nearly unconstrained.
    t_est = float(np.prod(2.0 * box) / N) ** (2.0 / n)
    thr = 4.0 * max(t, t_est)
    act_pp, act_pv = dpp <= thr, dpv <= thr
    x0 = np.concatenate([Y0.ravel(), [t]])
    x = x0
    cols = np.arange(n)
    for _ in range(_MAX_ROUNDS):
        Ii, Jj = iu[act_pp], ju[act_pp]
        P, Q = pi[act_pv], pv[act_pv]
        npp, npv = Ii.shape[0], P.shape[0]

        def cons(
            x: np.ndarray,
            Ii: np.ndarray = Ii,
            Jj: np.ndarray = Jj,
            P: np.ndarray = P,
            Q: np.ndarray = Q,
        ) -> np.ndarray:
            Y, tt = x[:-1].reshape(N, n), x[-1]
            return np.concatenate(
                [
                    np.sum((Y[Ii] - Y[Jj]) ** 2, axis=1) - tt,
                    np.sum((Y[P] - V[Q]) ** 2, axis=1) - tt,
                    bf - (Y @ nu.T).ravel(),
                ]
            )

        def cons_jac(
            x: np.ndarray,
            Ii: np.ndarray = Ii,
            Jj: np.ndarray = Jj,
            P: np.ndarray = P,
            Q: np.ndarray = Q,
            npp: int = npp,
            npv: int = npv,
        ) -> np.ndarray:
            Y = x[:-1].reshape(N, n)
            Jac = np.zeros((npp + npv + N * Nf, nvar))
            r = np.arange(npp)
            d = 2.0 * (Y[Ii] - Y[Jj])
            Jac[r[:, None], Ii[:, None] * n + cols] = d
            Jac[r[:, None], Jj[:, None] * n + cols] = -d
            Jac[r, -1] = -1.0
            r2 = npp + np.arange(npv)
            Jac[r2[:, None], P[:, None] * n + cols] = 2.0 * (Y[P] - V[Q])
            Jac[r2, -1] = -1.0
            Jac[npp + npv :] = Jf
            return Jac

        # Each round restarts from the start design: an iterate of an earlier round may have merged
        # points whose pair was omitted, and at coincident points the distance constraint has a
        # zero
        # gradient, so SLSQP could not separate them again (a stationary point with t = 0).
        x = _slsqp(x0, cons, cons_jac, bounds)
        Y, t = x[:-1].reshape(N, n), float(x[-1])
        if not np.isfinite(x).all():
            break
        dpp, dpv = _sq_dists(Y, V)
        if not (np.any(~act_pp & (dpp < t)) or np.any(~act_pv & (dpv < t))):
            break
        act_pp |= dpp < 4.0 * t
        act_pv |= dpv < 4.0 * t
    return x[:-1].reshape(N, n).copy()


def _rejection_start(
    gen: np.random.Generator, beta: np.ndarray, nu: np.ndarray, b: np.ndarray, N: int, rho: float
) -> np.ndarray:
    """``N`` points uniform in the zonotope by rejection from ``[-beta, beta]``, in draw order.

    See §5.3 step 3.
    """
    n = beta.shape[0]
    kept: list[np.ndarray] = []
    total = 0
    for _ in range(_MAX_BATCHES):
        Z = gen.uniform(-beta, beta, size=(N, n))
        ok = np.max(Z @ nu.T - b, axis=1) <= -_REJECT_TOL * rho
        if ok.any():
            kept.append(Z[ok])
            total += int(ok.sum())
        if total >= N:
            return np.concatenate(kept, axis=0)[:N]
    raise SolverError(
        "maximin design: rejection sampling of a start exhausted its batches",
        code="solver.failed",
        details={"batches": _MAX_BATCHES, "kept": total, "required": N},
    )


def _maximin(
    domain: ActiveDomain, N: int, n_starts: int, gen: np.random.Generator
) -> tuple[np.ndarray, float]:
    Vt = np.asarray(domain.vertices, dtype=np.float64)
    nu = np.asarray(domain.facet_normals, dtype=np.float64)
    b = np.asarray(domain.facet_offsets, dtype=np.float64)
    rho = float(np.max(np.linalg.norm(Vt, axis=1)))
    beta = np.max(np.abs(Vt), axis=0)
    V, bt = Vt / rho, b / rho
    children = _random.spawn(gen, n_starts)
    best, best_val, failed = None, -math.inf, 0
    for s in range(n_starts):
        Y0 = _rejection_start(children[s], beta, nu, b, N, rho) / rho
        Y = _maximin_local(Y0, V, nu, bt)
        if not (np.isfinite(Y).all() and np.all(Y @ nu.T - bt <= _FEAS_TOL)):
            failed += 1
            continue
        val = _min_sq(Y, V)
        if val > best_val:
            best, best_val = Y, val
    if best is None:
        raise SolverError(
            f"maximin design: none of the {n_starts} starts gave a feasible design",
            code="solver.failed",
            details={"n_starts": n_starts, "failed": failed},
        )
    if failed:
        _warn(
            ConvergenceWarning,
            f"maximin design: {failed} of {n_starts} starts left the domain and were discarded",
            code="convergence.failed_starts",
            details={"failed": failed, "n_starts": n_starts},
        )
    Y = best * rho
    proj = Y @ nu.T
    out = proj > b
    rows = np.flatnonzero(out.any(axis=1))
    for i in rows:
        Y[i] *= float(np.min(b[out[i]] / proj[i, out[i]]))
    return Y, math.sqrt(_min_sq(Y, Vt))


def _maximin_greedy(
    domain: ActiveDomain, N: int, gen: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Greedy farthest-point design [D-170].

    From the vertices, add the candidate farthest from the set, N times.

    The ``50 N`` candidates are uniform in the domain, by rejection from the
    bounding box, drawn from ``gen`` in batches of ``50 N`` rows.
    """
    Vt = np.asarray(domain.vertices, dtype=np.float64)
    nu = np.asarray(domain.facet_normals, dtype=np.float64)
    b = np.asarray(domain.facet_offsets, dtype=np.float64)
    rho = float(np.max(np.linalg.norm(Vt, axis=1)))
    beta = np.max(np.abs(Vt), axis=0)
    C = _rejection_start(gen, beta, nu, b, _GREEDY_CANDIDATES * N, rho)
    d2 = np.min(np.sum((C[:, None, :] - Vt[None, :, :]) ** 2, axis=2), axis=1)
    chosen = np.empty(N, dtype=np.int64)
    for i in range(N):
        c = int(np.argmax(d2))  # ties to the lower candidate index
        chosen[i] = c
        d2 = np.minimum(d2, np.sum((C - C[c]) ** 2, axis=1))
    Y = np.ascontiguousarray(C[chosen])
    return Y, math.sqrt(_min_sq(Y, Vt))


_Method = Literal["interval", "maximin_slsqp", "maximin_greedy", "gauss_hermite", "given"]


def _design_points(
    domain: ActiveDomain, N: int, n_starts: int, gen: np.random.Generator
) -> tuple[np.ndarray, _Method, float | None]:
    kind = domain.kind
    if kind not in _KINDS:
        raise UnsupportedError(
            f"active_design: domain kind {kind!r} is not supported",
            code="unsupported.space",
            details={"kind": kind},
        )
    if kind == "interval":
        return _interval(domain, N), "interval", None
    if kind == "unbounded":
        return _gauss_hermite(domain.subspace.n, N), "gauss_hermite", None
    n_facets = int(np.asarray(domain.facet_normals).shape[0])
    if N * (N - 1) // 2 + N * n_facets > _SLSQP_MAX_CONSTRAINTS:
        Y, dmin = _maximin_greedy(domain, N, gen)
        return Y, "maximin_greedy", dmin
    Y, dmin = _maximin(domain, N, n_starts, gen)
    return Y, "maximin_slsqp", dmin


def active_design(
    *,
    domain: ActiveDomain,
    n_points: int,
    n_starts: int = 3,
    rng: _random.RngLike = None,
) -> ActiveDesign:
    r"""Build a design of `n_points` points on an active domain.

    The kind of design follows the kind of domain: equispaced interior points
    on an interval (:math:`n = 1`), a maximin design on a zonotope
    (:math:`n \ge 2`), and tensor Gauss-Hermite nodes on an unbounded
    (Gaussian) domain.

    Parameters
    ----------
    domain : ActiveDomain
        The active domain (from :func:`~active_subspaces.active_domain`).
    n_points : int
        Number of design points :math:`N_q \ge 1`.
    n_starts : int, default 3
        Starts of the maximin design (zonotope, SLSQP branch only), each on
        its own child stream of `rng`.
    rng : None, int, SeedSequence or Generator
        Seed for the maximin design. Interval and Gauss-Hermite designs draw
        nothing; `rng` is still recorded.

    Returns
    -------
    ActiveDesign
        The points in normalized active coordinates, the method used and,
        for maximin designs, the smallest distance.

    Raises
    ------
    InputError
        ``input.parameter`` if `domain` is not an ActiveDomain or
        `n_points` or `n_starts` is not an integer :math:`\ge 1`.
    UnsupportedError
        ``unsupported.space`` if the domain kind has no design.
    SolverError
        ``solver.failed`` if no maximin start is usable, or rejection
        sampling of a start fails.

    Warns
    -----
    ConvergenceWarning
        ``convergence.failed_starts`` if some, but not all, maximin starts
        are unusable.

    See Also
    --------
    conditional_samples : Evaluate a model on a design.

    Notes
    -----
    **Interval** :math:`[a, b]`: :math:`y_k = a + k\,(b - a)/(N_q + 1)`,
    :math:`k = 1, \dots, N_q` (endpoints excluded). Method ``"interval"``.

    **Zonotope**: with vertices :math:`\hat y_v` and facets
    :math:`\nu_l^\top y \le b_l`, the maximin design solves, in epigraph form,

    .. math::

       \max_{y_1, \dots, y_{N_q},\, t}\ t \quad\text{s.t.}\quad
       \lVert y_i - y_j\rVert^2 \ge t\ (i < j),\quad
       \lVert y_i - \hat y_v\rVert^2 \ge t,\quad
       \nu_l^\top y_i \le b_l ,

    by SLSQP from `n_starts` feasible random starts, keeping the best usable
    one (method ``"maximin_slsqp"``). The 2016 package maximized the
    non-smooth minimum distance directly from starts that could lie outside
    the domain. When the program's constraint count
    :math:`N_q(N_q - 1)/2 + N_q N_f` exceeds 2000 (:math:`N_f` facets) the design is instead
    built greedily (method ``"maximin_greedy"``): starting from the
    vertices, it adds :math:`N_q` times the candidate farthest from the
    points chosen so far, among :math:`50 N_q` points uniform in the domain.

    **Unbounded**: the tensor Gauss-Hermite nodes for :math:`N(0, I_n)` with
    :math:`q` points per dimension, :math:`q` the largest integer with
    :math:`q^n \le N_q`, so the design may have fewer than :math:`N_q`
    points. Method ``"gauss_hermite"``.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(3)
    >>> sub = ac.subspace_from_basis([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], space=space)
    >>> dom = ac.active_domain(sub)
    >>> design = ac.active_design(domain=dom, n_points=5, rng=0)
    >>> dom.kind, design.method, design.points.shape
    ('zonotope', 'maximin_slsqp', (5, 2))
    >>> bool(np.all(ac.in_active_domain(dom, Y=design.points)))
    True

    On a Gaussian space the design is a tensor Gauss-Hermite grid.

    >>> gauss = ac.Gaussian(np.zeros(3), np.eye(3))
    >>> sub = ac.subspace_from_basis([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], space=gauss)
    >>> design = ac.active_design(domain=ac.active_domain(sub), n_points=10)
    >>> design.method, design.points.shape
    ('gauss_hermite', (9, 2))
    """
    domain = _check_domain(domain)
    n_points = _v.integer(n_points, "n_points", lo=1)
    n_starts = _v.integer(n_starts, "n_starts", lo=1)
    gen, rng_desc = _random.generator(rng)
    points, method, dmin = _design_points(domain, n_points, n_starts, gen)
    prov = _make_provenance(
        "active_design",
        {"domain": domain, "n_points": n_points, "n_starts": n_starts},
        space=domain.subspace.space,
        rng_state=rng_desc,
    )
    return ActiveDesign(
        domain=domain, points=points, method=method, min_distance=dmin, provenance=prov
    )


# ---------------------------------------------------------------------------
# Conditional samples (§3.5, §5.4)
# ---------------------------------------------------------------------------


def _given_design(domain: ActiveDomain, design: Any) -> np.ndarray:
    n = domain.subspace.n
    if isinstance(design, ActiveDesign):
        if not design.domain == domain:
            raise _param(
                "design: its domain differs from domain", "design", reason="domain_mismatch"
            )
        return np.array(design.points, dtype=np.float64, copy=True)
    arr = _v._as_2d(_v.coerce(design, "design"), "design", n, allow_1d=True)
    _v.check_finite(arr, "design")
    arr = np.array(arr, dtype=np.float64, order="C", copy=True)
    inside = np.asarray(in_active_domain(domain, Y=arr))
    if not inside.all():
        row = int(np.argmin(inside))
        raise InputError(
            f"design: row {row} lies outside the active domain",
            code="input.out_of_domain",
            details={"param": "design", "row": row},
        )
    return arr


def _conditional_samples(
    *,
    domain: Any,
    model: Any,
    design: Any,
    n_design: Any,
    n_mc: Any,
    include_vertices: Any,
    n_starts: Any,
    executor: Any,
    batch_size: Any,
    qoi: Any,
    gen: np.random.Generator,
    rng_desc: dict[str, Any],
) -> tuple[ConditionalSamples, Model, dict[str, Any]]:
    """Shared body of :func:`conditional_samples` and ``fit_surrogate_from_model``.

    ``gen`` is the entry generator.
    """
    domain = _check_domain(domain)
    sub = domain.subspace
    space = sub.space
    model = as_model(model, space=space)
    n_mc = _v.integer(n_mc, "n_mc", lo=2)
    if n_design is not None:
        n_design = _v.integer(n_design, "n_design", lo=1)
    if design is not None and n_design is not None:
        raise _param("give at most one of design and n_design", "n_design")
    include_vertices = _v.boolean(include_vertices, "include_vertices")
    n_starts = _v.integer(n_starts, "n_starts", lo=1)
    if batch_size is not None:
        batch_size = _v.integer(batch_size, "batch_size", lo=1)
    if executor is not None and not isinstance(executor, concurrent.futures.Executor):
        raise _param(
            "executor: expected None or a concurrent.futures.Executor, got "
            f"{type(executor).__name__}",
            "executor",
        )
    if qoi is None:
        qoi = sub.qoi
    elif not isinstance(qoi, str) or not qoi:
        raise _param("qoi: expected a non-empty string", "qoi")
    elif qoi != sub.qoi:
        raise _param(
            f"qoi: {qoi!r} differs from the subspace's {sub.qoi!r}", "qoi", reason="qoi_mismatch"
        )
    given = None if design is None else _given_design(domain, design)

    s_design, s_inverse, s_vertex = _random.spawn(gen, 3)
    n = sub.n
    if given is None:
        pts = active_design(
            domain=domain,
            n_points=n_design if n_design is not None else 5**n,
            n_starts=n_starts,
            rng=s_design,
        ).points
        pts = np.array(pts, dtype=np.float64, copy=True)
    else:
        pts = given
    Nq = pts.shape[0]
    inv = inverse_map(sub, Y=pts, n_draws=n_mc, rng=s_inverse)
    X_parts = [np.asarray(inv.X, dtype=np.float64)]
    idx_parts = [np.asarray(inv.index, dtype=np.int64)]
    Y_parts = [pts]
    counts = [np.full(Nq, n_mc, dtype=np.int64)]
    is_vertex = [np.zeros(Nq, dtype=bool)]
    if include_vertices and domain.kind in ("interval", "zonotope"):
        verts = np.asarray(domain.vertices, dtype=np.float64)
        pre = np.asarray(domain.vertex_preimages, dtype=np.float64)
        Nv = verts.shape[0]
        vchild = _random.spawn(s_vertex, Nv)
        vcounts = np.empty(Nv, dtype=np.int64)
        for v in range(Nv):
            sigma = pre[v]
            free = np.flatnonzero(sigma == 0)
            if free.size == 0:
                rows = sigma[None, :].copy()
            else:
                rows = np.tile(sigma, (n_mc, 1))
                rows[:, free] = vchild[v].uniform(-1.0, 1.0, size=(n_mc, free.size))
            X_parts.append(rows)
            idx_parts.append(np.full(rows.shape[0], Nq + v, dtype=np.int64))
            vcounts[v] = rows.shape[0]
        Y_parts.append(verts)
        counts.append(vcounts)
        is_vertex.append(np.ones(Nv, dtype=bool))
    X = np.ascontiguousarray(np.concatenate(X_parts, axis=0))
    index = np.concatenate(idx_parts)
    Xp = space.from_normalized(X)
    if isinstance(space, Uniform):
        Xp = np.clip(Xp, space.lower, space.upper)
    fvals = evaluate(model, X=Xp, space=space, executor=executor, batch_size=batch_size)
    cnt = np.concatenate(counts)
    means, var = _reduce(fvals, index, cnt)
    params = {
        "domain": domain,
        "model": model.name,
        "design": design
        if isinstance(design, ActiveDesign) or design is None
        else _v.digest(given),
        "n_design": n_design,
        "n_mc": n_mc,
        "include_vertices": include_vertices,
        "n_starts": n_starts,
        "batch_size": batch_size,
        "qoi": qoi,
    }
    prov = _make_provenance("conditional_samples", params, space=space, rng_state=rng_desc)
    samples = ConditionalSamples(
        domain=domain,
        Y=np.concatenate(Y_parts, axis=0),
        is_vertex=np.concatenate(is_vertex),
        X=X,
        f_values=fvals,
        index=index,
        counts=cnt,
        means=means,
        variances=var,
        n_mc=n_mc,
        qoi=qoi,
        provenance=prov,
    )
    return samples, model, params


def conditional_samples(
    *,
    domain: ActiveDomain,
    model: Model | ValueFn,
    design: ActiveDesign | ArrayLike | None = None,
    n_design: int | None = None,
    n_mc: int = 50,  # [D-171]
    include_vertices: bool = True,
    n_starts: int = 3,
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
    qoi: str | None = None,
    rng: _random.RngLike = None,
) -> ConditionalSamples:
    r"""Evaluate a model at draws of the inactive variables for each design point and vertex.

    For each design point :math:`y_k`, `n_mc` inactive points
    :math:`z_{kj}` are drawn from the conditional density
    :math:`\pi(z \mid y_k)` with :func:`~active_subspaces.inverse_map`, and
    the model is evaluated at :math:`x_{kj} = W_1 y_k + W_2 z_{kj}`. On a
    bounded domain the vertices of the domain follow the design points.

    Parameters
    ----------
    domain : ActiveDomain
        The active domain (from :func:`~active_subspaces.active_domain`).
    model : Model or callable
        A :class:`~active_subspaces.Model` on ``domain.subspace.space``, or a
        bare callable mapping physical points ``(L, m)`` to values ``(L,)``.
    design : ActiveDesign or array_like, shape (N_q, n), optional
        The design in normalized active coordinates; given points must lie
        in the domain. Default: :func:`active_design` with `n_design`
        points.
    n_design : int, optional
        Number of design points when `design` is not given; default
        :math:`5^n`. Do not give both.
    n_mc : int, default 50
        Draws of the inactive variables per design point, at least 2.
    include_vertices : bool, default True
        Add the vertices of a bounded domain.
    n_starts : int, default 3
        Starts of the maximin design, when it is built.
    executor : concurrent.futures.Executor, optional
        Runs the model evaluations; results do not depend on it.
    batch_size : int, optional
        Points per model call.
    qoi : str, optional
        Name of the quantity of interest; defaults to
        ``domain.subspace.qoi`` (a given value must equal it).
    rng : None, int, SeedSequence or Generator
        Seed. It is split into three independent streams: the design, the
        inactive draws and the vertex draws.

    Returns
    -------
    ConditionalSamples
        The points, the evaluated inputs (normalized), the model values, and
        the conditional means and variances per point.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad argument, a model on a different space,
        a `qoi` that does not match, an :class:`ActiveDesign` on another
        domain, or both `design` and `n_design`; ``input.shape``,
        ``input.dtype`` or ``input.nonfinite`` for a malformed `design`;
        ``input.out_of_domain`` if a design point lies outside the domain.
    ModelError
        ``model.*`` if the model raises or returns malformed or non-finite
        values.
    SolverError
        ``solver.failed`` if no maximin start is usable.
    UnsupportedError
        ``unsupported.space`` if the domain kind has no design.

    Warns
    -----
    ConvergenceWarning
        ``convergence.failed_starts`` if some maximin starts are unusable.
    AccuracyWarning
        ``accuracy.degenerate_fiber`` (from the inverse map) if a design point
        lies on the boundary of the domain.

    See Also
    --------
    active_subspaces.fit_surrogate_from_model : Fit a surrogate to the means.
    active_design : Designs on the active domain.

    Notes
    -----
    Per point,

    .. math::

       \bar f_k = \frac1{N_k} \sum_{j=1}^{N_k} f_{kj},\qquad
       \hat V_k = \frac1{N_k - 1} \sum_{j=1}^{N_k} (f_{kj} - \bar f_k)^2,

    with :math:`N_k` = ``counts[k]`` and :math:`\hat V_k = 0` when
    :math:`N_k = 1`. The preimage of a vertex in :math:`[-1, 1]^m` is a face of
    the cube when some inputs do not affect :math:`y` (zero rows of
    :math:`W_1`); those inputs are drawn uniformly on :math:`[-1, 1]`, `n_mc`
    times. A vertex with a unique preimage is evaluated once. The 2016
    package always evaluated a vertex once. On a uniform space the physical
    points are clipped to the box.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(3)
    >>> model = ac.test_functions.ridge([1.0, 0.5, 1.0], space=space)
    >>> sub = ac.subspace_from_basis([[1.0], [0.5], [0.0]], space=space)
    >>> cs = ac.conditional_samples(
    ...     domain=ac.active_domain(sub), model=model, n_design=3, n_mc=4, rng=0
    ... )
    >>> cs.counts.tolist()
    [4, 4, 4, 4, 4]
    >>> cs.means.shape, bool(np.all(cs.variances > 0))
    ((5,), True)
    """
    gen, rng_desc = _random.generator(rng)
    samples, _, _ = _conditional_samples(
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
    return samples
