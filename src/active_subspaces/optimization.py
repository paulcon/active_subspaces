r"""Minimization over the active variables and the lift to a full input.

:func:`minimize_active` minimizes a function :math:`g(y)` of the active
variables over the active domain: bounded Brent on an interval, SLSQP from
feasible multistarts on a zonotope, and BFGS on :math:`\mathbb{R}^n` for a
Gaussian space [1]_. :func:`lift` maps a minimizer :math:`y^*` to a full input
:math:`x^* = W_1 y^* + W_2 z^*` by minimizing, over the inactive variables
:math:`z`, a quadratic fitted on the leading eigen-directions, with its
curvature clipped to keep the problem convex. The lift has no published
reference; it is the 2016 package's heuristic with its sign and factor-1/2
errors corrected. :func:`minimize` runs both and, given a model, evaluates
:math:`f(x^*)` once. All points are in normalized coordinates unless a name
says ``physical``.

References
----------
.. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
   Reduction in Parameter Studies.* SIAM, 2015.
"""

from __future__ import annotations

import concurrent.futures
import math
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import scipy.optimize
import scipy.special
from numpy.typing import ArrayLike, NDArray

from active_subspaces import _random
from active_subspaces import _validation as _v
from active_subspaces.errors import (
    AccuracyWarning,
    ConvergenceWarning,
    InputError,
    SolverError,
    UnsupportedError,
    _warn,
)
from active_subspaces.geometry import ActiveDomain, _to_physical
from active_subspaces.models import Model, as_model, evaluate
from active_subspaces.results import (
    QoIResult,
    _invariant,
    _make_provenance,
    _register_result,
)
from active_subspaces.solvers import SolverBackend, get_backend, solve_lp, solve_qp
from active_subspaces.spaces import InputSpace
from active_subspaces.surrogates import Surrogate, _poly_fit, _quadratic_terms

__all__ = (
    "ActiveMinimizationResult",
    "LiftResult",
    "MinimizationResult",
    "lift",
    "minimize",
    "minimize_active",
)

_KINDS = ("interval", "zonotope", "unbounded")
_METHODS: dict[str, Literal["bounded_brent", "slsqp", "bfgs"]] = {
    "interval": "bounded_brent",
    "zonotope": "slsqp",
    "unbounded": "bfgs",
}
_STATUSES = ("converged", "max_iter", "failed")
_SCIPY_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError)
_FEAS_TOL = 1e-8
_PROJ_TOL = 1e-6
_BALL = 0.999
_MAX_ITER_STATUS = {"bounded_brent": 1, "slsqp": 9, "bfgs": 1}


# ---------------------------------------------------------------------------
# Result types (§4.3)
# ---------------------------------------------------------------------------


@_register_result
class ActiveMinimizationResult(QoIResult):
    r"""Minimizer of a function of the active variables over the active domain.

    Results are produced by :func:`minimize_active`; the constructor is used
    when a saved result is loaded.

    Parameters
    ----------
    domain : ActiveDomain
        The active domain minimized over.
    method : {"bounded_brent", "slsqp", "bfgs"}
        The local method: bounded Brent (interval), SLSQP (zonotope) or BFGS
        (Gaussian space).
    y_star : ndarray, shape (n,)
        The minimizer :math:`y^*` in normalized active coordinates (the best
        run's end point).
    value : float
        :math:`g(y^*)`.
    starts : ndarray, shape (S, n)
        Starting points in normalized active coordinates; shape ``(0, 1)`` for
        an interval.
    local_minima : ndarray, shape (K, n)
        Each run's end point (``K = S``, or 1 for an interval); NaN rows for
        failed runs.
    local_values : ndarray, shape (K,)
        Each run's final value; NaN for failed runs.
    run_status : tuple of str
        ``"converged"``, ``"max_iter"`` or ``"failed"`` for each run.
    best_run : int
        Index of the best run in ``local_minima`` (lowest index among ties).
    provenance : Provenance
        How the result was computed.
    qoi : str
        Name of the quantity of interest, ``domain.subspace.qoi``.

    See Also
    --------
    minimize_active : Compute the minimizer.
    minimize : Minimize and lift to a full input.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> sub = ac.subspace_from_basis([[1.0], [0.5], [0.0]], space=space)
    >>> res = ac.minimize_active(lambda Y: Y[:, 0] ** 2 - Y[:, 0], domain=ac.active_domain(sub))
    >>> print(res.method, np.round(res.y_star, 4), f"{res.value:.4f}", res.run_status)
    bounded_brent [0.5] -0.2500 ('converged',)
    """

    TYPE = "ActiveMinimizationResult"
    SCHEMA_VERSION = 1

    domain: ActiveDomain
    method: Literal["bounded_brent", "slsqp", "bfgs"]
    y_star: NDArray[np.float64]
    value: float
    starts: NDArray[np.float64]
    local_minima: NDArray[np.float64]
    local_values: NDArray[np.float64]
    run_status: tuple[str, ...]
    best_run: int

    def _check(self) -> None:
        dom = self.domain
        _invariant(isinstance(dom, ActiveDomain), "domain must be an ActiveDomain")
        n = dom.n
        _invariant(
            self.method == _METHODS[dom.kind],
            f"method {self.method!r} does not match a {dom.kind} domain",
        )
        _invariant(
            self.y_star.shape == (n,) and bool(np.isfinite(self.y_star).all()),
            "y_star must be finite (n,)",
        )
        _invariant(math.isfinite(self.value), "value must be finite")
        S = self.starts.shape[0] if self.starts.ndim == 2 else -1
        _invariant(self.starts.shape == (S, n), "starts must be (S, n)")
        K = 1 if dom.kind == "interval" else S
        _invariant((dom.kind == "interval") == (S == 0), "starts is empty exactly for an interval")
        _invariant(
            self.local_minima.shape == (K, n) and self.local_values.shape == (K,),
            "local_minima (K, n), local_values (K,)",
        )
        _invariant(
            len(self.run_status) == K and all(s in _STATUSES for s in self.run_status),
            "run_status: one status per run",
        )
        _invariant(
            0 <= self.best_run < K and self.run_status[self.best_run] != "failed",
            "best_run must index a non-failed run",
        )
        failed = np.array([s == "failed" for s in self.run_status], dtype=bool)
        _invariant(
            bool(np.all(np.isnan(self.local_values[failed]))), "failed runs have NaN values"
        )
        _invariant(
            bool(np.all(np.isfinite(self.local_values[~failed]))),
            "non-failed runs have finite values",
        )
        _invariant(
            bool(np.array_equal(self.local_minima[self.best_run], self.y_star)),
            "y_star is the best run's minimum",
        )
        _invariant(self.local_values[self.best_run] == self.value, "value is the best run's value")
        _invariant(self.qoi == dom.subspace.qoi, "qoi must equal domain.subspace.qoi")


@_register_result
class LiftResult(QoIResult):
    r"""The lift of an active point :math:`y` to a full input :math:`x`.

    Results are produced by :func:`lift`; the constructor is used when a saved
    result is loaded.

    Parameters
    ----------
    domain : ActiveDomain
        The active domain of `y`.
    y : ndarray, shape (n,)
        The point lifted, in normalized active coordinates (after any
        projection onto the domain).
    z : ndarray, shape (m - n,)
        :math:`z^*`, the minimizer of the fitted quadratic over the inactive
        variables.
    x : ndarray, shape (m,)
        :math:`x^* = W_1 y + W_2 z^*` in normalized coordinates (clipped to
        :math:`[-1, 1]^m` for a uniform space).
    lift_dim : int
        :math:`p`, the number of leading eigen-directions in the fit.
    n_samples : int
        Number :math:`M` of data points in the fit.
    intercept : float
        :math:`a_0` of the fitted quadratic.
    linear : ndarray, shape (p,)
        :math:`\gamma` of the fitted quadratic.
    hessian : ndarray, shape (p, p)
        :math:`H` of the fitted quadratic, symmetric.
    curvature_eigenvalues : ndarray, shape (p - n,)
        Eigenvalues :math:`\theta` of :math:`H_{\zeta\zeta}` before clipping.
    curvature_floor : float
        :math:`\tau` as used.
    n_clipped : int
        Number of :math:`\theta_i < \tau`.
    model_value : float
        :math:`q(W_p^\top x^*)`.
    projection_distance : float
        :math:`\lVert y - y_{requested} \rVert_2`; 0.0 when no projection was
        needed.
    provenance : Provenance
        How the result was computed.
    qoi : str
        Name of the quantity of interest, ``domain.subspace.qoi``.

    See Also
    --------
    lift : Compute the lift; its Notes give the formulas.

    Notes
    -----
    The fitted quadratic is
    :math:`q(u) = a_0 + \gamma^\top u + \tfrac12 u^\top H u` in
    :math:`u = W_p^\top x`, where :math:`W_p` holds the first :math:`p`
    eigenvectors.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, n=1)
    >>> res = ac.lift([0.5], domain=ac.active_domain(sub), X=X, f=f)
    >>> print(res.lift_dim, res.n_samples, res.n_clipped, res.x.shape)
    2 40 0 (3,)
    """

    TYPE = "LiftResult"
    SCHEMA_VERSION = 1

    domain: ActiveDomain
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    x: NDArray[np.float64]
    lift_dim: int
    n_samples: int
    intercept: float
    linear: NDArray[np.float64]
    hessian: NDArray[np.float64]
    curvature_eigenvalues: NDArray[np.float64]
    curvature_floor: float
    n_clipped: int
    model_value: float
    projection_distance: float

    @property
    def x_physical(self) -> NDArray[np.float64]:
        """The lifted point ``x`` in physical coordinates.

        For a uniform space the point is clipped to the box ``[lower, upper]``.

        Returns
        -------
        ndarray, shape (m,)
            ``x`` mapped to physical coordinates.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> sub = ac.active_subspace(grads=G, space=space, n=1)
        >>> res = ac.lift([0.5], domain=ac.active_domain(sub), X=X, f=f)
        >>> print(np.allclose(res.x_physical, space.from_normalized(res.x[None, :])[0]))
        True
        """
        out: NDArray[np.float64] = _to_physical(self.domain.subspace.space, self.x[None, :])[0]
        return out

    def _check(self) -> None:
        dom = self.domain
        _invariant(isinstance(dom, ActiveDomain), "domain must be an ActiveDomain")
        m, n, p = dom.m, dom.n, self.lift_dim
        _invariant(n + 1 <= p <= m, "lift_dim must satisfy n + 1 <= p <= m")
        _invariant(
            self.y.shape == (n,) and self.z.shape == (m - n,) and self.x.shape == (m,),
            "y (n,), z (m-n,), x (m,)",
        )
        _invariant(
            all(bool(np.isfinite(a).all()) for a in (self.y, self.z, self.x)),
            "y, z and x must be finite",
        )
        if dom.kind != "unbounded":
            _invariant(bool(np.all(np.abs(self.x) <= 1.0)), "x must lie in [-1, 1]^m")
        _invariant(self.n_samples >= (p + 1) * (p + 2) // 2, "n_samples must be >= (p+1)(p+2)/2")
        _invariant(
            self.linear.shape == (p,) and self.hessian.shape == (p, p),
            "linear (p,), hessian (p, p)",
        )
        _invariant(bool(np.array_equal(self.hessian, self.hessian.T)), "hessian must be symmetric")
        _invariant(
            self.curvature_eigenvalues.shape == (p - n,), "curvature_eigenvalues must be (p - n,)"
        )
        _invariant(
            self.curvature_floor > 0 and math.isfinite(self.curvature_floor),
            "curvature_floor must be positive",
        )
        _invariant(0 <= self.n_clipped <= p - n, "n_clipped must be in [0, p - n]")
        _invariant(self.projection_distance >= 0, "projection_distance must be >= 0")
        _invariant(self.qoi == dom.subspace.qoi, "qoi must equal domain.subspace.qoi")


@_register_result
class MinimizationResult(QoIResult):
    r"""Minimization over the active variables, the lift, and optionally :math:`f(x^*)`.

    Results are produced by :func:`minimize`; the constructor is used when a
    saved result is loaded.

    Parameters
    ----------
    active : ActiveMinimizationResult
        The minimization over the active domain.
    lift : LiftResult
        The lift of ``active.y_star`` to a full input.
    f_star : float
        :math:`g(y^*)`, the surrogate value at ``active.y_star``.
    f_model : float or None
        :math:`f(x^*)` from one model evaluation at ``x_star_physical``;
        ``None`` when no model was passed.
    provenance : Provenance
        How the result was computed.
    qoi : str
        Name of the quantity of interest, ``domain.subspace.qoi``.

    See Also
    --------
    minimize : Compute the result.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, n=1)
    >>> s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)
    >>> res = ac.minimize(s, domain=ac.active_domain(sub), X=X, f=f, model=model, rng=0)
    >>> print(res.active.method, res.x_star.shape, res.f_model < res.f_star)
    bounded_brent (3,) True
    """

    TYPE = "MinimizationResult"
    SCHEMA_VERSION = 1

    active: ActiveMinimizationResult
    lift: LiftResult
    f_star: float
    f_model: float | None

    @property
    def x_star(self) -> NDArray[np.float64]:
        """The minimizer :math:`x^*` in normalized coordinates.

        Returns
        -------
        ndarray, shape (m,)
            ``lift.x``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> sub = ac.active_subspace(grads=G, space=space, n=1)
        >>> s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)
        >>> res = ac.minimize(s, domain=ac.active_domain(sub), X=X, f=f, rng=0)
        >>> print(bool(abs(res.x_star).max() <= 1.0))
        True
        """
        return self.lift.x

    @property
    def x_star_physical(self) -> NDArray[np.float64]:
        """The minimizer :math:`x^*` in physical coordinates.

        Returns
        -------
        ndarray, shape (m,)
            ``lift.x_physical``; clipped to the box for a uniform space.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> sub = ac.active_subspace(grads=G, space=space, n=1)
        >>> s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)
        >>> res = ac.minimize(s, domain=ac.active_domain(sub), X=X, f=f, rng=0)
        >>> res.x_star_physical.shape
        (3,)
        """
        return self.lift.x_physical

    @property
    def y_star(self) -> NDArray[np.float64]:
        """The active minimizer :math:`y^*` as lifted.

        It differs from ``active.y_star`` by ``lift.projection_distance``, which
        is nonzero only when the lift had to project :math:`y^*` onto the domain.

        Returns
        -------
        ndarray, shape (n,)
            ``lift.y``, in normalized active coordinates.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> sub = ac.active_subspace(grads=G, space=space, n=1)
        >>> s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)
        >>> res = ac.minimize(s, domain=ac.active_domain(sub), X=X, f=f, rng=0)
        >>> print(np.array_equal(res.y_star, res.active.y_star))
        True
        """
        return self.lift.y

    def _check(self) -> None:
        _invariant(
            isinstance(self.active, ActiveMinimizationResult)
            and isinstance(self.lift, LiftResult),
            "active and lift results required",
        )
        _invariant(self.f_star == self.active.value, "f_star must equal active.value")
        _invariant(
            self.f_model is None or math.isfinite(self.f_model), "f_model must be finite or None"
        )
        _invariant(
            self.qoi == self.active.qoi == self.lift.qoi,
            "qoi must agree with the active and lift results",
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _param(message: str, param: str, **details: Any) -> InputError:
    return InputError(message, code="input.parameter", details={"param": param, **details})


def _check_domain(domain: Any) -> ActiveDomain:
    if not isinstance(domain, ActiveDomain):
        raise _param(f"domain: expected an ActiveDomain, got {type(domain).__name__}", "domain")
    if domain.kind not in _KINDS:
        raise UnsupportedError(
            f"optimization is not defined for an active domain of kind {domain.kind!r}",
            code="unsupported.space",
            details={"kind": domain.kind},
        )
    return domain


def _check_space(space: Any, domain: ActiveDomain) -> InputSpace:
    ref = domain.subspace.space
    if space is None:
        return ref
    if not (type(space) is type(ref) and space == ref):
        raise _param(
            "space: does not equal domain.subspace.space", "space", reason="space_mismatch"
        )
    return space


def _resolve_g(
    g: Any, grad: Any, domain: ActiveDomain
) -> tuple[Callable[..., Any], Callable[..., Any] | None]:
    if isinstance(g, Surrogate):
        sub = g.subspace
        if sub is None or not (type(sub) is type(domain.subspace) and sub == domain.subspace):
            raise _param(
                "surrogate: its subspace must equal domain.subspace (a surrogate on all inputs "
                "is not a function of y)",
                "surrogate",
                reason="subspace_mismatch",
            )
        return g.evaluate, g.gradient
    if not callable(g):
        raise _param(
            f"g: expected a Surrogate or a callable, got {type(g).__name__}",
            "g",
            got=type(g).__name__,
        )
    if grad is not None and not callable(grad):
        raise _param(f"grad: expected a callable or None, got {type(grad).__name__}", "grad")
    return g, grad


def _radius(dim: int) -> float:
    return float(math.sqrt(2.0 * scipy.special.gammaincinv(dim / 2, _BALL)))


def _extrapolation(what: str, norm: float, dim: int) -> None:
    r = _radius(dim)
    if norm > r:
        _warn(
            AccuracyWarning,
            f"{what} has norm {norm:.4g}, beyond the radius {r:.4g} of the 99.9% ball of N(0, "
            f"I_{dim}); "
            "the surrogate may be extrapolating",
            code="accuracy.extrapolation",
            details={"what": what, "norm": norm, "radius": r, "dim": dim},
        )


# ---------------------------------------------------------------------------
# Minimization over the active variables (§5.3)
# ---------------------------------------------------------------------------


class _UserError(Exception):
    """Carries an exception raised by the user's ``g`` or ``grad`` through SciPy unchanged."""

    def __init__(self, original: BaseException):
        super().__init__(original)
        self.original = original


def _wrap(fn: Callable[..., Any], scalar: bool) -> Callable[[np.ndarray], Any]:
    def call(y: np.ndarray) -> Any:
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        try:
            out = fn(y[None, :])
        except Exception as exc:  # the user's exception, re-raised unchanged by _local_minimize
            raise _UserError(exc) from exc
        out = np.asarray(out, dtype=np.float64)
        return float(out.reshape(-1)[0]) if scalar else np.array(out.reshape(-1), dtype=np.float64)

    return call


def _local_minimize(
    method: str,
    g1: Callable[..., Any],
    dg1: Callable[..., Any] | None,
    y0: np.ndarray | None,
    domain: ActiveDomain,
    max_iter: int,
) -> tuple[np.ndarray, float, str, str]:
    """One local run through SciPy; returns ``(y, value, status, message)`` [D-04].

    The only SciPy optimizer call site.
    """
    try:
        if method == "bounded_brent":
            lo, hi = sorted(float(v) for v in np.asarray(domain.vertices)[:, 0])
            res = scipy.optimize.minimize_scalar(
                lambda t: g1(np.array([t])),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-9, "maxiter": max_iter},
            )
            y = np.array([float(res.x)])
        elif method == "slsqp":
            A = np.asarray(domain.facet_normals, dtype=np.float64)
            b = np.asarray(domain.facet_offsets, dtype=np.float64)
            res = scipy.optimize.minimize(
                g1,
                y0,
                jac=dg1,
                method="SLSQP",
                constraints={"type": "ineq", "fun": lambda y: b - A @ y, "jac": lambda y: -A},
                options={"ftol": 1e-9, "maxiter": max_iter},
            )
            y = np.asarray(res.x, dtype=np.float64).reshape(-1)
        else:
            res = scipy.optimize.minimize(
                g1, y0, jac=dg1, method="BFGS", options={"maxiter": max_iter}
            )
            y = np.asarray(res.x, dtype=np.float64).reshape(-1)
    except _UserError as exc:
        orig = exc.original
        raise orig.with_traceback(orig.__traceback__) from orig.__cause__
    except _SCIPY_ERRORS as exc:
        return np.full(domain.n, np.nan), math.nan, "failed", f"{type(exc).__name__}: {exc}"
    value = float(np.asarray(res.fun).reshape(-1)[0])
    message = str(getattr(res, "message", ""))
    if res.success:
        status = "converged"
    elif int(getattr(res, "status", -1)) == _MAX_ITER_STATUS[method]:
        status = "max_iter"
    else:
        status = "failed"
    if not (np.isfinite(y).all() and math.isfinite(value)):
        status, message = "failed", f"non-finite point or value ({message})"
    elif method == "slsqp":
        A, b = np.asarray(domain.facet_normals), np.asarray(domain.facet_offsets)
        viol = float(np.max(A @ y - b))
        if viol > _FEAS_TOL * max(1.0, float(np.max(np.abs(b)))):
            status, message = "failed", f"violates the domain by {viol:.3g} ({message})"
    if status == "failed":
        return np.full(domain.n, np.nan), math.nan, status, message
    return y, value, status, message


def _minimize_active(
    g: Any, domain: ActiveDomain, grad: Any, n_starts: int, max_iter: int, rng: Any
) -> tuple[ActiveMinimizationResult, dict[str, Any] | None]:
    gfun, gradfun = _resolve_g(g, grad, domain)
    g1 = _wrap(gfun, scalar=True)
    dg1 = _wrap(gradfun, scalar=False) if gradfun is not None else None
    method = _METHODS[domain.kind]
    n, m = domain.n, domain.m
    desc = None
    if method == "bounded_brent":
        starts = np.zeros((0, n))
        runs = [_local_minimize(method, g1, dg1, None, domain, max_iter)]
    else:
        gen, desc = _random.generator(rng)
        kids = _random.spawn(gen, n_starts)
        W1 = np.array(domain.subspace.W1, dtype=np.float64)

        def draw(k: np.random.Generator) -> NDArray[np.float64]:
            if method == "slsqp":
                return k.uniform(-1.0, 1.0, m)
            return k.standard_normal(m)

        starts = np.array([draw(k) @ W1 for k in kids]).reshape(n_starts, n)
        runs = [
            _local_minimize(method, g1, dg1, starts[j].copy(), domain, max_iter)
            for j in range(n_starts)
        ]
    minima = np.array([r[0] for r in runs]).reshape(len(runs), n)
    values = np.array([r[1] for r in runs], dtype=np.float64)
    status = tuple(r[2] for r in runs)
    ok = np.array([s != "failed" for s in status])
    if not ok.any():
        raise SolverError(
            f"every one of the {len(runs)} {method} run(s) failed",
            code="solver.failed",
            details={"method": method, "runs": [{"status": r[2], "message": r[3]} for r in runs]},
        )
    best = int(np.argmin(np.where(ok, values, np.inf)))
    if not ok.all():
        bad = [int(j) for j in np.flatnonzero(~ok)]
        _warn(
            ConvergenceWarning,
            f"{len(bad)} of {len(runs)} {method} runs failed; the best of the others is used",
            code="convergence.failed_starts",
            details={"failed_runs": bad, "messages": [runs[j][3] for j in bad]},
        )
    if "max_iter" in status:
        hit = [j for j, s in enumerate(status) if s == "max_iter"]
        _warn(
            ConvergenceWarning,
            f"{len(hit)} {method} run(s) stopped at max_iter = {max_iter}; their last iterates "
            "are used",
            code="convergence.max_iter",
            details={"runs": hit, "max_iter": max_iter},
        )
    y_star = minima[best].copy()
    if method == "bfgs":
        _extrapolation("y*", float(np.linalg.norm(y_star)), n)
    result = ActiveMinimizationResult(
        domain=domain,
        method=method,
        y_star=y_star,
        value=float(values[best]),
        starts=starts,
        local_minima=minima,
        local_values=values,
        run_status=status,
        best_run=best,
        qoi=domain.subspace.qoi,
        provenance=_make_provenance(
            "minimize_active",
            {"g": g, "grad": grad, "n_starts": n_starts, "max_iter": max_iter},
            space=domain.subspace.space,
            rng_state=desc,
        ),
    )
    return result, desc


def minimize_active(
    g: Surrogate | Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    domain: ActiveDomain,
    grad: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    n_starts: int = 10,
    max_iter: int = 10_000,
    rng: _random.RngLike = None,
) -> ActiveMinimizationResult:
    r"""Minimize a function of the active variables over the active domain.

    Computes :math:`y^* \in \arg\min_{y \in \mathcal{Y}} g(y)` and
    :math:`g(y^*)`, where :math:`\mathcal{Y}` is the active domain: an interval
    or a zonotope for a uniform space, :math:`\mathbb{R}^n` for a Gaussian
    space [1]_.

    Parameters
    ----------
    g : Surrogate or callable
        A surrogate whose ``subspace`` equals ``domain.subspace`` (its
        ``evaluate`` and ``gradient`` are used), or a callable mapping
        normalized active coordinates ``(k, n)`` to values ``(k,)``.
    domain : ActiveDomain
        The active domain, from :func:`~active_subspaces.active_domain`.
    grad : callable, optional
        Gradient of `g`, mapping ``(k, n)`` to ``(k, n)`` in normalized active
        coordinates; ignored for a ``Surrogate``. Without it, SLSQP and BFGS
        use finite differences.
    n_starts : int, default 10
        Number of starts for a zonotope or Gaussian domain; ignored for an
        interval.
    max_iter : int, default 10000
        Iteration limit of each local run.
    rng : None, int, SeedSequence or Generator, optional
        Random source of the starts (start ``j`` uses child ``j``); not used
        for an interval.

    Returns
    -------
    ActiveMinimizationResult
        ``y_star``, ``value`` and every run's start, end point and status.

    Raises
    ------
    InputError
        ``input.parameter`` if `domain` is not an ``ActiveDomain``, if
        `n_starts` or `max_iter` is not a positive integer, if `g` is neither
        a surrogate nor callable, if `grad` is not callable, or if the
        surrogate's subspace is not ``domain.subspace``.
    SolverError
        ``solver.failed`` if every run failed.

    Warns
    -----
    ConvergenceWarning
        ``convergence.failed_starts`` if some runs failed (the others are
        used); ``convergence.max_iter`` if a run stopped at `max_iter` (its
        last iterate is used).
    AccuracyWarning
        ``accuracy.extrapolation`` for a Gaussian domain when
        :math:`\lVert y^* \rVert_2` exceeds the radius of the 99.9% probability
        ball of :math:`N(0, I_n)`.

    See Also
    --------
    minimize : Minimize and lift the minimizer to a full input.

    Notes
    -----
    ==========================  ==================================  ==========================
    Domain                      Method                              Starts
    ==========================  ==================================  ==========================
    interval :math:`[l, u]`     bounded Brent                       one run, no start
    zonotope :math:`Ay \le b`   SLSQP with the linear constraints   :math:`y_0 = W_1^\top x_0`
    :math:`\mathbb{R}^n`        BFGS                                :math:`y_0 = W_1^\top x_0`
    ==========================  ==================================  ==========================

    with :math:`x_0 \sim \rho` (uniform on :math:`[-1, 1]^m` or
    :math:`N(0, I_m)`), so every start is feasible. :math:`y^*` is the best
    value over the runs that did not fail (lowest index among ties). A run
    fails if SciPy reports failure, raises a numerical error, or ends at a
    non-finite or infeasible point. Exceptions raised by `g` or `grad`
    propagate unchanged. For a Gaussian space the minimization is over all of
    :math:`\mathbb{R}^n`, so a surrogate may be extrapolating; the library warns
    when :math:`\lVert y^* \rVert_2 > \sqrt{\chi^2_{n, 0.999}}` but does not
    constrain :math:`y`.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    On an interval:

    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> sub = ac.subspace_from_basis([[1.0], [0.5], [0.0]], space=space)
    >>> res = ac.minimize_active(lambda Y: Y[:, 0] ** 2 - Y[:, 0], domain=ac.active_domain(sub))
    >>> print(res.method, np.round(res.y_star, 4), f"{res.value:.4f}")
    bounded_brent [0.5] -0.2500

    On a zonotope, with a gradient and three starts:

    >>> sub = ac.subspace_from_basis([[1.0, 0.0], [0.5, 1.0], [0.0, 0.2]], space=space)
    >>> res = ac.minimize_active(
    ...     lambda Y: np.sum((Y - 0.1) ** 2, axis=1),
    ...     grad=lambda Y: 2 * (Y - 0.1),
    ...     domain=ac.active_domain(sub),
    ...     n_starts=3,
    ...     rng=0,
    ... )
    >>> print(res.method, np.round(res.y_star, 4), res.starts.shape)
    slsqp [0.1 0.1] (3, 2)
    """
    domain = _check_domain(domain)
    n_starts = _v.integer(n_starts, "n_starts", lo=1)
    max_iter = _v.integer(max_iter, "max_iter", lo=1)
    return _minimize_active(g, domain, grad, n_starts, max_iter, rng)[0]


# ---------------------------------------------------------------------------
# Lift (§3.6, §5.4)
# ---------------------------------------------------------------------------


def _project_y(y: np.ndarray, W1: np.ndarray, be: SolverBackend) -> np.ndarray:
    """``W1^T x_hat`` for ``x_hat`` minimizing ``||W1^T x - y||_1`` over the cube.

    An LP in ``(x, s)``.
    """
    m, n = W1.shape
    eye = np.eye(n)
    A_ub = np.block([[W1.T, -eye], [-W1.T, -eye]])
    b_ub = np.concatenate([y, -y])
    c = np.concatenate([np.zeros(m), np.ones(n)])
    bounds = np.vstack([np.tile([-1.0, 1.0], (m, 1)), np.tile([0.0, np.inf], (n, 1))])
    out = solve_lp(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, backend=be)
    assert out.x is not None  # solve_lp returns only an optimal output, which has x
    y_proj: np.ndarray = W1.T @ np.clip(out.x[:m], -1.0, 1.0)
    return y_proj


def _lift_qp(
    Ht: np.ndarray,
    ct: np.ndarray,
    y: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    be: SolverBackend,
) -> np.ndarray:
    s = W1 @ y
    A_ub = np.vstack([W2, -W2])
    b_ub = np.concatenate([1.0 - s, 1.0 + s])
    return np.asarray(solve_qp(Ht, ct, A_ub=A_ub, b_ub=b_ub, backend=be).x, dtype=np.float64)


def _lift(
    y: Any,
    domain: ActiveDomain,
    X: Any,
    f: Any,
    space: Any,
    lift_dim: Any,
    curvature_floor: Any,
    be: SolverBackend,
) -> LiftResult:
    space = _check_space(space, domain)
    sub = domain.subspace
    m, n = sub.m, sub.n
    d = _v.data(space, X=X, f=f, require=("X", "f"))
    yv = _v.vector(y, "y", size=n)
    y_digest = _v.digest(yv)
    uniform = domain.kind != "unbounded"
    if uniform:
        A, b = np.asarray(domain.facet_normals), np.asarray(domain.facet_offsets)
        excess = float(np.max(A @ yv - b))
        if excess > _FEAS_TOL * max(1.0, float(np.max(np.abs(b)))):
            raise InputError(
                "y: lies outside the active domain (a facet inequality is violated by "
                f"{excess:.3g})",
                code="input.out_of_domain",
                details={"param": "y", "violation": excess},
            )
    p = (
        (n + 2 if m - n > 2 else n + 1)
        if lift_dim is None
        else _v.integer(lift_dim, "lift_dim", lo=n + 1, hi=m)
    )
    _v.min_samples(d.M, (p + 1) * (p + 2) // 2, f"a quadratic lift model in p = {p} variables")
    delta = _v.real(curvature_floor, "curvature_floor", lo=0.0, hi=1.0, lo_open=True, hi_open=True)

    assert d.X is not None  # required above
    assert d.f is not None
    W = np.array(sub.W, dtype=np.float64)
    W1, W2 = W[:, :n], W[:, n:]
    U = d.X @ W[:, :p]
    indices, coef = _poly_fit(U, d.f, degree=2)
    a0, gamma, H = _quadratic_terms(indices, coef)
    H = 0.5 * (H + H.T)

    Qz = H[n:, n:]
    theta, V = np.linalg.eigh(Qz)
    tmax = float(np.max(np.abs(theta)))
    # constant-model branch: inactive curvature at rounding level [D-110, amended 2026-10-29]
    scale = max(1.0, float(np.linalg.norm(H, 2)), float(np.linalg.norm(gamma)))
    constant = tmax <= 64.0 * np.finfo(np.float64).eps * scale
    tau = 1.0 if constant else delta * tmax
    n_clipped = int(np.count_nonzero(theta < tau))
    if n_clipped:
        _warn(
            AccuracyWarning,
            f"lift: {n_clipped} of {p - n} curvature eigenvalues of the fitted quadratic were "
            "below the floor "
            f"tau = {tau:.3g} and were raised to it",
            code="accuracy.curvature_clipped",
            details={"theta": [float(t) for t in theta], "tau": tau},
        )
    theta_p = np.maximum(theta, tau)
    Qp = (V * theta_p) @ V.T
    Qp = 0.5 * (Qp + Qp.T)

    def c_of(yy: np.ndarray) -> np.ndarray:
        return np.zeros(p - n) if constant else gamma[n:] + H[n:, :n] @ yy

    projection = 0.0
    if not uniform:
        zeta = -(V @ ((V.T @ c_of(yv)) / theta_p))
        z = np.concatenate([zeta, np.zeros(m - p)])
        x = W1 @ yv + W2 @ z
        _extrapolation("x*", float(np.linalg.norm(x)), m)
    else:
        Ht = np.zeros((m - n, m - n))
        Ht[: p - n, : p - n] = Qp
        Ht[p - n :, p - n :] = tau * np.eye(m - p)

        def ct(yy: np.ndarray) -> np.ndarray:
            return np.concatenate([c_of(yy), np.zeros(m - p)])

        try:
            z = _lift_qp(Ht, ct(yv), yv, W1, W2, be)
        except SolverError as exc:
            if exc.code != "solver.infeasible":
                raise
            y_req = yv
            yv = _project_y(y_req, W1, be)
            projection = float(np.linalg.norm(yv - y_req))
            limit = _PROJ_TOL * (1.0 + float(np.linalg.norm(y_req)))
            if projection > limit:
                raise SolverError(
                    "lift: the fiber of y is empty and the nearest point of the domain is "
                    f"{projection:.3g} away",
                    code="solver.infeasible",
                    details={"projection_distance": projection, "limit": limit},
                ) from exc
            try:
                z = _lift_qp(Ht, ct(yv), yv, W1, W2, be)
            except SolverError as exc2:
                raise SolverError(
                    "lift: the QP over the fiber failed after projecting y onto the domain "
                    f"({exc2})",
                    code="solver.infeasible",
                    details={"projection_distance": projection, "cause": exc2.code},
                ) from exc2
        x = np.clip(W1 @ yv + W2 @ z, -1.0, 1.0)
    u = W[:, :p].T @ x
    model_value = float(a0 + gamma @ u + 0.5 * u @ H @ u)
    params = {
        "lift_dim": p,
        "curvature_floor": delta,
        "backend": str(getattr(be, "name", type(be).__name__)),
    }
    digests = dict(d.digests)
    digests["y"] = y_digest
    return LiftResult(
        domain=domain,
        y=yv,
        z=z,
        x=x,
        lift_dim=p,
        n_samples=d.M,
        intercept=float(a0),
        linear=gamma,
        hessian=H,
        curvature_eigenvalues=theta,
        curvature_floor=float(tau),
        n_clipped=n_clipped,
        model_value=model_value,
        projection_distance=projection,
        qoi=sub.qoi,
        provenance=_make_provenance("lift", params, space=space, digests=digests),
    )


def lift(
    y: ArrayLike,
    *,
    domain: ActiveDomain,
    X: ArrayLike,
    f: ArrayLike,
    space: InputSpace | None = None,
    lift_dim: int | None = None,
    curvature_floor: float = 1e-2,
    backend: SolverBackend | str | None = None,
) -> LiftResult:
    r"""Lift an active point :math:`y` to a full input :math:`x = W_1 y + W_2 z`.

    A quadratic in the leading :math:`p` eigen-directions is fitted to the data
    by least squares, and :math:`z` minimizes it over the inactive variables
    (with its curvature clipped so that the problem is convex). For a Gaussian
    space the minimizer has a closed form; for a uniform space it solves a
    convex QP over the fiber :math:`-1 \le W_1 y + W_2 z \le 1`.
    Deterministic. The lift is a heuristic with no published reference.

    Parameters
    ----------
    y : array_like, shape (n,)
        The active point, in normalized active coordinates; it must lie in the
        domain.
    domain : ActiveDomain
        The active domain, from :func:`~active_subspaces.active_domain`.
    X : array_like, shape (M, m)
        Inputs of the fit, in physical coordinates.
    f : array_like, shape (M,) or (M, 1)
        Values of the model at `X`.
    space : InputSpace, optional
        The input space of `X`; if given it must equal
        ``domain.subspace.space`` (the default).
    lift_dim : int, optional
        :math:`p`, with :math:`n + 1 \le p \le m`; default :math:`n + 2` if
        :math:`m - n > 2`, else :math:`n + 1`.
    curvature_floor : float, default 0.01
        :math:`\delta \in (0, 1)`, the relative floor of the curvature
        eigenvalues.
    backend : SolverBackend, "scipy" or None, optional
        Solver for the QP and the projection LP (uniform space).

    Returns
    -------
    LiftResult
        ``x`` (normalized), ``x_physical``, ``z`` and the fitted quadratic.

    Raises
    ------
    InputError
        ``input.parameter`` for an invalid `domain`, `space` (``space_mismatch``),
        `lift_dim`, `curvature_floor` or `backend`; ``input.shape``,
        ``input.dtype``, ``input.nonfinite``, ``input.size_mismatch``,
        ``input.vector_qoi`` or ``input.out_of_space`` for invalid `X`, `f` or
        `y`; ``input.out_of_domain`` if `y` lies outside a uniform domain;
        ``input.too_few_samples`` if :math:`M < (p + 1)(p + 2) / 2`.
    NumericalError
        ``numerical.singular`` if the quadratic basis is rank-deficient on the
        data; ``numerical.nonfinite`` if the fit is not finite.
    SolverError
        ``solver.infeasible`` if the fiber of `y` is empty even after
        projecting `y` onto the domain; other ``solver.*`` codes if the QP
        fails.

    Warns
    -----
    AccuracyWarning
        ``accuracy.curvature_clipped`` if curvature eigenvalues were raised to
        the floor; ``accuracy.extrapolation`` for a Gaussian space when
        :math:`\lVert x^* \rVert_2` exceeds the radius of the 99.9% probability
        ball of :math:`N(0, I_m)`.

    See Also
    --------
    minimize : Minimize over the active variables, then lift.

    Notes
    -----
    Let :math:`W_p` hold the first :math:`p` eigenvectors and
    :math:`u = W_p^\top x = (y, \zeta)`, :math:`\zeta \in \mathbb{R}^{p-n}`.

    1. Fit :math:`q(u) = a_0 + \gamma^\top u + \tfrac12 u^\top H u` by least
       squares on the normalized data.
    2. For fixed :math:`y`, :math:`q = \text{const} + c(y)^\top \zeta +
       \tfrac12 \zeta^\top Q \zeta` with
       :math:`c(y) = \gamma_\zeta + H_{\zeta y} y` and :math:`Q = H_{\zeta\zeta}`.
    3. With :math:`Q = V \operatorname{diag}(\theta) V^\top`, clip
       :math:`Q_+ = V \operatorname{diag}(\max(\theta_i, \tau)) V^\top`,
       :math:`\tau = \delta \max_i |\theta_i|`. If
       :math:`\max_i |\theta_i| \le 64\, \varepsilon_{mach}
       \max(1, \lVert H \rVert_2, \lVert \gamma \rVert_2)` (no inactive
       curvature beyond round-off, e.g. an exact ridge), the lift takes
       :math:`c = 0` and :math:`\tau = 1`, i.e. the minimum-norm :math:`z`.
    4. With :math:`z = (\zeta, z_b)`, :math:`z_b \in \mathbb{R}^{m-p}`, minimize
       :math:`\tfrac12 z^\top \tilde H z + \tilde c^\top z`,
       :math:`\tilde H = \operatorname{blockdiag}(Q_+, \tau I_{m-p})`,
       :math:`\tilde c = (c(y), 0)`: for a Gaussian space
       :math:`\zeta^* = -Q_+^{-1} c(y)`, :math:`z_b^* = 0`; for a uniform space
       subject to :math:`-1 \le W_1 y + W_2 z \le 1`.
    5. :math:`x^* = W_1 y + W_2 z^*`.

    If the fiber is empty by round-off (at a vertex of the zonotope), `y` is
    first projected onto the domain by a linear program and the distance is
    recorded in ``projection_distance``.

    *Departures from the 2016 package.* The 2016 lift returned
    :math:`+Q^{-1} c` instead of :math:`-Q^{-1} c`, minimized
    :math:`c^\top z + z^\top Q z` (missing the factor 1/2) in the uniform case,
    used :math:`Q + 0.01 I` (possibly indefinite and not scale-aware), and
    penalized the unmodeled directions by :math:`0.01 \lVert z_b \rVert^2`;
    v1 uses the formulas above.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, n=1)
    >>> res = ac.lift([0.5], domain=ac.active_domain(sub), X=X, f=f)
    >>> print(np.round(sub.W1.T @ res.x, 6))
    [0.5]
    >>> print(res.x_physical.shape, res.projection_distance)
    (3,) 0.0
    """
    domain = _check_domain(domain)
    be = get_backend(backend)
    return _lift(y, domain, X, f, space, lift_dim, curvature_floor, be)


# ---------------------------------------------------------------------------
# minimize (§5.5)
# ---------------------------------------------------------------------------


def minimize(
    surrogate: Surrogate | Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    domain: ActiveDomain,
    X: ArrayLike,
    f: ArrayLike,
    space: InputSpace | None = None,
    grad: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    model: Model | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    n_starts: int = 10,
    max_iter: int = 10_000,
    lift_dim: int | None = None,
    curvature_floor: float = 1e-2,
    backend: SolverBackend | str | None = None,
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
    rng: _random.RngLike = None,
) -> MinimizationResult:
    r"""Minimize a surrogate over the active variables and lift the minimizer.

    Runs :func:`minimize_active`, then :func:`lift` of its ``y_star``, then, if
    `model` is given, one model evaluation at ``x_star_physical``. No local
    refinement in the full input space is done.

    Parameters
    ----------
    surrogate : Surrogate or callable
        The function :math:`g` of the active variables, as `g` in
        :func:`minimize_active`.
    domain : ActiveDomain
        The active domain, from :func:`~active_subspaces.active_domain`.
    X : array_like, shape (M, m)
        Inputs for the lift's quadratic fit, in physical coordinates.
    f : array_like, shape (M,) or (M, 1)
        Values of the model at `X`.
    space : InputSpace, optional
        The input space of `X`; if given it must equal
        ``domain.subspace.space``.
    grad : callable, optional
        As in :func:`minimize_active`.
    model : Model or callable, optional
        A ``Model`` whose space equals ``domain.subspace.space``, or a batch
        callable on physical points ``(k, m) -> (k,)``; evaluated once at
        :math:`x^*`.
    n_starts : int, default 10
        As in :func:`minimize_active`.
    max_iter : int, default 10000
        As in :func:`minimize_active`.
    lift_dim : int, optional
        As in :func:`lift`.
    curvature_floor : float, default 0.01
        As in :func:`lift`.
    backend : SolverBackend, "scipy" or None, optional
        As in :func:`lift`.
    executor : concurrent.futures.Executor, optional
        Passed to :func:`~active_subspaces.evaluate`; ignored without a model.
    batch_size : int, optional
        Passed to :func:`~active_subspaces.evaluate`; ignored without a model.
    rng : None, int, SeedSequence or Generator, optional
        Random source of the starts, as in :func:`minimize_active`.

    Returns
    -------
    MinimizationResult
        ``f_star`` :math:`= g(y^*)`, ``f_model`` :math:`= f(x^*)` (or ``None``),
        ``x_star``, ``x_star_physical``, ``y_star`` and the two partial
        results.

    Raises
    ------
    InputError
        As in :func:`minimize_active` and :func:`lift`; ``input.parameter``
        also for a `model` whose space differs from ``domain.subspace.space``
        or an invalid `batch_size`.
    ModelError
        ``model.*`` if the model raises, or returns a value of the wrong shape
        or dtype, or one that is not finite.
    NumericalError
        As in :func:`lift`.
    SolverError
        As in :func:`minimize_active` and :func:`lift`.

    Warns
    -----
    ConvergenceWarning
        As in :func:`minimize_active`.
    AccuracyWarning
        As in :func:`minimize_active` and :func:`lift`.

    See Also
    --------
    minimize_active : The minimization over the active domain.
    lift : The lift of :math:`y^*` to :math:`x^*`.

    Notes
    -----
    ``f_star`` is the surrogate value :math:`g(y^*)`, which approximates the
    conditional average of :math:`f` over the inactive variables at
    :math:`y^*`; ``f_model`` is the model at the single point :math:`x^*` the
    lift chose, and can differ from it.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, n=1)
    >>> s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)
    >>> res = ac.minimize(s, domain=ac.active_domain(sub), X=X, f=f, model=model, rng=0)
    >>> print(np.round(res.x_star_physical, 2))
    [0.49 0.99 1.97]
    >>> print(f"{res.f_star:.3f} {res.f_model:.3f}")
    0.226 0.001
    """
    domain = _check_domain(domain)
    _check_space(space, domain)
    n_starts = _v.integer(n_starts, "n_starts", lo=1)
    max_iter = _v.integer(max_iter, "max_iter", lo=1)
    be = get_backend(backend)
    mdl = None if model is None else as_model(model, space=domain.subspace.space)
    if mdl is not None and batch_size is not None:
        _v.integer(batch_size, "batch_size", lo=1)
    active, desc = _minimize_active(surrogate, domain, grad, n_starts, max_iter, rng)
    lifted = _lift(active.y_star, domain, X, f, space, lift_dim, curvature_floor, be)
    f_model = None
    if mdl is not None:
        out = evaluate(
            mdl,
            X=lifted.x_physical[None, :],
            space=domain.subspace.space,
            executor=executor,
            batch_size=batch_size,
        )
        f_model = float(out[0])
    params = {
        "surrogate": surrogate,
        "n_starts": n_starts,
        "max_iter": max_iter,
        "lift_dim": lifted.lift_dim,
        "curvature_floor": curvature_floor,
        "backend": str(getattr(be, "name", type(be).__name__)),
        "model": None if mdl is None else mdl.name,
        "batch_size": batch_size,
    }
    return MinimizationResult(
        active=active,
        lift=lifted,
        f_star=active.value,
        f_model=f_model,
        qoi=domain.subspace.qoi,
        provenance=_make_provenance(
            "minimize", params, space=domain.subspace.space, rng_state=desc
        ),
    )
