"""Solver backends for linear and quadratic programs.

Two operations make a backend::

    LP:  min_v  c^T v
    QP:  min_v  1/2 v^T H v + g^T v
    s.t. A_ub v <= b_ub,  A_eq v = b_eq,  lower <= v <= upper

Each constraint block is optional, and absent ``bounds`` mean free variables
(unlike ``scipy.optimize.linprog``, whose default is ``v >= 0``). The library
calls a backend only through :func:`solve_lp` and :func:`solve_qp`, which
validate the arrays, call the backend and turn every status other than
``"optimal"`` into a :class:`~active_subspaces.errors.SolverError`. SciPy is the
only built-in backend; another one is any object with the methods of
:class:`SolverBackend`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

from active_subspaces import _validation as _v
from active_subspaces.errors import InputError, SolverError

__all__ = (
    "ScipyBackend",
    "SolverBackend",
    "SolverOutput",
    "get_backend",
    "solve_lp",
    "solve_qp",
)

_STATUSES = ("optimal", "infeasible", "unbounded", "max_iter", "failed")
_LINPROG_STATUS = {0: "optimal", 1: "max_iter", 2: "infeasible", 3: "unbounded", 4: "failed"}
_QP_METHODS = ("slsqp", "trust-constr")
_SCIPY_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError)

_F64 = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Output and protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolverOutput:
    r"""Outcome of one LP or QP solve.

    A value type returned by backends and by :func:`solve_lp` and
    :func:`solve_qp`. It is not a :class:`~active_subspaces.Result`: it has no
    provenance and is not serialized.

    Parameters
    ----------
    x : array_like, shape (k,), float, or None
        The minimizer when ``status == "optimal"``, the last iterate when
        ``"max_iter"`` (if the backend has one), else ``None``. Stored as a
        read-only float64 copy.
    objective : float or None
        The objective at ``x``: :math:`c^\top v` for an LP,
        :math:`\tfrac12 v^\top H v + g^\top v` for a QP.
    status : {"optimal", "infeasible", "unbounded", "max_iter", "failed"}
        Outcome of the solve.
    message : str
        The backend's message.
    n_iter : int or None
        Iterations used, when the backend reports them.

    Raises
    ------
    InputError
        ``input.parameter`` for an unknown ``status``.

    See Also
    --------
    solve_lp : Solve a linear program.
    solve_qp : Solve a convex quadratic program.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> out = ac.SolverOutput([1.0, 2.0], 3.0, "optimal", "done", 4)
    >>> out.x, out.x.flags.writeable
    (array([1., 2.]), False)
    """

    x: NDArray[np.float64] | None
    objective: float | None
    status: str
    message: str
    n_iter: int | None

    def __post_init__(self) -> None:
        """Validate the fields."""
        _v.choice(self.status, "status", _STATUSES)
        if self.x is not None:
            object.__setattr__(
                self, "x", _v.freeze_array(np.ravel(np.asarray(self.x, dtype=np.float64)))
            )
        if self.objective is not None:
            object.__setattr__(self, "objective", float(self.objective))
        object.__setattr__(self, "message", str(self.message))
        if self.n_iter is not None:
            object.__setattr__(self, "n_iter", int(self.n_iter))

    def __eq__(self, other: object) -> bool:
        """Compare field by field, arrays with NaN equal to NaN."""
        if not isinstance(other, SolverOutput):
            return NotImplemented
        same_x = (self.x is None and other.x is None) or (
            self.x is not None and other.x is not None and np.array_equal(self.x, other.x)
        )
        return (
            same_x
            and self.objective == other.objective
            and self.status == other.status
            and self.message == other.message
            and self.n_iter == other.n_iter
        )


@runtime_checkable
class SolverBackend(Protocol):
    r"""Protocol of an LP/QP backend.

    A backend is any object with a ``name`` and the methods ``lp`` and ``qp``.
    It receives validated, C-contiguous float64 arrays, must not modify them,
    and reports every solver outcome through :attr:`SolverOutput.status`
    rather than by raising. Pass a backend as the ``backend`` argument of the
    functions that solve programs.

    Parameters
    ----------
    *args, **kwargs
        A protocol is not instantiated; implement its members instead.

    See Also
    --------
    ScipyBackend : The built-in backend.
    get_backend : Resolve a ``backend`` argument.

    Notes
    -----
    The programs are

    .. math::

        \text{LP: } \min_v c^\top v, \qquad
        \text{QP: } \min_v \tfrac12 v^\top H v + g^\top v,

    subject to :math:`A_{ub} v \le b_{ub}`, :math:`A_{eq} v = b_{eq}` and
    :math:`\ell \le v \le u`, where each block is optional and absent bounds
    mean free variables.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> isinstance(ac.ScipyBackend(), ac.SolverBackend)
    True
    """

    @property
    def name(self) -> str:
        """Name of the backend, used in error messages."""

    def lp(
        self,
        c: _F64,
        A_ub: _F64 | None = None,
        b_ub: _F64 | None = None,
        A_eq: _F64 | None = None,
        b_eq: _F64 | None = None,
        bounds: _F64 | None = None,
    ) -> SolverOutput:
        """Solve a linear program and report the outcome in ``status``.

        Parameters
        ----------
        c : numpy.ndarray, shape (k,), float64
            Cost vector.
        A_ub : numpy.ndarray, shape (r, k), float64, optional
            Inequality matrix, given together with ``b_ub``.
        b_ub : numpy.ndarray, shape (r,), float64, optional
            Inequality right-hand side.
        A_eq : numpy.ndarray, shape (e, k), float64, optional
            Equality matrix, given together with ``b_eq``.
        b_eq : numpy.ndarray, shape (e,), float64, optional
            Equality right-hand side.
        bounds : numpy.ndarray, shape (k, 2), float64, optional
            Rows ``(lower, upper)``, possibly infinite; ``None`` means free.

        Returns
        -------
        SolverOutput
            The outcome; never raises for a solver failure.

        See Also
        --------
        ScipyBackend.lp : The built-in implementation.
        """

    def qp(
        self,
        H: _F64,
        g: _F64,
        A_ub: _F64 | None = None,
        b_ub: _F64 | None = None,
        A_eq: _F64 | None = None,
        b_eq: _F64 | None = None,
        bounds: _F64 | None = None,
    ) -> SolverOutput:
        """Solve a convex quadratic program and report the outcome in ``status``.

        Parameters
        ----------
        H : numpy.ndarray, shape (k, k), float64
            Symmetric positive semidefinite Hessian.
        g : numpy.ndarray, shape (k,), float64
            Linear term.
        A_ub : numpy.ndarray, shape (r, k), float64, optional
            Inequality matrix, given together with ``b_ub``.
        b_ub : numpy.ndarray, shape (r,), float64, optional
            Inequality right-hand side.
        A_eq : numpy.ndarray, shape (e, k), float64, optional
            Equality matrix, given together with ``b_eq``.
        b_eq : numpy.ndarray, shape (e,), float64, optional
            Equality right-hand side.
        bounds : numpy.ndarray, shape (k, 2), float64, optional
            Rows ``(lower, upper)``, possibly infinite; ``None`` means free.

        Returns
        -------
        SolverOutput
            The outcome; never raises for a solver failure.

        See Also
        --------
        ScipyBackend.qp : The built-in implementation.
        """


# ---------------------------------------------------------------------------
# Validation (spec 10 §5.6)
# ---------------------------------------------------------------------------


def _bad(message: str, param: str, reason: str, **details: Any) -> InputError:
    return InputError(
        message, code="input.parameter", details={"param": param, "reason": reason, **details}
    )


def _array(
    a: Any, name: str, ndim: int, shape: tuple[int | None, ...], *, finite: bool = True
) -> _F64:
    try:
        arr = _v.coerce(a, name)
    except InputError as exc:
        raise _bad(exc.message, name, "dtype") from exc
    if arr.ndim != ndim or any(
        s is not None and s != g for s, g in zip(shape, arr.shape, strict=False)
    ):
        expected = (
            "("
            + ", ".join("?" if s is None else str(s) for s in shape)
            + ("," if ndim == 1 else "")
            + ")"
        )
        raise _bad(
            f"{name}: expected shape {expected}, got {arr.shape}",
            name,
            "shape",
            got=list(arr.shape),
        )
    if finite and not np.isfinite(arr).all():
        raise _bad(f"{name}: contains NaN or inf", name, "nonfinite")
    return np.array(arr, dtype=np.float64, order="C", copy=True)


def _pair(A: Any, b: Any, nA: str, nb: str, k: int) -> tuple[_F64 | None, _F64 | None]:
    if A is None and b is None:
        return None, None
    if A is None or b is None:
        raise _bad(f"{nA} and {nb} must be given together", nA if A is None else nb, "pair")
    A = _array(A, nA, 2, (None, k))
    b = _array(b, nb, 1, (A.shape[0],))
    return A, b


def _bounds(bounds: Any, k: int) -> _F64 | None:
    if bounds is None:
        return None
    B = _array(bounds, "bounds", 2, (k, 2), finite=False)
    lo, up = B[:, 0], B[:, 1]
    if np.isnan(B).any():
        raise _bad("bounds: contains NaN", "bounds", "nonfinite")
    if np.any(lo == np.inf) or np.any(up == -np.inf):
        raise _bad(
            "bounds: lower bounds must be < +inf and upper bounds > -inf", "bounds", "bounds"
        )
    if np.any(lo > up):
        raise _bad(
            "bounds: some lower bound exceeds its upper bound",
            "bounds",
            "bounds",
            row=int(np.argmax(lo > up)),
        )
    return B


def _check_H(H: Any, k: int) -> _F64:
    H = _array(H, "H", 2, (k, k))
    scale = float(np.abs(H).sum(axis=1).max()) if k else 0.0
    if np.abs(H - H.T).max(initial=0.0) > 1e-12 * scale:
        raise _bad("H: not symmetric", "H", "not_symmetric")
    H = 0.5 * (H + H.T)
    eig = np.linalg.eigvalsh(H)
    norm2 = float(np.abs(eig).max(initial=0.0))
    if eig.min(initial=0.0) < -1e-10 * max(1.0, norm2):
        raise _bad(
            f"H: not positive semidefinite (smallest eigenvalue {eig.min():.3g})",
            "H",
            "not_psd",
            min_eigenvalue=float(eig.min()),
        )
    out: _F64 = H
    return out


@dataclass(frozen=True, slots=True)
class _Problem:
    kind: str
    c: _F64  # c (LP) or g (QP)
    H: _F64 | None
    A_ub: _F64 | None
    b_ub: _F64 | None
    A_eq: _F64 | None
    b_eq: _F64 | None
    bounds: _F64 | None

    @property
    def shape(self) -> list[int]:
        return [
            int(self.c.shape[0]),
            0 if self.A_ub is None else int(self.A_ub.shape[0]),
            0 if self.A_eq is None else int(self.A_eq.shape[0]),
        ]


def _validate(
    kind: str, H: Any, c: Any, A_ub: Any, b_ub: Any, A_eq: Any, b_eq: Any, bounds: Any
) -> _Problem:
    name = "c" if kind == "lp" else "g"
    c = _array(c, name, 1, (None,))
    k = c.shape[0]
    if k < 1:
        raise _bad(f"{name}: expected at least one variable", name, "shape", got=[k])
    Hv = _check_H(H, k) if kind == "qp" else None
    A_ub, b_ub = _pair(A_ub, b_ub, "A_ub", "b_ub", k)
    A_eq, b_eq = _pair(A_eq, b_eq, "A_eq", "b_eq", k)
    return _Problem(kind, c, Hv, A_ub, b_ub, A_eq, b_eq, _bounds(bounds, k))


# ---------------------------------------------------------------------------
# The SciPy backend
# ---------------------------------------------------------------------------


def _linprog_bounds(bounds: _F64 | None, k: int) -> list[tuple[float | None, float | None]]:
    if bounds is None:
        return [(None, None)] * k
    return [
        (None if np.isinf(lo) else float(lo), None if np.isinf(up) else float(up))
        for lo, up in bounds
    ]


def _violation(p: _Problem, v: _F64) -> float:
    parts = [0.0]
    if p.A_ub is not None and p.b_ub is not None:  # given together (_pair)
        parts.append(float(np.max(p.A_ub @ v - p.b_ub, initial=0.0)))
    if p.A_eq is not None and p.b_eq is not None:
        parts.append(float(np.max(np.abs(p.A_eq @ v - p.b_eq), initial=0.0)))
    if p.bounds is not None:
        with np.errstate(invalid="ignore"):
            parts.append(float(np.max(p.bounds[:, 0] - v, initial=0.0)))
            parts.append(float(np.max(v - p.bounds[:, 1], initial=0.0)))
    return max(parts)


def _feas_tol(p: _Problem) -> float:
    pieces = [np.zeros(1)]
    for b in (p.b_ub, p.b_eq, None if p.bounds is None else p.bounds.ravel()):
        if b is not None:
            pieces.append(np.abs(b[np.isfinite(b)]))
    return 1e-8 * max(1.0, float(np.max(np.concatenate(pieces))))


def _qp_objective(H: _F64, g: _F64, v: _F64) -> float:
    return float(0.5 * v @ (H @ v) + g @ v)


@dataclass(frozen=True, slots=True)
class ScipyBackend:
    r"""The SciPy backend: HiGHS for LPs, SLSQP or ``trust-constr`` for QPs.

    Parameters
    ----------
    name : str, default "scipy"
        Name used in error messages.
    qp_method : {"slsqp", "trust-constr"}, default "slsqp"
        The first QP method; the other is the fallback.
    tol : float, default 1e-10
        QP tolerance, positive (SLSQP ``ftol``; ``trust-constr`` ``gtol`` and
        ``xtol``).
    max_iter : int, default 1000
        SLSQP iteration limit; ``trust-constr`` gets ``10 * max_iter``.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad field.

    See Also
    --------
    SolverBackend : The protocol it implements.
    solve_lp : Solve a linear program with a backend.
    solve_qp : Solve a quadratic program with a backend.

    Notes
    -----
    LPs go to ``scipy.optimize.linprog(method="highs")``. A QP with no
    constraints and free variables is solved in closed form,
    :math:`v = -H^{+} g` with eigenvalues below :math:`10^{-12}\lambda_{\max}`
    dropped, and is reported unbounded when :math:`\|Hv + g\|` exceeds
    :math:`10^{-8}\max(1, \|g\|)`. Otherwise a feasible start is found by an
    LP with zero cost, and ``qp_method`` runs from it, then the other method
    if the first fails or its answer violates the constraints by more than
    :math:`10^{-8}` times the largest finite right-hand side or bound (at
    least 1).

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> backend = ac.ScipyBackend(qp_method="trust-constr")
    >>> out = ac.solve_qp(np.eye(2), [-2.0, -2.0], A_ub=[[1.0, 1.0]], b_ub=[1.0],
    ...                   backend=backend)
    >>> out.status, out.x.round(4)
    ('optimal', array([0.5, 0.5]))
    """

    name: str = "scipy"
    qp_method: str = "slsqp"
    tol: float = 1e-10
    max_iter: int = 1_000

    def __post_init__(self) -> None:
        """Validate the fields."""
        if not isinstance(self.name, str):
            raise _bad("name: expected a string", "name", "type")
        _v.choice(self.qp_method, "qp_method", _QP_METHODS)
        _v.real(self.tol, "tol", lo=0.0, lo_open=True)
        _v.integer(self.max_iter, "max_iter", lo=1)

    # -- LP ---------------------------------------------------------------

    def lp(
        self,
        c: ArrayLike,
        A_ub: ArrayLike | None = None,
        b_ub: ArrayLike | None = None,
        A_eq: ArrayLike | None = None,
        b_eq: ArrayLike | None = None,
        bounds: ArrayLike | None = None,
    ) -> SolverOutput:
        """Minimize ``c @ v`` subject to the given blocks with ``linprog(method="highs")``.

        Parameters
        ----------
        c : array_like, shape (k,), float
            Cost vector.
        A_ub : array_like, shape (r, k), float, optional
            Inequality matrix, given together with ``b_ub``.
        b_ub : array_like, shape (r,), float, optional
            Inequality right-hand side.
        A_eq : array_like, shape (e, k), float, optional
            Equality matrix, given together with ``b_eq``.
        b_eq : array_like, shape (e,), float, optional
            Equality right-hand side.
        bounds : array_like, shape (k, 2), float, optional
            Rows ``(lower, upper)``, possibly infinite; ``None`` means free.

        Returns
        -------
        SolverOutput
            The outcome, with any status; this method does not raise for a
            solver failure.

        Raises
        ------
        InputError
            ``input.parameter`` for malformed arrays.

        See Also
        --------
        solve_lp : Validates, calls a backend and raises on failure.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> out = ac.ScipyBackend().lp([1.0], A_ub=[[1.0]], b_ub=[-1.0], bounds=[[0.0, 1.0]])
        >>> out.status, out.x
        ('infeasible', None)
        """
        p = _validate("lp", None, c, A_ub, b_ub, A_eq, b_eq, bounds)
        k = p.c.shape[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = scipy.optimize.linprog(
                    p.c,
                    A_ub=p.A_ub,
                    b_ub=p.b_ub,
                    A_eq=p.A_eq,
                    b_eq=p.b_eq,
                    bounds=_linprog_bounds(p.bounds, k),
                    method="highs",
                )
        except _SCIPY_ERRORS as exc:
            return SolverOutput(
                None, None, "failed", f"linprog raised {type(exc).__name__}: {exc}", None
            )
        status = _LINPROG_STATUS.get(int(res.status), "failed")
        x = None
        if status in ("optimal", "max_iter") and res.x is not None:
            x = np.asarray(res.x, dtype=np.float64)
            if x.shape != (k,) or not np.isfinite(x).all():
                x = None
                status = "failed" if status == "optimal" else status
        objective = None if x is None else float(p.c @ x)
        nit = getattr(res, "nit", None)
        return SolverOutput(
            x, objective, status, str(res.message), None if nit is None else int(nit)
        )

    # -- QP ---------------------------------------------------------------

    def qp(
        self,
        H: ArrayLike,
        g: ArrayLike,
        A_ub: ArrayLike | None = None,
        b_ub: ArrayLike | None = None,
        A_eq: ArrayLike | None = None,
        b_eq: ArrayLike | None = None,
        bounds: ArrayLike | None = None,
    ) -> SolverOutput:
        """Minimize ``0.5 v @ H @ v + g @ v`` subject to the given blocks.

        Parameters
        ----------
        H : array_like, shape (k, k), float
            Symmetric positive semidefinite Hessian.
        g : array_like, shape (k,), float
            Linear term.
        A_ub : array_like, shape (r, k), float, optional
            Inequality matrix, given together with ``b_ub``.
        b_ub : array_like, shape (r,), float, optional
            Inequality right-hand side.
        A_eq : array_like, shape (e, k), float, optional
            Equality matrix, given together with ``b_eq``.
        b_eq : array_like, shape (e,), float, optional
            Equality right-hand side.
        bounds : array_like, shape (k, 2), float, optional
            Rows ``(lower, upper)``, possibly infinite; ``None`` means free.

        Returns
        -------
        SolverOutput
            The outcome, with any status; this method does not raise for a
            solver failure.

        Raises
        ------
        InputError
            ``input.parameter`` for malformed arrays, or an ``H`` that is not
            symmetric or not positive semidefinite.

        See Also
        --------
        solve_qp : Validates, calls a backend and raises on failure.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> out = ac.ScipyBackend().qp(np.eye(2), [-1.0, -2.0])
        >>> out.status, out.x, out.message
        ('optimal', array([1., 2.]), 'closed-form pseudoinverse solution')
        """
        p = _validate("qp", H, g, A_ub, b_ub, A_eq, b_eq, bounds)
        Hm, gv = p.H, p.c
        assert Hm is not None  # _validate("qp", ...) always sets H
        free = p.bounds is None or bool(np.all(np.isinf(p.bounds)))
        if p.A_ub is None and p.A_eq is None and free:
            return self._unconstrained(Hm, gv)

        start = self._start(p)
        if isinstance(start, SolverOutput):
            return start
        order = (self.qp_method, _QP_METHODS[1 - _QP_METHODS.index(self.qp_method)])
        tol = _feas_tol(p)
        last: tuple[_F64 | None, bool, str, int | None] = (None, False, "", None)
        for method in order:
            x, success, at_limit, message, nit = self._run(method, p, start)
            if success and x is not None and np.isfinite(x).all() and _violation(p, x) <= tol:
                return SolverOutput(x, _qp_objective(Hm, gv, x), "optimal", message, nit)
            if success and x is not None:
                message = (
                    f"{message} (constraint violation {_violation(p, x):.3g} exceeds {tol:.3g})"
                )
            last = (x, at_limit, message, nit)
        x, at_limit, message, nit = last
        if at_limit and x is not None and np.isfinite(x).all():
            return SolverOutput(x, _qp_objective(Hm, gv, x), "max_iter", message, nit)
        return SolverOutput(None, None, "failed", message, nit)

    @staticmethod
    def _unconstrained(H: _F64, g: _F64) -> SolverOutput:
        lam, V = np.linalg.eigh(H)
        lam_max = float(np.max(lam, initial=0.0))
        keep = lam > 1e-12 * lam_max if lam_max > 0 else np.zeros(lam.shape, dtype=bool)
        inv = np.zeros_like(lam)
        inv[keep] = 1.0 / lam[keep]
        v = -(V @ (inv * (V.T @ g)))
        resid = float(np.linalg.norm(H @ v + g))
        if resid > 1e-8 * max(1.0, float(np.linalg.norm(g))):
            return SolverOutput(
                None, None, "unbounded", f"unconstrained QP is unbounded (residual {resid:.3g})", 0
            )
        return SolverOutput(
            v, _qp_objective(H, g, v), "optimal", "closed-form pseudoinverse solution", 0
        )

    @staticmethod
    def _start(p: _Problem) -> _F64 | SolverOutput:
        k = p.c.shape[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = scipy.optimize.linprog(
                    np.zeros(k),
                    A_ub=p.A_ub,
                    b_ub=p.b_ub,
                    A_eq=p.A_eq,
                    b_eq=p.b_eq,
                    bounds=_linprog_bounds(p.bounds, k),
                    method="highs",
                )
        except _SCIPY_ERRORS as exc:
            return SolverOutput(
                None, None, "failed", f"feasible-start LP raised {type(exc).__name__}: {exc}", None
            )
        status = _LINPROG_STATUS.get(int(res.status), "failed")
        if status == "infeasible":
            return SolverOutput(
                None,
                None,
                "infeasible",
                f"feasible-start LP: {res.message}",
                getattr(res, "nit", None),
            )
        if status != "optimal" or res.x is None:
            return SolverOutput(
                None,
                None,
                "failed",
                f"feasible-start LP: {res.message}",
                getattr(res, "nit", None),
            )
        return np.asarray(res.x, dtype=np.float64)

    def _run(
        self, method: str, p: _Problem, x0: _F64
    ) -> tuple[_F64 | None, bool, bool, str, int | None]:
        H, g = p.H, p.c
        assert H is not None  # only called for a QP

        def fun(v: _F64) -> Any:
            return 0.5 * v @ (H @ v) + g @ v

        def jac(v: _F64) -> _F64:
            return H @ v + g

        bnds = None if p.bounds is None else scipy.optimize.Bounds(p.bounds[:, 0], p.bounds[:, 1])
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                if method == "slsqp":
                    cons = []
                    if p.A_ub is not None:
                        A, b = p.A_ub, p.b_ub
                        cons.append(
                            {"type": "ineq", "fun": lambda v: b - A @ v, "jac": lambda v: -A}
                        )
                    if p.A_eq is not None:
                        Ae, be = p.A_eq, p.b_eq
                        cons.append(
                            {"type": "eq", "fun": lambda v: Ae @ v - be, "jac": lambda v: Ae}
                        )
                    res = scipy.optimize.minimize(
                        fun,
                        x0.copy(),
                        jac=jac,
                        method="SLSQP",
                        bounds=bnds,
                        constraints=cons,
                        options={"ftol": self.tol, "maxiter": self.max_iter},
                    )
                    at_limit = int(res.status) == 9
                else:
                    cons = []
                    if p.A_ub is not None:
                        cons.append(scipy.optimize.LinearConstraint(p.A_ub, -np.inf, p.b_ub))
                    if p.A_eq is not None:
                        cons.append(scipy.optimize.LinearConstraint(p.A_eq, p.b_eq, p.b_eq))
                    res = scipy.optimize.minimize(
                        fun,
                        x0.copy(),
                        jac=jac,
                        hess=lambda v: H,
                        method="trust-constr",
                        bounds=bnds,
                        constraints=cons,
                        options={
                            "gtol": self.tol,
                            "xtol": self.tol,
                            "maxiter": 10 * self.max_iter,
                        },
                    )
                    at_limit = int(res.status) == 0
        except _SCIPY_ERRORS as exc:
            return None, False, False, f"{method} raised {type(exc).__name__}: {exc}", None
        x = None if res.x is None else np.asarray(res.x, dtype=np.float64).ravel()
        nit = getattr(res, "nit", None)
        return (
            x,
            bool(res.success),
            at_limit,
            f"{method}: {res.message}",
            None if nit is None else int(nit),
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def get_backend(backend: SolverBackend | str | None) -> SolverBackend:
    """Resolve a ``backend`` argument to a backend object.

    Parameters
    ----------
    backend : SolverBackend, "scipy" or None
        ``None`` or ``"scipy"`` give a default :class:`ScipyBackend`; an object
        with the members of :class:`SolverBackend` is returned unchanged.

    Returns
    -------
    SolverBackend
        The backend.

    Raises
    ------
    InputError
        ``input.parameter`` for any other value.

    See Also
    --------
    ScipyBackend : The built-in backend.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> ac.get_backend(None)
    ScipyBackend(name='scipy', qp_method='slsqp', tol=1e-10, max_iter=1000)
    """
    if backend is None or (isinstance(backend, str) and backend == "scipy"):
        return ScipyBackend()
    if not isinstance(backend, str) and isinstance(backend, SolverBackend):
        return backend
    raise InputError(
        f"backend: expected None, 'scipy' or a SolverBackend object, got {backend!r}",
        code="input.parameter",
        details={"param": "backend", "expected": ["scipy"], "got": repr(backend)},
    )


def _finish(p: _Problem, backend: SolverBackend, out: Any) -> SolverOutput:
    name = str(getattr(backend, "name", type(backend).__name__))
    label = p.kind.upper()
    if not isinstance(out, SolverOutput):
        status, message, n_iter = (
            "failed",
            f"backend returned {type(out).__name__}, not SolverOutput",
            None,
        )
    else:
        status, message, n_iter = out.status, out.message, out.n_iter
        if status == "optimal":
            k = p.c.shape[0]
            if out.x is not None and out.x.shape == (k,) and np.isfinite(out.x).all():
                return out
            status, message = (
                "failed",
                f"backend reported optimal without a finite ({k},) solution: {message}",
            )
    text = f"{label} solve by backend {name!r} ended with status {status!r}: {message}"
    details = {
        "problem": p.kind,
        "backend": name,
        "status": status,
        "message": message,
        "n_iter": n_iter,
        "shape": p.shape,
    }
    if status == "infeasible":
        raise SolverError(text, code="solver.infeasible", details=details)
    if status == "unbounded":
        raise SolverError(text, code="solver.unbounded", details=details)
    raise SolverError(text, code="solver.failed", details=details)  # "max_iter" and "failed"


def solve_lp(
    c: ArrayLike,
    A_ub: ArrayLike | None = None,
    b_ub: ArrayLike | None = None,
    A_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    bounds: ArrayLike | None = None,
    *,
    backend: SolverBackend | str | None = None,
) -> SolverOutput:
    r"""Solve a linear program.

    Parameters
    ----------
    c : array_like, shape (k,), float
        Cost vector, ``k >= 1``, finite.
    A_ub : array_like, shape (r, k), float, optional
        Inequality matrix, given together with ``b_ub``.
    b_ub : array_like, shape (r,), float, optional
        Inequality right-hand side.
    A_eq : array_like, shape (e, k), float, optional
        Equality matrix, given together with ``b_eq``.
    b_eq : array_like, shape (e,), float, optional
        Equality right-hand side.
    bounds : array_like, shape (k, 2), float, optional
        Rows ``(lower, upper)``; ``-inf`` and ``+inf`` are allowed. ``None``
        means every variable is free (unlike ``scipy.optimize.linprog``, whose
        default is :math:`v \ge 0`).
    backend : SolverBackend, "scipy" or None, keyword-only
        The backend; ``None`` uses :class:`ScipyBackend`.

    Returns
    -------
    SolverOutput
        The solution, always with ``status == "optimal"``.

    Raises
    ------
    InputError
        ``input.parameter`` for malformed arrays (``details["reason"]`` names
        the problem) or a bad backend.
    SolverError
        ``solver.infeasible``, ``solver.unbounded``, or ``solver.failed``
        (iteration limit or other failure); ``details`` hold the status,
        message and problem shape.

    See Also
    --------
    solve_qp : Solve a convex quadratic program.

    Notes
    -----
    The program is

    .. math::

        \min_v c^\top v \quad \text{subject to} \quad A_{ub} v \le b_{ub},\;
        A_{eq} v = b_{eq},\; \ell \le v \le u.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> out = ac.solve_lp([1.0, 1.0], bounds=[[0.0, 1.0], [-1.0, 2.0]])
    >>> out.status, out.x, out.objective
    ('optimal', array([ 0., -1.]), -1.0)
    """
    p = _validate("lp", None, c, A_ub, b_ub, A_eq, b_eq, bounds)
    be = get_backend(backend)
    out = be.lp(p.c, A_ub=p.A_ub, b_ub=p.b_ub, A_eq=p.A_eq, b_eq=p.b_eq, bounds=p.bounds)
    return _finish(p, be, out)


def solve_qp(
    H: ArrayLike,
    g: ArrayLike,
    A_ub: ArrayLike | None = None,
    b_ub: ArrayLike | None = None,
    A_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    bounds: ArrayLike | None = None,
    *,
    backend: SolverBackend | str | None = None,
) -> SolverOutput:
    r"""Solve a convex quadratic program.

    Parameters
    ----------
    H : array_like, shape (k, k), float
        Hessian, symmetric to :math:`10^{-12}` times its largest absolute row
        sum and positive semidefinite to :math:`-10^{-10}\max(1, \|H\|_2)`; it
        is symmetrized before use.
    g : array_like, shape (k,), float
        Linear term, ``k >= 1``.
    A_ub : array_like, shape (r, k), float, optional
        Inequality matrix, given together with ``b_ub``.
    b_ub : array_like, shape (r,), float, optional
        Inequality right-hand side.
    A_eq : array_like, shape (e, k), float, optional
        Equality matrix, given together with ``b_eq``.
    b_eq : array_like, shape (e,), float, optional
        Equality right-hand side.
    bounds : array_like, shape (k, 2), float, optional
        Rows ``(lower, upper)``; infinite entries allowed. ``None`` means free.
    backend : SolverBackend, "scipy" or None, keyword-only
        The backend; ``None`` uses :class:`ScipyBackend`.

    Returns
    -------
    SolverOutput
        The solution, always with ``status == "optimal"``.

    Raises
    ------
    InputError
        ``input.parameter`` for malformed arrays or a bad backend;
        ``details["reason"]`` is ``"not_symmetric"`` or ``"not_psd"`` for a
        bad ``H``.
    SolverError
        ``solver.infeasible``, ``solver.unbounded`` or ``solver.failed``, as
        for :func:`solve_lp`.

    See Also
    --------
    solve_lp : Solve a linear program.

    Notes
    -----
    The program is

    .. math::

        \min_v \tfrac12 v^\top H v + g^\top v \quad \text{subject to} \quad
        A_{ub} v \le b_{ub},\; A_{eq} v = b_{eq},\; \ell \le v \le u.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> out = ac.solve_qp(np.eye(2), [-2.0, -2.0], A_ub=[[1.0, 1.0]], b_ub=[1.0])
    >>> out.status, out.x.round(4), round(out.objective, 4)
    ('optimal', array([0.5, 0.5]), -1.75)
    """
    p = _validate("qp", H, g, A_ub, b_ub, A_eq, b_eq, bounds)
    assert p.H is not None  # _validate("qp", ...) always sets H
    be = get_backend(backend)
    out = be.qp(p.H, p.c, A_ub=p.A_ub, b_ub=p.b_ub, A_eq=p.A_eq, b_eq=p.b_eq, bounds=p.bounds)
    return _finish(p, be, out)
