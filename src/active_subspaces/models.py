"""The model contract and model evaluation.

A :class:`Model` bundles batch callables for the values and gradients of a
scalar quantity of interest (QoI) with the input space they are defined on.
Callables see **physical** coordinates: ``value`` maps ``(k, m) -> (k,)`` (or
``(k, 1)``), ``grad`` maps ``(k, m) -> (k, m)`` and the optional
``value_and_grad`` returns both from one call.

:func:`evaluate`, :func:`evaluate_grads` and :func:`evaluate_with_grads` run
a model at many points, in the calling thread or on any
``concurrent.futures.Executor``. Rows are split into batches from ``M`` and
``batch_size`` only, and outputs are assembled and errors reported in batch
order, so results never depend on the executor or its worker count.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import math
import re
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from active_subspaces import _validation as _v
from active_subspaces.errors import InputError, ModelError, UnsupportedError
from active_subspaces.results import QoIResult, _invariant, _register_result
from active_subspaces.spaces import Gaussian, InputSpace, Uniform

__all__ = (
    "GradFn",
    "Model",
    "ReferenceResults",
    "ValueAndGradFn",
    "ValueFn",
    "as_model",
    "evaluate",
    "evaluate_grads",
    "evaluate_with_grads",
    "pointwise",
)

ValueFn = Callable[[NDArray[np.float64]], ArrayLike]
"""Batch value callable: physical points ``(k, m)`` to values ``(k,)`` or ``(k, 1)``."""
GradFn = Callable[[NDArray[np.float64]], ArrayLike]
"""Batch gradient callable: physical points ``(k, m)`` to physical gradients ``(k, m)``."""
ValueAndGradFn = Callable[[NDArray[np.float64]], tuple[ArrayLike, ArrayLike]]
"""Batch callable returning the pair (values ``(k,)``, physical gradients ``(k, m)``)."""

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ORTHO_TOL = 1e-10


def _param_error(message: str, param: str, **details: Any) -> InputError:
    return InputError(message, code="input.parameter", details={"param": param, **details})


# ---------------------------------------------------------------------------
# Reference results (§4.2)
# ---------------------------------------------------------------------------


@_register_result
class ReferenceResults(QoIResult):
    r"""Stored reference values of a model: the spectrum of :math:`C`, and moments of :math:`f`.

    The built-in test problems carry one, so estimates can be compared with a
    known answer. ``exact`` values come from closed forms; the others from a
    large Monte Carlo sample, with standard errors. Instances are produced by
    the library; the constructor is used by deserialization.

    Parameters
    ----------
    model : str
        Registry name of the model the values belong to.
    exact : bool
        Whether the values are closed-form. Exactly then are
        ``eigenvalue_se``, ``mean_se``, ``n_samples`` and ``seed`` all ``None``.
    eigenvalues : numpy.ndarray, shape (m,), float64
        Eigenvalues of :math:`C = \mathbb{E}_\rho[\nabla_x f \, \nabla_x f^\top]`
        in normalized coordinates, non-negative and non-increasing.
    eigenvectors : numpy.ndarray, shape (m, m), float64
        Orthonormal eigenvectors (columns), normalized coordinates.
    eigenvalue_se : numpy.ndarray, shape (m,), float64, or None
        Standard errors of the Monte Carlo eigenvalues.
    mean : float
        Mean of :math:`f` under the input density.
    mean_se : float or None
        Standard error of ``mean``.
    variance : float
        Variance of :math:`f` under the input density.
    n_samples : int or None
        Monte Carlo sample size.
    seed : int or None
        Seed of the Monte Carlo sample.
    provenance : Provenance, keyword-only
        How the values were produced.
    qoi : str, keyword-only, default "f"
        Name of the quantity of interest.

    See Also
    --------
    Model : Carries reference results in ``reference_results``.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> ref = ac.test_functions.builtin_models().get("borehole").reference_results
    >>> ref.exact, ref.eigenvalues.shape, ref.n_samples
    (False, (8,), 1000000)
    """

    TYPE = "ReferenceResults"
    SCHEMA_VERSION = 1
    model: str
    exact: bool
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    eigenvalue_se: NDArray[np.float64] | None
    mean: float
    mean_se: float | None
    variance: float
    n_samples: int | None
    seed: int | None

    def _check(self) -> None:
        lam, W = self.eigenvalues, self.eigenvectors
        _invariant(
            lam.ndim == 1 and lam.shape[0] >= 1, "eigenvalues must be a non-empty 1-D array"
        )
        m = lam.shape[0]
        _invariant(W.shape == (m, m), "eigenvectors must be (m, m)")
        _invariant(
            bool(np.all(np.isfinite(lam))) and bool(np.all(lam >= 0)),
            "eigenvalues must be finite and >= 0",
        )
        _invariant(bool(np.all(np.diff(lam) <= 0)), "eigenvalues must be non-increasing")
        _invariant(bool(np.all(np.isfinite(W))), "eigenvectors must be finite")
        _invariant(
            np.max(np.abs(W.T @ W - np.eye(m))) <= _ORTHO_TOL,
            "eigenvectors must have orthonormal columns",
        )
        mc = (self.eigenvalue_se, self.mean_se, self.n_samples, self.seed)
        _invariant(
            self.exact == all(v is None for v in mc)
            and (self.exact or all(v is not None for v in mc)),
            "exact must hold exactly when eigenvalue_se, mean_se, n_samples and seed are all None",
        )
        if self.eigenvalue_se is not None:
            se = self.eigenvalue_se
            _invariant(se.shape == (m,), "eigenvalue_se must be (m,)")
            _invariant(
                bool(np.all(np.isfinite(se))) and bool(np.all(se >= 0)),
                "eigenvalue_se must be finite and >= 0",
            )
        _invariant(math.isfinite(self.mean), "mean must be finite")
        _invariant(
            math.isfinite(self.variance) and self.variance >= 0, "variance must be finite and >= 0"
        )
        if self.mean_se is not None:
            _invariant(
                math.isfinite(self.mean_se) and self.mean_se >= 0,
                "mean_se must be finite and >= 0",
            )
        if self.n_samples is not None:
            _invariant(self.n_samples >= 1, "n_samples must be >= 1")
        _invariant(self.model != "", "model must be a non-empty string")


# ---------------------------------------------------------------------------
# The model type (§4.1, §5.1)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class Model:
    """A scalar quantity of interest on an input space, as batch callables.

    All arguments are keyword-only. The callables receive physical points and
    return values and physical gradients; the library maps to normalized
    coordinates itself.

    Parameters
    ----------
    space : Uniform or Gaussian
        The input space; fixes the dimension ``m`` and the physical coordinates.
    value : callable, optional
        Batch values, physical points ``(k, m)`` to ``(k,)`` or ``(k, 1)``.
    grad : callable, optional
        Batch gradients with respect to the physical inputs, ``(k, m)`` to
        ``(k, m)``.
    value_and_grad : callable, optional
        Both from one call, ``(k, m)`` to the pair ``((k,), (k, m))``. At least
        one of ``value`` and ``value_and_grad`` is required.
    name : str, optional
        Registry name matching ``^[a-z][a-z0-9_]{0,63}$``; required to put the
        model in a :class:`~active_subspaces.ModelRegistry`.
    description : str, default ""
        Free text.
    input_descriptions : sequence of str, optional
        One description per input, length ``m``.
    qoi : str, default "f"
        Non-empty name of the quantity of interest; results computed from the
        model carry it.
    qoi_units : str, optional
        Units of the quantity of interest.
    qoi_description : str, default ""
        Free text about the quantity of interest.
    cost : float, optional
        Non-negative cost of one evaluation, in any unit, for planning.
    batch_size : int, optional
        Default rows per call, and the largest ``k`` the callables accept.
    references : sequence of str, default ()
        Literature for the model, as non-empty strings.
    reference_results : ReferenceResults, optional
        Known values for the model, with ``m`` eigenvalues and the same
        ``qoi`` (and ``model`` equal to ``name`` when a name is given).

    Raises
    ------
    InputError
        ``input.parameter`` for any invalid field.

    See Also
    --------
    evaluate : Evaluate the values at many points.
    as_model : Wrap a bare callable.
    pointwise : Turn a per-point function into a batch callable.

    Notes
    -----
    Models compare by identity. They pickle when their callables do, so a
    model defined with module-level functions runs on a process pool.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> model = ac.Model(
    ...     space=space,
    ...     value=lambda X: X[:, 0] * X[:, 1],
    ...     grad=lambda X: X[:, ::-1],
    ...     name="product",
    ... )
    >>> model.dim, model.has_grad
    (2, True)
    >>> ac.evaluate(model, X=[[1.0, 2.0], [0.5, 0.5]])
    array([2.  , 0.25])
    """

    space: InputSpace
    value: ValueFn | None = None
    grad: GradFn | None = None
    value_and_grad: ValueAndGradFn | None = None
    name: str | None = None
    description: str = ""
    input_descriptions: tuple[str, ...] | None = None
    qoi: str = "f"
    qoi_units: str | None = None
    qoi_description: str = ""
    cost: float | None = None
    batch_size: int | None = None
    references: tuple[str, ...] = ()
    reference_results: ReferenceResults | None = None

    def __post_init__(self) -> None:
        """Validate the fields."""
        if not isinstance(self.space, (Uniform, Gaussian)):
            raise _param_error(
                "space: expected a Uniform or Gaussian input space, got "
                f"{type(self.space).__name__}",
                "space",
                got=type(self.space).__name__,
            )
        m = self.space.dim
        if self.value is None and self.value_and_grad is None:
            raise _param_error("a Model needs value or value_and_grad", "value")
        for key in ("value", "grad", "value_and_grad"):
            fn = getattr(self, key)
            if fn is not None and not callable(fn):
                raise _param_error(f"{key}: expected a callable, got {type(fn).__name__}", key)
        if self.name is not None and (
            not isinstance(self.name, str) or not _NAME_RE.match(self.name)
        ):
            raise _param_error(
                f"name: expected None or a string matching {_NAME_RE.pattern}, got {self.name!r}",
                "name",
            )
        for key in ("description", "qoi_description"):
            if not isinstance(getattr(self, key), str):
                raise _param_error(f"{key}: expected a string", key)
        if not isinstance(self.qoi, str) or not self.qoi:
            raise _param_error("qoi: expected a non-empty string", "qoi")
        if self.qoi_units is not None and not isinstance(self.qoi_units, str):
            raise _param_error("qoi_units: expected a string or None", "qoi_units")
        if self.input_descriptions is not None:
            desc = self.input_descriptions
            if isinstance(desc, str) or not hasattr(desc, "__iter__"):
                raise _param_error(
                    "input_descriptions: expected a sequence of strings", "input_descriptions"
                )
            desc = tuple(desc)
            if len(desc) != m or not all(isinstance(s, str) for s in desc):
                raise _param_error(
                    f"input_descriptions: expected {m} strings, got {len(desc)} items",
                    "input_descriptions",
                    expected=m,
                )
            object.__setattr__(self, "input_descriptions", desc)
        if self.cost is not None:
            object.__setattr__(self, "cost", _v.real(self.cost, "cost", lo=0.0))
        if self.batch_size is not None:
            object.__setattr__(self, "batch_size", _v.integer(self.batch_size, "batch_size", lo=1))
        refs = self.references
        if isinstance(refs, str) or not hasattr(refs, "__iter__"):
            raise _param_error("references: expected a sequence of strings", "references")
        refs = tuple(refs)
        if not all(isinstance(r, str) and r for r in refs):
            raise _param_error("references: expected non-empty strings", "references")
        object.__setattr__(self, "references", refs)
        rr = self.reference_results
        if rr is not None:
            if not isinstance(rr, ReferenceResults):
                raise _param_error(
                    "reference_results: expected a ReferenceResults or None", "reference_results"
                )
            if rr.eigenvalues.shape[0] != m:
                raise _param_error(
                    f"reference_results: expected {m} eigenvalues, got {rr.eigenvalues.shape[0]}",
                    "reference_results",
                )
            if rr.qoi != self.qoi:
                raise _param_error(
                    f"reference_results: qoi {rr.qoi!r} differs from the model's {self.qoi!r}",
                    "reference_results",
                )
            if self.name is not None and rr.model != self.name:
                raise _param_error(
                    f"reference_results: model {rr.model!r} differs from the model name "
                    f"{self.name!r}",
                    "reference_results",
                )

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by fields."""
        return (
            _rebuild_model,
            ({f.name: getattr(self, f.name) for f in dataclasses.fields(self)},),
        )

    @property
    def dim(self) -> int:
        """Number of inputs ``m``, equal to ``space.dim``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ac.Model(space=ac.Uniform.reference(3), value=lambda X: X.sum(axis=1)).dim
        3
        """
        return self.space.dim

    @property
    def has_grad(self) -> bool:
        """Whether gradients are available: ``grad`` or ``value_and_grad`` is set.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ac.Model(space=ac.Uniform.reference(3), value=lambda X: X.sum(axis=1)).has_grad
        False
        """
        return self.grad is not None or self.value_and_grad is not None

    def describe(self) -> dict[str, Any]:
        """Return JSON-compatible metadata of the model, without the callables.

        Returns
        -------
        dict
            Keys ``name``, ``description``, ``dim``, ``space`` (its
            ``to_dict``), ``input_descriptions``, ``qoi``, ``qoi_units``,
            ``qoi_description``, ``has_grad``, ``has_value_and_grad``, ``cost``,
            ``batch_size``, ``references`` and ``reference_results`` (its
            ``to_dict`` or ``None``).

        See Also
        --------
        ModelRegistry.describe : The metadata of every model in a registry.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> model = ac.Model(space=ac.Uniform.reference(2), value=lambda X: X[:, 0], name="x1")
        >>> d = model.describe()
        >>> d["name"], d["dim"], d["has_grad"]
        ('x1', 2, False)
        """
        rr = self.reference_results
        return {
            "name": self.name,
            "description": self.description,
            "dim": self.dim,
            "space": self.space.to_dict(),
            "input_descriptions": None
            if self.input_descriptions is None
            else list(self.input_descriptions),
            "qoi": self.qoi,
            "qoi_units": self.qoi_units,
            "qoi_description": self.qoi_description,
            "has_grad": self.has_grad,
            "has_value_and_grad": self.value_and_grad is not None,
            "cost": self.cost,
            "batch_size": self.batch_size,
            "references": list(self.references),
            "reference_results": None if rr is None else rr.to_dict(),
        }


def _rebuild_model(kwargs: dict[str, Any]) -> Model:
    return Model(**kwargs)


# ---------------------------------------------------------------------------
# Resolving a model (§4.3.1)
# ---------------------------------------------------------------------------


def as_model(model: Model | ValueFn, *, space: InputSpace | None = None) -> Model:
    """Resolve a ``Model``, or a bare batch value callable with a space, to a ``Model``.

    Parameters
    ----------
    model : Model or callable
        A ``Model``, returned unchanged, or a batch value callable mapping
        physical points ``(k, m)`` to ``(k,)``.
    space : Uniform or Gaussian, optional, keyword-only
        Required with a callable. With a ``Model`` it is optional and must
        equal ``model.space``.

    Returns
    -------
    Model
        The model; a callable becomes ``Model(space=space, value=model)``.

    Raises
    ------
    InputError
        ``input.parameter``: ``model`` is neither a ``Model`` nor callable, a
        callable comes without a space, or ``space`` differs from the model's
        (``details["reason"] == "space_mismatch"``).

    See Also
    --------
    Model : The model contract.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> model = ac.as_model(lambda X: X.sum(axis=1), space=ac.Uniform.reference(2))
    >>> type(model).__name__, model.has_grad
    ('Model', False)
    """
    if isinstance(model, Model):
        if space is not None and not (type(space) is type(model.space) and space == model.space):
            raise InputError(
                "space: does not equal the model's space",
                code="input.parameter",
                details={"param": "space", "reason": "space_mismatch"},
            )
        return model
    if callable(model):
        if space is None:
            raise _param_error("space: required when model is a bare callable", "space")
        if not isinstance(space, (Uniform, Gaussian)):
            raise _param_error(
                f"space: expected a Uniform or Gaussian input space, got {type(space).__name__}",
                "space",
            )
        return Model(space=space, value=model)
    raise _param_error(
        f"model: expected a Model or a callable, got {type(model).__name__}",
        "model",
        got=type(model).__name__,
    )


# ---------------------------------------------------------------------------
# Wrapping a pointwise function (§4.4)
# ---------------------------------------------------------------------------


class _Pointwise:
    """Batch wrapper around a per-point function; pickles whenever ``fun`` does."""

    __slots__ = ("fun", "output")

    def __init__(self, fun: Callable[[NDArray[np.float64]], Any], output: str) -> None:
        self.fun = fun
        self.output = output

    def __getstate__(self) -> tuple[Any, str]:
        return (self.fun, self.output)

    def __setstate__(self, state: tuple[Any, str]) -> None:
        self.fun, self.output = state

    def __call__(self, X: Any) -> Any:
        X = np.asarray(X)
        rows = [np.array(X[i], dtype=np.float64, order="C", copy=True) for i in range(X.shape[0])]
        if self.output == "value":
            return np.stack([_scalar(self.fun(x)) for x in rows])
        if self.output == "grad":
            return np.stack([np.asarray(self.fun(x)) for x in rows])
        vals, grads = [], []
        for x in rows:
            out = self.fun(x)
            if not isinstance(out, (tuple, list)) or len(out) != 2:
                raise ValueError(
                    "pointwise value_and_grad: fun must return a pair (value, gradient)"
                )
            vals.append(_scalar(out[0]))
            grads.append(np.asarray(out[1]))
        return np.stack(vals), np.stack(grads)

    def __repr__(self) -> str:
        return f"pointwise({self.fun!r}, output={self.output!r})"


def _scalar(v: Any) -> NDArray[Any]:
    a = np.asarray(v)
    if a.size != 1:
        raise ValueError(
            f"pointwise: expected a number or an array of size 1, got shape {a.shape}"
        )
    return a.reshape(())


def pointwise(
    fun: Callable[[NDArray[np.float64]], Any],
    *,
    output: Literal["value", "grad", "value_and_grad"] = "value",
) -> Callable[[NDArray[np.float64]], Any]:
    """Wrap a per-point function into a batch callable.

    Parameters
    ----------
    fun : callable
        A function of one physical point, shape ``(m,)``.
    output : {"value", "grad", "value_and_grad"}, default "value", keyword-only
        What ``fun`` returns: a number, a physical gradient ``(m,)``, or the
        pair (number, gradient).

    Returns
    -------
    callable
        A batch callable on physical points ``(k, m)`` returning ``(k,)``,
        ``(k, m)`` or the pair, usable as ``value``, ``grad`` or
        ``value_and_grad`` of a :class:`Model`. It calls ``fun`` once per row,
        each time on a fresh writeable copy of the row, and pickles whenever
        ``fun`` does.

    Raises
    ------
    InputError
        ``input.parameter`` if ``fun`` is not callable or ``output`` is
        unknown.

    See Also
    --------
    Model : Takes batch callables.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> value = ac.pointwise(lambda x: float(np.sum(x**2)))
    >>> model = ac.Model(space=ac.Uniform.reference(2), value=value)
    >>> ac.evaluate(model, X=[[0.5, 0.5], [1.0, 0.0]])
    array([0.5, 1. ])
    """
    if not callable(fun):
        raise _param_error(f"fun: expected a callable, got {type(fun).__name__}", "fun")
    _v.choice(output, "output", ("value", "grad", "value_and_grad"))
    return _Pointwise(fun, output)


# ---------------------------------------------------------------------------
# Evaluation (§4.3, §5.2-5.4)
# ---------------------------------------------------------------------------


def _qualname(exc: BaseException) -> str:
    cls = type(exc)
    mod = cls.__module__
    return cls.__qualname__ if mod == "builtins" else f"{mod}.{cls.__qualname__}"


class _Task:
    """One callable to run on one batch: which callable, its role and the rows."""

    __slots__ = ("fn", "label", "start", "stop", "want")

    def __init__(
        self, label: str, fn: Callable[..., Any], start: int, stop: int, want: tuple[bool, bool]
    ) -> None:
        self.label = label  # "value" | "grad" | "value_and_grad"
        self.fn = fn
        self.start = start
        self.stop = stop
        self.want = want  # (values wanted, gradients wanted)


def _plan(model: Model, request: str) -> list[tuple[str, Any, tuple[bool, bool]]]:
    """Return the callables run per batch for ``request`` (§5.2), in call order.

    Raises before any evaluation.
    """
    if request == "value":
        if model.value is not None:
            return [("value", model.value, (True, False))]
        return [("value_and_grad", model.value_and_grad, (True, False))]
    if not model.has_grad:
        raise UnsupportedError(
            "the model provides neither grad nor value_and_grad; use finite differences "
            "explicitly",
            code="unsupported.model_gradient",
            details={"model": model.name},
        )
    if request == "grad":
        if model.grad is not None:
            return [("grad", model.grad, (False, True))]
        return [("value_and_grad", model.value_and_grad, (False, True))]
    if model.value_and_grad is not None:
        return [("value_and_grad", model.value_and_grad, (True, True))]
    return [("value", model.value, (True, False)), ("grad", model.grad, (False, True))]


def _details(task: _Task, model: Model, **extra: Any) -> dict[str, Any]:
    return {"model": model.name, "batch": [task.start, task.stop], "callable": task.label, **extra}


def _raised(exc: BaseException, task: _Task, model: Model) -> ModelError:
    err = ModelError(
        f"the model's {task.label} callable raised {_qualname(exc)} on rows [{task.start}, "
        f"{task.stop}): {exc}",
        code="model.raised",
        details=_details(task, model, exception=_qualname(exc)),
    )
    err.__cause__ = exc
    return err


def _broken(exc: BaseException, task: _Task, model: Model) -> ModelError:
    err = ModelError(
        f"the executor failed to run the {task.label} callable on rows [{task.start}, "
        f"{task.stop}): {exc}",
        code="model.executor",
        details=_details(task, model, exception=_qualname(exc)),
    )
    err.__cause__ = exc
    return err


def _as_numeric(a: Any, task: _Task, model: Model, what: str) -> np.ndarray:
    try:
        arr = np.asarray(a)
    except (TypeError, ValueError) as exc:
        raise ModelError(
            f"the model's {task.label} callable returned {what} that is not an array ({exc})",
            code="model.shape",
            details=_details(task, model),
        ) from exc
    if arr.dtype.kind not in "iuf":
        raise ModelError(
            f"the model's {task.label} callable returned {what} of dtype {arr.dtype}; expected "
            "integer or floating",
            code="model.dtype",
            details=_details(task, model, got=str(arr.dtype)),
        )
    return np.asarray(arr, dtype=np.float64)


def _check_finite(arr: np.ndarray, task: _Task, model: Model, what: str) -> None:
    ok = np.isfinite(arr)
    if ok.all():
        return
    local = int(np.argwhere(~ok)[0][0])
    row = task.start + local
    raise ModelError(
        f"the model's {task.label} callable returned NaN or inf {what} at row {row}",
        code="model.nonfinite",
        details=_details(task, model, row=row),
    )


def _check_output(
    out: Any, task: _Task, model: Model
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate one batch's output (§5.3 step 6); return the requested parts as float64."""
    k, m = task.stop - task.start, model.dim
    want_f, want_g = task.want
    if task.label == "value_and_grad":
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise ModelError(
                "the model's value_and_grad callable must return a pair (values, gradients)",
                code="model.shape",
                details=_details(task, model, expected="pair", got=type(out).__name__),
            )
        fv, gv = out
    elif task.label == "value":
        fv, gv = out, None
    else:
        fv, gv = None, out
    f = g = None
    if want_f:
        f = _as_numeric(fv, task, model, "values")
        if not (f.shape == (k,) or f.shape == (k, 1)):
            raise ModelError(
                f"the model's {task.label} callable returned values of shape {f.shape}; expected "
                f"({k},) or ({k}, 1)",
                code="model.shape",
                details=_details(task, model, expected=[k], got=list(f.shape)),
            )
        f = f.reshape(k)
        _check_finite(f, task, model, "values")
    if want_g:
        g = _as_numeric(gv, task, model, "gradients")
        if g.shape == (k,) and m == 1:
            g = g.reshape(k, 1)
        if g.shape != (k, m):
            raise ModelError(
                f"the model's {task.label} callable returned gradients of shape {g.shape}; "
                f"expected ({k}, {m})",
                code="model.shape",
                details=_details(task, model, expected=[k, m], got=list(g.shape)),
            )
        _check_finite(g, task, model, "gradients")
    return f, g


def _evaluate(
    model: Model | ValueFn,
    request: str,
    X: ArrayLike,
    space: InputSpace | None,
    executor: concurrent.futures.Executor | None,
    batch_size: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    # 1. resolve the model and check arguments
    model = as_model(model, space=space)
    if executor is not None and not isinstance(executor, concurrent.futures.Executor):
        raise _param_error(
            "executor: expected None or a concurrent.futures.Executor, got "
            f"{type(executor).__name__}",
            "executor",
        )
    if batch_size is not None:
        batch_size = _v.integer(batch_size, "batch_size", lo=1)
        if model.batch_size is not None and batch_size > model.batch_size:
            raise _param_error(
                f"batch_size: {batch_size} exceeds the model's batch_size {model.batch_size}",
                "batch_size",
                expected=[1, model.batch_size],
                got=batch_size,
            )
    # 2. callables (a missing gradient raises here)
    plan = _plan(model, request)
    # 3. points, physical
    Xp = _v.points(X, model.space, "X", normalize=False)
    M, m = Xp.shape
    # 4. partition [D-125]
    b = (
        batch_size
        if batch_size is not None
        else (model.batch_size if model.batch_size is not None else M)
    )
    tasks = [
        _Task(label, fn, s, min(s + b, M), want)
        for s in range(0, M, b)
        for (label, fn, want) in plan
    ]
    f_out = np.empty(M) if request in ("value", "both") else None
    g_out = np.empty((M, m)) if request in ("grad", "both") else None

    def store(task: _Task, out: Any) -> None:
        f, g = _check_output(out, task, model)
        if f is not None and f_out is not None:
            f_out[task.start : task.stop] = f
        if g is not None and g_out is not None:
            g_out[task.start : task.stop] = g

    def rows(task: _Task) -> np.ndarray:
        return np.array(Xp[task.start : task.stop], dtype=np.float64, order="C", copy=True)

    # 5-7. run, validate in batch order, assemble
    if executor is None:
        for task in tasks:
            try:
                out = task.fn(rows(task))
            except Exception as exc:
                raise _raised(exc, task, model) from exc
            store(task, out)
        return f_out, g_out

    futures: list[concurrent.futures.Future[Any]] = []
    try:
        for task in tasks:
            try:
                futures.append(executor.submit(task.fn, rows(task)))
            except concurrent.futures.BrokenExecutor as exc:
                raise _broken(exc, task, model) from exc
            except Exception as exc:
                raise _raised(exc, task, model) from exc
        for task, fut in zip(tasks, futures, strict=False):
            try:
                out = fut.result()
            except concurrent.futures.BrokenExecutor as exc:
                raise _broken(exc, task, model) from exc
            except concurrent.futures.CancelledError as exc:
                raise _broken(exc, task, model) from exc
            except Exception as exc:
                raise _raised(exc, task, model) from exc
            store(task, out)
    except BaseException:
        for fut in futures:
            fut.cancel()
        raise
    return f_out, g_out


def evaluate(
    model: Model | ValueFn,
    *,
    X: ArrayLike,
    space: InputSpace | None = None,
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
) -> NDArray[np.float64]:
    """Evaluate a model's values at many points.

    Parameters
    ----------
    model : Model or callable
        A ``Model``, or a bare batch value callable together with ``space``.
    X : array_like, shape (M, m), float, keyword-only
        Physical points, validated against the model's space (finite and
        inside its support).
    space : Uniform or Gaussian, optional, keyword-only
        Required with a bare callable; with a ``Model`` it must equal
        ``model.space``.
    executor : concurrent.futures.Executor, optional, keyword-only
        Where batches run; ``None`` runs them in order in the calling thread.
    batch_size : int, optional, keyword-only
        Rows per call, at most ``model.batch_size``; default
        ``model.batch_size``, else all rows in one call.

    Returns
    -------
    numpy.ndarray, shape (M,), float64
        The values, a new writeable array. They do not depend on the executor
        or its worker count.

    Raises
    ------
    InputError
        ``input.dtype``, ``input.shape``, ``input.nonfinite`` or
        ``input.out_of_space`` for a bad ``X``; ``input.parameter`` for a bad
        model, space, executor or batch size.
    ModelError
        ``model.raised`` if a callable raised (the original exception is the
        ``__cause__``); ``model.shape``, ``model.dtype`` or ``model.nonfinite``
        for a bad output; ``model.executor`` if the executor failed. The first
        failing batch, in batch order, is reported; ``details`` names the
        model, the batch's row range and the callable.

    See Also
    --------
    evaluate_grads : Evaluate the gradients.
    evaluate_with_grads : Evaluate both.

    Notes
    -----
    Rows are split into consecutive batches of ``batch_size`` rows; each
    callable receives a fresh C-contiguous float64 copy of its rows.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> ac.evaluate(lambda X: X[:, 0] * X[:, 1], X=[[1.0, 2.0], [0.5, 0.5]], space=space)
    array([2.  , 0.25])
    """
    f, _ = _evaluate(model, "value", X, space, executor, batch_size)
    assert f is not None
    return f


def evaluate_grads(
    model: Model | ValueFn,
    *,
    X: ArrayLike,
    space: InputSpace | None = None,
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
) -> NDArray[np.float64]:
    """Evaluate a model's gradients at many points.

    Parameters
    ----------
    model : Model or callable
        A ``Model`` with ``grad`` or ``value_and_grad``. A bare callable is a
        value callable and so has no gradient.
    X : array_like, shape (M, m), float, keyword-only
        Physical points, validated against the model's space.
    space : Uniform or Gaussian, optional, keyword-only
        As for :func:`evaluate`.
    executor : concurrent.futures.Executor, optional, keyword-only
        As for :func:`evaluate`.
    batch_size : int, optional, keyword-only
        As for :func:`evaluate`.

    Returns
    -------
    numpy.ndarray, shape (M, m), float64
        Gradients with respect to the physical inputs, a new array. Uses
        ``grad`` if set, else ``value_and_grad``.

    Raises
    ------
    UnsupportedError
        ``unsupported.model_gradient``, before any call, if the model has
        neither ``grad`` nor ``value_and_grad``. Finite differences are never
        substituted silently.
    InputError
        As for :func:`evaluate`.
    ModelError
        As for :func:`evaluate`.

    See Also
    --------
    evaluate : Evaluate the values.
    active_subspaces.finite_difference_gradients : Gradients by finite differences.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> model = ac.Model(space=space, value=lambda X: X[:, 0] * X[:, 1], grad=lambda X: X[:, ::-1])
    >>> ac.evaluate_grads(model, X=[[1.0, 2.0], [0.5, 0.5]])
    array([[2. , 1. ],
           [0.5, 0.5]])
    """
    _, g = _evaluate(model, "grad", X, space, executor, batch_size)
    assert g is not None
    return g


def evaluate_with_grads(
    model: Model | ValueFn,
    *,
    X: ArrayLike,
    space: InputSpace | None = None,
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate a model's values and gradients at many points.

    Parameters
    ----------
    model : Model or callable
        A ``Model`` with ``grad`` or ``value_and_grad``.
    X : array_like, shape (M, m), float, keyword-only
        Physical points, validated against the model's space.
    space : Uniform or Gaussian, optional, keyword-only
        As for :func:`evaluate`.
    executor : concurrent.futures.Executor, optional, keyword-only
        As for :func:`evaluate`.
    batch_size : int, optional, keyword-only
        As for :func:`evaluate`.

    Returns
    -------
    f : numpy.ndarray, shape (M,), float64
        The values.
    grads : numpy.ndarray, shape (M, m), float64
        Gradients with respect to the physical inputs.

    Raises
    ------
    UnsupportedError
        ``unsupported.model_gradient`` if the model has no gradient.
    InputError
        As for :func:`evaluate`.
    ModelError
        As for :func:`evaluate`.

    See Also
    --------
    evaluate : Evaluate the values.
    evaluate_grads : Evaluate the gradients.

    Notes
    -----
    Uses ``value_and_grad`` if set, else ``value`` then ``grad`` on each
    batch.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> model = ac.Model(space=space, value=lambda X: X[:, 0] * X[:, 1], grad=lambda X: X[:, ::-1])
    >>> f, G = ac.evaluate_with_grads(model, X=[[1.0, 2.0]])
    >>> f, G
    (array([2.]), array([[2., 1.]]))
    """
    f, g = _evaluate(model, "both", X, space, executor, batch_size)
    assert f is not None
    assert g is not None
    return f, g
