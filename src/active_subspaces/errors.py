"""Errors, warnings and their stable codes.

Every error and warning raised by the library carries a ``code`` of the form
``<family>.<snake_case>`` and a JSON-compatible ``details`` mapping. Codes are
permanent: compare them, not messages. The table of codes is in the
documentation. Warnings point at the first line of the caller's code outside
the package.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Mapping
from types import FrameType
from typing import Any

from active_subspaces import _codes

__all__ = (
    "AccuracyWarning",
    "ActiveSubspacesDeprecationWarning",
    "ActiveSubspacesError",
    "ActiveSubspacesWarning",
    "ConvergenceWarning",
    "ExperimentalWarning",
    "InputError",
    "ModelError",
    "NumericalError",
    "SchemaError",
    "SolverError",
    "UnsupportedError",
    "ValidityWarning",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _rebuild(
    cls: type[ActiveSubspacesError] | type[ActiveSubspacesWarning],
    message: str,
    code: str,
    details: dict[str, Any],
) -> ActiveSubspacesError | ActiveSubspacesWarning:
    return cls(message, code=code, details=details)


class ActiveSubspacesError(Exception):
    """Base class of every library error.

    Every error carries a stable ``code`` of the form ``<family>.<snake_case>``
    and a JSON-compatible ``details`` mapping. ``str(err)`` is the message
    followed by the code in brackets. Catch this class to catch any error the
    library raises; catch a subclass (or its built-in base, such as
    ``ValueError``) to catch one family. Errors pickle by message, code and
    details, so they cross process boundaries intact.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"<family>.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    InputError : Invalid input (``input.*``).
    SchemaError : A serialized result cannot be decoded (``schema.*``).
    UnsupportedError : Operation not defined for this input (``unsupported.*``).
    NumericalError : Failed factorization or non-finite values (``numerical.*``).
    SolverError : Optimization backend failure (``solver.*``).
    ModelError : The user's model failed (``model.*``).
    ActiveSubspacesWarning : The base class of the warnings.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> try:
    ...     ac.Uniform([0.0], [-1.0])
    ... except ac.ActiveSubspacesError as err:
    ...     print(err.code, type(err).__name__)
    input.parameter InputError
    """

    family: str = ""

    def __init__(self, message: str, *, code: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.message = str(message)
        self.code = code
        self.details = dict(details) if details is not None else {}

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by message, code and details."""
        return (_rebuild, (type(self), self.message, self.code, self.details))

    def __str__(self) -> str:
        """Return the message with the code in brackets."""
        return f"{self.message} [{self.code}]"


class InputError(ActiveSubspacesError, ValueError):
    """Invalid input: a bad array, parameter, space or sample count.

    Code family ``input.*``. Also a ``ValueError``. Raised before any
    computation starts, so no partial result exists.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"input.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    ActiveSubspacesError : The base class.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> err = ac.InputError("X must be 2-D", code="input.shape", details={"ndim": 1})
    >>> print(err)
    X must be 2-D [input.shape]
    >>> isinstance(err, ac.ActiveSubspacesError), isinstance(err, ValueError)
    (True, True)
    """

    family = "input"


class SchemaError(ActiveSubspacesError, ValueError):
    """A serialized result cannot be decoded.

    Code family ``schema.*``: an unknown type tag, a schema version newer than
    this library, or a malformed dictionary or file. Also a ``ValueError``.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"schema.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    active_subspaces.load : Load a result from a file.
    active_subspaces.from_dict : Rebuild a result from a dictionary.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> err = ac.SchemaError(
    ...     "unknown result type",
    ...     code="schema.unknown_type", details={"type": "Foo"},
    ... )
    >>> print(err)
    unknown result type [schema.unknown_type]
    >>> isinstance(err, ac.ActiveSubspacesError), isinstance(err, ValueError)
    (True, True)
    """

    family = "schema"


class UnsupportedError(ActiveSubspacesError, NotImplementedError):
    """An operation is not defined for this kind of input.

    Code family ``unsupported.*``, for example an operation that needs a
    uniform input space applied to a Gaussian one. Also a
    ``NotImplementedError``.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"unsupported.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    ActiveSubspacesError : The base class.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> err = ac.UnsupportedError("needs a Uniform space", code="unsupported.space", details=None)
    >>> print(err)
    needs a Uniform space [unsupported.space]
    >>> isinstance(err, ac.ActiveSubspacesError), isinstance(err, NotImplementedError)
    (True, True)
    """

    family = "unsupported"


class NumericalError(ActiveSubspacesError, RuntimeError):
    """A factorization failed or a computation produced non-finite values.

    Code family ``numerical.*``. Raised when finite inputs lead to NaN or
    infinity, or a required factorization or solve fails. Also a
    ``RuntimeError``.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"numerical.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    ActiveSubspacesError : The base class.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> err = ac.NumericalError(
    ...     "Cholesky factorization failed",
    ...     code="numerical.singular", details=None,
    ... )
    >>> print(err)
    Cholesky factorization failed [numerical.singular]
    >>> isinstance(err, ac.ActiveSubspacesError), isinstance(err, RuntimeError)
    (True, True)
    """

    family = "numerical"


class SolverError(ActiveSubspacesError, RuntimeError):
    """An optimization backend failed or the problem is infeasible.

    Code family ``solver.*``. ``details`` holds the backend's status and
    message where it reports them. Also a ``RuntimeError``.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"solver.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    active_subspaces.solvers.solve_lp : Solve a linear program.
    active_subspaces.solvers.solve_qp : Solve a convex quadratic program.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> err = ac.SolverError(
    ...     "constraints infeasible",
    ...     code="solver.infeasible", details={"status": 2},
    ... )
    >>> print(err)
    constraints infeasible [solver.infeasible]
    >>> isinstance(err, ac.ActiveSubspacesError), isinstance(err, RuntimeError)
    (True, True)
    """

    family = "solver"


class ModelError(ActiveSubspacesError, RuntimeError):
    """The user's model raised or returned bad values.

    Code family ``model.*``. When the model itself raised, the original
    exception is the ``__cause__`` and ``details`` holds the indices of the
    failing rows. Also a ``RuntimeError``.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"model.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    active_subspaces.models.evaluate : Evaluates a model and applies the failure policy.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> err = ac.ModelError(
    ...     "the model returned NaN",
    ...     code="model.nonfinite", details={"rows": [3]},
    ... )
    >>> print(err)
    the model returned NaN [model.nonfinite]
    >>> isinstance(err, ac.ActiveSubspacesError), isinstance(err, RuntimeError)
    (True, True)
    """

    family = "model"


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class ActiveSubspacesWarning(UserWarning):
    """Base class of every library warning.

    Every warning carries a stable ``code`` and a JSON-compatible ``details``
    mapping, like the errors. The library issues warnings so that they point at
    the first line of the caller's code outside the package. Filter by class, or
    by code through the warning's message (``str(w)`` ends with the code in
    brackets). A subclass of ``UserWarning``.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"<family>.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    ValidityWarning : A method used outside its assumptions (``validity.*``).
    AccuracyWarning : Accuracy likely degraded (``accuracy.*``).
    ConvergenceWarning : Early stop or failed starts (``convergence.*``).
    ExperimentalWarning : An experimental module was imported.
    ActiveSubspacesError : The base class of the errors.

    Examples
    --------
    >>> import warnings
    >>> import active_subspaces as ac
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", ac.AccuracyWarning)  # silence one family
    ...     pass
    >>> issubclass(ac.AccuracyWarning, ac.ActiveSubspacesWarning)
    True
    """

    family: str = ""

    def __init__(self, message: str, *, code: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.message = str(message)
        self.code = code
        self.details = dict(details) if details is not None else {}

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by message, code and details."""
        return (_rebuild, (type(self), self.message, self.code, self.details))

    def __str__(self) -> str:
        """Return the message with the code in brackets."""
        return f"{self.message} [{self.code}]"


class ValidityWarning(ActiveSubspacesWarning):
    """A method was used outside its assumptions.

    Code family ``validity.*``: for example an estimator whose theory needs
    Gaussian inputs applied to a uniform space, or a bootstrap with
    non-uniform quadrature weights. The result is returned but its
    interpretation is not supported by the theory.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"validity.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    ActiveSubspacesWarning : The base class.

    Examples
    --------
    >>> import warnings
    >>> import active_subspaces as ac
    >>> w = ac.ValidityWarning("pHd assumes Gaussian inputs", code="validity.assumption")
    >>> print(w)
    pHd assumes Gaussian inputs [validity.assumption]
    >>> isinstance(w, ac.ActiveSubspacesWarning), w.code.startswith("validity.")
    (True, True)
    """

    family = "validity"


class AccuracyWarning(ActiveSubspacesWarning):
    """The result is valid but its accuracy is likely degraded.

    Code family ``accuracy.*``: for example a rank-deficient least-squares
    fit, correlated hit-and-run draws, or renormalized quadrature weights.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"accuracy.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    ActiveSubspacesWarning : The base class.

    Examples
    --------
    >>> import warnings
    >>> import active_subspaces as ac
    >>> w = ac.AccuracyWarning(
    ...     "least-squares fit is rank-deficient",
    ...     code="accuracy.rank_deficient",
    ... )
    >>> print(w)
    least-squares fit is rank-deficient [accuracy.rank_deficient]
    >>> isinstance(w, ac.ActiveSubspacesWarning), w.code.startswith("accuracy.")
    (True, True)
    """

    family = "accuracy"


class ConvergenceWarning(ActiveSubspacesWarning):
    """An iteration stopped early or some starts failed.

    Code family ``convergence.*``. The best iterate, or the result of the
    starts that succeeded, is returned.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"convergence.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    ActiveSubspacesWarning : The base class.

    Examples
    --------
    >>> import warnings
    >>> import active_subspaces as ac
    >>> w = ac.ConvergenceWarning("iteration limit reached", code="convergence.max_iter")
    >>> print(w)
    iteration limit reached [convergence.max_iter]
    >>> isinstance(w, ac.ActiveSubspacesWarning), w.code.startswith("convergence.")
    (True, True)
    """

    family = "convergence"


class ExperimentalWarning(ActiveSubspacesWarning):
    """An experimental module was imported.

    Code family ``experimental.*``. Issued once per process, on the first
    import of ``active_subspaces.experimental``. Experimental modules carry no
    compatibility promise.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"experimental.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    ActiveSubspacesWarning : The base class.

    Examples
    --------
    >>> import warnings
    >>> import active_subspaces as ac
    >>> w = ac.ExperimentalWarning("experimental module imported", code="experimental.module")
    >>> print(w)
    experimental module imported [experimental.module]
    >>> isinstance(w, ac.ActiveSubspacesWarning), w.code.startswith("experimental.")
    (True, True)
    """

    family = "experimental"


class ActiveSubspacesDeprecationWarning(ActiveSubspacesWarning, DeprecationWarning):
    """A public name, parameter or default that will change in the next major release.

    Code family ``deprecated.*``. Also a ``DeprecationWarning``, so Python's
    default filters (PEP 565) show it in scripts and notebooks. Version 1.0
    issues none.

    Parameters
    ----------
    message : str
        Human-readable description of the problem.
    code : str, keyword-only
        Stable code ``"deprecated.<name>"`` from the code table. Programs should
        compare codes, not messages.
    details : Mapping[str, Any] or None, keyword-only
        JSON-compatible facts about the failure (for example the offending
        shape). Stored as a new ``dict``; ``None`` gives ``{}``.

    Attributes
    ----------
    message : str
        The message without the code.
    code : str
        The stable code.
    details : dict[str, Any]
        The facts passed as ``details``.

    See Also
    --------
    ActiveSubspacesWarning : The base class.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> w = ac.ActiveSubspacesDeprecationWarning("renamed", code="deprecated.parameter")
    >>> isinstance(w, DeprecationWarning), isinstance(w, ac.ActiveSubspacesWarning)
    (True, True)
    """

    family = "deprecated"


_FAMILIES: dict[str, type[ActiveSubspacesError] | type[ActiveSubspacesWarning]] = {
    cls.family: cls
    for cls in (
        InputError,
        SchemaError,
        UnsupportedError,
        NumericalError,
        SolverError,
        ModelError,
        ValidityWarning,
        AccuracyWarning,
        ConvergenceWarning,
        ExperimentalWarning,
        ActiveSubspacesDeprecationWarning,
    )
}


def _check_code_table() -> None:
    """Check, at import, that each ``CODES`` row names a class defined here and its family."""
    by_name = {cls.__name__: cls for cls in _FAMILIES.values()}
    for code, (class_name, meaning) in _codes.CODES.items():
        family, _, rest = code.partition(".")
        cls = by_name.get(class_name)
        if cls is None or cls.family != family or not rest or not meaning:
            raise RuntimeError(
                f"bad row in active_subspaces._codes.CODES: {code!r} -> {class_name!r}"
            )


_check_code_table()


def _registered_codes() -> dict[str, str]:
    """Return ``{code: meaning}`` for every code of the static table."""
    return {code: meaning for code, (_, meaning) in _codes.CODES.items()}


def _family_class(code: str) -> type[ActiveSubspacesError] | type[ActiveSubspacesWarning]:
    """Return the error or warning class that owns the family of ``code``."""
    return _FAMILIES[code.split(".", 1)[0]]


# ---------------------------------------------------------------------------
# Issuing warnings
# ---------------------------------------------------------------------------

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_USE_SKIP_PREFIXES = sys.version_info >= (3, 12)


def _is_internal_frame(frame: FrameType) -> bool:
    filename = frame.f_code.co_filename
    return "importlib" in filename and "_bootstrap" in filename


def _in_package(frame: FrameType) -> bool:
    return os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_DIR + os.sep)


def _stacklevel() -> int:
    """Stack level of the first frame outside the package.

    Counted from ``_warn``'s call to ``warnings.warn``.
    """
    frame: FrameType | None = sys._getframe(2)  # caller of _warn
    level = 2
    while frame is not None and (_in_package(frame) or _is_internal_frame(frame)):
        if not _is_internal_frame(frame):
            level += 1
        frame = frame.f_back
    return level


def _warn(
    category: type[ActiveSubspacesWarning],
    message: str,
    *,
    code: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Issue ``category(message, code=, details=)`` attributed to the first outside caller."""
    instance = category(message, code=code, details=details)
    if _USE_SKIP_PREFIXES:
        # skip_file_prefixes exists from Python 3.12, which _USE_SKIP_PREFIXES checks
        warnings.warn(  # type: ignore[call-overload,unused-ignore]
            instance, stacklevel=2, skip_file_prefixes=(_PACKAGE_DIR + os.sep,)
        )
    else:
        warnings.warn(instance, stacklevel=_stacklevel())
