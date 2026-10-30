"""Results: immutable records of a computation, with provenance and serialization.

Every public function that computes something returns a subclass of
:class:`Result`. A result is a frozen dataclass whose arrays are owned,
read-only copies; derived quantities are properties. Two results compare equal
when their fields agree bitwise (provenance excluded). Each result records how
it was produced in a :class:`Provenance`.

Results serialize to JSON-compatible dicts (:meth:`Result.to_dict`,
:func:`from_dict`) and to uncompressed ``.npz`` files (:meth:`Result.save`,
:func:`load`). Loading dispatches on the type tag stored in the dict or file and
never unpickles.
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib
import json
import math
import os
import platform
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, BinaryIO, ClassVar, Literal, NoReturn, Self, Union, cast

import numpy as np
import numpy.typing as npt
import scipy

from active_subspaces import _validation as _v
from active_subspaces._version import __version__
from active_subspaces.errors import SchemaError
from active_subspaces.spaces import Gaussian, InputSpace, Uniform, _space_from_dict

# Developer notes (spec 04)
# ------------------------
# Defining a result type::
#
#     import numpy as np
#     from numpy.typing import NDArray
#
#     from active_subspaces.results import QoIResult, _invariant, _register_result
#     from active_subspaces.spaces import InputSpace
#
#     @_register_result
#     class SubspaceResult(QoIResult):
#         TYPE = "SubspaceResult"          # permanent tag
#         SCHEMA_VERSION = 1
#         eigenvalues: NDArray[np.float64]
#         W: NDArray[np.float64]
#         n: int
#         space: InputSpace
#
#         @property
#         def W1(self):                     # derived values are properties
#             return self.W[:, : self.n]
#
#         def _check(self):                 # invariants; no super() calls
#             _invariant(self.W.shape == (self.space.dim,) * 2, "W must be (m, m)")
#
# Field annotations drive conversion and decoding. Allowed: ``NDArray[np.float64 |
# np.int64 | np.bool_]`` (a bare ``NDArray`` means float64), ``int``, ``float``,
# ``bool``, ``str``, ``Literal[...]``, ``tuple[T, ...]``, ``tuple[T1, T2]``,
# ``X | None``, an input space (``InputSpace``, ``Uniform``, ``Gaussian``) and any
# ``Result`` subclass. Do not define ``__post_init__`` (use ``_check``) and avoid
# zero-argument ``super()`` in slotted dataclasses.
#
# Foundation helper index
# -----------------------
# Public (``__all__``): ``Result``, ``QoIResult``, ``Provenance``, ``from_dict``, ``load``.
# Private, for package modules::
#
#     # class decorator: frozen, slotted, eq=False, repr=False dataclass
#     _register_result(cls) -> cls
#     # fields-dict -> fields-dict, k -> k+1
#     _register_migration(type_tag: str, from_version: int) -> decorator
#     # use in _check(); raises _InvariantError (an AssertionError)
#     _invariant(condition, message: str) -> None
#     _InvariantError(AssertionError)
#     _make_provenance(function: str, parameters: Mapping | None = None, *, space=None,
#                      rng_state: Mapping[str, Any] | None = None, digests: Mapping | None = None)
#                      -> Provenance
#         # function: public name, "active_subspaces." is prefixed if missing
#         # parameters: non-data arguments (arrays become digests)
#         # digests: _validation.data(...).digests
#         # rng_state: description from _random.generator(rng) (None if deterministic)
#     _registered_types() -> dict[str, type]   # copy of the TYPE -> class table
#     _EXPERIMENTAL_TYPES                      # tag -> module that load/from_dict import [D-141]
#
# The array codec (``freeze_array``, ``encode_array``, ``decode_array``,
# ``summarize``) lives in ``active_subspaces._validation``.

__all__ = ("Provenance", "QoIResult", "Result", "from_dict", "load")

_ARRAY_DTYPES = _v._ARRAY_DTYPES
_NONFINITE = _v._NONFINITE
_invalid = _v.schema_invalid
_float_json = _v.float_json
_is_tagged = _v.is_tagged

#: Experimental result tags and the module that defines each [D-141]. ``load``
#: and ``from_dict`` import the module on demand when they meet an unregistered
#: tag listed here (the module's import issues its ExperimentalWarning once).
_EXPERIMENTAL_TYPES: Mapping[str, str] = types.MappingProxyType(
    {"SketchedSubspace": "active_subspaces.experimental.gradfree"}
)


class _InvariantError(AssertionError):
    """A result invariant failed (a library bug when raised from a constructor)."""


def _invariant(condition: Any, message: str) -> None:
    """Raise :class:`_InvariantError` with ``message`` unless ``condition`` holds.

    Use in ``_check``.
    """
    if not condition:
        raise _InvariantError(message)


# ---------------------------------------------------------------------------
# Arrays (the codec itself lives in _validation)
# ---------------------------------------------------------------------------


def _retag(obj: Any, binary: bool) -> Any:
    """Copy a JSON-like tree, re-encoding every tagged array (or ndarray) in the requested form."""
    if isinstance(obj, np.ndarray):
        return _v.encode_array(obj, binary=binary)
    if isinstance(obj, Mapping):
        if _is_tagged(obj):
            return _v.encode_array(_v.decode_array(obj), binary=binary)
        return {k: _retag(v, binary) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_retag(v, binary) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Frozen JSON trees (for provenance)
# ---------------------------------------------------------------------------


class _FrozenDict(dict[str, Any]):
    """A read-only dict that still pickles and JSON-encodes as a dict."""

    __slots__ = ()

    def _ro(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("provenance mappings are read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _ro

    def __reduce__(self) -> tuple[Any, ...]:
        return (_FrozenDict, (dict(self),))

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: Any) -> Self:
        return self


class _FrozenList(list[Any]):
    """A read-only list that still pickles, JSON-encodes and compares as a list."""

    __slots__ = ()

    def _ro(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("provenance lists are read-only")

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = sort = (
        reverse
    ) = _ro
    __iadd__ = __imul__ = _ro

    def __reduce__(self) -> tuple[Any, ...]:
        return (_FrozenList, (list(self),))

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: Any) -> Self:
        return self


def _freeze(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return _FrozenDict({str(k): _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return _FrozenList(_freeze(v) for v in obj)
    return obj


def _thaw(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_thaw(v) for v in obj]
    return obj


def _jsonable(obj: Any) -> Any:
    """Convert a parameter value to JSON-compatible data; arrays become digests."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _float_json(float(obj))
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fiub":
            return _v.digest(obj)
        return repr(obj)
    if isinstance(obj, (Uniform, Gaussian)):
        return obj.to_dict(binary=True)
    if isinstance(obj, Result):
        return {"result": type(obj).TYPE}
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.random.Generator):
        return {"generator": type(obj.bit_generator).__name__}
    if callable(obj):
        name = getattr(obj, "__qualname__", None) or getattr(type(obj), "__qualname__", "callable")
        return {"callable": f"{getattr(obj, '__module__', '') or ''}.{name}".lstrip(".")}
    return repr(obj)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

_PROV_KEYS = (
    "library_version",
    "function",
    "parameters",
    "input_space",
    "rng",
    "numpy_version",
    "scipy_version",
    "python_version",
    "platform",
    "created",
)


@dataclass(frozen=True, slots=True)
class Provenance:
    """How a result was produced.

    Every result carries one, built by the function that made it. Provenance
    is excluded from result equality, and its mappings are read-only.

    Parameters
    ----------
    library_version : str
        Version of ``active_subspaces`` that produced the result.
    function : str
        Qualified public name of the producing function, for example
        ``"active_subspaces.active_subspace"``.
    parameters : Mapping[str, Any]
        The call's non-data arguments, converted to JSON-compatible values.
        Arrays are replaced by digests ``{"shape", "dtype", "sha256"}`` of the
        physical arrays.
    input_space : Mapping or None
        The input space as its :meth:`~active_subspaces.Uniform.to_dict`, or
        ``None``.
    rng : Mapping or None
        Description of the random generator on entry, enough to replay it;
        ``None`` for deterministic functions.
    numpy_version, scipy_version, python_version : str
        Versions of the numerical stack.
    platform : str
        ``platform.platform()``.
    created : str
        Creation time, UTC, ISO 8601.

    See Also
    --------
    Result : The base class of every result.

    Notes
    -----
    Results are reproducible bitwise from the recorded parameters, data and
    ``rng`` on the same platform and versions; ``created`` is not.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(3)
    >>> X = space.sample(40, rng=0)
    >>> a = np.array([1.0, 0.5, 0.1])
    >>> res = ac.active_subspace(grads=np.outer(2 * (X @ a), a), space=space)
    >>> prov = res.provenance
    >>> prov.function, sorted(prov.parameters["grads"])
    ('active_subspaces.active_subspace', ['dtype', 'sha256', 'shape'])
    >>> prov.rng is None  # no bootstrap, so the call was deterministic
    True
    """

    library_version: str
    function: str
    parameters: Mapping[str, Any]
    input_space: Mapping[str, Any] | None
    rng: Mapping[str, Any] | None
    numpy_version: str
    scipy_version: str
    python_version: str
    platform: str
    created: str

    def __post_init__(self) -> None:
        """Freeze the parameters, space and rng description."""
        object.__setattr__(self, "parameters", _freeze(_jsonable(dict(self.parameters))))
        if self.input_space is not None:
            # stored in the space.to_dict() form (list data) so the attribute equals it [spec 04
            # §4.3]
            object.__setattr__(self, "input_space", _freeze(_retag(self.input_space, False)))
        if self.rng is not None:
            object.__setattr__(self, "rng", _freeze(self.rng))

    def to_dict(self, *, binary: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible dict of the provenance.

        Parameters
        ----------
        binary : bool, default False
            Encode the arrays of ``input_space`` as base64 instead of lists.

        Returns
        -------
        dict
            One key per field, with plain ``dict`` and ``list`` values.

        See Also
        --------
        Provenance.from_dict : The inverse.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform.reference(2)
        >>> res = ac.active_subspace(grads=np.array([[1.0, 0.0], [2.0, 0.1]]), space=space)
        >>> res.provenance.to_dict()["input_space"]["type"]
        'Uniform'
        """
        d = {k: _thaw(getattr(self, k)) for k in _PROV_KEYS}
        if d["input_space"] is not None:
            d["input_space"] = _retag(d["input_space"], binary)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Provenance:
        """Rebuild provenance from the output of :meth:`to_dict`.

        Parameters
        ----------
        d : Mapping
            A dict with exactly the provenance keys.

        Returns
        -------
        Provenance
            The rebuilt record.

        Raises
        ------
        SchemaError
            ``schema.invalid`` if keys are missing or extra, or a value has the
            wrong type.

        See Also
        --------
        Provenance.to_dict : The inverse.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform.reference(2)
        >>> res = ac.active_subspace(grads=np.array([[1.0, 0.0], [2.0, 0.1]]), space=space)
        >>> prov = ac.Provenance.from_dict(res.provenance.to_dict())
        >>> prov == res.provenance
        True
        """
        if not isinstance(d, Mapping) or set(d) != set(_PROV_KEYS):
            got = sorted(d) if isinstance(d, Mapping) else type(d).__name__
            raise _invalid(f"provenance must have keys {list(_PROV_KEYS)}, got {got}")
        for k in _PROV_KEYS:
            if k in ("parameters", "input_space", "rng"):
                continue
            if not isinstance(d[k], str):
                raise _invalid(f"provenance.{k} must be a string")
        if not isinstance(d["parameters"], Mapping):
            raise _invalid("provenance.parameters must be a mapping")
        for k in ("input_space", "rng"):
            if d[k] is not None and not isinstance(d[k], Mapping):
                raise _invalid(f"provenance.{k} must be a mapping or null")
        return cls(**{k: d[k] for k in _PROV_KEYS})


def _make_provenance(
    function: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    space: Any = None,
    rng_state: Mapping[str, Any] | None = None,
    digests: Mapping[str, Any] | None = None,
) -> Provenance:
    """Build :class:`Provenance` for a call to public ``function``.

    ``parameters`` are the call's non-data arguments (arrays become digests);
    ``digests`` the input digests from ``_validation.data(...).digests``;
    ``rng_state`` the description from ``_random.generator`` (``None`` if deterministic).
    """
    if rng_state is not None and not isinstance(rng_state, Mapping):
        raise TypeError("rng_state must be the description dict from _random.generator")
    if not function.startswith("active_subspaces."):
        function = f"active_subspaces.{function}"
    params = dict(parameters or {})
    if digests:
        params.update(digests)
    return Provenance(
        library_version=__version__,
        function=function,
        parameters=params,
        input_space=None if space is None else space.to_dict(),
        rng=rng_state,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        created=datetime.datetime.now(datetime.UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# Field codecs
# ---------------------------------------------------------------------------


def _space_types() -> tuple[type[Uniform], type[Gaussian], type[InputSpace]]:
    return Uniform, Gaussian, InputSpace


def _array_dtype(ann: Any) -> Any:
    """Return the declared dtype if ``ann`` is an array annotation, else ``None``."""
    if ann is np.ndarray or ann is npt.NDArray:
        return np.float64
    origin = typing.get_origin(ann)
    if origin is npt.NDArray:
        args = typing.get_args(ann)
        return _norm_dtype(args[0]) if args else np.float64
    if origin is np.ndarray:
        args = typing.get_args(ann)
        if len(args) == 2:
            inner = typing.get_args(args[1])
            return _norm_dtype(inner[0]) if inner else np.float64
        return np.float64
    return None


def _norm_dtype(t: Any) -> Any:
    for dt in _ARRAY_DTYPES.values():
        if t is dt:
            return dt
    if isinstance(t, typing.TypeVar):
        return np.float64
    raise TypeError(f"result arrays must be float64, int64 or bool, got {t!r}")


def _kind(ann: Any) -> tuple[Any, ...]:
    """Compile a field annotation into a codec descriptor."""
    dt = _array_dtype(ann)
    if dt is not None:
        return ("array", dt)
    origin = typing.get_origin(ann)
    if origin in (Union, types.UnionType):
        args = typing.get_args(ann)
        non_none = [a for a in args if a is not type(None)]
        Uniform, Gaussian, InputSpace = _space_types()
        if all(a in (Uniform, Gaussian, InputSpace) for a in non_none):
            inner: tuple[Any, ...] = ("space",)
        elif len(non_none) == 1:
            inner = _kind(non_none[0])
        else:
            inner = ("union", tuple(_kind(a) for a in non_none))
        return ("optional", inner) if len(non_none) < len(args) else inner
    if ann in (int, float, bool, str):
        return ("scalar", ann)
    if ann is type(None):
        return ("none",)
    if origin is Literal:
        return ("literal", typing.get_args(ann))
    if ann is tuple:
        raise TypeError("tuple fields need element types, e.g. tuple[float, ...]")
    if origin is tuple:
        args = typing.get_args(ann)
        if len(args) == 2 and args[1] is Ellipsis:
            return ("tuple", _kind(args[0]))
        return ("tuple_fixed", tuple(_kind(a) for a in args))
    Uniform, Gaussian, InputSpace = _space_types()
    if ann in (Uniform, Gaussian, InputSpace):
        return ("space",)
    if isinstance(ann, type) and issubclass(ann, Result):
        return ("result", ann)
    raise TypeError(f"unsupported result field annotation {ann!r}")


def _coerce(kind: tuple[Any, ...], value: Any, name: str) -> Any:
    """Convert a constructor argument to its stored form (frozen array, tuple, Python scalar)."""
    tag = kind[0]
    if tag == "array":
        if not isinstance(value, np.ndarray) and not isinstance(
            value, (list, tuple, int, float, bool)
        ):
            raise TypeError(f"{name}: expected an array, got {type(value).__name__}")
        return _v.freeze_array(value, kind[1])
    if tag == "optional":
        return None if value is None else _coerce(kind[1], value, name)
    if tag == "scalar":
        t = kind[1]
        if t is bool:
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name}: expected bool, got {type(value).__name__}")
            return bool(value)
        if t is int:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name}: expected int, got {type(value).__name__}")
            return int(value)
        if t is float:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{name}: expected float, got {type(value).__name__}")
            return float(value)
        if not isinstance(value, str):
            raise TypeError(f"{name}: expected str, got {type(value).__name__}")
        return value
    if tag == "none":
        if value is not None:
            raise TypeError(f"{name}: expected None")
        return None
    if tag == "literal":
        if value not in kind[1]:
            raise TypeError(f"{name}: expected one of {list(kind[1])}, got {value!r}")
        return value
    if tag == "tuple":
        if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__iter__"):
            raise TypeError(f"{name}: expected a sequence")
        return tuple(_coerce(kind[1], v, name) for v in value)
    if tag == "tuple_fixed":
        vals = tuple(value)
        if len(vals) != len(kind[1]):
            raise TypeError(f"{name}: expected {len(kind[1])} items, got {len(vals)}")
        return tuple(_coerce(k, v, name) for k, v in zip(kind[1], vals, strict=False))
    if tag == "space":
        Uniform, Gaussian, _ = _space_types()
        if not isinstance(value, (Uniform, Gaussian)):
            raise TypeError(f"{name}: expected an input space, got {type(value).__name__}")
        return value
    if tag == "result":
        if not isinstance(value, kind[1]):
            raise TypeError(f"{name}: expected {kind[1].__name__}, got {type(value).__name__}")
        return value
    if tag == "union":
        for k in kind[1]:
            try:
                return _coerce(k, value, name)
            except TypeError:
                continue
        raise TypeError(f"{name}: value {value!r} matches no member of its union type")
    raise TypeError(f"unknown field kind {tag}")


def _encode(kind: tuple[Any, ...], value: Any, binary: bool) -> Any:
    tag = kind[0]
    if value is None:
        return None
    if tag == "array":
        return _v.encode_array(value, binary=binary)
    if tag == "optional":
        return _encode(kind[1], value, binary)
    if tag == "scalar" and kind[1] is float:
        return _float_json(value)
    if tag in ("scalar", "literal", "none"):
        return value
    if tag == "tuple":
        return [_encode(kind[1], v, binary) for v in value]
    if tag == "tuple_fixed":
        return [_encode(k, v, binary) for k, v in zip(kind[1], value, strict=False)]
    if tag in ("space", "result"):
        return value.to_dict(binary=binary)
    if tag == "union":
        for k in kind[1]:
            try:
                _coerce(k, value, "")
            except TypeError:
                continue
            return _encode(k, value, binary)
    raise TypeError(f"cannot encode field kind {tag}")


def _decode(kind: tuple[Any, ...], value: Any, name: str) -> Any:
    tag = kind[0]
    if tag == "optional":
        return None if value is None else _decode(kind[1], value, name)
    if tag == "array":
        arr = _v.decode_array(value)
        if arr.dtype != np.dtype(kind[1]):
            want = np.dtype(kind[1]).name
            raise _invalid(
                f"field {name!r}: expected dtype {want}, got {arr.dtype.name}", field=name
            )
        return arr
    if tag == "scalar":
        t = kind[1]
        if t is float:
            if isinstance(value, str) and value in _NONFINITE:
                return _NONFINITE[value]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        elif t is int:
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        elif isinstance(value, t):
            return value
        raise _invalid(
            f"field {name!r}: expected {t.__name__}, got {type(value).__name__}", field=name
        )
    if tag == "none":
        if value is None:
            return None
        raise _invalid(f"field {name!r}: expected null", field=name)
    if tag == "literal":
        if value in kind[1] and isinstance(value, str):
            return value
        raise _invalid(
            f"field {name!r}: expected one of {list(kind[1])}, got {value!r}", field=name
        )
    if tag in ("tuple", "tuple_fixed"):
        if not isinstance(value, list):
            raise _invalid(f"field {name!r}: expected a list", field=name)
        if tag == "tuple":
            return tuple(_decode(kind[1], v, name) for v in value)
        if len(value) != len(kind[1]):
            raise _invalid(f"field {name!r}: expected {len(kind[1])} items", field=name)
        return tuple(_decode(k, v, name) for k, v in zip(kind[1], value, strict=False))
    if tag == "space":
        return _space_from_dict(value)
    if tag == "result":
        return kind[1].from_dict(value)
    if tag == "union":
        for k in kind[1]:
            try:
                return _decode(k, value, name)
            except SchemaError:
                continue
        raise _invalid(f"field {name!r}: value matches no member of its union type", field=name)
    raise _invalid(f"field {name!r}: unknown kind")


# ---------------------------------------------------------------------------
# Comparison and repr
# ---------------------------------------------------------------------------


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        return bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b, strict=False))
    if type(a) is not type(b):
        return False
    return bool(a == b)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Result]] = {}
_Migration = Callable[[dict[str, Any]], dict[str, Any]]
_MIGRATIONS: dict[tuple[str, int], _Migration] = {}
_CODECS: dict[type, dict[str, tuple[Any, ...]]] = {}


def _registered_types() -> dict[str, type[Result]]:
    """Return a copy of the ``TYPE -> class`` table."""
    return dict(_REGISTRY)


def _register_migration(type_tag: str, from_version: int) -> Callable[[_Migration], _Migration]:
    """Register a pure ``fields -> fields`` migration of ``type_tag``.

    It migrates from ``from_version`` to ``from_version + 1``.
    """

    def deco(fn: _Migration) -> _Migration:
        key = (type_tag, int(from_version))
        if key in _MIGRATIONS and _MIGRATIONS[key] is not fn:
            raise TypeError(f"migration {key} already registered")
        _MIGRATIONS[key] = fn
        return fn

    return deco


_ResultT = typing.TypeVar("_ResultT", bound=type)


@typing.dataclass_transform(frozen_default=True, eq_default=False)
def _register_result(cls: _ResultT) -> _ResultT:
    """Class decorator for result types: frozen, slotted dataclass with the base's eq/repr.

    Registers ``TYPE``.
    """
    if "TYPE" not in cls.__dict__ or "SCHEMA_VERSION" not in cls.__dict__:
        raise TypeError(f"{cls.__name__} must define TYPE and SCHEMA_VERSION")
    out = cast(_ResultT, dataclass(frozen=True, slots=True, eq=False, repr=False)(cls))
    try:
        _codecs(out)  # check the field annotations now ...
    except NameError:
        pass  # ... unless they name something not yet importable; then on first use
    return out


def _codecs(cls: type) -> dict[str, tuple[Any, ...]]:
    try:
        return _CODECS[cls]
    except KeyError:
        pass
    params = getattr(cls, "__dataclass_params__", None)
    if (
        params is None
        or not params.frozen
        or params.eq
        or params.repr
        or "__slots__" not in cls.__dict__
    ):
        raise TypeError(f"{cls.__name__} must be declared with @_register_result")
    hints = typing.get_type_hints(cls)
    codecs: dict[str, tuple[Any, ...]] = {}
    for f in dataclasses.fields(cls):
        if not f.init:
            raise TypeError(
                f"{cls.__name__}.{f.name}: result fields must be init fields; use a property for "
                "derived values"
            )
        if f.name == "provenance":
            continue
        codecs[f.name] = _kind(hints[f.name])
    cls.__hash__ = None  # type: ignore[assignment]
    _CODECS[cls] = codecs
    return codecs


def _lookup_type(tag: str) -> type[Result]:
    """Return the class registered for ``tag``; import an experimental module on demand [D-141]."""
    target = _REGISTRY.get(tag)
    if target is None and tag in _EXPERIMENTAL_TYPES:
        importlib.import_module(_EXPERIMENTAL_TYPES[tag])
        target = _REGISTRY.get(tag)
    if target is None:
        raise SchemaError(
            f"unknown result type {tag!r}", code="schema.unknown_type", details={"type": tag}
        )
    return target


def _rebuild_result(cls: type[Result], kwargs: dict[str, Any]) -> Result:
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Result base
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Result:
    """Base class of every result: an immutable record of a computation.

    Concrete result types (for example :class:`~active_subspaces.SubspaceResult`)
    are frozen, slotted dataclasses registered under a permanent type tag
    ``TYPE`` with a ``SCHEMA_VERSION``. Construct them by calling the library's
    functions; ``Result`` itself is abstract.

    Parameters
    ----------
    provenance : Provenance, keyword-only
        How the result was produced.

    Attributes
    ----------
    TYPE : str
        Permanent type tag, for example ``"SubspaceResult"``.
    SCHEMA_VERSION : int
        Version of the serialized layout of this type, starting at 1.

    Raises
    ------
    TypeError
        If ``Result`` (or an unregistered base) is instantiated directly, or a
        field has the wrong type.

    See Also
    --------
    QoIResult : Base of results that depend on QoI values.
    load : Load a result from a ``.npz`` file.
    from_dict : Rebuild a result from a dict.

    Notes
    -----
    Arrays are owned, read-only copies whose writeable flag cannot be set
    again. ``==`` compares every field except provenance, arrays bitwise (NaN
    equal to NaN); results are not hashable. ``repr`` prints scalars in full
    and arrays by dtype, shape and at most four entries. Results pickle through
    their constructor, so their invariants are checked again on unpickling.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(2)
    >>> res = ac.active_subspace(grads=np.array([[1.0, 0.0], [2.0, 0.1]]), space=space)
    >>> isinstance(res, ac.Result), res.TYPE, res.SCHEMA_VERSION
    (True, 'SubspaceResult', 2)
    >>> res.eigenvalues.flags.writeable
    False
    """

    TYPE: ClassVar[str]
    SCHEMA_VERSION: ClassVar[int]
    provenance: Provenance = field(kw_only=True, repr=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register a subclass that defines ``TYPE``."""
        super(Result, cls).__init_subclass__(**kwargs)
        if "TYPE" not in cls.__dict__:
            return
        tag = cls.__dict__["TYPE"]
        version = cls.__dict__.get("SCHEMA_VERSION")
        if not isinstance(tag, str) or not tag:
            raise TypeError(f"{cls.__name__}.TYPE must be a non-empty string")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise TypeError(f"{cls.__name__}.SCHEMA_VERSION must be an integer >= 1")
        old = _REGISTRY.get(tag)
        # dataclass(slots=True) re-creates the class (briefly without its qualname); that is not a
        # duplicate
        if old is not None and (old.__module__, old.__name__) != (cls.__module__, cls.__name__):
            raise TypeError(
                f"result TYPE {tag!r} already registered by {old.__module__}.{old.__qualname__}"
            )
        if old is not None:
            _CODECS.pop(old, None)
        _REGISTRY[tag] = cls

    def __post_init__(self) -> None:
        """Coerce the fields to their stored form and check the invariants."""
        cls = type(self)
        if cls is Result or "TYPE" not in vars(cls):
            raise TypeError("instantiate a registered result type, not an abstract base")
        for name, kind in _codecs(cls).items():
            object.__setattr__(self, name, _coerce(kind, getattr(self, name), name))
        if not isinstance(self.provenance, Provenance):
            raise TypeError(
                f"provenance must be a Provenance, got {type(self.provenance).__name__}"
            )
        if hasattr(self, "qoi"):
            _invariant(
                isinstance(self.qoi, str) and self.qoi != "", "qoi must be a non-empty string"
            )
        self._check()

    def _check(self) -> None:
        """Enforce the type's invariants with :func:`_invariant`; override in subclasses."""

    def __eq__(self, other: object) -> bool:
        """Compare field by field; provenance is excluded."""
        if type(self) is not type(other):
            return NotImplemented
        return all(_values_equal(getattr(self, n), getattr(other, n)) for n in _codecs(type(self)))

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """Return a summary: scalars in full, arrays by dtype, shape and first entries."""
        parts = [f"{n}={_v.summarize(getattr(self, n))}" for n in _codecs(type(self))]
        return f"{type(self).__name__}({', '.join(parts)})"

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle through the constructor, so invariants are checked again."""
        kwargs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return (_rebuild_result, (type(self), kwargs))

    def to_dict(self, *, binary: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible dict of the result.

        Parameters
        ----------
        binary : bool, default False
            Encode arrays as base64 of their little-endian bytes instead of
            flat lists. Both forms decode to bitwise-equal arrays.

        Returns
        -------
        dict
            Keys ``"type"``, ``"schema_version"``, ``"library_version"``,
            ``"fields"`` and ``"provenance"``. Arrays are tagged dicts
            ``{"__ndarray__": 1, "dtype", "shape", "data" | "b64"}``;
            non-finite floats are the strings ``"NaN"``, ``"Infinity"`` and
            ``"-Infinity"``, so ``json.dumps(..., allow_nan=False)`` succeeds.

        See Also
        --------
        from_dict : Rebuild any result from such a dict.
        save : Write the result to a ``.npz`` file.

        Examples
        --------
        >>> import json
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform.reference(2)
        >>> res = ac.active_subspace(grads=np.array([[1.0, 0.0], [2.0, 0.1]]), space=space)
        >>> d = res.to_dict()
        >>> d["type"], sorted(d["fields"])[:3]
        ('SubspaceResult', ['bootstrap_dets', 'bootstrap_distances', 'bootstrap_eigenvalues'])
        >>> ac.from_dict(json.loads(json.dumps(d))) == res
        True
        """
        codecs = _codecs(type(self))
        return {
            "type": type(self).TYPE,
            "schema_version": type(self).SCHEMA_VERSION,
            "library_version": __version__,
            "fields": {n: _encode(k, getattr(self, n), binary) for n, k in codecs.items()},
            "provenance": self.provenance.to_dict(binary=binary),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        """Rebuild a result from the output of :meth:`to_dict`.

        Parameters
        ----------
        d : Mapping
            A result dict. Older schema versions are migrated forward.

        Returns
        -------
        Result
            The result, of the type named by ``d["type"]``, which must be this
            class or a subclass of it.

        Raises
        ------
        SchemaError
            ``schema.unknown_type`` for an unknown tag; ``schema.too_new`` if
            the schema version is newer than this library supports;
            ``schema.invalid`` for a malformed dict, a tag of the wrong class,
            or fields that break the type's invariants.

        See Also
        --------
        to_dict : The inverse.
        active_subspaces.from_dict : The same, for any result type.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform.reference(2)
        >>> res = ac.active_subspace(grads=np.array([[1.0, 0.0], [2.0, 0.1]]), space=space)
        >>> ac.SubspaceResult.from_dict(res.to_dict(binary=True)) == res
        True
        """
        if not isinstance(d, Mapping):
            raise _invalid(f"expected a mapping, got {type(d).__name__}")
        tag = d.get("type")
        if not isinstance(tag, str):
            raise _invalid("missing or non-string 'type' tag")
        target = _lookup_type(tag)
        if cls is not Result and not issubclass(target, cls):
            raise _invalid(f"expected a {cls.__name__} dict, got type {tag!r}", type=tag)
        return cast(Self, target._decode_dict(d))  # target is cls or a subclass, checked above

    @classmethod
    def _decode_dict(cls, d: Mapping[str, Any]) -> Self:
        expected = {"type", "schema_version", "library_version", "fields", "provenance"}
        if set(d) != expected:
            raise _invalid(f"result dict must have keys {sorted(expected)}, got {sorted(d)}")
        version = d["schema_version"]
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise _invalid(f"bad schema_version {version!r}")
        if version > cls.SCHEMA_VERSION:
            raise SchemaError(
                f"{cls.TYPE} schema version {version} is newer than this library supports "
                f"({cls.SCHEMA_VERSION})",
                code="schema.too_new",
                details={
                    "type": cls.TYPE,
                    "schema_version": version,
                    "supported": cls.SCHEMA_VERSION,
                },
            )
        fields_in = d["fields"]
        if not isinstance(fields_in, Mapping):
            raise _invalid("'fields' must be a mapping")
        fields_in = dict(fields_in)
        while version < cls.SCHEMA_VERSION:
            mig = _MIGRATIONS.get((cls.TYPE, version))
            if mig is None:
                raise _invalid(f"no migration for {cls.TYPE} from schema version {version}")
            fields_in = mig(fields_in)
            version += 1
        codecs = _codecs(cls)
        unknown = sorted(set(fields_in) - set(codecs))
        missing = sorted(set(codecs) - set(fields_in))
        if unknown or missing:
            raise _invalid(
                f"{cls.TYPE}: unknown fields {unknown}, missing fields {missing}",
                unknown=unknown,
                missing=missing,
            )
        kwargs = {n: _decode(k, fields_in[n], n) for n, k in codecs.items()}
        prov = Provenance.from_dict(d["provenance"])
        try:
            return cls(**kwargs, provenance=prov)
        except SchemaError:
            raise
        except (
            _InvariantError,
            AssertionError,
            TypeError,
            ValueError,
            IndexError,
            KeyError,
        ) as exc:
            raise _invalid(f"{cls.TYPE}: invalid fields ({exc})") from exc

    def save(self, file: str | os.PathLike[str] | BinaryIO) -> None:
        """Write this result to one uncompressed ``.npz`` file.

        Parameters
        ----------
        file : str, os.PathLike or binary file object
            A path, used exactly as given (no extension is added), or a file
            opened for binary writing.

        Returns
        -------
        None
            The file is written.

        See Also
        --------
        load : Read the file back.
        to_dict : The JSON-compatible form.

        Notes
        -----
        The archive holds the arrays of :meth:`to_dict` as ``.npy`` entries and
        the rest of the dict as a JSON string in ``__meta__``. It contains no
        pickled objects, so :func:`load` reads it with ``allow_pickle=False``.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform.reference(2)
        >>> res = ac.active_subspace(grads=np.array([[1.0, 0.0], [2.0, 0.1]]), space=space)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = os.path.join(tmp, "subspace.npz")
        ...     res.save(path)
        ...     ac.load(path) == res
        True
        """
        arrays: list[np.ndarray] = []

        def extract(obj: Any) -> Any:
            if isinstance(obj, Mapping):
                if _is_tagged(obj):
                    arrays.append(np.ascontiguousarray(_v.decode_array(obj)))
                    return {"__npz__": f"a{len(arrays) - 1}"}
                return {k: extract(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [extract(v) for v in obj]
            return obj

        meta = json.dumps(extract(self.to_dict(binary=True)), allow_nan=False)
        payload = {"__meta__": np.array(meta)}
        payload.update({f"a{i}": a for i, a in enumerate(arrays)})
        if isinstance(file, (str, os.PathLike)):
            with open(file, "wb") as fh:
                np.savez(fh, **payload)  # type: ignore[arg-type]  # stubs match **payload to allow_pickle
        else:
            np.savez(file, **payload)  # type: ignore[arg-type]  # as above


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class QoIResult(Result):
    """Base class of results that depend on values of a quantity of interest.

    Adds the keyword-only field ``qoi``, the name of the scalar quantity of
    interest the result describes. Results computed from several QoI are kept
    apart by this name.

    Parameters
    ----------
    provenance : Provenance, keyword-only
        How the result was produced.
    qoi : str, keyword-only, default "f"
        Non-empty name of the quantity of interest.

    See Also
    --------
    Result : The base class.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(2)
    >>> G = np.array([[1.0, 0.0], [2.0, 0.1]])
    >>> res = ac.active_subspace(grads=G, space=space, qoi="lift")
    >>> isinstance(res, ac.QoIResult), res.qoi
    (True, 'lift')
    """

    qoi: str = field(default="f", kw_only=True)


# ---------------------------------------------------------------------------
# Package-level loaders
# ---------------------------------------------------------------------------


def from_dict(d: Mapping[str, Any]) -> Result:
    """Rebuild any result from its dict, dispatching on the ``"type"`` tag.

    Parameters
    ----------
    d : Mapping
        The output of :meth:`Result.to_dict`, possibly after a JSON round trip.

    Returns
    -------
    Result
        A result of the type named by the tag.

    Raises
    ------
    SchemaError
        ``schema.unknown_type``, ``schema.too_new`` or ``schema.invalid``, as
        for :meth:`Result.from_dict`.

    See Also
    --------
    Result.to_dict : The inverse.
    load : Load a result from a file.

    Notes
    -----
    A tag defined in ``active_subspaces.experimental`` imports that module on
    demand, which issues ``ExperimentalWarning`` once.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Gaussian.reference(2)
    >>> res = ac.active_subspace(grads=np.array([[1.0, 0.5], [2.0, 0.1]]), space=space)
    >>> type(ac.from_dict(res.to_dict())).__name__
    'SubspaceResult'
    """
    return Result.from_dict(d)


def load(file: str | os.PathLike[str] | BinaryIO) -> Result:
    """Load a result written by :meth:`Result.save`.

    Parameters
    ----------
    file : str, os.PathLike or binary file object
        A path, used exactly as given, or a file opened for binary reading.

    Returns
    -------
    Result
        A result of the type stored in the file.

    Raises
    ------
    SchemaError
        ``schema.invalid`` if the file is not a result archive or is
        malformed; ``schema.unknown_type`` or ``schema.too_new`` as for
        :func:`from_dict`.

    See Also
    --------
    Result.save : Write a result.
    from_dict : Rebuild a result from a dict.

    Notes
    -----
    The file is read with ``numpy.load(..., allow_pickle=False)``, so loading
    an untrusted file never executes code.

    Examples
    --------
    >>> import io
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform.reference(2)
    >>> res = ac.active_subspace(grads=np.array([[1.0, 0.0], [2.0, 0.1]]), space=space)
    >>> buf = io.BytesIO()
    >>> res.save(buf)
    >>> _ = buf.seek(0)
    >>> ac.load(buf) == res
    True
    """
    if isinstance(file, (str, os.PathLike)):
        with open(file, "rb") as fh:
            return _load_from(fh)
    return _load_from(file)


def _load_from(fh: BinaryIO) -> Result:
    import zipfile

    try:
        npz = np.load(fh, allow_pickle=False)
    except (ValueError, zipfile.BadZipFile, EOFError) as exc:
        raise _invalid(f"not a valid result file ({exc})") from exc
    if not isinstance(npz, np.lib.npyio.NpzFile):
        raise _invalid("not an .npz archive")
    with npz:
        try:
            if "__meta__" not in npz.files:
                raise _invalid("missing __meta__ entry")
            meta_arr = npz["__meta__"]
            if meta_arr.ndim != 0 or meta_arr.dtype.kind != "U":
                raise _invalid("__meta__ must be a 0-d unicode array")
            meta = json.loads(str(meta_arr[()]))
            arrays = {k: npz[k] for k in npz.files if k != "__meta__"}
        except (ValueError, zipfile.BadZipFile, EOFError, KeyError) as exc:
            if isinstance(exc, SchemaError):
                raise
            raise _invalid(f"cannot read result file ({exc})") from exc

    def inject(obj: Any) -> Any:
        if isinstance(obj, dict):
            if set(obj) == {"__npz__"}:
                key = obj["__npz__"]
                if key not in arrays:
                    raise _invalid(f"missing array {key!r}")
                return arrays[key]
            return {k: inject(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [inject(v) for v in obj]
        return obj

    return from_dict(inject(meta))
