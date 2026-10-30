"""Private validation, digest and array-codec helpers (specs 02 and 04).

Every public data-taking function validates its inputs with these helpers
before any computation. They raise :class:`~active_subspaces.errors.InputError`
with the codes of spec 02 §6 and never warn. Use :func:`data` to validate a
whole call at once (it applies the check order of spec 02 §5.2 across arrays);
the single-array helpers are for functions that take one array. The array
codec used by spaces and results also lives here (layer 0, below ``spaces``).

Foundation helper index
-----------------------
Arrays (spec 02 §4.3), all return new float64 arrays::

    points(X, space, name="X", *, normalize=True) -> ndarray (M, m)   # physical if not normalize
    values(f, M=None, name="f") -> ndarray (M,)
    gradients(G, space, M=None, name="grads") -> ndarray (M, m)       # normalized
    quad_weights(w, M, name="quad_weights") -> ndarray (M,)           # None -> 1/M
    sample_weights(w, M, name="sample_weights") -> ndarray (M,)       # None -> ones
    data(space, *, X=None, f=None, grads=None, quad_weights=None, sample_weights=None,
         require=(), use_quad_weights=False, use_sample_weights=False) -> Data
    Data(M, m, X, f, grads, quad_weights, sample_weights, digests)    # frozen record
    matrix(a, name, *, rows=None, cols=None, finite=True) -> ndarray 2-D
    vector(a, name, *, size=None, finite=True) -> ndarray 1-D
    coerce(a, name) -> ndarray float64 (may alias the input; never write to it)
    check_finite(a, name) -> None                                     # input.nonfinite
    digest(a) -> {"shape", "dtype", "sha256"}                         # of the physical array

Scalars, choices and spaces (``details["param"] == name``)::

    integer(n, name, *, lo=None, hi=None) -> int                      # operator.index; no bool
    dimension(n, m, lo=1, hi=None, name="n") -> int                   # lo <= n <= m-1 (or hi)
    real(x, name, *, lo=None, hi=None, lo_open=False, hi_open=False, finite=True) -> float
    boolean(b, name) -> bool
    choice(value, name, options) -> str
    min_samples(M, minimum, what) -> None                             # input.too_few_samples
    check_space(space, name="space") -> space                         # duck-typed on kind/dim

Array codec and repr (spec 04 §5.1-5.4; used by ``spaces`` and ``results``)::

    freeze_array(a, dtype=np.float64) -> read-only owned C-contiguous array (view of a frozen base)
    encode_array(a, *, binary=False)
        -> tagged dict {"__ndarray__": 1, "dtype", "shape", "data" | "b64"}
    decode_array(d) -> read-only array                                # SchemaError schema.invalid
    is_tagged(obj) -> bool
    float_json(v) -> float | "NaN" | "Infinity" | "-Infinity"
    schema_invalid(message, **details) -> SchemaError                 # code schema.invalid
    summarize(value) -> str                                           # short repr of a field value
"""

from __future__ import annotations

import base64
import hashlib
import math
import numbers
import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from active_subspaces.errors import InputError, SchemaError

__all__ = ()

F64 = NDArray[np.float64]  # private alias for float64 arrays of any shape

QUAD_WEIGHT_TOL = 1e-10
_BAD_KINDS = {
    "b": "bool",
    "c": "complex",
    "O": "object",
    "U": "string",
    "S": "bytes",
    "M": "datetime",
    "m": "timedelta",
    "V": "void",
}


# ---------------------------------------------------------------------------
# Low-level pieces
# ---------------------------------------------------------------------------


def coerce(a: Any, name: str) -> F64:
    """Return ``a`` as a float64 ndarray (may alias the caller's array; never write to it);
    reject bad dtypes.
    """
    if a is None:
        raise InputError(
            f"{name}: expected an array, got None",
            code="input.dtype",
            details={"param": name, "got": "None"},
        )
    try:
        arr = np.asarray(a)
    except (TypeError, ValueError) as exc:
        raise InputError(
            f"{name}: cannot be converted to an array ({exc})",
            code="input.dtype",
            details={"param": name},
        ) from exc
    kind = arr.dtype.kind
    if kind in _BAD_KINDS:
        raise InputError(
            f"{name}: unsupported dtype {arr.dtype} ({_BAD_KINDS[kind]}); expected a real "
            "numeric array",
            code="input.dtype",
            details={"param": name, "got": str(arr.dtype)},
        )
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    return arr


def _shape_error(
    name: str, expected: str, got: tuple[int, ...], details_expected: Any = None
) -> InputError:
    return InputError(
        f"{name}: expected shape {expected}, got {tuple(got)}",
        code="input.shape",
        details={
            "param": name,
            "expected": details_expected if details_expected is not None else expected,
            "got": list(got),
        },
    )


def _size_error(name: str, M: int, got: int) -> InputError:
    return InputError(
        f"{name}: expected {M} rows to match the other arrays, got {got}",
        code="input.size_mismatch",
        details={"param": name, "expected": int(M), "got": int(got)},
    )


def check_finite(a: NDArray[Any], name: str) -> None:
    """Raise ``input.nonfinite`` naming the first offending row if ``a`` holds NaN or ±inf."""
    ok = np.isfinite(a)
    if ok.all():
        return
    bad = np.argwhere(~ok)[0]
    details: dict[str, Any] = {"param": name, "index": [int(i) for i in bad]}
    if a.ndim >= 1:
        details["row"] = int(bad[0])
    raise InputError(
        f"{name}: contains NaN or inf (first at index {tuple(int(i) for i in bad)})",
        code="input.nonfinite",
        details=details,
    )


def _as_2d(arr: F64, name: str, m: int, allow_1d: bool) -> F64:
    if arr.ndim == 1 and allow_1d and m == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[1] != m:
        raise _shape_error(name, f"(M, {m})", arr.shape, ["M", m])
    if arr.shape[0] < 1:
        raise _shape_error(name, f"(M, {m}) with M >= 1", arr.shape, ["M", m])
    return arr


def matrix(
    a: Any, name: str, *, rows: int | None = None, cols: int | None = None, finite: bool = True
) -> F64:
    """Coerce a 2-D float64 array with optional fixed ``rows``/``cols``; returns a new array."""
    arr = coerce(a, name)
    if (
        arr.ndim != 2
        or (rows is not None and arr.shape[0] != rows)
        or (cols is not None and arr.shape[1] != cols)
    ):
        exp = f"({'r' if rows is None else rows}, {'c' if cols is None else cols})"
        raise _shape_error(name, exp, arr.shape, [rows, cols])
    if finite:
        check_finite(arr, name)
    return np.array(arr, dtype=np.float64, order="C", copy=True)


def vector(a: Any, name: str, *, size: int | None = None, finite: bool = True) -> F64:
    """Coerce a 1-D float64 array with optional fixed ``size``; returns a new array."""
    arr = coerce(a, name)
    if arr.ndim != 1 or (size is not None and arr.shape[0] != size):
        raise _shape_error(name, f"({'n' if size is None else size},)", arr.shape, [size])
    if finite:
        check_finite(arr, name)
    return np.array(arr, dtype=np.float64, order="C", copy=True)


# ---------------------------------------------------------------------------
# Array helpers of spec 02 §4.3
# ---------------------------------------------------------------------------


def check_space(space: Any, name: str = "space") -> Any:
    """Require an input-space object (duck-typed: a string ``kind`` and an integer
    ``dim >= 1``); return it.
    """
    kind = getattr(space, "kind", None)
    dim = getattr(space, "dim", None)
    if not isinstance(kind, str) or isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise InputError(
            f"{name}: expected an input space (Uniform or Gaussian), got {type(space).__name__}",
            code="input.parameter",
            details={"param": name, "got": type(space).__name__},
        )
    return space


def _check_support(Xphys: F64, space: Any, name: str) -> None:
    inside = space.contains(Xphys)
    if inside.all():
        return
    n_outside = int((~inside).sum())
    row = int(np.argmin(inside))
    violations = space._support_violations(Xphys[row : row + 1])[0]
    col = int(np.argmax(violations))
    value = float(Xphys[row, col])
    bounds = space._bounds() if hasattr(space, "_bounds") else (space.lower, space.upper)
    lower, upper = float(bounds[0][col]), float(bounds[1][col])
    side, bound = ("lower", lower) if value < lower else ("upper", upper)
    label = space.names[col]
    op = "<" if side == "lower" else ">"
    raise InputError(
        f"{name}: {n_outside} of {Xphys.shape[0]} rows lie outside the support; first: row {row}, "
        f"input {label!r} (column {col}) = {value!r} {op} {side} {bound!r}",
        code="input.out_of_space",
        details={
            "param": name,
            "row": row,
            "coordinate": col,
            "value": value,
            "n_outside": n_outside,
            "column": col,
            "name": label,
            "bound": bound,
            "side": side,
        },
    )


def points(X: Any, space: Any, name: str = "X", *, normalize: bool = True) -> F64:
    """Validate physical points ``(M, m)`` against ``space``; return them normalized (a new array).

    With ``normalize=False`` the same checks run and the coerced physical
    float64 ``(M, m)`` array is returned instead, also a new array (spec 12
    uses this to pass physical points to a model [D-58]).
    """
    check_space(space)
    arr = _as_2d(coerce(X, name), name, space.dim, allow_1d=True)
    check_finite(arr, name)
    _check_support(arr, space, name)
    if not normalize:
        return np.array(arr, dtype=np.float64, order="C", copy=True)
    out: F64 = space._to_normalized(arr)
    return out


def values(f: Any, M: int | None = None, name: str = "f") -> F64:
    """Validate scalar QoI values ``(M,)`` or ``(M, 1)``; return a new float64 ``(M,)`` array."""
    arr = _values_shape(coerce(f, name), name)
    if M is not None and arr.shape[0] != M:
        raise _size_error(name, M, arr.shape[0])
    check_finite(arr, name)
    return np.array(arr, dtype=np.float64, copy=True)


def _values_shape(arr: F64, name: str) -> F64:
    if arr.ndim == 2:
        if arr.shape[1] != 1:
            raise InputError(
                f"{name}: expected shape (M,) or (M, 1), got {arr.shape}; vector-valued QoI are "
                "not supported",
                code="input.vector_qoi",
                details={"param": name, "expected": ["M"], "got": list(arr.shape)},
            )
        arr = arr[:, 0]
    if arr.ndim != 1 or arr.shape[0] < 1:
        raise _shape_error(name, "(M,) or (M, 1) with M >= 1", arr.shape, ["M"])
    return arr


def gradients(G: Any, space: Any, M: int | None = None, name: str = "grads") -> F64:
    """Validate physical gradients ``(M, m)`` and return them in normalized coordinates (a new
    array).
    """
    check_space(space)
    arr = _as_2d(coerce(G, name), name, space.dim, allow_1d=True)
    if M is not None and arr.shape[0] != M:
        raise _size_error(name, M, arr.shape[0])
    check_finite(arr, name)
    out: F64 = space._grads_to_normalized(arr)
    return out


def _weights_shape(arr: F64, M: int, name: str) -> None:
    if arr.ndim != 1:
        raise _shape_error(name, f"({M},)", arr.shape, [M])
    if arr.shape[0] != M:
        raise _size_error(name, M, arr.shape[0])


def _check_quad(arr: F64, name: str) -> None:
    if np.any(arr < 0):
        i = int(np.argmax(arr < 0))
        raise InputError(
            f"{name}: entry {i} is negative ({arr[i]!r})",
            code="input.weights",
            details={"param": name, "row": i},
        )
    s = math.fsum(arr.tolist())
    if abs(s - 1.0) > QUAD_WEIGHT_TOL:
        raise InputError(
            f"{name}: quadrature weights must sum to 1 within {QUAD_WEIGHT_TOL:g}, got sum {s!r}",
            code="input.weights",
            details={"param": name, "sum": s},
        )


def _check_sample(arr: F64, name: str) -> None:
    if np.any(arr < 0):
        i = int(np.argmax(arr < 0))
        raise InputError(
            f"{name}: entry {i} is negative ({arr[i]!r})",
            code="input.weights",
            details={"param": name, "row": i},
        )
    if not np.any(arr > 0):
        raise InputError(
            f"{name}: at least one weight must be positive",
            code="input.weights",
            details={"param": name},
        )


def quad_weights(w: Any, M: int, name: str = "quad_weights") -> F64:
    """Validate quadrature weights (``None`` → ``1/M``): shape ``(M,)``, finite, ``>= 0``, sum
    1 within 1e-10.
    """
    if w is None:
        return np.full(M, 1.0 / M)
    arr = coerce(w, name)
    _weights_shape(arr, M, name)
    check_finite(arr, name)
    _check_quad(arr, name)
    return np.array(arr, dtype=np.float64, copy=True)


def sample_weights(w: Any, M: int, name: str = "sample_weights") -> F64:
    """Validate regression weights (``None`` → ones): shape ``(M,)``, finite, ``>= 0``, one
    entry ``> 0``.
    """
    if w is None:
        return np.ones(M)
    arr = coerce(w, name)
    _weights_shape(arr, M, name)
    check_finite(arr, name)
    _check_sample(arr, name)
    return np.array(arr, dtype=np.float64, copy=True)


def digest(a: Any) -> dict[str, Any]:
    """Return ``{"shape", "dtype": "float64", "sha256"}`` of the C-contiguous float64 bytes of
    ``a``.
    """
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    h = hashlib.sha256(arr.tobytes(order="C")).hexdigest()
    return {"shape": [int(s) for s in arr.shape], "dtype": "float64", "sha256": h}


# ---------------------------------------------------------------------------
# Array codec and repr summaries (spec 04 §5.1-5.4)
# ---------------------------------------------------------------------------

_ARRAY_DTYPES: dict[str, type[np.generic]] = {
    "float64": np.float64,
    "int64": np.int64,
    "bool": np.bool_,
}
_NONFINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


def schema_invalid(message: str, **details: Any) -> SchemaError:
    """Return (not raise) a ``SchemaError`` with code ``schema.invalid``."""
    return SchemaError(message, code="schema.invalid", details=details)


def freeze_array(a: Any, dtype: Any = np.float64) -> NDArray[Any]:
    """Return an owned C-contiguous copy of ``a`` as ``dtype`` whose writeable flag can never
    be re-enabled.
    """
    arr = np.asarray(a, dtype=dtype, order="C")
    # an array over an immutable bytes object: neither it nor its .base can be made writeable again
    base = np.frombuffer(arr.tobytes(order="C"), dtype=arr.dtype)
    return base.reshape(arr.shape)


def _dtype_name(a: NDArray[Any]) -> str:
    for name, dt in _ARRAY_DTYPES.items():
        if a.dtype == dt:
            return name
    raise TypeError(f"arrays in results must be float64, int64 or bool, got {a.dtype}")


def float_json(v: float) -> float | str:
    """A float as JSON: finite values unchanged, non-finite as ``"NaN"``, ``"Infinity"``,
    ``"-Infinity"``.
    """
    if math.isfinite(v):
        return v
    return "NaN" if math.isnan(v) else ("Infinity" if v > 0 else "-Infinity")


def encode_array(a: NDArray[Any], *, binary: bool = False) -> dict[str, Any]:
    """Encode an array as a tagged dict: flat C-order list, or base64 of little-endian bytes if
    ``binary``.
    """
    a = np.asarray(a)
    name = _dtype_name(a)
    out: dict[str, Any] = {"__ndarray__": 1, "dtype": name, "shape": [int(s) for s in a.shape]}
    if binary:
        le = np.ascontiguousarray(a, dtype=np.dtype(_ARRAY_DTYPES[name]).newbyteorder("<"))
        out["b64"] = base64.b64encode(le.tobytes()).decode("ascii")
    else:
        flat = a.ravel(order="C").tolist()
        if name == "float64" and not np.isfinite(a).all():
            flat = [float_json(v) for v in flat]
        out["data"] = flat
    return out


def is_tagged(obj: Any) -> bool:
    """Whether ``obj`` is a tagged-array dict."""
    return isinstance(obj, Mapping) and "__ndarray__" in obj


def _decode_float_list(data: list[Any] | tuple[Any, ...]) -> F64:
    out = np.empty(len(data), dtype=np.float64)
    for i, v in enumerate(data):
        if isinstance(v, str):
            if v not in _NONFINITE:
                raise schema_invalid(f"bad float entry {v!r} in array")
            out[i] = _NONFINITE[v]
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[i] = v
        else:
            raise schema_invalid(f"bad float entry {v!r} in array")
    return out


def decode_array(d: Any) -> NDArray[Any]:
    """Decode a tagged array dict (list or base64 form) or a raw ndarray into a read-only array."""
    if isinstance(d, np.ndarray):
        if d.dtype.kind not in "fib" or d.dtype.hasobject:
            raise schema_invalid(f"unsupported array dtype {d.dtype}")
        to: type[np.generic] = {"f": np.float64, "i": np.int64, "b": np.bool_}[d.dtype.kind]
        return freeze_array(d, to)
    if not is_tagged(d):
        raise schema_invalid("expected a tagged array")
    keys = set(d)
    if (
        d.get("__ndarray__") != 1
        or not ({"__ndarray__", "dtype", "shape"} < keys)
        or len(keys) != 4
        or not (keys & {"data", "b64"})
    ):
        raise schema_invalid(f"malformed tagged array with keys {sorted(keys)}")
    name = d["dtype"]
    if name not in _ARRAY_DTYPES:
        raise schema_invalid(f"unsupported array dtype {name!r}")
    shape = d["shape"]
    if not isinstance(shape, (list, tuple)) or not all(
        isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in shape
    ):
        raise schema_invalid(f"bad array shape {shape!r}")
    size = math.prod(shape)
    dt: np.dtype[Any] = np.dtype(_ARRAY_DTYPES[name])
    if "b64" in d:
        if not isinstance(d["b64"], str):
            raise schema_invalid("b64 payload must be a string")
        try:
            raw = base64.b64decode(d["b64"].encode("ascii"), validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise schema_invalid("bad base64 payload") from exc
        if len(raw) != size * dt.itemsize:
            raise schema_invalid("base64 payload size does not match shape")
        flat = np.frombuffer(raw, dtype=dt.newbyteorder("<"))
    else:
        data = d["data"]
        if not isinstance(data, (list, tuple)) or len(data) != size:
            raise schema_invalid("array data must be a flat list matching the shape")
        if name == "float64":
            flat = _decode_float_list(data)
        elif name == "int64":
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in data):
                raise schema_invalid("int64 array data must be integers")
            flat = np.array(data, dtype=np.int64)
        else:
            if not all(isinstance(v, bool) for v in data):
                raise schema_invalid("bool array data must be booleans")
            flat = np.array(data, dtype=np.bool_)
    return freeze_array(np.asarray(flat).reshape(shape), dt)


def summarize(value: Any) -> str:
    """Short repr for a field value: arrays as dtype, shape and at most four entries; results
    by type tag.
    """
    if isinstance(value, np.ndarray):
        flat = value.ravel()
        head = ", ".join(repr(v.item()) for v in flat[:4])
        more = ", …" if flat.size > 4 else ""
        return f"array({value.dtype}, shape={value.shape}, [{head}{more}])"
    tag = getattr(type(value), "TYPE", None)
    if isinstance(tag, str) and hasattr(value, "provenance"):
        return f"<{tag}>"
    if isinstance(value, tuple):
        items = [summarize(v) for v in value[:4]]
        if len(value) > 4:
            items.append(f"… ({len(value)} items)")
        return "(" + ", ".join(items) + ("," if len(value) == 1 else "") + ")"
    return repr(value)


# ---------------------------------------------------------------------------
# Whole-call validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Data:
    """Validated, normalized arrays of one call, with digests of the physical inputs."""

    M: int
    m: int
    X: F64 | None
    f: F64 | None
    grads: F64 | None
    quad_weights: F64 | None
    sample_weights: F64 | None
    digests: dict[str, Any]


def data(
    space: Any,
    *,
    X: Any = None,
    f: Any = None,
    grads: Any = None,
    quad_weights: Any = None,
    sample_weights: Any = None,
    require: Iterable[str] = (),
    use_quad_weights: bool = False,
    use_sample_weights: bool = False,
) -> Data:
    """Validate a call's ``X``/``f``/``grads``/weights in spec 02 order; return normalized arrays
    and digests.

    ``require`` names arrays that must be given. Weights default (``1/M`` or ones)
    only when ``use_quad_weights``/``use_sample_weights`` is set; otherwise an
    absent weight array stays ``None``. ``digests`` maps each given array name
    to :func:`digest` of its coerced physical array.
    """
    check_space(space)
    m = space.dim
    given = {
        "X": X,
        "f": f,
        "grads": grads,
        "quad_weights": quad_weights,
        "sample_weights": sample_weights,
    }
    for key in require:
        if given[key] is None:
            raise InputError(f"{key}: required", code="input.parameter", details={"param": key})
    # 1. type and dtype
    arrs = {k: coerce(v, k) for k, v in given.items() if v is not None}
    if not any(k in arrs for k in ("X", "f", "grads")):
        raise InputError(
            "at least one of X, f, grads is required",
            code="input.parameter",
            details={"param": "X"},
        )
    # 2. shape
    if "X" in arrs:
        arrs["X"] = _as_2d(arrs["X"], "X", m, allow_1d=True)
    if "grads" in arrs:
        arrs["grads"] = _as_2d(arrs["grads"], "grads", m, allow_1d=True)
    if "f" in arrs:
        arrs["f"] = _values_shape(arrs["f"], "f")
    for k in ("quad_weights", "sample_weights"):
        if k in arrs and arrs[k].ndim != 1:
            raise _shape_error(k, "(M,)", arrs[k].shape, ["M"])
    # 3. sizes agree
    order = [k for k in ("X", "f", "grads", "quad_weights", "sample_weights") if k in arrs]
    M = int(arrs[order[0]].shape[0])
    for k in order[1:]:
        if arrs[k].shape[0] != M:
            raise _size_error(k, M, arrs[k].shape[0])
    # 4. finiteness
    for k in order:
        check_finite(arrs[k], k)
    # 5. support
    if "X" in arrs:
        _check_support(arrs["X"], space, "X")
    # 6. weights
    if "quad_weights" in arrs:
        _check_quad(arrs["quad_weights"], "quad_weights")
    if "sample_weights" in arrs:
        _check_sample(arrs["sample_weights"], "sample_weights")

    digests: dict[str, Any] = {k: digest(arrs[k]) for k in order}
    qw = (
        np.array(arrs["quad_weights"], copy=True)
        if "quad_weights" in arrs
        else (np.full(M, 1.0 / M) if use_quad_weights else None)
    )
    sw = (
        np.array(arrs["sample_weights"], copy=True)
        if "sample_weights" in arrs
        else (np.ones(M) if use_sample_weights else None)
    )
    return Data(
        M=M,
        m=m,
        X=space._to_normalized(arrs["X"]) if "X" in arrs else None,
        f=np.array(arrs["f"], copy=True) if "f" in arrs else None,
        grads=space._grads_to_normalized(arrs["grads"]) if "grads" in arrs else None,
        quad_weights=qw,
        sample_weights=sw,
        digests=digests,
    )


# ---------------------------------------------------------------------------
# Scalars and choices
# ---------------------------------------------------------------------------


def integer(n: Any, name: str, *, lo: int | None = None, hi: int | None = None) -> int:
    """Return ``operator.index(n)`` (bool rejected), required to satisfy ``lo <= n <= hi``."""
    if isinstance(n, (bool, np.bool_)):
        raise InputError(
            f"{name}: expected an integer, got bool",
            code="input.parameter",
            details={"param": name, "got": repr(n)},
        )
    try:
        val = operator.index(n)
    except TypeError as exc:
        raise InputError(
            f"{name}: expected an integer, got {type(n).__name__}",
            code="input.parameter",
            details={"param": name, "got": repr(n)},
        ) from exc
    if (lo is not None and val < lo) or (hi is not None and val > hi):
        rng = f"[{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]"
        raise InputError(
            f"{name}: expected an integer in {rng}, got {val}",
            code="input.parameter",
            details={"param": name, "expected": [lo, hi], "got": val},
        )
    return int(val)


def dimension(n: Any, m: int, lo: int = 1, hi: int | None = None, name: str = "n") -> int:
    """Validate an active dimension: an integer with
    ``lo <= n <= (m - 1 if hi is None else hi)``.
    """
    return integer(n, name, lo=lo, hi=(m - 1 if hi is None else hi))


def real(
    x: Any,
    name: str,
    *,
    lo: float | None = None,
    hi: float | None = None,
    lo_open: bool = False,
    hi_open: bool = False,
    finite: bool = True,
) -> float:
    """Return a real scalar as ``float`` (bool rejected), optionally finite and within bounds."""
    if isinstance(x, (bool, np.bool_)) or not isinstance(
        x, (numbers.Real, np.floating, np.integer)
    ):
        raise InputError(
            f"{name}: expected a real number, got {type(x).__name__}",
            code="input.parameter",
            details={"param": name},
        )
    val = float(x)
    if finite and not math.isfinite(val):
        raise InputError(
            f"{name}: expected a finite number, got {val!r}",
            code="input.parameter",
            details={"param": name, "got": str(val)},
        )
    bad = (
        (lo is not None and (val <= lo if lo_open else val < lo))
        or (hi is not None and (val >= hi if hi_open else val > hi))
        or math.isnan(val)
    )
    if bad:
        lb = "(" if lo_open else "["
        rb = ")" if hi_open else "]"
        raise InputError(
            f"{name}: expected a number in {lb}{lo if lo is not None else '-inf'}, "
            f"{hi if hi is not None else 'inf'}{rb}, got {val!r}",
            code="input.parameter",
            details={
                "param": name,
                "expected": [lo, hi],
                "got": val if math.isfinite(val) else str(val),
            },
        )
    return val


def boolean(b: Any, name: str) -> bool:
    """Require a Python or NumPy bool; return ``bool``."""
    if not isinstance(b, (bool, np.bool_)):
        raise InputError(
            f"{name}: expected a bool, got {type(b).__name__}",
            code="input.parameter",
            details={"param": name},
        )
    return bool(b)


def choice(value: Any, name: str, options: Iterable[str]) -> str:
    """Require ``value`` to be one of the strings in ``options``; return it."""
    opts = tuple(options)
    if not isinstance(value, str) or value not in opts:
        raise InputError(
            f"{name}: expected one of {list(opts)}, got {value!r}",
            code="input.parameter",
            details={"param": name, "expected": list(opts), "got": repr(value)},
        )
    return value


def min_samples(M: int, minimum: int, what: str) -> None:
    """Raise ``input.too_few_samples`` if ``M < minimum``; ``what`` names the method."""
    if M < minimum:
        raise InputError(
            f"{what} needs at least {minimum} samples, got {M}",
            code="input.too_few_samples",
            details={"method": what, "minimum": int(minimum), "got": int(M)},
        )
