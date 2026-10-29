r"""Input spaces: ``Uniform`` and ``Gaussian`` with their affine maps.

A space owns the map :math:`\xi = Ax + b` between physical inputs :math:`\xi`
and normalized inputs :math:`x`. Every computation is done, and every result
is stored, in normalized coordinates. The reference density :math:`\rho` of
:math:`x` is uniform on :math:`[-1, 1]^m` for ``Uniform`` and :math:`N(0, I_m)`
for ``Gaussian``.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, Self

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

from active_subspaces import _random
from active_subspaces import _validation as _v
from active_subspaces.errors import InputError, SchemaError

__all__ = ("Gaussian", "InputSpace", "Uniform")

_SUPPORT_RTOL = 1e-12
_SYMMETRY_RTOL = 1e-12

_F64 = NDArray[np.float64]


class InputSpace(Protocol):
    r"""Protocol shared by :class:`Uniform` and :class:`Gaussian`.

    An input space describes the density of the physical inputs
    :math:`\xi \in \mathbb{R}^m` and owns the affine map
    :math:`\xi = Ax + b` to normalized inputs :math:`x`, whose reference
    density :math:`\rho` is fixed by the kind of space. Functions that take a
    ``space`` argument accept any object with these members; v1 provides
    exactly two implementations.

    Parameters
    ----------
    *args, **kwargs
        A protocol is not instantiated. Construct :class:`Uniform` or
        :class:`Gaussian` instead.

    See Also
    --------
    Uniform : Independent uniform inputs on a box.
    Gaussian : Gaussian inputs with a given mean and covariance.

    Notes
    -----
    In row form, with points and gradients as the rows of ``(M, m)`` arrays,

    .. math::

        X_{\mathrm{norm}} = (X - \mathbf{1} b^\top) A^{-\top}, \qquad
        G_{\mathrm{norm}} = G A, \qquad G = G_{\mathrm{norm}} A^{-1}.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> def describe(space: ac.InputSpace) -> str:
    ...     return f"{space.kind} in {space.dim} dimensions"
    >>> describe(ac.Uniform([0.0, 0.0], [1.0, 2.0]))
    'uniform in 2 dimensions'
    """

    @property
    def kind(self) -> str:
        """Kind of space: ``"uniform"`` or ``"gaussian"``."""

    @property
    def names(self) -> tuple[str, ...]:
        """Input names, length ``m``; default ``("x1", ..., "xm")``."""

    @property
    def units(self) -> tuple[str | None, ...]:
        """Input units, length ``m``; default ``(None,) * m``."""

    @property
    def dim(self) -> int:
        """Number of inputs ``m``."""

    @property
    def scale(self) -> NDArray[np.float64]:
        """The matrix ``A`` of the map ``xi = A x + b``, read-only ``(m, m)``."""

    @property
    def shift(self) -> NDArray[np.float64]:
        """The shift ``b`` of the map ``xi = A x + b``, read-only ``(m,)``."""

    @property
    def reference_covariance(self) -> NDArray[np.float64]:
        """Covariance of the reference density in normalized coordinates, ``(m, m)``."""

    def to_normalized(self, X: ArrayLike) -> NDArray[np.float64]:
        """Map physical points to normalized coordinates.

        Parameters
        ----------
        X : array_like, shape (M, m), float
            Physical points, one per row.

        Returns
        -------
        numpy.ndarray, shape (M, m), float64
            Normalized points, a new array.

        See Also
        --------
        Uniform.to_normalized : The implementation, with an example.
        """

    def from_normalized(self, X: ArrayLike) -> NDArray[np.float64]:
        """Map normalized points to physical coordinates.

        Parameters
        ----------
        X : array_like, shape (M, m), float
            Normalized points, one per row.

        Returns
        -------
        numpy.ndarray, shape (M, m), float64
            Physical points, a new array.

        See Also
        --------
        Uniform.from_normalized : The implementation, with an example.
        """

    def grads_to_normalized(self, G: ArrayLike) -> NDArray[np.float64]:
        """Map physical gradients to normalized gradients.

        Parameters
        ----------
        G : array_like, shape (M, m), float
            Physical gradients, one per row.

        Returns
        -------
        numpy.ndarray, shape (M, m), float64
            Normalized gradients ``G A``, a new array.

        See Also
        --------
        Uniform.grads_to_normalized : The implementation, with an example.
        """

    def grads_to_physical(self, G: ArrayLike) -> NDArray[np.float64]:
        """Map normalized gradients to physical gradients.

        Parameters
        ----------
        G : array_like, shape (M, m), float
            Normalized gradients, one per row.

        Returns
        -------
        numpy.ndarray, shape (M, m), float64
            Physical gradients ``G A^{-1}``, a new array.

        See Also
        --------
        Uniform.grads_to_physical : The implementation, with an example.
        """

    def directions_to_physical(self, W: ArrayLike) -> NDArray[np.float64]:
        """Map normalized directions to an orthonormal physical basis.

        Parameters
        ----------
        W : array_like, shape (m, k), float
            Normalized directions as columns, ``1 <= k <= m``.

        Returns
        -------
        numpy.ndarray, shape (m, k), float64
            Orthonormal basis of ``span(A^{-T} W)``, sign-oriented.

        See Also
        --------
        Uniform.directions_to_physical : The implementation, with the formula.
        """

    def contains(self, X: ArrayLike, *, normalized: bool = False) -> NDArray[np.bool_]:
        """Return whether each point lies in the support.

        Parameters
        ----------
        X : array_like, shape (M, m), float
            Points, one per row.
        normalized : bool, default False
            Read ``X`` in normalized coordinates.

        Returns
        -------
        numpy.ndarray, shape (M,), bool
            ``True`` for each row inside the support.

        See Also
        --------
        Uniform.contains : The implementation, with the tolerance.
        """

    def sample(
        self, M: int, *, rng: _random.RngLike = None, normalized: bool = False
    ) -> NDArray[np.float64]:
        """Draw independent points from the input density.

        Parameters
        ----------
        M : int
            Number of points, at least 1.
        rng : None, int, numpy.random.SeedSequence or numpy.random.Generator, optional
            Source of randomness.
        normalized : bool, default False
            Draw from the reference density instead.

        Returns
        -------
        numpy.ndarray, shape (M, m), float64
            The points, physical unless ``normalized``.

        See Also
        --------
        Uniform.sample : The implementation, with an example.
        """

    def to_dict(self, *, binary: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible dict with a ``"type"`` tag.

        Parameters
        ----------
        binary : bool, default False
            Encode arrays as base64 instead of lists.

        Returns
        -------
        dict
            The encoded space.

        See Also
        --------
        Uniform.to_dict : The implementation, with an example.
        """


def _sign_pivots(W: _F64) -> NDArray[np.intp]:
    """Row of each column's sign pivot: the first entry within sqrt(eps) of the largest magnitude.

    The tolerance makes ties that are exact in exact arithmetic platform-independent [D-81].
    A column with a NaN keeps the plain argmax.
    """
    a = np.abs(W)
    amax = a.max(axis=0)
    first = np.argmax(a >= amax * (1.0 - np.sqrt(np.finfo(np.float64).eps)), axis=0)
    return np.where(np.isnan(amax), np.argmax(a, axis=0), first)


def _orient_columns(W: ArrayLike) -> NDArray[np.float64]:
    """Flip column signs so each column's sign pivot (see ``_sign_pivots``) is positive.

    Returns a new array.
    """
    W = np.array(W, dtype=np.float64, copy=True)
    if W.size == 0:
        return W
    idx = _sign_pivots(W)
    signs = np.sign(W[idx, np.arange(W.shape[1])])
    signs[signs == 0] = 1.0
    out: _F64 = W * signs
    return out


def _param_error(msg: str, **details: Any) -> InputError:
    return InputError(msg, code="input.parameter", details=details)


def _labels(names: Any, units: Any, m: int) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    if names is None:
        nm = tuple(f"x{j + 1}" for j in range(m))
    else:
        if isinstance(names, str):
            raise _param_error("names: expected a sequence of strings", param="names")
        nm = tuple(names)
        if len(nm) != m or not all(isinstance(s, str) and s for s in nm) or len(set(nm)) != m:
            raise _param_error(
                f"names: expected {m} unique non-empty strings", param="names", expected=m
            )
    if units is None:
        un: tuple[str | None, ...] = (None,) * m
    else:
        if isinstance(units, str):
            raise _param_error("units: expected a sequence of strings or None", param="units")
        un = tuple(units)
        if len(un) != m or not all(u is None or isinstance(u, str) for u in un):
            raise _param_error(f"units: expected {m} strings or None", param="units", expected=m)
    return nm, un


def _param_vector(a: Any, name: str) -> _F64:
    try:
        arr = _v.coerce(a, name)
    except InputError as exc:
        raise _param_error(str(exc.message), param=name) from exc
    if arr.ndim != 1 or arr.shape[0] < 1:
        raise _param_error(
            f"{name}: expected a 1-D array with at least one entry, got shape {arr.shape}",
            param=name,
        )
    if not np.all(np.isfinite(arr)):
        raise _param_error(f"{name}: must be finite", param=name)
    return arr


def _points_shape(X: Any, m: int, name: str) -> _F64:
    arr = _v.coerce(X, name)
    if arr.ndim == 1 and m == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[1] != m:
        raise InputError(
            f"{name}: expected shape (M, {m}), got {arr.shape}",
            code="input.shape",
            details={"param": name, "expected": ["M", m], "got": list(arr.shape)},
        )
    _v.check_finite(arr, name)
    return arr


class _SpaceBase:
    """Shared behavior of the concrete spaces: maps, support, sampling, equality, serialization."""

    __slots__ = ()
    kind: ClassVar[str]
    _ARRAYS: ClassVar[tuple[str, ...]]

    if typing.TYPE_CHECKING:
        # Provided by the concrete dataclasses; declared here for the type checker only.
        names: tuple[str, ...]
        units: tuple[str | None, ...]

        def __init__(
            self,
            *arrays: ArrayLike,
            names: Sequence[str] | None,
            units: Sequence[str | None] | None,
        ) -> None: ...
        @property
        def shift(self) -> _F64: ...
        def _to_normalized(self, X: _F64) -> _F64: ...
        def _from_normalized(self, X: _F64) -> _F64: ...
        def _grads_to_normalized(self, G: _F64) -> _F64: ...
        def _grads_to_physical(self, G: _F64) -> _F64: ...
        def _inv_transpose(self, W: _F64) -> _F64: ...
        def _support_violations(self, X: _F64) -> NDArray[np.bool_]: ...
        def _draw(self, gen: np.random.Generator, M: int) -> _F64: ...

    # --- maps (public versions validate; private ones assume (M, m) float64 finite) ---

    @property
    def dim(self) -> int:
        """Number of inputs ``m``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ac.Uniform([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).dim
        3
        """
        return int(self.shift.shape[0])

    def to_normalized(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Map physical points to normalized coordinates.

        Parameters
        ----------
        X : array_like, shape (M, m), float
            Points in physical coordinates, one per row. A 1-D array is read
            as ``(M, 1)`` when ``m == 1``. Points outside the support are not
            rejected here.

        Returns
        -------
        numpy.ndarray, shape (M, m), float64
            Normalized points :math:`(X - \mathbf{1}b^\top)A^{-\top}`, a new array.

        Raises
        ------
        InputError
            ``input.dtype``, ``input.shape`` or ``input.nonfinite`` for a bad ``X``.

        See Also
        --------
        from_normalized : The inverse map.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 10.0], [2.0, 20.0])
        >>> space.to_normalized([[1.0, 15.0], [2.0, 10.0]])
        array([[ 0.,  0.],
               [ 1., -1.]])
        """
        return self._to_normalized(_points_shape(X, self.dim, "X"))

    def from_normalized(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Map normalized points to physical coordinates.

        Parameters
        ----------
        X : array_like, shape (M, m), float
            Points in normalized coordinates, one per row.

        Returns
        -------
        numpy.ndarray, shape (M, m), float64
            Physical points :math:`X A^\top + \mathbf{1} b^\top`, a new array.

        Raises
        ------
        InputError
            ``input.dtype``, ``input.shape`` or ``input.nonfinite`` for a bad ``X``.

        See Also
        --------
        to_normalized : The inverse map.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 10.0], [2.0, 20.0])
        >>> space.from_normalized([[0.0, 0.0], [1.0, -1.0]])
        array([[ 1., 15.],
               [ 2., 10.]])
        """
        return self._from_normalized(_points_shape(X, self.dim, "X"))

    def grads_to_normalized(self, G: ArrayLike) -> NDArray[np.float64]:
        r"""Map physical gradients to normalized gradients.

        Parameters
        ----------
        G : array_like, shape (M, m), float
            Gradients :math:`\nabla_\xi f` with respect to the physical inputs,
            one per row.

        Returns
        -------
        numpy.ndarray, shape (M, m), float64
            Gradients :math:`\nabla_x f = A^\top \nabla_\xi f` as rows, that is
            :math:`G A`, a new array.

        Raises
        ------
        InputError
            ``input.dtype``, ``input.shape`` or ``input.nonfinite`` for a bad ``G``.

        See Also
        --------
        grads_to_physical : The inverse map.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 10.0], [2.0, 20.0])
        >>> space.grads_to_normalized([[1.0, 1.0]])
        array([[1., 5.]])
        """
        return self._grads_to_normalized(_points_shape(G, self.dim, "G"))

    def grads_to_physical(self, G: ArrayLike) -> NDArray[np.float64]:
        r"""Map normalized gradients to physical gradients.

        Parameters
        ----------
        G : array_like, shape (M, m), float
            Gradients :math:`\nabla_x f` with respect to the normalized inputs,
            one per row.

        Returns
        -------
        numpy.ndarray, shape (M, m), float64
            Gradients :math:`\nabla_\xi f = A^{-\top} \nabla_x f` as rows, that
            is :math:`G A^{-1}`, a new array.

        Raises
        ------
        InputError
            ``input.dtype``, ``input.shape`` or ``input.nonfinite`` for a bad ``G``.

        See Also
        --------
        grads_to_normalized : The inverse map.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 10.0], [2.0, 20.0])
        >>> space.grads_to_physical([[1.0, 5.0]])
        array([[1., 1.]])
        """
        return self._grads_to_physical(_points_shape(G, self.dim, "G"))

    def directions_to_physical(self, W: ArrayLike) -> NDArray[np.float64]:
        r"""Return an orthonormal basis of the physical view of normalized directions.

        Parameters
        ----------
        W : array_like, shape (m, k), float
            Directions in normalized coordinates as columns, with
            ``1 <= k <= m`` (for example ``SubspaceResult.W1``).

        Returns
        -------
        numpy.ndarray, shape (m, k), float64
            Orthonormal basis of :math:`\operatorname{span}(A^{-\top} W)` in
            physical coordinates, each column signed so that its
            largest-magnitude entry is positive.

        Raises
        ------
        InputError
            ``input.shape`` if ``W`` is not ``(m, k)`` with ``1 <= k <= m``;
            ``input.dtype`` or ``input.nonfinite`` for a bad ``W``.

        Notes
        -----
        If :math:`f(\xi) \approx h(a^\top \xi)`, then in normalized coordinates
        :math:`f \approx h((A^\top a)^\top x + a^\top b)`, so the normalized
        direction is :math:`w \propto A^\top a` and :math:`a \propto A^{-\top} w`.
        The basis is the thin QR factor of :math:`A^{-\top} W`. The eigenvalues
        have no physical counterpart. Physical coordinates carry units, so the
        entries of physical directions are not comparable across inputs; the
        normalized ``W`` is the scale-free measure of importance.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 10.0], [2.0, 20.0])
        >>> w = np.array([[1.0], [1.0]]) / np.sqrt(2.0)
        >>> space.directions_to_physical(w).round(4)
        array([[0.9806],
               [0.1961]])
        """
        m = self.dim
        arr = _v.coerce(W, "W")
        if arr.ndim != 2 or arr.shape[0] != m or not (1 <= arr.shape[1] <= m):
            raise InputError(
                f"W: expected shape ({m}, k) with 1 <= k <= {m}, got {arr.shape}",
                code="input.shape",
                details={"param": "W", "expected": [m, "k"], "got": list(arr.shape)},
            )
        _v.check_finite(arr, "W")
        Q, _ = np.linalg.qr(self._inv_transpose(arr), mode="reduced")
        return _orient_columns(Q)

    def contains(self, X: ArrayLike, *, normalized: bool = False) -> NDArray[np.bool_]:
        r"""Return whether each point lies in the support of the input density.

        Parameters
        ----------
        X : array_like, shape (M, m), float
            Points, one per row, in physical coordinates (or normalized ones
            if ``normalized``).
        normalized : bool, default False
            Read ``X`` in normalized coordinates.

        Returns
        -------
        numpy.ndarray, shape (M,), bool
            ``True`` for each row inside the support.

        Raises
        ------
        InputError
            ``input.dtype``, ``input.shape`` or ``input.nonfinite`` for a bad ``X``.

        Notes
        -----
        For ``Uniform`` a point is inside when
        :math:`\ell_j - \tau_j \le \xi_j \le u_j + \tau_j` for every :math:`j`,
        with :math:`\tau_j = 10^{-12} (u_j - \ell_j)`. A ``Gaussian`` space has
        full support, so every finite point is inside.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [1.0, 1.0])
        >>> space.contains([[0.5, 0.5], [1.5, 0.5]])
        array([ True, False])
        """
        arr = _points_shape(X, self.dim, "X")
        if normalized:
            arr = self._from_normalized(arr)
        out: NDArray[np.bool_] = ~self._support_violations(arr).any(axis=1)
        return out

    def sample(
        self, M: int, *, rng: _random.RngLike = None, normalized: bool = False
    ) -> NDArray[np.float64]:
        r"""Draw independent points from the input density.

        Parameters
        ----------
        M : int
            Number of points, at least 1.
        rng : None, int, numpy.random.SeedSequence or numpy.random.Generator, optional
            Source of randomness. ``None`` draws fresh OS entropy; an integer
            seeds a new generator, so equal seeds give equal draws.
        normalized : bool, default False
            Return draws :math:`x \sim \rho` from the reference density instead
            of physical points.

        Returns
        -------
        numpy.ndarray, shape (M, m), float64
            The points, physical unless ``normalized``.

        Raises
        ------
        InputError
            ``input.parameter`` for a bad ``M`` or ``rng``.

        Notes
        -----
        ``Uniform`` draws ``rng.uniform(-1, 1, (M, m))`` and ``Gaussian`` draws
        ``rng.standard_normal((M, m))``; each is then mapped by
        :meth:`from_normalized` unless ``normalized``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 10.0], [2.0, 20.0])
        >>> X = space.sample(100, rng=0)
        >>> X.shape, bool(space.contains(X).all())
        ((100, 2), True)
        """
        M = _v.integer(M, "M", lo=1)
        gen, _ = _random.generator(rng)
        Z = self._draw(gen, M)
        return Z if normalized else self._from_normalized(Z)

    # --- equality, repr, pickling ---

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        for name in self._ARRAYS:
            a, b = getattr(self, name), getattr(other, name)
            if a.shape != b.shape or not np.array_equal(a, b):
                return False
        return self.names == other.names and self.units == other.units

    __hash__ = None  # type: ignore[assignment]  # spaces hold arrays and are not hashable

    def __repr__(self) -> str:
        parts = [f"{n}={_v.summarize(getattr(self, n))}" for n in self._ARRAYS]
        default_names = tuple(f"x{j + 1}" for j in range(self.dim))
        if self.names != default_names:
            parts.append(f"names={self.names!r}")
        if any(u is not None for u in self.units):
            parts.append(f"units={self.units!r}")
        return f"{type(self).__name__}(m={self.dim}, {', '.join(parts)})"

    def __reduce__(self) -> tuple[Any, ...]:
        arrays = (getattr(self, n) for n in self._ARRAYS)
        args = (*arrays, self.names, self.units)
        return (type(self), args)

    # --- serialization ---

    def to_dict(self, *, binary: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible dict with a ``"type"`` tag.

        Parameters
        ----------
        binary : bool, default False
            Encode the arrays as base64 of their little-endian bytes instead of
            flat lists.

        Returns
        -------
        dict
            Keys ``"type"`` (the class name), the parameter arrays as tagged
            array dicts, ``"names"`` and ``"units"``.

        See Also
        --------
        from_dict : The inverse.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> d = ac.Uniform([0.0], [2.0]).to_dict()
        >>> d["type"], d["lower"]["data"], d["names"]
        ('Uniform', [0.0], ['x1'])
        """
        d: dict[str, Any] = {"type": type(self).__name__}
        for n in self._ARRAYS:
            d[n] = _v.encode_array(getattr(self, n), binary=binary)
        d["names"] = list(self.names)
        d["units"] = list(self.units)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        """Rebuild a space from the output of :meth:`to_dict`.

        Parameters
        ----------
        d : Mapping
            A dict with exactly the keys that :meth:`to_dict` writes, whose
            ``"type"`` is this class's name.

        Returns
        -------
        Uniform or Gaussian
            A space of this class, equal to the one that was encoded.

        Raises
        ------
        SchemaError
            ``schema.invalid`` if the dict is malformed, has the wrong type tag
            or encodes invalid parameters.

        See Also
        --------
        to_dict : The inverse.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Gaussian([0.0, 1.0], [1.0, 4.0])
        >>> ac.Gaussian.from_dict(space.to_dict(binary=True)) == space
        True
        """
        expected = {"type", *cls._ARRAYS, "names", "units"}
        if not isinstance(d, Mapping) or set(d) != expected:
            got = sorted(d) if isinstance(d, Mapping) else type(d).__name__
            raise _v.schema_invalid(
                f"{cls.__name__} dict must have keys {sorted(expected)}, got {got}"
            )
        if d["type"] != cls.__name__:
            raise _v.schema_invalid(f"expected type {cls.__name__!r}, got {d['type']!r}")
        arrays = [_v.decode_array(d[n]) for n in cls._ARRAYS]
        names, units = d["names"], d["units"]
        if not isinstance(names, (list, tuple)) or not isinstance(units, (list, tuple)):
            raise _v.schema_invalid("names and units must be lists")
        try:
            return cls(*arrays, names=tuple(names), units=tuple(units))
        except InputError as exc:
            raise _v.schema_invalid(f"invalid {cls.__name__}: {exc.message}") from exc


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Uniform(_SpaceBase):
    r"""Independent uniform inputs on a box, normalized to :math:`[-1, 1]^m`.

    Parameters
    ----------
    lower : array_like, shape (m,), float
        Lower bounds :math:`\ell` in physical units, finite, ``m >= 1``.
    upper : array_like, shape (m,), float
        Upper bounds :math:`u` in physical units, finite, with
        ``lower < upper`` in every coordinate.
    names : sequence of str, optional
        ``m`` unique non-empty input names. Default ``("x1", ..., "xm")``.
    units : sequence of str or None, optional
        ``m`` unit labels, ``None`` for a dimensionless input. Default all
        ``None``.

    Attributes
    ----------
    lower, upper : numpy.ndarray, shape (m,), float64
        Read-only copies of the bounds.
    names : tuple of str
        The input names.
    units : tuple of str or None
        The unit labels.
    kind : str
        ``"uniform"``.

    Raises
    ------
    InputError
        ``input.parameter`` if the bounds are not 1-D, finite, of equal length
        and strictly ordered, or if ``names`` or ``units`` have the wrong
        length or type.

    See Also
    --------
    Gaussian : Gaussian inputs.
    Uniform.reference : The normalized box :math:`[-1, 1]^m`.

    Notes
    -----
    The physical density is uniform on :math:`\prod_j [\ell_j, u_j]`. The map
    :math:`\xi = Ax + b` has

    .. math::

        A = \operatorname{diag}\Bigl(\frac{u - \ell}{2}\Bigr), \qquad
        b = \frac{u + \ell}{2},

    so :math:`x` is uniform on :math:`[-1, 1]^m` with density :math:`2^{-m}`
    and covariance :math:`\Sigma = \tfrac13 I` [1]_. Spaces are immutable and
    compare equal when their arrays (bitwise), names and units agree; they are
    not hashable.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 10.0], [2.0, 20.0], names=["a", "b"])
    >>> space.dim, space.names, space.units
    (2, ('a', 'b'), (None, None))
    >>> space.to_normalized([[2.0, 10.0]])
    array([[ 1., -1.]])
    """

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    names: tuple[str, ...] = None  # type: ignore[assignment]  # tuple after __post_init__
    units: tuple[str | None, ...] = None  # type: ignore[assignment]  # tuple after __post_init__
    _half: NDArray[np.float64] = field(init=False, repr=False)
    _mid: NDArray[np.float64] = field(init=False, repr=False)

    kind: ClassVar[Literal["uniform"]] = "uniform"
    _ARRAYS: ClassVar[tuple[str, ...]] = ("lower", "upper")

    if typing.TYPE_CHECKING:
        # The dataclass __init__ accepts anything array-like; __post_init__ converts.
        def __init__(
            self,
            lower: ArrayLike,
            upper: ArrayLike,
            names: Sequence[str] | None = None,
            units: Sequence[str | None] | None = None,
        ) -> None: ...

    def __post_init__(self) -> None:
        """Validate and freeze the parameters."""
        lo = _param_vector(self.lower, "lower")
        up = _param_vector(self.upper, "upper")
        if lo.shape != up.shape:
            raise _param_error(
                f"lower and upper must have equal length, got {lo.shape[0]} and {up.shape[0]}",
                param="upper",
            )
        if not np.all(lo < up):
            j = int(np.argmin(lo < up))
            raise _param_error(
                f"lower < upper required; fails at coordinate {j}", param="upper", coordinate=j
            )
        names, units = _labels(self.names, self.units, lo.shape[0])
        object.__setattr__(self, "lower", _v.freeze_array(lo))
        object.__setattr__(self, "upper", _v.freeze_array(up))
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "_half", _v.freeze_array((up - lo) / 2.0))
        object.__setattr__(self, "_mid", _v.freeze_array((up + lo) / 2.0))

    @classmethod
    def reference(cls, m: int) -> Uniform:
        """Return the reference space :math:`[-1, 1]^m`, whose maps are the identity.

        Parameters
        ----------
        m : int
            Number of inputs, at least 1.

        Returns
        -------
        Uniform
            The space with ``lower = -1`` and ``upper = 1``, so ``A = I`` and
            ``b = 0``. Pass it when the data are already normalized.

        Raises
        ------
        InputError
            ``input.parameter`` if ``m < 1``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ref = ac.Uniform.reference(2)
        >>> ref.lower, ref.upper
        (array([-1., -1.]), array([1., 1.]))
        """
        m = _v.integer(m, "m", lo=1)
        return cls(-np.ones(m), np.ones(m))

    @property
    def scale(self) -> NDArray[np.float64]:
        r"""The matrix :math:`A = \operatorname{diag}((u - \ell)/2)`, read-only ``(m, m)``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ac.Uniform([0.0, 10.0], [2.0, 20.0]).scale
        array([[1., 0.],
               [0., 5.]])
        """
        return _v.freeze_array(np.diag(self._half))

    @property
    def shift(self) -> NDArray[np.float64]:
        r"""The shift :math:`b = (u + \ell)/2`, read-only ``(m,)``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ac.Uniform([0.0, 10.0], [2.0, 20.0]).shift
        array([ 1., 15.])
        """
        return self._mid

    @property
    def reference_covariance(self) -> NDArray[np.float64]:
        r"""The covariance :math:`\Sigma = I/3` of the reference density, read-only ``(m, m)``.

        It whitens SIR, SAVE and pHd and enters QPHD.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ac.Uniform([0.0], [2.0]).reference_covariance.round(4)
        array([[0.3333]])
        """
        return _v.freeze_array(np.eye(self.dim) / 3.0)

    def _to_normalized(self, X: _F64) -> _F64:
        return (X - self._mid) / self._half

    def _from_normalized(self, X: _F64) -> _F64:
        return X * self._half + self._mid

    def _grads_to_normalized(self, G: _F64) -> _F64:
        return G * self._half

    def _grads_to_physical(self, G: _F64) -> _F64:
        return G / self._half

    def _inv_transpose(self, W: _F64) -> _F64:
        return W / self._half[:, None]

    def _support_violations(self, X: _F64) -> NDArray[np.bool_]:
        tau = _SUPPORT_RTOL * (self.upper - self.lower)
        return (X < self.lower - tau) | (X > self.upper + tau)

    def _draw(self, gen: np.random.Generator, M: int) -> _F64:
        return gen.uniform(-1.0, 1.0, (M, self.dim))


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Gaussian(_SpaceBase):
    r"""Gaussian inputs :math:`N(\mu, \Gamma)`, normalized to :math:`N(0, I_m)`.

    Parameters
    ----------
    mean : array_like, shape (m,), float
        Mean :math:`\mu` in physical units, finite, ``m >= 1``.
    cov : array_like, shape (m, m) or (m,), float
        Covariance :math:`\Gamma`: a symmetric positive definite ``(m, m)``
        matrix, or a vector of ``m`` positive variances for independent
        inputs.
    names : sequence of str, optional
        ``m`` unique non-empty input names. Default ``("x1", ..., "xm")``.
    units : sequence of str or None, optional
        ``m`` unit labels, ``None`` for a dimensionless input. Default all
        ``None``.

    Attributes
    ----------
    mean : numpy.ndarray, shape (m,), float64
        Read-only copy of the mean.
    cov : numpy.ndarray, shape (m, m) or (m,), float64
        Read-only copy of the covariance as given.
    names : tuple of str
        The input names.
    units : tuple of str or None
        The unit labels.
    kind : str
        ``"gaussian"``.

    Raises
    ------
    InputError
        ``input.parameter`` if ``mean`` is not 1-D and finite, if ``cov`` has
        the wrong shape, is not finite, is not symmetric to
        :math:`\max_{ij} |\Gamma_{ij} - \Gamma_{ji}| \le 10^{-12} \max_{ij} |\Gamma_{ij}|`,
        is not positive definite (the Cholesky factorization fails) or has a
        non-positive variance, or if ``names`` or ``units`` are bad.

    See Also
    --------
    Uniform : Independent uniform inputs.
    Gaussian.reference : The standard normal space :math:`N(0, I_m)`.

    Notes
    -----
    The map :math:`\xi = Ax + b` has :math:`A = L`, the lower Cholesky factor
    with :math:`\Gamma = L L^\top`, and :math:`b = \mu`, so
    :math:`x \sim N(0, I_m)` and :math:`\Sigma = I` [1]_. Any :math:`L' = LQ`
    with :math:`Q` orthogonal would also factor :math:`\Gamma`; it would rotate
    the normalized eigenvectors by :math:`Q^\top` and leave the eigenvalues and
    the physical subspace unchanged. Because the Cholesky factor is used, the
    normalized coordinates of a non-diagonal :math:`\Gamma` depend on the order
    of the inputs. Spaces are immutable and not hashable.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Gaussian([1.0, 2.0], [[4.0, 1.0], [1.0, 2.0]])
    >>> space.scale.round(4)
    array([[2.    , 0.    ],
           [0.5   , 1.3229]])
    >>> space.sample(3, rng=0).round(3)
    array([[ 1.251,  1.888],
           [ 2.281,  2.459],
           [-0.071,  2.211]])
    """

    mean: NDArray[np.float64]
    cov: NDArray[np.float64]
    names: tuple[str, ...] = None  # type: ignore[assignment]  # tuple after __post_init__
    units: tuple[str | None, ...] = None  # type: ignore[assignment]  # tuple after __post_init__
    _L: NDArray[np.float64] = field(init=False, repr=False)
    _diag: bool = field(init=False, repr=False)

    kind: ClassVar[Literal["gaussian"]] = "gaussian"
    _ARRAYS: ClassVar[tuple[str, ...]] = ("mean", "cov")

    if typing.TYPE_CHECKING:
        # The dataclass __init__ accepts anything array-like; __post_init__ converts.
        def __init__(
            self,
            mean: ArrayLike,
            cov: ArrayLike,
            names: Sequence[str] | None = None,
            units: Sequence[str | None] | None = None,
        ) -> None: ...

    def __post_init__(self) -> None:
        """Validate and freeze the parameters."""
        mu = _param_vector(self.mean, "mean")
        m = mu.shape[0]
        try:
            cov = _v.coerce(self.cov, "cov")
        except InputError as exc:
            raise _param_error(exc.message, param="cov") from exc
        if not np.all(np.isfinite(cov)):
            raise _param_error("cov: must be finite", param="cov")
        if cov.shape == (m,):
            if not np.all(cov > 0):
                raise _param_error("cov: variances must be positive", param="cov")
            L = np.diag(np.sqrt(cov))
            diag = True
        elif cov.shape == (m, m):
            scale = np.max(np.abs(cov))
            if np.max(np.abs(cov - cov.T)) > _SYMMETRY_RTOL * scale:
                raise _param_error("cov: must be symmetric", param="cov")
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as exc:
                raise _param_error(
                    "cov: must be positive definite (Cholesky failed)", param="cov"
                ) from exc
            diag = False
        else:
            raise _param_error(
                f"cov: expected shape ({m}, {m}) or ({m},), got {cov.shape}", param="cov"
            )
        names, units = _labels(self.names, self.units, m)
        object.__setattr__(self, "mean", _v.freeze_array(mu))
        object.__setattr__(self, "cov", _v.freeze_array(cov))
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "_L", _v.freeze_array(L))
        object.__setattr__(self, "_diag", diag)

    @classmethod
    def reference(cls, m: int) -> Gaussian:
        """Return the reference space :math:`N(0, I_m)`, whose maps are the identity.

        Parameters
        ----------
        m : int
            Number of inputs, at least 1.

        Returns
        -------
        Gaussian
            The space with zero mean and unit variances, so ``A = I`` and
            ``b = 0``. Pass it when the data are already normalized.

        Raises
        ------
        InputError
            ``input.parameter`` if ``m < 1``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ref = ac.Gaussian.reference(2)
        >>> ref.mean, ref.cov
        (array([0., 0.]), array([1., 1.]))
        """
        m = _v.integer(m, "m", lo=1)
        return cls(np.zeros(m), np.ones(m))

    @property
    def scale(self) -> NDArray[np.float64]:
        r"""The matrix :math:`A = L`, the lower Cholesky factor of ``cov``, read-only ``(m, m)``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ac.Gaussian([0.0, 0.0], [4.0, 9.0]).scale
        array([[2., 0.],
               [0., 3.]])
        """
        return self._L

    @property
    def shift(self) -> NDArray[np.float64]:
        r"""The shift :math:`b = \mu`, read-only ``(m,)``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ac.Gaussian([1.0, -1.0], [1.0, 1.0]).shift
        array([ 1., -1.])
        """
        return self.mean

    @property
    def reference_covariance(self) -> NDArray[np.float64]:
        r"""The covariance :math:`\Sigma = I` of the reference density, read-only ``(m, m)``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> ac.Gaussian.reference(2).reference_covariance
        array([[1., 0.],
               [0., 1.]])
        """
        return _v.freeze_array(np.eye(self.dim))

    def _sd(self) -> _F64:
        return np.diagonal(self._L)

    def _to_normalized(self, X: _F64) -> _F64:
        D = X - self.mean
        if self._diag:
            return D / self._sd()
        out: _F64 = scipy.linalg.solve_triangular(self._L, D.T, lower=True).T
        return out

    def _from_normalized(self, X: _F64) -> _F64:
        if self._diag:
            return X * self._sd() + self.mean
        return X @ self._L.T + self.mean

    def _grads_to_normalized(self, G: _F64) -> _F64:
        if self._diag:
            return G * self._sd()
        return G @ self._L

    def _grads_to_physical(self, G: _F64) -> _F64:
        if self._diag:
            return G / self._sd()
        out: _F64 = scipy.linalg.solve_triangular(self._L, G.T, lower=True, trans="T").T
        return out

    def _inv_transpose(self, W: _F64) -> _F64:
        if self._diag:
            return W / self._sd()[:, None]
        out: _F64 = scipy.linalg.solve_triangular(self._L, W, lower=True, trans="T")
        return out

    def _support_violations(self, X: _F64) -> NDArray[np.bool_]:
        return np.zeros(X.shape, dtype=bool)

    def _draw(self, gen: np.random.Generator, M: int) -> _F64:
        return gen.standard_normal((M, self.dim))


_SPACE_TYPES: dict[str, type[Uniform] | type[Gaussian]] = {
    "Uniform": Uniform,
    "Gaussian": Gaussian,
}


def _space_from_dict(d: Mapping[str, Any]) -> Uniform | Gaussian:
    """Rebuild a ``Uniform`` or ``Gaussian`` from its dict, dispatching on ``"type"``."""
    if not isinstance(d, Mapping) or d.get("type") not in _SPACE_TYPES:
        tag = d.get("type") if isinstance(d, Mapping) else None
        raise SchemaError(
            f"unknown input space type {tag!r}", code="schema.invalid", details={"type": repr(tag)}
        )
    return _SPACE_TYPES[d["type"]].from_dict(d)
