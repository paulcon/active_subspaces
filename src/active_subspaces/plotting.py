"""Diagnostic plots of an active-subspace analysis.

Every function takes a library result as its first argument, draws on a new
:class:`matplotlib.figure.Figure` (or on a caller's ``ax``) and returns
``(fig, ax)``. Nothing is shown, saved or written: no ``show()``, no files, no
``rcParams`` writes, and ``matplotlib.pyplot`` is never imported. Colors come
from the caller's property cycle and colormap at call time, so a
``matplotlib.style.context`` around the call restyles the plot; the ``*_kw``
mappings override each element's defaults. Data arrays ``X`` are in physical
coordinates and are normalized with the space bound to the result.

Bootstrap bands are percentile intervals by default. The eigenvalue bootstrap
is inconsistent at repeated or zero eigenvalues (Beran and Srivastava, 1985;
Eaton and Tyler, 1991), so a band there is descriptive only. A zero
eigenvalue cannot be drawn on the logarithmic axis and is masked.

:data:`PLOTS` is the closed catalogue of named plots, and
:func:`figure_to_bytes` renders a figure to PNG or SVG bytes. PNG bytes are
reproducible for the same figure, ``rcParams`` and library versions; SVG
element ids may differ between calls unless the caller sets
``rcParams["svg.hashsalt"]``.

Examples
--------
>>> import active_subspaces as ac
>>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
>>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
>>> X = space.sample(40, rng=0)
>>> f, G = ac.evaluate_with_grads(model, X=X)
>>> sub = ac.active_subspace(grads=G, space=space, nboot=50, rng=0)
>>> fig, ax = ac.plotting.plot_eigenvalues(sub)
>>> type(fig).__name__
'Figure'
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, cast

import matplotlib
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors
import matplotlib.figure
import matplotlib.patches
import matplotlib.ticker
import numpy as np
import scipy.spatial
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from active_subspaces import _validation as _v
from active_subspaces.errors import InputError, UnsupportedError
from active_subspaces.subspaces import (
    SubspaceResult,
    bootstrap_interval,
    dimension_criterion,
    mean_distance,
)
from active_subspaces.surrogates import Surrogate, predict

__all__ = (
    "PLOTS",
    "figure_to_bytes",
    "plot_active_domain",
    "plot_dimension_criterion",
    "plot_eigenvalues",
    "plot_eigenvectors",
    "plot_subspace_errors",
    "plot_sufficient_summary",
    "plot_surrogate_fit",
)

#: Documented, not exported: the band choices of the eigenvalue and distance plots.
Interval = Literal["percentile", "minmax", "none"]

_INTERVALS = ("percentile", "minmax", "none")
_LINE: dict[str, Any] = {"color": "C0", "marker": "o", "linestyle": "-"}
_BAND: dict[str, Any] = {"color": "C0", "alpha": 0.25, "linewidth": 0}
_DOMAIN: dict[str, Any] = {"color": "0.2", "linewidth": 1.5, "fill": False}
_MARKER: dict[str, Any] = {"color": "0.5", "linestyle": ":", "linewidth": 1}
_RASTER_M = 5000
_MAX_NAMED_TICKS = 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _param(message: str, param: str, **details: Any) -> InputError:
    return InputError(message, code="input.parameter", details={"param": param, **details})


def _root(ax: Any) -> Figure:
    try:
        root: Figure = ax.get_figure(root=True)
        return root
    except TypeError:  # pragma: no cover - matplotlib < 3.10 has no root= keyword
        fig = ax.figure
        while not isinstance(fig, Figure):
            fig = fig.figure
        return fig


def _check_subspace(subspace: Any, name: str = "subspace") -> SubspaceResult:
    if not isinstance(subspace, SubspaceResult):
        raise _param(f"{name}: expected a SubspaceResult, got {type(subspace).__name__}", name)
    return subspace


def _figsize(figsize: Any) -> tuple[float, float] | None:
    if figsize is None:
        return None
    try:
        w, h = figsize
    except (TypeError, ValueError):
        raise _param("figsize: expected (width, height) in inches", "figsize") from None
    return (
        _v.real(w, "figsize", lo=0.0, lo_open=True),
        _v.real(h, "figsize", lo=0.0, lo_open=True),
    )


def _axes(ax: Any, figsize: Any) -> tuple[Figure, Axes]:
    """Return the root figure and the Axes to draw into.

    A new constrained-layout Figure when ``ax`` is None.
    """
    if ax is None:
        fig = Figure(figsize=_figsize(figsize), layout="constrained")
        return fig, fig.add_subplot()
    if not isinstance(ax, Axes):
        raise _param(f"ax: expected a matplotlib Axes or None, got {type(ax).__name__}", "ax")
    if figsize is not None:
        raise _param("give at most one of ax and figsize", "figsize", reason="ax_and_figsize")
    return _root(ax), ax


def _cycle_color(value: Any) -> Any:
    """Resolve a property-cycle color ``"Cn"`` now.

    The caller's style at call time is the one drawn [D-129].
    """
    if isinstance(value, str) and len(value) > 1 and value[0] == "C" and value[1:].isdigit():
        return matplotlib.colors.to_rgba(value)
    return value


_COLOR_KEYS = (
    "color",
    "c",
    "markerfacecolor",
    "mfc",
    "markeredgecolor",
    "mec",
    "facecolor",
    "fc",
    "edgecolor",
    "ec",
)


def _kw(kw: Any, defaults: Mapping[str, Any], name: str) -> dict[str, Any]:
    if kw is not None and not isinstance(kw, Mapping):
        raise _param(f"{name}: expected a mapping or None, got {type(kw).__name__}", name)
    merged = {**defaults, **(kw or {})}
    for key in _COLOR_KEYS:
        if key in merged:
            merged[key] = _cycle_color(merged[key])
    return merged


def _label(label: Any, default: str, name: str) -> str:
    if label is None:
        return default
    if not isinstance(label, str):
        raise _param(f"{name}: expected a string or None, got {type(label).__name__}", name)
    return label


def _band(replicates: np.ndarray | None, interval: str, level: float, K: int) -> np.ndarray | None:
    """``(2, K)`` band rows [lower, upper], or None for ``interval == "none"``."""
    if interval == "none":
        return None
    if replicates is None:
        raise InputError(
            f"interval={interval!r} needs bootstrap replicates; the result has none (pass nboot "
            "> 0)",
            code="input.no_bootstrap",
            details={"param": "interval"},
        )
    kind = cast(Literal["percentile", "minmax"], interval)
    return bootstrap_interval(replicates[:, :K], kind=kind, level=level)


def _log_limits(ax: Axes, arrays: Sequence[np.ndarray], top: float | None = None) -> None:
    """y-limits of a log axis from the positive data, padded by one decade."""
    pos = np.concatenate([np.ravel(a) for a in arrays]) if arrays else np.zeros(0)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    if pos.size == 0:
        if top is not None:
            ax.set_ylim(top=top)
        return
    lo, hi = float(np.min(pos)) / 10.0, float(np.max(pos)) * 10.0
    ax.set_ylim(lo, hi if top is None else top)


def _index_axis(ax: Axes, K: int, lo: float = 0.5) -> None:
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlim(lo, K + 0.5)


def _scatter_defaults(M: int) -> dict[str, Any]:
    d: dict[str, Any] = {"s": matplotlib.rcParams["lines.markersize"] ** 2, "alpha": 0.8}
    if M > _RASTER_M:
        d["rasterized"] = True
    return d


def _hull(vertices: np.ndarray) -> np.ndarray:
    """Vertices of the convex hull of 2-D points, counter-clockwise."""
    V = np.asarray(vertices, dtype=np.float64)
    hull: np.ndarray = V[scipy.spatial.ConvexHull(V).vertices]
    return hull


def _line_only(kw: Mapping[str, Any]) -> dict[str, Any]:
    """``domain_kw`` without patch-only keys, for the interval endpoints drawn with ``axvline``."""
    return {
        k: v
        for k, v in kw.items()
        if k not in ("fill", "facecolor", "fc", "edgecolor", "ec", "hatch")
    }


# ---------------------------------------------------------------------------
# Plots of a SubspaceResult
# ---------------------------------------------------------------------------


def plot_eigenvalues(
    subspace: SubspaceResult,
    *,
    interval: Interval | None = None,
    level: float = 0.95,
    k_max: int | None = None,
    mark_dimension: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    line_kw: Mapping[str, Any] | None = None,
    band_kw: Mapping[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot eigenvalue magnitudes on a log axis, with a bootstrap band.

    Draws :math:`\lvert\lambda_k\rvert` against :math:`k = 1, \dots, K`. For an
    estimator with signed eigenvalues, a negative :math:`\lambda_k` is drawn
    with an open marker and a legend entry. A zero magnitude cannot be drawn
    on the log axis and is masked.

    Parameters
    ----------
    subspace : SubspaceResult
        The subspace; not a ``"basis"`` result (its eigenvalues are NaN).
    interval : {"percentile", "minmax", "none"}, optional
        Band kind; default ``"percentile"`` when the result has bootstrap
        replicates, else ``"none"``. ``"minmax"`` reproduces the 2016 band.
    level : float, default 0.95
        Level of the percentile band, ``0 < level < 1``.
    k_max : int, optional
        Plot :math:`\lambda_1, \dots, \lambda_K` with ``K = min(k_max, m)``;
        default ``m``.
    mark_dimension : bool, default True
        Draw a dotted vertical line at ``n + 1/2``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; default: a new constrained-layout figure.
    figsize : (float, float), optional
        Width and height in inches of the new figure; not allowed with `ax`.
    line_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the line, merged
        over the defaults ``{"color": "C0", "marker": "o", "linestyle": "-"}``.
    band_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the band (``fill_between``), merged
        over the defaults ``{"color": "C0", "alpha": 0.25, "linewidth": 0}``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The root figure (new, or the figure of `ax`).
    ax : matplotlib.axes.Axes
        The Axes drawn into.

    Raises
    ------
    InputError
        ``input.parameter`` for an invalid argument (including a bad `ax`,
        `figsize` or ``*_kw``, or both `ax` and `figsize`), or for a ``"basis"`` result;
        ``input.no_bootstrap`` if `interval` asks for a band and the result
        has no replicates.

    See Also
    --------
    active_subspaces.bootstrap_interval : The band.

    Notes
    -----
    The percentile band at level :math:`\alpha` is
    :math:`[q_{(1-\alpha)/2}, q_{(1+\alpha)/2}]` of the replicates
    :math:`\lvert\lambda_k^{(b)}\rvert`, :math:`b = 1, \dots, B`; the
    min/max band is :math:`[\min_b \lvert\lambda_k^{(b)}\rvert,
    \max_b \lvert\lambda_k^{(b)}\rvert]`, which widens as :math:`B` grows.
    The eigenvalue bootstrap is inconsistent at repeated or zero
    eigenvalues [1]_, [2]_, so a band there is descriptive only.

    References
    ----------
    .. [1] R. Beran, M. S. Srivastava. Bootstrap tests and confidence regions
       for functions of a covariance matrix. *Ann. Statist.* 13(1):95–115,
       1985 (details to verify).
    .. [2] M. L. Eaton, D. E. Tyler. On Wielandt's inequality and its
       application to the asymptotic distribution of the eigenvalues of a
       random symmetric matrix. *Ann. Statist.* 19(1):260–271, 1991 (details
       to verify).

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=50, rng=0)
    >>> fig, ax = ac.plotting.plot_eigenvalues(sub)
    >>> print(ax.get_yscale(), ax.get_xlabel())
    log index $k$
    """
    sub = _check_subspace(subspace)
    if sub.estimator == "basis":
        raise _param(
            "a 'basis' result has no eigenvalues to plot", "subspace", reason="no_eigenvalues"
        )
    if interval is not None:
        interval = cast(Interval, _v.choice(interval, "interval", _INTERVALS))
    level = _v.real(level, "level", lo=0.0, hi=1.0, lo_open=True, hi_open=True)
    m = sub.m
    K = m if k_max is None else min(_v.integer(k_max, "k_max", lo=1), m)
    mark_dimension = _v.boolean(mark_dimension, "mark_dimension")
    lkw = _kw(line_kw, _LINE, "line_kw")
    bkw = _kw(band_kw, _BAND, "band_kw")
    if interval is None:
        interval = "percentile" if sub.bootstrap_eigenvalues is not None else "none"
    reps = None if sub.bootstrap_eigenvalues is None else np.abs(sub.bootstrap_eigenvalues)
    band = _band(reps, interval, level, K)
    fig, ax = _axes(ax, figsize)

    lam = np.asarray(sub.eigenvalues[:K], dtype=np.float64)
    mag = np.abs(lam)
    ks = np.arange(1, K + 1)
    neg = lam < 0
    ax.set_yscale("log", nonpositive="mask")
    if band is not None:
        ax.fill_between(ks, band[0], band[1], **bkw)
    (line,) = ax.plot(ks, mag, **{"markevery": [int(i) for i in np.flatnonzero(~neg)], **lkw})
    if np.any(neg):
        (open_markers,) = ax.plot(
            ks[neg],
            mag[neg],
            linestyle="none",
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            markeredgecolor=line.get_color(),
            markerfacecolor="none",
            label="negative eigenvalue",
        )
        ax.legend(handles=[open_markers])
    if mark_dimension:
        ax.axvline(sub.n + 0.5, **_MARKER)
    _index_axis(ax, K)
    _log_limits(ax, [mag] if band is None else [mag, band])
    ax.set_xlabel("index $k$")
    ax.set_ylabel(
        "eigenvalue $\\lambda_k$" if not np.any(neg) else "eigenvalue magnitude $|\\lambda_k|$"
    )
    return fig, ax


def plot_subspace_errors(
    subspace: SubspaceResult,
    *,
    interval: Interval | None = None,
    level: float = 0.95,
    k_max: int | None = None,
    mark_dimension: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    line_kw: Mapping[str, Any] | None = None,
    band_kw: Mapping[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot the bootstrap mean subspace distance on a log axis, with a band.

    Draws :math:`\bar\delta_k` against :math:`k = 1, \dots, K`; the upper
    limit of the axis is 1.

    Parameters
    ----------
    subspace : SubspaceResult
        The subspace; it must hold bootstrap replicates.
    interval : {"percentile", "minmax", "none"}, optional
        Band kind; default ``"percentile"``.
    level : float, default 0.95
        Level of the percentile band, ``0 < level < 1``.
    k_max : int, optional
        Plot ``k = 1, ..., K`` with ``K = min(k_max, m - 1)``; default
        ``m - 1``.
    mark_dimension : bool, default True
        Draw a dotted vertical line at ``n``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; default: a new constrained-layout figure.
    figsize : (float, float), optional
        Width and height in inches of the new figure; not allowed with `ax`.
    line_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the line, merged
        over the defaults ``{"color": "C0", "marker": "o", "linestyle": "-"}``.
    band_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the band (``fill_between``), merged
        over the defaults ``{"color": "C0", "alpha": 0.25, "linewidth": 0}``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The root figure (new, or the figure of `ax`).
    ax : matplotlib.axes.Axes
        The Axes drawn into.

    Raises
    ------
    InputError
        ``input.parameter`` for an invalid argument (including a bad `ax`,
        `figsize` or ``*_kw``, or both `ax` and `figsize`);
        ``input.no_bootstrap`` if the result has no bootstrap replicates.

    See Also
    --------
    active_subspaces.mean_distance : The plotted mean.

    Notes
    -----
    For each :math:`k` the replicates are
    :math:`\delta_k^{(b)} = \lVert W_{1,k}^\top W_{2,k}^{(b)} \rVert_2`, the
    sine of the largest principal angle between the estimated
    :math:`k`-dimensional subspace and its :math:`b`-th bootstrap replicate,
    and the plot shows

    .. math::

        \bar\delta_k = \frac{1}{B} \sum_{b=1}^{B} \delta_k^{(b)},
        \qquad 0 \le \bar\delta_k \le 1,

    with a percentile or min/max band of the :math:`\delta_k^{(b)}` [1]_.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=50, rng=0)
    >>> fig, ax = ac.plotting.plot_subspace_errors(sub)
    >>> print(ax.get_yscale(), ax.get_ylim()[1])
    log 1.0
    """
    sub = _check_subspace(subspace)
    interval = cast(
        Interval, "percentile" if interval is None else _v.choice(interval, "interval", _INTERVALS)
    )
    level = _v.real(level, "level", lo=0.0, hi=1.0, lo_open=True, hi_open=True)
    kmax = None if k_max is None else _v.integer(k_max, "k_max", lo=1)
    mark_dimension = _v.boolean(mark_dimension, "mark_dimension")
    lkw = _kw(line_kw, _LINE, "line_kw")
    bkw = _kw(band_kw, _BAND, "band_kw")
    delta = mean_distance(sub)  # raises input.no_bootstrap
    K = sub.m - 1 if kmax is None else min(kmax, sub.m - 1)
    band = _band(sub.bootstrap_distances, interval, level, K) if K > 0 else None
    fig, ax = _axes(ax, figsize)

    ks = np.arange(1, K + 1)
    ax.set_yscale("log", nonpositive="mask")
    if band is not None:
        ax.fill_between(ks, band[0], band[1], **bkw)
    ax.plot(ks, delta[:K], **lkw)
    if mark_dimension:
        ax.axvline(sub.n, **_MARKER)
    _index_axis(ax, max(K, 1))
    _log_limits(ax, [delta[:K]] if band is None else [delta[:K], band], top=1.0)
    ax.set_xlabel("subspace dimension $k$")
    ax.set_ylabel("subspace distance")
    return fig, ax


def plot_eigenvectors(
    subspace: SubspaceResult,
    *,
    indices: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    interval: Interval | None = None,
    level: float = 0.95,
    ax: Axes | Sequence[Axes] | None = None,
    figsize: tuple[float, float] | None = None,
    line_kw: Mapping[str, Any] | None = None,
    band_kw: Mapping[str, Any] | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    r"""Plot eigenvector components in normalized coordinates, one panel per eigenvector.

    Panel :math:`j` shows the components :math:`w_{ij}`,
    :math:`i = 1, \dots, m`, with a zero line and the vertical range
    :math:`[-1.05, 1.05]` (since :math:`\lVert w_j \rVert_2 = 1`). The
    normalized components are the scale-free measure of input importance;
    physical directions carry units and are not plotted. The eigenvectors
    are drawn with the signs the result carries; nothing is re-signed. With
    ``interval``, each panel also shows a band of the sign-aligned bootstrap
    replicates of its eigenvector (``bootstrap_eigenvectors``) [D-196].

    Parameters
    ----------
    subspace : SubspaceResult
        The subspace.
    indices : sequence of int, optional
        0-based columns of ``subspace.eigenvectors``, unique; default
        ``range(min(n, 4))``. The panel labels are 1-based (:math:`w_1`, ...).
    labels : sequence of str, optional
        ``m`` tick labels for the inputs; default ``space.names`` when
        ``m <= 40``, else integer ticks.
    interval : {"percentile", "minmax", "none"}, optional
        A band per component from ``subspace.bootstrap_eigenvectors[:, :, j]``
        through :func:`~active_subspaces.bootstrap_interval`; default (or
        ``"none"``) no band. Every index must have stored replicates
        (``active_subspace(..., bootstrap_vectors=k)``, index ``< k``).
    level : float, default 0.95
        Level of the ``"percentile"`` band, in :math:`(0, 1)`.
    ax : matplotlib.axes.Axes or sequence of Axes, optional
        One Axes per index; default: a new figure with one panel per index,
        stacked vertically.
    figsize : (float, float), optional
        Width and height in inches of the new figure; not allowed with `ax`.
    line_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the line, merged
        over the defaults ``{"color": "C0", "marker": "o", "linestyle": "-"}``.
    band_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the band (``fill_between``), merged
        over the defaults ``{"color": "C0", "alpha": 0.25, "linewidth": 0}``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The root figure.
    axes : tuple of matplotlib.axes.Axes
        One per index, top to bottom.

    Raises
    ------
    InputError
        ``input.parameter`` for an invalid argument (including a bad `ax`,
        `figsize` or ``*_kw``, or both `ax` and `figsize`),
        including repeated or out-of-range `indices`, or `labels` of the
        wrong length; and for a band on an index without eigenvector
        replicates (``details["reason"] == "no_eigenvector_replicates"``).

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=50, rng=0)
    >>> fig, axes = ac.plotting.plot_eigenvectors(sub, indices=[0, 1])
    >>> print(len(axes), axes[0].get_ylabel())
    2 $w_{1}$

    With sign-aligned bootstrap replicates of the first eigenvector:

    >>> sub1 = ac.active_subspace(grads=G, space=space, nboot=50, bootstrap_vectors=1, rng=0)
    >>> fig, axes = ac.plotting.plot_eigenvectors(sub1, indices=[0], interval="percentile")
    >>> len(axes[0].collections)
    1
    """
    sub = _check_subspace(subspace)
    m = sub.m
    if indices is None:
        idx = list(range(min(sub.n, 4)))
    else:
        if isinstance(indices, (str, bytes)) or not isinstance(indices, (Sequence, np.ndarray)):
            raise _param("indices: expected a sequence of integers", "indices")
        idx = [_v.integer(j, "indices", lo=0, hi=m - 1) for j in indices]
        if not idx or len(set(idx)) != len(idx):
            raise _param("indices: expected at least one index, without repeats", "indices")
    if labels is not None:
        if (
            isinstance(labels, (str, bytes))
            or not isinstance(labels, (Sequence, np.ndarray))
            or len(labels) != m
        ):
            raise _param(f"labels: expected a sequence of {m} strings", "labels", expected=m)
        tick_labels: list[str] | None = [str(s) for s in labels]
    else:
        tick_labels = list(sub.space.names) if m <= _MAX_NAMED_TICKS else None
    lkw = _kw(line_kw, _LINE, "line_kw")
    bkw = _kw(band_kw, _BAND, "band_kw")
    if interval is not None:
        interval = cast(Interval, _v.choice(interval, "interval", _INTERVALS))
    level = _v.real(level, "level", lo=0.0, hi=1.0, lo_open=True, hi_open=True)
    bands: dict[int, np.ndarray] = {}
    if interval not in (None, "none"):
        V = sub.bootstrap_eigenvectors
        stored = 0 if V is None else V.shape[2]
        missing = [j for j in idx if j >= stored]
        if missing:
            raise _param(
                f"interval: no bootstrap eigenvector replicates for indices {missing}; pass "
                f"bootstrap_vectors > {max(missing)} to active_subspace",
                "interval",
                reason="no_eigenvector_replicates",
                indices=missing,
            )
        assert V is not None  # stored > 0
        kind = cast(Literal["percentile", "minmax"], interval)
        bands = {j: bootstrap_interval(V[:, :, j], kind=kind, level=level) for j in idx}
    k = len(idx)
    if ax is None:
        size = _figsize(figsize)
        if size is None:
            size = (float(matplotlib.rcParams["figure.figsize"][0]), 1.2 + 1.8 * k)
        fig = Figure(figsize=size, layout="constrained")
        axes = tuple(fig.subplots(k, 1, sharex=True, squeeze=False)[:, 0])
    else:
        seq = (ax,) if isinstance(ax, Axes) else ax
        if isinstance(seq, (str, bytes)) or not isinstance(seq, (Sequence, np.ndarray)):
            raise _param(
                f"ax: expected an Axes or a sequence of Axes, got {type(ax).__name__}", "ax"
            )
        axes = tuple(seq)
        if len(axes) != k or not all(isinstance(a, Axes) for a in axes):
            raise _param(f"ax: expected {k} Axes, one per index", "ax", expected=k)
        if figsize is not None:
            raise _param("give at most one of ax and figsize", "figsize", reason="ax_and_figsize")
        fig = _root(axes[0])

    W = sub.eigenvectors
    ii = np.arange(1, m + 1)
    for a, j in zip(axes, idx, strict=False):
        a.axhline(0.0, color="0.5", linewidth=0.8)
        if j in bands:
            a.fill_between(ii, bands[j][0], bands[j][1], **bkw)
        a.plot(ii, W[:, j], **lkw)
        a.set_ylim(-1.05, 1.05)
        a.set_xlim(0.5, m + 0.5)
        a.set_ylabel(f"$w_{{{j + 1}}}$")
        if tick_labels is not None:
            a.set_xticks(ii, tick_labels)
        else:
            a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for a in axes[:-1]:
        a.tick_params(axis="x", labelbottom=False)  # a boolean [P-09]
    bottom = axes[-1]
    bottom.tick_params(axis="x", labelbottom=True)
    if tick_labels is not None:
        bottom.tick_params(axis="x", labelrotation=90)
    else:
        bottom.set_xlabel("input $i$")
    return fig, axes


def plot_sufficient_summary(
    subspace: SubspaceResult,
    *,
    X: ArrayLike,
    f: ArrayLike,
    dims: int | None = None,
    domain: Any = None,
    qoi_label: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    scatter_kw: Mapping[str, Any] | None = None,
    domain_kw: Mapping[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    r"""Draw a sufficient summary plot in one or two active variables.

    With one active variable, a scatter of :math:`f_i` against
    :math:`y_{i1}`; with two, a scatter of :math:`(y_{i1}, y_{i2})` coloured
    by :math:`f_i`, with a colorbar. An active domain of matching dimension
    can be overlaid: the interval endpoints, or the zonotope polygon (an
    unbounded domain draws nothing).

    Parameters
    ----------
    subspace : SubspaceResult
        The subspace; its eigenvectors define the active variables.
    X : array_like, shape (M, m)
        Inputs in **physical** coordinates, normalized with
        ``subspace.space``.
    f : array_like, shape (M,) or (M, 1)
        Values of the quantity of interest at `X`.
    dims : {1, 2}, optional
        Number of active variables to plot; default ``min(n, 2)``. It may
        exceed ``n`` (to look for structure along :math:`w_2`), not ``m``.
    domain : ActiveDomain, optional
        Boundary overlay; ``domain.subspace`` must equal `subspace` and
        ``domain.n`` must equal `dims`.
    qoi_label : str, optional
        Label of the values; default ``subspace.qoi``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; default: a new constrained-layout figure.
    figsize : (float, float), optional
        Width and height in inches of the new figure; not allowed with `ax`.
    scatter_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the scatter, merged
        over the defaults ``{"s": rcParams["lines.markersize"] ** 2, "alpha": 0.8}``.
    domain_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the domain boundary, merged
        over the defaults ``{"color": "0.2", "linewidth": 1.5, "fill": False}``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The root figure (new, or the figure of `ax`).
    ax : matplotlib.axes.Axes
        The Axes drawn into.

    Raises
    ------
    InputError
        ``input.parameter`` for an invalid argument (including a bad `ax`,
        `figsize` or ``*_kw``, or both `ax` and `figsize`),
        including a `domain` of another subspace or dimension;
        ``input.shape``, ``input.dtype``, ``input.nonfinite``,
        ``input.out_of_space`` or ``input.size_mismatch`` for invalid `X`
        or `f`.

    Notes
    -----
    With :math:`x = A^{-1}(\xi - b)` the normalized inputs, the active
    variables are

    .. math::

        y = W_{1:d}^\top x, \qquad W_{1:d} = [w_1\ \cdots\ w_d],
        \quad d \in \lbrace 1, 2 \rbrace .

    These are the sufficient summary plots of [1]_.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=50, rng=0)
    >>> fig, ax = ac.plotting.plot_sufficient_summary(sub, X=X, f=f)
    >>> print(ax.get_xlabel())
    $y_1 = w_1^\top x$
    >>> sub2 = ac.with_dimension(sub, 2)
    >>> fig, ax = ac.plotting.plot_sufficient_summary(
    ...     sub2, X=X, f=f, domain=ac.active_domain(sub2)
    ... )
    >>> print(ax.get_xlabel(), ax.get_ylabel(), len(fig.axes))
    $y_1$ $y_2$ 2
    """
    sub = _check_subspace(subspace)
    data = _v.data(sub.space, X=X, f=f, require=("X", "f"))
    x, fv = data.X, data.f
    assert x is not None  # required above
    assert fv is not None
    m = sub.m
    if dims is None:
        d = min(sub.n, 2)
    else:
        d = _v.integer(dims, "dims", lo=1, hi=2)
        if d > m:
            raise _param(
                f"dims: {d} exceeds the number of inputs m={m}",
                "dims",
                expected=[1, min(2, m)],
                got=d,
            )
    label = _label(qoi_label, sub.qoi, "qoi_label")
    if domain is not None:
        if getattr(domain, "subspace", None) != sub:
            raise _param(
                "domain: built from a different subspace result",
                "domain",
                reason="subspace_mismatch",
            )
        if domain.n != d:
            raise _param(
                f"domain: dimension {domain.n} does not match dims={d}",
                "domain",
                reason="dimension_mismatch",
            )
    skw = _kw(scatter_kw, _scatter_defaults(data.M), "scatter_kw")
    dkw = _kw(domain_kw, _DOMAIN, "domain_kw")
    fig, ax = _axes(ax, figsize)

    y = x @ sub.eigenvectors[:, :d]
    if d == 1:
        ax.scatter(y[:, 0], fv, **skw)
        ax.set_xlabel("$y_1 = w_1^\\top x$")
        ax.set_ylabel(label)
    else:
        sc = ax.scatter(y[:, 0], y[:, 1], c=fv, **skw)
        ax.figure.colorbar(sc, ax=ax, label=label)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("$y_1$")
        ax.set_ylabel("$y_2$")
    if domain is not None and domain.kind != "unbounded":
        if domain.kind == "interval":
            for v in np.asarray(domain.vertices, dtype=np.float64)[:, 0]:
                ax.axvline(float(v), **_line_only(dkw))
        else:
            ax.add_patch(matplotlib.patches.Polygon(_hull(domain.vertices), closed=True, **dkw))
    return fig, ax


# ---------------------------------------------------------------------------
# Active domain, surrogate fit, dimension criteria
# ---------------------------------------------------------------------------


def plot_active_domain(
    domain: Any,
    *,
    design: ArrayLike | None = None,
    triangulate: bool = False,
    points: ArrayLike | None = None,
    values: ArrayLike | None = None,
    qoi_label: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    domain_kw: Mapping[str, Any] | None = None,
    scatter_kw: Mapping[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot a two-dimensional zonotope active domain, with an optional design, mesh and points.

    Draws the polygon :math:`\mathcal Y = W_1^\top [-1, 1]^m` (the convex
    hull of its vertices); optionally a design with the Delaunay
    triangulation of the design together with the vertices, and points
    coloured by values.

    Parameters
    ----------
    domain : ActiveDomain
        A domain with ``n == 2`` and ``kind == "zonotope"``.
    design : array_like, shape (N, 2), optional
        Design points in **normalized active** coordinates.
    triangulate : bool, default False
        Draw the Delaunay mesh of the design and the vertices; needs `design`.
    points : array_like, shape (K, 2), optional
        Points in **normalized active** coordinates, coloured by `values`;
        give both or neither.
    values : array_like, shape (K,), optional
        Values at `points`.
    qoi_label : str, optional
        Colorbar label; default ``domain.subspace.qoi``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; default: a new constrained-layout figure.
    figsize : (float, float), optional
        Width and height in inches of the new figure; not allowed with `ax`.
    domain_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the domain boundary, merged
        over the defaults ``{"color": "0.2", "linewidth": 1.5, "fill": False}``.
    scatter_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the scatter, merged
        over the defaults ``{"s": rcParams["lines.markersize"] ** 2, "alpha": 0.8}``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The root figure (new, or the figure of `ax`).
    ax : matplotlib.axes.Axes
        The Axes drawn into.

    Raises
    ------
    UnsupportedError
        ``unsupported.space`` for an unbounded domain (Gaussian space).
    InputError
        ``input.parameter`` for an invalid argument (including a bad `ax`,
        `figsize` or ``*_kw``, or both `ax` and `figsize`),
        including a domain that is not a two-dimensional zonotope,
        `triangulate` without `design`, or only one of `points` and
        `values`; ``input.shape``, ``input.dtype``, ``input.nonfinite`` or
        ``input.size_mismatch`` for invalid `design`, `points` or `values`.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=50, rng=0)
    >>> dom = ac.active_domain(ac.with_dimension(sub, 2))
    >>> design = ac.active_design(domain=dom, n_points=6, rng=0)
    >>> fig, ax = ac.plotting.plot_active_domain(dom, design=design.points, triangulate=True)
    >>> print(len(ax.patches), ax.get_xlabel(), ax.get_ylabel())
    1 $y_1$ $y_2$
    """
    kind = getattr(domain, "kind", None)
    if kind == "unbounded":
        raise UnsupportedError(
            "plot_active_domain: a Gaussian space has an unbounded active domain",
            code="unsupported.space",
            details={"kind": kind},
        )
    if kind != "zonotope" or getattr(domain, "n", None) != 2:
        raise _param(
            "domain: expected an ActiveDomain with n == 2 and kind 'zonotope'",
            "domain",
            kind=str(kind),
        )
    triangulate = _v.boolean(triangulate, "triangulate")
    D = None if design is None else _v.matrix(design, "design", cols=2)
    if triangulate and D is None:
        raise _param("triangulate=True needs a design", "triangulate")
    if (points is None) != (values is None):
        raise _param(
            "give both points and values, or neither", "values" if values is None else "points"
        )
    P = vals = None
    if points is not None:
        P = _v.matrix(points, "points", cols=2)
        vals = _v.values(values, P.shape[0], name="values")
    label = _label(qoi_label, domain.subspace.qoi, "qoi_label")
    dkw = _kw(domain_kw, _DOMAIN, "domain_kw")
    skw = _kw(scatter_kw, _scatter_defaults(0 if P is None else P.shape[0]), "scatter_kw")
    fig, ax = _axes(ax, figsize)

    V = np.asarray(domain.vertices, dtype=np.float64)
    hull = _hull(V)
    if D is not None and triangulate:
        tri = scipy.spatial.Delaunay(np.vstack([D, V]))
        edges = {
            tuple(sorted((int(s[a]), int(s[b]))))
            for s in tri.simplices
            for a, b in ((0, 1), (1, 2), (0, 2))
        }
        pts = tri.points
        segs = [pts[[i, j]] for i, j in sorted(edges)]
        ax.add_collection(
            matplotlib.collections.LineCollection(segs, colors="0.8", linewidths=0.8, zorder=1)
        )
    ax.add_patch(matplotlib.patches.Polygon(hull, closed=True, **dkw))
    if D is not None:
        ax.plot(
            D[:, 0],
            D[:, 1],
            linestyle="none",
            marker="o",
            color=_cycle_color("C0"),
            label="design",
        )
    if P is not None:
        sc = ax.scatter(P[:, 0], P[:, 1], c=vals, **skw)
        ax.figure.colorbar(sc, ax=ax, label=label)
    lo, hi = hull.min(axis=0), hull.max(axis=0)
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
    return fig, ax


def plot_surrogate_fit(
    surrogate: Surrogate,
    *,
    X: ArrayLike,
    f: ArrayLike,
    qoi_label: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    scatter_kw: Mapping[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot surrogate predictions against observed values, with the line ``g = f``.

    A scatter of :math:`g(W_1^\top x_i)` against :math:`f_i`, with a dashed
    diagonal and equal aspect.

    Parameters
    ----------
    surrogate : Surrogate
        Any object satisfying the
        :class:`~active_subspaces.surrogates.Surrogate` protocol, including a
        :class:`~active_subspaces.ReducedModel`.
    X : array_like, shape (M, m)
        Inputs in **physical** coordinates, validated against
        ``surrogate.space``.
    f : array_like, shape (M,) or (M, 1)
        Observed values at `X`.
    qoi_label : str, optional
        Label of the values; default ``surrogate.qoi``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; default: a new constrained-layout figure.
    figsize : (float, float), optional
        Width and height in inches of the new figure; not allowed with `ax`.
    scatter_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the scatter, merged
        over the defaults ``{"s": rcParams["lines.markersize"] ** 2, "alpha": 0.8}``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The root figure (new, or the figure of `ax`).
    ax : matplotlib.axes.Axes
        The Axes drawn into; x is the observed value, y the prediction.

    Raises
    ------
    InputError
        ``input.parameter`` for an invalid argument (including a bad `ax`,
        `figsize` or ``*_kw``, or both `ax` and `figsize`),
        including a `surrogate` that does not satisfy the protocol;
        ``input.shape``, ``input.dtype``, ``input.nonfinite``,
        ``input.out_of_space`` or ``input.size_mismatch`` for invalid `X`
        or `f`.

    See Also
    --------
    active_subspaces.predict : The predictions plotted.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=50, rng=0)
    >>> s = ac.fit_surrogate(
    ...     X=X, f=f, subspace=ac.with_dimension(sub, 1), kind="poly", degree=2
    ... )
    >>> fig, ax = ac.plotting.plot_surrogate_fit(s, X=X, f=f)
    >>> print(ax.get_xlabel(), "|", ax.get_ylabel())
    observed f | surrogate f
    """
    if not isinstance(surrogate, Surrogate):
        raise _param(
            "surrogate: expected an object satisfying the Surrogate protocol, got "
            f"{type(surrogate).__name__}",
            "surrogate",
        )
    data = _v.data(surrogate.space, X=X, f=f, require=("X", "f"))
    fv = data.f
    assert fv is not None  # required above
    g = predict(surrogate, X)
    label = _label(qoi_label, surrogate.qoi, "qoi_label")
    skw = _kw(scatter_kw, _scatter_defaults(data.M), "scatter_kw")
    fig, ax = _axes(ax, figsize)

    ax.scatter(fv, g, **skw)
    lo = float(min(np.min(fv), np.min(g)))
    hi = float(max(np.max(fv), np.max(g)))
    ax.plot([lo, hi], [lo, hi], color="0.5", linestyle="--", linewidth=1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"observed {label}")
    ax.set_ylabel(f"surrogate {label}")
    return fig, ax


_CRITERION_LABELS = {
    ("evg", "linear"): "eigenvalue gap $|\\lambda_k| - |\\lambda_{k+1}|$",
    ("evg", "log"): "log eigenvalue gap $\\log|\\lambda_k| - \\log|\\lambda_{k+1}|$",
    ("rs", None): "RS error bound $\\varepsilon_k$",
    ("li", None): "ladle criterion $g(k)$",
}


def plot_dimension_criterion(
    subspace: SubspaceResult,
    *,
    rule: Any,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    line_kw: Mapping[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot a dimension rule's criterion against the candidate dimension, marking the choice.

    The criterion is computed by
    :func:`~active_subspaces.dimension_criterion`; the selected dimension
    :math:`\hat k` is marked with a dotted line and an open marker. The
    linear eigenvalue gap, and an RS bound with all values positive, are
    drawn on a log axis.

    Parameters
    ----------
    subspace : SubspaceResult
        The subspace; not a ``"basis"`` result.
    rule : str or DimensionRule
        ``"evg"``, ``"rs"``, ``"li"`` or an instance such as
        ``EVG(gap="log")``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; default: a new constrained-layout figure.
    figsize : (float, float), optional
        Width and height in inches of the new figure; not allowed with `ax`.
    line_kw : mapping, optional
        Keyword arguments for the matplotlib call that draws the line, merged
        over the defaults ``{"color": "C0", "marker": "o", "linestyle": "-"}``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The root figure (new, or the figure of `ax`).
    ax : matplotlib.axes.Axes
        The Axes drawn into.

    Raises
    ------
    InputError
        ``input.parameter`` for an invalid argument (including a bad `ax`,
        `figsize` or ``*_kw``, or both `ax` and `figsize`),
        or for a ``"basis"`` result; ``input.no_bootstrap`` for ``"rs"`` and
        ``"li"`` without bootstrap replicates.

    See Also
    --------
    active_subspaces.dimension_criterion : The plotted criterion.

    Notes
    -----
    With :math:`\mu_k = \lvert\lambda_k\rvert`, the criteria are the
    eigenvalue gap :math:`\mu_k - \mu_{k+1}` (``"evg"``, maximized; or the
    gap of the logarithms for ``EVG(gap="log")``), the RS error bound

    .. math::

        \varepsilon_k = \Bigl(\sum_{j \le k} \mu_j\Bigr)^{1/2} \bar\delta_k
        + \Bigl(\sum_{j > k} \mu_j\Bigr)^{1/2}

    (``"rs"``, minimized) [1]_, and the ladle criterion :math:`g(k)` of Luo
    and Li (``"li"``, minimized; :math:`\hat k` may be 0) [2]_.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.
    .. [2] W. Luo, B. Li. Combining eigenvalues and variation of eigenvectors
       for order determination. *Biometrika* 103(4):875–887, 2016 (details
       to verify).

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=50, rng=0)
    >>> fig, ax = ac.plotting.plot_dimension_criterion(sub, rule="rs")
    >>> print(ax.get_xlabel(), "|", ax.get_ylabel())
    dimension $k$ | RS error bound $\varepsilon_k$
    """
    sub = _check_subspace(subspace)
    lkw = _kw(line_kw, _LINE, "line_kw")
    if ax is not None and not isinstance(ax, Axes):
        raise _param(f"ax: expected a matplotlib Axes or None, got {type(ax).__name__}", "ax")
    ks, values, k_hat = dimension_criterion(sub, rule=rule)
    name = rule.lower() if isinstance(rule, str) else rule.name
    gap = getattr(rule, "gap", "linear") if name == "evg" else None
    fig, ax = _axes(ax, figsize)

    (line,) = ax.plot(ks, values, **lkw)
    log = (name == "evg" and gap == "linear") or (
        name == "rs" and values.size > 0 and bool(np.all(values > 0))
    )
    if log:
        ax.set_yscale("log", nonpositive="mask")
    hit = np.flatnonzero(ks == k_hat)
    if hit.size:
        ax.axvline(k_hat, **_MARKER)
        ax.plot(
            [k_hat],
            values[hit[:1]],
            linestyle="none",
            marker=line.get_marker() if line.get_marker() not in (None, "None", "", " ") else "o",
            markersize=2.0 * float(matplotlib.rcParams["lines.markersize"]),
            markeredgecolor=line.get_color(),
            markerfacecolor="none",
            label=f"selected $k = {k_hat}$",
        )
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("dimension $k$")
    ax.set_ylabel(_CRITERION_LABELS[(name, gap)])
    return fig, ax


# ---------------------------------------------------------------------------
# Rendering and the catalogue
# ---------------------------------------------------------------------------


def figure_to_bytes(
    fig: Figure | Axes, *, format: Literal["png", "svg"] = "png", dpi: float = 150.0
) -> bytes:
    r"""Render a figure to PNG or SVG bytes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, SubFigure or Axes
        The figure; an Axes or SubFigure renders its root figure.
    format : {"png", "svg"}, default "png"
        Output format.
    dpi : float, default 150.0
        Resolution in dots per inch, ``> 0`` (PNG only).

    Returns
    -------
    bytes
        The rendered image, with a tight bounding box.

    Raises
    ------
    InputError
        ``input.parameter`` if `fig` is not a Figure, SubFigure or Axes, or
        for a bad `format` or `dpi`.

    Notes
    -----
    PNG bytes are reproducible for the same figure, ``rcParams`` and library
    versions on one platform. SVG output carries no date, but its element
    ids differ between calls unless the caller sets
    ``rcParams["svg.hashsalt"]``.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> sub = ac.active_subspace(grads=G, space=space, nboot=50, rng=0)
    >>> fig, ax = ac.plotting.plot_eigenvalues(sub)
    >>> ac.plotting.figure_to_bytes(fig)[:4]
    b'\x89PNG'
    >>> ac.plotting.figure_to_bytes(ax, format="svg")[:5]
    b'<?xml'
    """
    if isinstance(fig, (Axes, matplotlib.figure.SubFigure)):
        root = _root(fig)
    elif isinstance(fig, Figure):
        root = fig
    else:
        raise _param(f"fig: expected a Figure, SubFigure or Axes, got {type(fig).__name__}", "fig")
    fmt = _v.choice(format, "format", ("png", "svg"))
    dpi = _v.real(dpi, "dpi", lo=0.0, lo_open=True)
    meta = {"Date": None} if fmt == "svg" else None
    buf = io.BytesIO()
    root.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", metadata=meta)
    return buf.getvalue()


PLOTS: Mapping[str, Callable[..., tuple[Figure, Any]]] = MappingProxyType(
    {
        "eigenvalues": plot_eigenvalues,
        "subspace_errors": plot_subspace_errors,
        "eigenvectors": plot_eigenvectors,
        "sufficient_summary": plot_sufficient_summary,
        "active_domain": plot_active_domain,
        "surrogate_fit": plot_surrogate_fit,
        "dimension_criterion": plot_dimension_criterion,
    }
)
"""Read-only mapping from plot name to plot function: the closed catalogue of named plots.

The names are ``"eigenvalues"``, ``"subspace_errors"``, ``"eigenvectors"``,
``"sufficient_summary"``, ``"active_domain"``, ``"surrogate_fit"`` and
``"dimension_criterion"``; each value is the ``plot_*`` function of the same
name.

Examples
--------
>>> import active_subspaces as ac
>>> sorted(ac.plotting.PLOTS)[:3]
['active_domain', 'dimension_criterion', 'eigenvalues']
>>> ac.plotting.PLOTS["eigenvalues"] is ac.plotting.plot_eigenvalues
True
"""
