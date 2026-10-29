"""Unit tests of spec 13 (plotting): drawn data, labels, errors and side effects; no image
comparisons.
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import scipy.spatial
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

import active_subspaces as asub
from active_subspaces import plotting as P
from active_subspaces import subspaces as S
from active_subspaces.errors import InputError, UnsupportedError
from active_subspaces.geometry import active_domain
from active_subspaces.surrogates import fit_surrogate, predict

SRC = Path(__file__).resolve().parents[2] / "src"
M_IN = 5
NAMES = ("alpha", "beta", "gamma", "delta", "eps")


# ---------------------------------------------------------------------------
# Side effects (autouse, spec 13 §8)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = matplotlib.rcParams.copy()
    backend = dict.__getitem__(matplotlib.rcParams, "backend")
    had_pyplot = "matplotlib.pyplot" in sys.modules
    yield
    assert dict.__eq__(rc, matplotlib.rcParams), "rcParams changed"
    assert dict.__getitem__(matplotlib.rcParams, "backend") == backend
    assert had_pyplot or "matplotlib.pyplot" not in sys.modules, "matplotlib.pyplot was imported"
    assert os.listdir(tmp_path) == [], "a file was written"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

A = np.array([1.0, 0.5, 0.2, 0.1, 0.05])
B = np.array([0.0, 1.0, 0.0, 0.3, 0.0])


def _f(x):
    return np.exp(x @ A) + (x @ B) ** 2 + 0.05 * np.sum(x**2, axis=1)


def _g(x):
    return np.exp(x @ A)[:, None] * A + 2.0 * (x @ B)[:, None] * B + 0.1 * x


@pytest.fixture(scope="module")
def space():
    lo = np.array([-1.0, 0.0, 10.0, -50.0, 0.5])
    return asub.Uniform(lo, lo + np.array([2.0, 0.01, 30.0, 400.0, 1.0]), names=NAMES)


@pytest.fixture(scope="module")
def data(space):
    X = space.sample(150, rng=np.random.default_rng(3))
    x = space.to_normalized(X)
    return X, _f(x), space.grads_to_physical(_g(x))


@pytest.fixture(scope="module")
def res(space, data):
    X, f, G = data
    return asub.active_subspace(space=space, X=X, f=f, grads=G, n=2, nboot=30, rng=1, qoi="q")


@pytest.fixture(scope="module")
def res1(space, data):
    X, f, G = data
    return asub.active_subspace(space=space, X=X, f=f, grads=G, n=1, nboot=30, rng=1, qoi="q")


@pytest.fixture(scope="module")
def res_noboot(space, data):
    X, f, G = data
    return asub.active_subspace(space=space, X=X, f=f, grads=G, n=2)


@pytest.fixture(scope="module")
def basis(space):
    return asub.subspace_from_basis(np.eye(M_IN)[:, :2], space=space)


@pytest.fixture(scope="module")
def dom(res):
    return active_domain(res)


@pytest.fixture
def signed(space, make_provenance):
    """A hand-built indefinite result: one negative and one zero eigenvalue."""
    Bn = 6
    gen = np.random.default_rng(0)
    Q, _ = np.linalg.qr(gen.standard_normal((M_IN, M_IN)))
    lam = np.array([3.0, -1.0, 0.5, 0.0, 0.0])
    return S.SubspaceResult(
        space=space,
        estimator="asx",
        estimator_options=(),
        eigenvalues=lam,
        eigenvectors=Q,
        n=2,
        dimension_rule="fixed",
        dimension_options=(),
        criterion=None,
        n_samples=10,
        nboot=Bn,
        bootstrap_eigenvalues=lam + 0.1 * gen.standard_normal((Bn, M_IN)),
        bootstrap_distances=gen.uniform(0.01, 0.5, (Bn, M_IN - 1)),
        bootstrap_dets=gen.uniform(0.5, 1.0, (Bn, M_IN - 1)),
        provenance=make_provenance(),
    )


def _code(exc_info) -> str:
    return exc_info.value.code


def _band_ok(poly, ks, lower, upper, rtol=1e-15):
    V = poly.get_paths()[0].vertices
    for k, lo, hi in zip(ks, lower, upper, strict=False):
        ys = V[V[:, 0] == k, 1]
        np.testing.assert_allclose([ys.min(), ys.max()], [lo, hi], rtol=rtol, atol=0)


def _data_lines(ax):
    """Lines drawn from data (exclude axhline/axvline, whose x or y data are lists)."""
    return [
        ln
        for ln in ax.get_lines()
        if isinstance(ln.get_xdata(), np.ndarray) and len(ln.get_xdata()) > 0
    ]


def _vlines(ax):
    return [ln for ln in ax.get_lines() if list(ln.get_ydata()) == [0, 1]]


# ---------------------------------------------------------------------------
# Return contract (V13.2)
# ---------------------------------------------------------------------------


def _calls(res, dom, data, surrogate):
    X, f, _ = data
    return {
        "eigenvalues": ((res,), {}),
        "subspace_errors": ((res,), {}),
        "eigenvectors": ((res,), {"indices": [0]}),
        "sufficient_summary": ((res,), {"X": X, "f": f}),
        "active_domain": ((dom,), {}),
        "surrogate_fit": ((surrogate,), {"X": X, "f": f}),
        "dimension_criterion": ((res,), {"rule": "evg"}),
    }


@pytest.fixture(scope="module")
def surrogate(res, data):
    X, f, _ = data
    return fit_surrogate(X=X, f=f, subspace=res, kind="poly", degree=2)


@pytest.mark.parametrize("name", sorted(P.PLOTS))
def test_return_contract(name, res, dom, data, surrogate):
    args, kw = _calls(res, dom, data, surrogate)[name]
    fn = P.PLOTS[name]
    fig, ax = fn(*args, **kw)
    assert type(fig) is Figure
    first = ax[0] if name == "eigenvectors" else ax
    assert isinstance(first, Axes)
    assert first in fig.axes
    if name == "eigenvectors":
        assert isinstance(ax, tuple)
        assert len(ax) == 1

    caller = Figure()
    cax = caller.add_subplot()
    fig2, ax2 = fn(*args, ax=cax, **kw)
    assert fig2 is caller
    assert (ax2[0] if name == "eigenvectors" else ax2) is cax

    sub = Figure().subfigures(1, 2)[1]
    sax = sub.add_subplot()
    fig3, _ = fn(*args, ax=sax, **kw)
    assert fig3 is sub.get_figure(root=True)

    fig4, _ = fn(*args, figsize=(3.0, 2.0), **kw)
    assert tuple(fig4.get_size_inches()) == (3.0, 2.0)
    with pytest.raises(InputError) as e:
        fn(*args, ax=cax, figsize=(3.0, 2.0), **kw)
    assert _code(e) == "input.parameter"
    with pytest.raises(InputError) as e:
        fn(*args, ax="not an axes", **kw)
    assert _code(e) == "input.parameter"
    with pytest.raises(InputError) as e:
        fn(*args, figsize=(0.0, 2.0), **kw)
    assert _code(e) == "input.parameter"


def test_default_figsize(res):
    fig, _ = P.plot_eigenvalues(res)
    np.testing.assert_allclose(fig.get_size_inches(), matplotlib.rcParams["figure.figsize"])
    fig, _ = P.plot_eigenvectors(res, indices=[0, 1, 2])
    np.testing.assert_allclose(
        fig.get_size_inches(), [matplotlib.rcParams["figure.figsize"][0], 1.2 + 1.8 * 3]
    )
    with matplotlib.rc_context({"figure.figsize": (4.0, 3.0)}):
        fig, _ = P.plot_eigenvalues(res)
    np.testing.assert_allclose(fig.get_size_inches(), [4.0, 3.0])


def test_first_argument_checked(data):
    for fn in (P.plot_eigenvalues, P.plot_subspace_errors, P.plot_eigenvectors):
        with pytest.raises(InputError) as e:
            fn(np.eye(3))
        assert _code(e) == "input.parameter"
    with pytest.raises(InputError):
        P.plot_sufficient_summary(np.eye(5), X=data[0], f=data[1])
    with pytest.raises(InputError):
        P.plot_dimension_criterion(np.eye(5), rule="evg")


# ---------------------------------------------------------------------------
# Eigenvalues (V13.3)
# ---------------------------------------------------------------------------


def test_eigenvalues(res, res_noboot):
    _, ax = P.plot_eigenvalues(res)
    (line,) = _data_lines(ax)
    np.testing.assert_array_equal(line.get_xdata(), np.arange(1, M_IN + 1))
    np.testing.assert_array_equal(line.get_ydata(), np.abs(res.eigenvalues))
    assert ax.get_yscale() == "log"
    assert ax.get_xlim() == (0.5, M_IN + 0.5)
    (band,) = [c for c in ax.collections if isinstance(c, PolyCollection)]
    iv = asub.bootstrap_interval(np.abs(res.bootstrap_eigenvalues), kind="percentile", level=0.95)
    _band_ok(band, range(1, M_IN + 1), iv[0], iv[1])
    assert ax.get_xlabel() == "index $k$"
    assert ax.get_ylabel() == "eigenvalue $\\lambda_k$"
    assert ax.get_legend() is None
    (vl,) = _vlines(ax)
    assert vl.get_xdata()[0] == res.n + 0.5
    lo, hi = ax.get_ylim()
    pos = np.concatenate([np.abs(res.eigenvalues), iv.ravel()])
    pos = pos[pos > 0]
    np.testing.assert_allclose([lo, hi], [pos.min() / 10, pos.max() * 10])

    _, ax = P.plot_eigenvalues(res, interval="minmax", k_max=3, mark_dimension=False)
    (band,) = [c for c in ax.collections if isinstance(c, PolyCollection)]
    reps = np.abs(res.bootstrap_eigenvalues[:, :3])
    _band_ok(band, range(1, 4), reps.min(axis=0), reps.max(axis=0))
    assert len(_data_lines(ax)[0].get_ydata()) == 3
    assert not _vlines(ax)

    _, ax = P.plot_eigenvalues(res, level=0.5)
    iv = asub.bootstrap_interval(np.abs(res.bootstrap_eigenvalues), level=0.5)
    _band_ok(ax.collections[0], range(1, M_IN + 1), iv[0], iv[1])

    _, ax = P.plot_eigenvalues(res, interval="none")
    assert not ax.collections
    _, ax = P.plot_eigenvalues(res_noboot)
    assert not ax.collections
    _, ax = P.plot_eigenvalues(res, k_max=99)
    assert len(_data_lines(ax)[0].get_ydata()) == M_IN
    with pytest.raises(InputError) as e:
        P.plot_eigenvalues(res_noboot, interval="percentile")
    assert _code(e) == "input.no_bootstrap"
    with pytest.raises(InputError) as e:
        P.plot_eigenvalues(res_noboot, interval="minmax")
    assert _code(e) == "input.no_bootstrap"
    for bad in (
        {"interval": "bca"},
        {"level": 1.0},
        {"level": 0.0},
        {"k_max": 0},
        {"mark_dimension": 1},
        {"line_kw": 3},
    ):
        with pytest.raises(InputError) as e:
            P.plot_eigenvalues(res, **bad)
        assert _code(e) == "input.parameter", bad


def test_eigenvalues_signed(signed):
    _, ax = P.plot_eigenvalues(signed)
    main, opened = _data_lines(ax)
    np.testing.assert_array_equal(main.get_ydata(), np.abs(signed.eigenvalues))
    assert main.get_markevery() == [0, 2, 3, 4]
    np.testing.assert_array_equal(opened.get_xdata(), [2])
    np.testing.assert_array_equal(opened.get_ydata(), [1.0])
    assert opened.get_markerfacecolor() == "none"
    assert opened.get_linestyle() == "None"
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["negative eigenvalue"]
    assert ax.get_ylabel() == "eigenvalue magnitude $|\\lambda_k|$"
    # zero eigenvalues are masked by the log axis, not removed from the data
    assert ax.get_yscale() == "log"
    pos = np.concatenate(
        [[3.0, 1.0, 0.5], np.abs(np.asarray(ax.collections[0].get_paths()[0].vertices[:, 1]))]
    )
    assert ax.get_ylim()[0] <= pos[pos > 0].min() / 10 * (1 + 1e-12)


# ---------------------------------------------------------------------------
# Subspace errors (V13.4)
# ---------------------------------------------------------------------------


def test_subspace_errors(res, res_noboot):
    _, ax = P.plot_subspace_errors(res)
    (line,) = _data_lines(ax)
    np.testing.assert_array_equal(line.get_xdata(), np.arange(1, M_IN))
    np.testing.assert_allclose(line.get_ydata(), res.bootstrap_distances.mean(axis=0), rtol=1e-14)
    np.testing.assert_array_equal(line.get_ydata(), asub.mean_distance(res))
    assert ax.get_ylim()[1] == 1.0
    assert ax.get_yscale() == "log"
    iv = asub.bootstrap_interval(res.bootstrap_distances)
    _band_ok(ax.collections[0], range(1, M_IN), iv[0], iv[1])
    assert [ln.get_xdata()[0] for ln in _vlines(ax)] == [res.n]
    assert ax.get_xlabel() == "subspace dimension $k$"
    assert ax.get_ylabel() == "subspace distance"

    _, ax = P.plot_subspace_errors(res, interval="minmax", k_max=2, mark_dimension=False)
    reps = res.bootstrap_distances[:, :2]
    _band_ok(ax.collections[0], range(1, 3), reps.min(axis=0), reps.max(axis=0))
    assert not _vlines(ax)
    _, ax = P.plot_subspace_errors(res, interval="none")
    assert not ax.collections
    with pytest.raises(InputError) as e:
        P.plot_subspace_errors(res_noboot)
    assert _code(e) == "input.no_bootstrap"
    with pytest.raises(InputError) as e:
        P.plot_subspace_errors(res, k_max=0)
    assert _code(e) == "input.parameter"


# ---------------------------------------------------------------------------
# Eigenvectors (V13.5)
# ---------------------------------------------------------------------------


def test_eigenvectors(res):
    fig, axes = P.plot_eigenvectors(res)
    assert len(axes) == min(res.n, 4) == 2
    assert list(fig.axes) == list(axes)
    for j, a in enumerate(axes):
        (line,) = _data_lines(a)
        np.testing.assert_array_equal(line.get_xdata(), np.arange(1, M_IN + 1))
        np.testing.assert_array_equal(line.get_ydata(), res.eigenvectors[:, j])
        assert a.get_ylim() == (-1.05, 1.05)
        assert a.get_ylabel() == f"$w_{{{j + 1}}}$"
    top, bottom = axes
    assert [t.get_text() for t in bottom.get_xticklabels()] == list(NAMES)
    assert all(t.get_rotation() == 90 for t in bottom.get_xticklabels())
    assert not any(t.label1.get_visible() for t in top.xaxis.get_major_ticks())
    assert all(t.label1.get_visible() for t in bottom.xaxis.get_major_ticks())

    _, axes = P.plot_eigenvectors(res, indices=[4, 0, 2], labels=list("abcde"))
    assert len(axes) == 3
    np.testing.assert_array_equal(_data_lines(axes[0])[0].get_ydata(), res.eigenvectors[:, 4])
    assert axes[0].get_ylabel() == "$w_{5}$"
    assert [t.get_text() for t in axes[2].get_xticklabels()] == list("abcde")

    caller = Figure()
    cax = tuple(caller.subplots(2, 1)[:])
    fig, axes = P.plot_eigenvectors(res, indices=[0, 1], ax=cax)
    assert fig is caller
    assert axes == cax
    assert not any(t.label1.get_visible() for t in cax[0].xaxis.get_major_ticks())

    for bad in (
        {"indices": []},
        {"indices": [0, 0]},
        {"indices": [5]},
        {"indices": [-1]},
        {"indices": "0"},
        {"labels": list("abc")},
        {"indices": [0, 1], "ax": cax[0]},
        {"indices": [0], "ax": cax},
    ):
        with pytest.raises(InputError) as e:
            P.plot_eigenvectors(res, **bad)
        assert _code(e) == "input.parameter", bad


def test_eigenvectors_many_inputs():
    m = 41
    sp = asub.Uniform(np.zeros(m), np.ones(m))
    b = asub.subspace_from_basis(np.eye(m)[:, :1], space=sp)
    _, (ax,) = P.plot_eigenvectors(b)
    assert isinstance(ax.xaxis.get_major_locator(), matplotlib.ticker.MaxNLocator)
    assert ax.get_xlabel() == "input $i$"


# ---------------------------------------------------------------------------
# Sufficient summary (V13.6)
# ---------------------------------------------------------------------------


def test_sufficient_summary_1d(res, data, space):
    X, f, _ = data
    fig, ax = P.plot_sufficient_summary(res, X=X, f=f[:, None], dims=1)
    (sc,) = [c for c in ax.collections if isinstance(c, PathCollection)]
    off = sc.get_offsets()
    np.testing.assert_allclose(
        off[:, 0], space.to_normalized(X) @ res.eigenvectors[:, 0], rtol=0, atol=1e-14
    )
    np.testing.assert_array_equal(off[:, 1], f)
    assert ax.get_xlabel() == "$y_1 = w_1^\\top x$"
    assert ax.get_ylabel() == "q"
    assert len(fig.axes) == 1
    assert not sc.get_rasterized()
    _, ax = P.plot_sufficient_summary(res, X=X, f=f, dims=1, qoi_label="drag")
    assert ax.get_ylabel() == "drag"


def test_sufficient_summary_2d(res, data, space, dom):
    X, f, _ = data
    fig, ax = P.plot_sufficient_summary(res, X=X, f=f)
    assert len(fig.axes) == 2  # plot and colorbar
    (sc,) = [c for c in ax.collections if isinstance(c, PathCollection)]
    np.testing.assert_allclose(
        sc.get_offsets(), space.to_normalized(X) @ res.eigenvectors[:, :2], rtol=0, atol=1e-14
    )
    np.testing.assert_array_equal(sc.get_array(), f)
    assert ax.get_xlabel() == "$y_1$"
    assert ax.get_ylabel() == "$y_2$"
    assert ax.get_aspect() == 1.0
    assert fig.axes[1].get_ylabel() == "q"

    _, ax = P.plot_sufficient_summary(res, X=X, f=f, domain=dom)
    (poly,) = ax.patches
    assert isinstance(poly, Polygon)
    V = np.asarray(dom.vertices)
    expected = V[scipy.spatial.ConvexHull(V).vertices]
    np.testing.assert_allclose(poly.get_xy()[:-1], expected, rtol=0, atol=1e-15)
    assert set(map(tuple, np.round(expected, 12))) == set(
        map(tuple, np.round(V, 12))
    )  # a zonotope's vertices are its hull
    assert not poly.get_fill()


def test_sufficient_summary_domains(res, res1, data, space, dom):
    X, f, G = data
    other = asub.active_subspace(space=space, X=X, f=f, grads=G, n=2, nboot=30, rng=2, qoi="q")
    with pytest.raises(InputError) as e:
        P.plot_sufficient_summary(other, X=X, f=f, domain=dom)
    assert _code(e) == "input.parameter"
    with pytest.raises(InputError) as e:
        P.plot_sufficient_summary(res, X=X, f=f, dims=1, domain=dom)
    assert _code(e) == "input.parameter"

    dom1 = active_domain(res1)
    assert dom1.kind == "interval"
    _, ax = P.plot_sufficient_summary(res1, X=X, f=f, domain=dom1, domain_kw={"color": "k"})
    xs = sorted(ln.get_xdata()[0] for ln in _vlines(ax))
    np.testing.assert_allclose(xs, np.sort(np.asarray(dom1.vertices)[:, 0]))
    assert all(to_rgba(ln.get_color()) == to_rgba("k") for ln in _vlines(ax))

    gs = asub.Gaussian(np.zeros(M_IN), np.eye(M_IN))
    Xg = gs.sample(40, rng=np.random.default_rng(1))
    rg = asub.active_subspace(space=gs, grads=_g(Xg), X=Xg, f=_f(Xg), n=2)
    dg = active_domain(rg)
    assert dg.kind == "unbounded"
    _, ax = P.plot_sufficient_summary(rg, X=Xg, f=_f(Xg), domain=dg)
    assert not ax.patches
    assert not _vlines(ax)


def test_sufficient_summary_errors(res, basis, data, space):
    X, f, _ = data
    _, ax = P.plot_sufficient_summary(basis, X=X, f=f)
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(),
        space.to_normalized(X) @ basis.eigenvectors[:, :2],
        atol=1e-14,
    )
    for bad, code in (
        ({"dims": 3}, "input.parameter"),
        ({"dims": 0}, "input.parameter"),
        ({"f": f[:-1]}, "input.size_mismatch"),
        ({"f": np.c_[f, f]}, "input.vector_qoi"),
        ({"X": X[:, :3]}, "input.shape"),
        ({"X": X + 1e3}, "input.out_of_space"),
        ({"f": np.where(np.arange(f.size) == 0, np.nan, f)}, "input.nonfinite"),
        ({"X": X.astype(str)}, "input.dtype"),
        ({"qoi_label": 3}, "input.parameter"),
        ({"scatter_kw": [1]}, "input.parameter"),
    ):
        kw = {"X": X, "f": f, **bad}
        with pytest.raises(InputError) as e:
            P.plot_sufficient_summary(res, **kw)
        assert _code(e) == code, bad
    m1 = asub.Uniform([0.0], [1.0])
    r1 = asub.subspace_from_basis(np.ones((1, 1)), space=m1)
    with pytest.raises(InputError) as e:
        P.plot_sufficient_summary(r1, X=np.ones((3, 1)) / 2, f=np.ones(3), dims=2)
    assert _code(e) == "input.parameter"


def test_sufficient_summary_rasterized(res, space):
    X = space.sample(5001, rng=np.random.default_rng(0))
    _, ax = P.plot_sufficient_summary(res, X=X, f=_f(space.to_normalized(X)), dims=1)
    assert ax.collections[0].get_rasterized()


# ---------------------------------------------------------------------------
# Active domain (V13.7)
# ---------------------------------------------------------------------------


def test_active_domain(dom, res):
    fig, ax = P.plot_active_domain(dom)
    (poly,) = ax.patches
    V = np.asarray(dom.vertices)
    np.testing.assert_allclose(
        poly.get_xy()[:-1], V[scipy.spatial.ConvexHull(V).vertices], atol=1e-15
    )
    lo, hi = V.min(axis=0), V.max(axis=0)
    pad = 0.05 * (hi - lo)
    np.testing.assert_allclose(ax.get_xlim(), [lo[0] - pad[0], hi[0] + pad[0]])
    np.testing.assert_allclose(ax.get_ylim(), [lo[1] - pad[1], hi[1] + pad[1]])
    assert ax.get_aspect() == 1.0
    assert ax.get_xlabel() == "$y_1$"
    assert len(fig.axes) == 1

    design = 0.3 * V[:4] + 0.01
    fig, ax = P.plot_active_domain(dom, design=design, triangulate=True)
    (mesh,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    tri = scipy.spatial.Delaunay(np.vstack([design, V]))
    edges = {
        frozenset(e) for s in tri.simplices for e in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2]))
    }
    assert len(mesh.get_segments()) == len(edges)
    (pts,) = _data_lines(ax)
    np.testing.assert_array_equal(np.c_[pts.get_xdata(), pts.get_ydata()], design)

    P_ = 0.2 * V[:3]
    fig, ax = P.plot_active_domain(dom, points=P_, values=[1.0, 2.0, 3.0])
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == res.qoi
    (sc,) = [c for c in ax.collections if isinstance(c, PathCollection)]
    np.testing.assert_array_equal(sc.get_offsets(), P_)
    np.testing.assert_array_equal(sc.get_array(), [1.0, 2.0, 3.0])
    fig, _ = P.plot_active_domain(dom, points=P_, values=[1.0, 2.0, 3.0], qoi_label="lift")
    assert fig.axes[1].get_ylabel() == "lift"


def test_active_domain_errors(dom, res1):
    V = np.asarray(dom.vertices)
    for bad, code in (
        ({"triangulate": True}, "input.parameter"),
        ({"points": V[:2]}, "input.parameter"),
        ({"values": [1.0, 2.0]}, "input.parameter"),
        ({"points": V[:2], "values": [1.0, 2.0, 3.0]}, "input.size_mismatch"),
        ({"design": V[:, :1]}, "input.shape"),
        ({"design": np.full((2, 2), np.nan)}, "input.nonfinite"),
        ({"points": V[:2].astype(complex), "values": [1.0, 2.0]}, "input.dtype"),
        ({"triangulate": 1}, "input.parameter"),
    ):
        with pytest.raises(InputError) as e:
            P.plot_active_domain(dom, **bad)
        assert _code(e) == code, bad
    with pytest.raises(InputError) as e:
        P.plot_active_domain(active_domain(res1))
    assert _code(e) == "input.parameter"
    gs = asub.Gaussian(np.zeros(3), np.eye(3))
    rg = asub.subspace_from_basis(np.eye(3)[:, :2], space=gs)
    with pytest.raises(UnsupportedError) as e:
        P.plot_active_domain(active_domain(rg))
    assert _code(e) == "unsupported.space"


# ---------------------------------------------------------------------------
# Surrogate fit and a ReducedModel (V13.8)
# ---------------------------------------------------------------------------


class _ReducedModelStub:
    """Minimal duck-typed stand-in for the spec 11 ReducedModel: the Surrogate protocol by
    delegation.
    """

    def __init__(self, surrogate):
        self.surrogate = surrogate
        self.space = surrogate.space
        self.subspace = surrogate.subspace
        self.qoi = surrogate.qoi

    @property
    def active_dim(self):
        return self.subspace.n

    def evaluate(self, Y):
        return self.surrogate.evaluate(Y)

    def gradient(self, Y):
        return self.surrogate.gradient(Y)


def test_surrogate_fit_and_reduced_model(surrogate, data):
    X, f, _ = data
    _, ax = P.plot_surrogate_fit(surrogate, X=X, f=f)
    (sc,) = [c for c in ax.collections if isinstance(c, PathCollection)]
    g = predict(surrogate, X)
    np.testing.assert_array_equal(sc.get_offsets()[:, 0], f)
    np.testing.assert_array_equal(sc.get_offsets()[:, 1], g)
    (diag,) = _data_lines(ax)
    lo, hi = min(f.min(), g.min()), max(f.max(), g.max())
    np.testing.assert_allclose(diag.get_xydata(), [[lo, lo], [hi, hi]])
    assert ax.get_xlabel() == "observed q"
    assert ax.get_ylabel() == "surrogate q"
    assert ax.get_aspect() == 1.0

    rm = _ReducedModelStub(surrogate)
    _, ax2 = P.plot_surrogate_fit(rm, X=X, f=f, qoi_label="y")
    np.testing.assert_array_equal(ax2.collections[0].get_offsets(), sc.get_offsets())
    assert ax2.get_xlabel() == "observed y"

    with pytest.raises(InputError) as e:
        P.plot_surrogate_fit(object(), X=X, f=f)
    assert _code(e) == "input.parameter"
    with pytest.raises(InputError) as e:
        P.plot_surrogate_fit(surrogate, X=X, f=f[:-2])
    assert _code(e) == "input.size_mismatch"
    with pytest.raises(InputError) as e:
        P.plot_surrogate_fit(surrogate, X=X - 1e4, f=f)
    assert _code(e) == "input.out_of_space"


# ---------------------------------------------------------------------------
# Dimension criterion (V13.9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule", "label", "log"),
    [
        ("evg", "eigenvalue gap $|\\lambda_k| - |\\lambda_{k+1}|$", True),
        (
            S.EVG(gap="log"),
            "log eigenvalue gap $\\log|\\lambda_k| - \\log|\\lambda_{k+1}|$",
            False,
        ),
        ("rs", "RS error bound $\\varepsilon_k$", True),
        ("li", "ladle criterion $g(k)$", False),
    ],
)
def test_dimension_criterion(rule, label, log, res):
    ks, values, k_hat = asub.dimension_criterion(res, rule=rule)
    _, ax = P.plot_dimension_criterion(res, rule=rule)
    main, sel = _data_lines(ax)
    np.testing.assert_array_equal(main.get_xdata(), ks)
    np.testing.assert_array_equal(main.get_ydata(), values)
    np.testing.assert_array_equal(sel.get_xdata(), [k_hat])
    np.testing.assert_array_equal(sel.get_ydata(), values[ks == k_hat])
    assert sel.get_markersize() > main.get_markersize()
    assert [ln.get_xdata()[0] for ln in _vlines(ax)] == [k_hat]
    assert ax.get_ylabel() == label
    assert ax.get_xlabel() == "dimension $k$"
    assert (ax.get_yscale() == "log") is log


def test_dimension_criterion_zero_and_errors(make_provenance, res_noboot, basis):
    space = asub.Gaussian.reference(4)
    Bn = 5
    hand = S.SubspaceResult(
        space=space,
        estimator="as",
        estimator_options=(),
        eigenvalues=np.array([1e-3, 1e-4, 1e-5, 1e-6]),
        eigenvectors=np.eye(4),
        n=1,
        dimension_rule="fixed",
        dimension_options=(),
        criterion=None,
        n_samples=10,
        nboot=Bn,
        bootstrap_eigenvalues=np.zeros((Bn, 4)),
        bootstrap_distances=np.ones((Bn, 3)),
        bootstrap_dets=np.zeros((Bn, 3)),
        provenance=make_provenance(),
    )
    _, ax = P.plot_dimension_criterion(hand, rule="li")
    assert [ln.get_xdata()[0] for ln in _vlines(ax)] == [0]
    assert _data_lines(ax)[0].get_xdata()[0] == 0
    for r in ("rs", "li"):
        with pytest.raises(InputError) as e:
            P.plot_dimension_criterion(res_noboot, rule=r)
        assert _code(e) == "input.no_bootstrap"
    with pytest.raises(InputError) as e:
        P.plot_dimension_criterion(basis, rule="evg")
    assert _code(e) == "input.parameter"
    with pytest.raises(InputError) as e:
        P.plot_dimension_criterion(res_noboot, rule="nope")
    assert _code(e) == "input.parameter"


def test_dimension_criterion_m1():
    sp = asub.Uniform([0.0], [1.0])
    X = np.linspace(0.1, 0.9, 6)[:, None]
    r = asub.active_subspace(space=sp, X=X, f=X[:, 0], grads=np.ones((6, 1)), n=1)
    _, ax = P.plot_dimension_criterion(r, rule="evg")
    assert not _data_lines(ax)
    assert not _vlines(ax)


# ---------------------------------------------------------------------------
# "basis" results (V13.10)
# ---------------------------------------------------------------------------


def test_basis_results(basis, data):
    with pytest.raises(InputError) as e:
        P.plot_eigenvalues(basis)
    assert _code(e) == "input.parameter"
    assert e.value.details["reason"] == "no_eigenvalues"
    _, axes = P.plot_eigenvectors(basis)
    assert len(axes) == 2
    P.plot_sufficient_summary(basis, X=data[0], f=data[1])
    with pytest.raises(InputError) as e:
        P.plot_subspace_errors(basis)
    assert _code(e) == "input.no_bootstrap"


# ---------------------------------------------------------------------------
# figure_to_bytes (V13.11), style (V13.12), catalogue (V13.13), threads (V13.14)
# ---------------------------------------------------------------------------


def test_figure_to_bytes(res):
    fig, ax = P.plot_eigenvalues(res)
    png = P.figure_to_bytes(fig)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert P.figure_to_bytes(fig) == png
    assert P.figure_to_bytes(ax) == png
    assert P.figure_to_bytes(P.plot_eigenvalues(res)[0]) == png
    assert P.figure_to_bytes(fig, dpi=50) != png
    svg = P.figure_to_bytes(fig, format="svg")
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert b"dc:date" not in svg
    svg2 = P.figure_to_bytes(fig, format="svg")
    assert len(ET.fromstring(svg2)) == len(root)

    sub = Figure().subfigures(1, 2)[0]
    P.plot_eigenvalues(res, ax=sub.add_subplot())
    assert P.figure_to_bytes(sub).startswith(b"\x89PNG")
    for bad in ({"format": "pdf"}, {"dpi": 0}, {"dpi": -1.0}, {"format": None}):
        with pytest.raises(InputError) as e:
            P.figure_to_bytes(fig, **bad)
        assert _code(e) == "input.parameter", bad
    with pytest.raises(InputError) as e:
        P.figure_to_bytes("figure")
    assert _code(e) == "input.parameter"


def test_style_hook(res, data):
    X, f, _ = data
    with matplotlib.rc_context({"axes.prop_cycle": cycler(color=["red"])}):
        _, ax = P.plot_eigenvalues(res)
    assert to_rgba(_data_lines(ax)[0].get_color()) == to_rgba("red")
    assert to_rgba(ax.collections[0].get_facecolor()[0], alpha=1) == to_rgba("red")
    _, ax = P.plot_eigenvalues(res, line_kw={"color": "k"}, band_kw={"alpha": 0.5})
    assert to_rgba(_data_lines(ax)[0].get_color()) == to_rgba("k")
    assert ax.collections[0].get_alpha() == 0.5
    _, ax = P.plot_eigenvalues(res)
    assert to_rgba(_data_lines(ax)[0].get_color()) == to_rgba("C0")
    with matplotlib.rc_context({"axes.prop_cycle": cycler(color=["green"])}):
        _, ax = P.plot_sufficient_summary(res, X=X, f=f, dims=1)
        _, ax2 = P.plot_eigenvectors(res, line_kw={"color": "C0"})
    assert to_rgba(ax.collections[0].get_facecolor()[0], alpha=1) == to_rgba("green")
    assert to_rgba(_data_lines(ax2[0])[0].get_color()) == to_rgba("green")


def test_catalogue():
    assert set(P.PLOTS) == {
        "eigenvalues",
        "subspace_errors",
        "eigenvectors",
        "sufficient_summary",
        "active_domain",
        "surrogate_fit",
        "dimension_criterion",
    }
    for name, fn in P.PLOTS.items():
        assert fn is getattr(P, f"plot_{name}")
    with pytest.raises(TypeError):
        P.PLOTS["x"] = P.plot_eigenvalues
    assert P.__all__ == tuple(sorted(P.__all__))
    assert set(P.__all__) == {"PLOTS", "figure_to_bytes", *(f"plot_{n}" for n in P.PLOTS)}
    assert asub.plotting is P
    assert "plotting" in asub.__all__


def test_threads(res):
    def render(_):
        return P.figure_to_bytes(P.plot_eigenvalues(res)[0])

    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(render, range(8)))
    assert all(b == out[0] for b in out)


# ---------------------------------------------------------------------------
# No pyplot, no figure registry, no files (V13.1), in fresh interpreters
# ---------------------------------------------------------------------------

_SCRIPT = r"""
import os, sys
import numpy as np
import active_subspaces as asub
from active_subspaces import plotting as P
from active_subspaces.geometry import active_domain
from active_subspaces.surrogates import fit_surrogate
sp = asub.Uniform(np.zeros(4), np.arange(1.0, 5.0))
X = sp.sample(60, rng=np.random.default_rng(0))
x = sp.to_normalized(X)
a = np.array([1.0, 0.5, 0.2, 0.1])
f = np.exp(x @ a) + 0.1 * x[:, 1] ** 2
G = sp.grads_to_physical(np.exp(x @ a)[:, None] * a + 0.2 * x[:, 1:2] * np.eye(4)[1])
res = asub.active_subspace(space=sp, X=X, f=f, grads=G, n=2, nboot=10, rng=0)
dom = active_domain(res)
sur = fit_surrogate(X=X, f=f, subspace=res, kind="poly")
WITH_PYPLOT = sys.argv[1] == "pyplot"
if WITH_PYPLOT:
    import matplotlib.pyplot as plt
    pfig, pax = plt.subplots()
    nums = plt.get_fignums()
figs = [
    P.plot_eigenvalues(res)[0], P.plot_subspace_errors(res)[0], P.plot_eigenvectors(res)[0],
    P.plot_sufficient_summary(res, X=X, f=f, domain=dom)[0],
    P.plot_active_domain(dom, design=dom.vertices / 2, triangulate=True)[0],
    P.plot_surrogate_fit(sur, X=X, f=f)[0], P.plot_dimension_criterion(res, rule="rs")[0],
]
for fig in figs:
    P.figure_to_bytes(fig); P.figure_to_bytes(fig, format="svg")
if WITH_PYPLOT:
    P.plot_eigenvalues(res, ax=pax)
    assert plt.get_fignums() == nums, (plt.get_fignums(), nums)
    plt.close(pfig)
    assert plt.get_fignums() == []
else:
    assert "matplotlib.pyplot" not in sys.modules
assert os.listdir(".") == []
"""


@pytest.mark.parametrize("mode", ["no_pyplot", "pyplot"])
def test_no_pyplot_and_no_files_subprocess(mode, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    if mode == "pyplot":
        env["MPLBACKEND"] = "agg"
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, "-W", "error", "-c", _SCRIPT, mode],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=run_dir,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert proc.stderr == ""
    run_dir.rmdir()
