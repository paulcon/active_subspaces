"""Statistical tests of the inverse-map samplers (spec 08 §8, spec 17; fixed seeds, D-78)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

import active_subspaces as asub

ALPHA = 1e-3


def _sub(m, n, seed=0, space=None):
    rng = np.random.default_rng(seed)
    space = asub.Uniform.reference(m) if space is None else space
    return asub.subspace_from_basis(rng.standard_normal((m, n)), space=space)


# ---------------------------------------------------------------------------
# Exact 2-D polygon geometry (for the hit-and-run reference)
# ---------------------------------------------------------------------------


def _clip(poly, a, b):
    """Sutherland-Hodgman: the part of polygon ``poly`` (k, 2) with a.z <= b."""
    out = []
    k = len(poly)
    for i in range(k):
        p, q = poly[i], poly[(i + 1) % k]
        fp, fq = a @ p - b, a @ q - b
        if fp <= 0:
            out.append(p)
        if fp * fq < 0:
            out.append(p + (fp / (fp - fq)) * (q - p))
    return np.array(out) if out else np.zeros((0, 2))


def _area_moments(poly):
    """Area, centroid and second moment matrix E[z z^T] * area of a convex polygon."""
    if len(poly) < 3:
        return 0.0, np.zeros(2), np.zeros((2, 2))
    area, first, second = 0.0, np.zeros(2), np.zeros((2, 2))
    o = poly[0]
    for i in range(1, len(poly) - 1):
        p, q = poly[i], poly[i + 1]
        A = 0.5 * ((p - o)[0] * (q - o)[1] - (p - o)[1] * (q - o)[0])
        verts = np.array([o, p, q])
        c = verts.mean(axis=0)
        S = (verts.T @ verts + 9 * np.outer(c, c)) / 12.0  # E[z z^T] on a triangle
        area += A
        first += A * c
        second += A * S
    return area, first / area, second


def _fiber_polygon(W2, s):
    poly = np.array([[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]])
    for j in range(W2.shape[0]):
        poly = _clip(poly, W2[j], 1.0 - s[j])
        poly = _clip(poly, -W2[j], 1.0 + s[j])
    return poly


# ---------------------------------------------------------------------------
# Hit-and-run (m = 4, n = 2: the fiber is a polygon)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hit_and_run_case():
    sub = _sub(4, 2, seed=21)
    dom = asub.active_domain(sub)
    y = 0.35 * dom.vertices[1] + 0.2 * dom.vertices[4]
    chains, n_draws = 20, 200
    res = asub.inverse_map(sub, Y=np.tile(y, (chains, 1)), n_draws=n_draws, thin=10, rng=20261028)
    return sub, y, res


def test_hit_and_run_cell_frequencies(hit_and_run_case):
    sub, y, res = hit_and_run_case
    assert res.sampler == "hit_and_run"
    W2, s = sub.W2, sub.W1 @ y
    poly = _fiber_polygon(W2, s)
    total, _, _ = _area_moments(poly)
    Z = res.Z
    # cells: a 4 x 4 grid over the bounding box of the polygon
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    gx = np.linspace(lo[0], hi[0], 5)
    gy = np.linspace(lo[1], hi[1], 5)
    expected, observed = [], []
    for i in range(4):
        for j in range(4):
            cell = poly
            cell = _clip(cell, np.array([-1.0, 0.0]), -gx[i])
            cell = _clip(cell, np.array([1.0, 0.0]), gx[i + 1])
            cell = _clip(cell, np.array([0.0, -1.0]), -gy[j])
            cell = _clip(cell, np.array([0.0, 1.0]), gy[j + 1])
            area = _area_moments(cell)[0]
            inside = (Z[:, 0] >= gx[i]) & (Z[:, 0] < gx[i + 1] if i < 3 else Z[:, 0] <= gx[i + 1])
            inside &= (Z[:, 1] >= gy[j]) & (Z[:, 1] < gy[j + 1] if j < 3 else Z[:, 1] <= gy[j + 1])
            expected.append(area / total)
            observed.append(int(inside.sum()))
    expected = np.array(expected) * Z.shape[0]
    observed = np.array(observed)
    keep = expected >= 5
    other_e, other_o = expected[~keep].sum(), observed[~keep].sum()
    e = np.append(expected[keep], other_e) if other_e > 0 else expected[keep]
    o = np.append(observed[keep], other_o) if other_e > 0 else observed[keep]
    assert o.sum() == Z.shape[0]
    _, p = scipy.stats.chisquare(o, e * o.sum() / e.sum())
    assert p > ALPHA


def test_hit_and_run_moments(hit_and_run_case):
    sub, y, res = hit_and_run_case
    poly = _fiber_polygon(sub.W2, sub.W1 @ y)
    area, mean, second = _area_moments(poly)
    cov = second / area - np.outer(mean, mean)
    Z = res.Z
    N = Z.shape[0]
    se = np.sqrt(np.diag(cov) / N)
    assert np.all(
        np.abs(Z.mean(axis=0) - mean) <= 6 * se
    )  # 6 se: thinned draws are mildly correlated
    np.testing.assert_allclose(np.cov(Z.T), cov, rtol=0.15, atol=0.05 * np.max(np.abs(cov)))


def test_hit_and_run_feasible(hit_and_run_case):
    sub, y, res = hit_and_run_case
    assert np.all(np.abs(res.X) <= 1.0)
    np.testing.assert_allclose(res.X @ sub.W1, np.tile(y, (res.X.shape[0], 1)), atol=1e-10)


def test_hit_and_run_without_thinning_warns():
    sub = _sub(4, 2, seed=21)
    with pytest.warns(asub.AccuracyWarning) as rec:
        asub.inverse_map(sub, Y=np.zeros((1, 2)), n_draws=10, thin=1, rng=1)
    assert [w.message.code for w in rec] == ["accuracy.correlated_draws"]


def test_hit_and_run_higher_d_mean():
    """d = 4: the fiber at y = 0 is symmetric about 0, so the draws have mean 0."""
    sub = _sub(6, 2, seed=5)
    res = asub.inverse_map(sub, Y=np.zeros((10, 2)), n_draws=200, rng=3)
    Z = res.Z
    se = Z.std(axis=0) / np.sqrt(Z.shape[0])
    assert np.all(np.abs(Z.mean(axis=0)) <= 10 * se)  # loose: draws within a chain are correlated
    assert np.all(np.abs(res.X) <= 1.0)


# ---------------------------------------------------------------------------
# Exact interval (d = 1)
# ---------------------------------------------------------------------------


def test_exact_interval_uniform_ks():
    sub = _sub(3, 2, seed=7)
    dom = asub.active_domain(sub)
    W1, w2 = sub.W1, sub.W2[:, 0]
    for k, y in enumerate([0.3 * dom.vertices[0], 0.7 * dom.vertices[2] + 0.1 * dom.vertices[3]]):
        res = asub.inverse_map(sub, Y=y[None, :], n_draws=2000, rng=100 + k)
        s = W1 @ y
        a, c = (-1 - s) / w2, (1 - s) / w2
        lo, hi = np.max(np.minimum(a, c)), np.min(np.maximum(a, c))
        t = (res.X - s) @ w2
        assert np.all(t >= lo - 1e-12)
        assert np.all(t <= hi + 1e-12)
        assert scipy.stats.kstest((t - lo) / (hi - lo), "uniform").pvalue > ALPHA


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------


def test_gaussian_conditional(gaussian_space):
    space = gaussian_space(4, seed=2)
    sub = _sub(4, 2, space=space)
    gen = np.random.default_rng(4)
    Y = gen.standard_normal((500, 2))
    res = asub.inverse_map(sub, Y=Y, n_draws=4, rng=8)
    Z = res.Z
    for j in range(2):
        assert scipy.stats.kstest(Z[:, j], "norm").pvalue > ALPHA
    Yr = Y[res.index]
    N = Z.shape[0]
    for i in range(2):
        for j in range(2):
            r = np.corrcoef(Z[:, j], Yr[:, i])[0, 1]
            assert abs(r) < 4 / np.sqrt(N)
    assert abs(np.corrcoef(Z[:, 0], Z[:, 1])[0, 1]) < 4 / np.sqrt(N)
