"""Known answers for spec 09 designs and conditional samples (V09.9, V09.12)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import _quadrature


def _ref(m):
    return asub.Uniform(-np.ones(m), np.ones(m))


# ---------------------------------------------------------------------------
# V09.9 interval and Gauss-Hermite designs
# ---------------------------------------------------------------------------


def test_interval_design():
    sp = _ref(3)
    sub = asub.subspace_from_basis(np.array([[1.0], [-2.0], [0.5]]), space=sp)
    dom = asub.active_domain(sub)
    assert dom.kind == "interval"
    a, b = sorted(dom.vertices[:, 0])
    d = asub.active_design(domain=dom, n_points=4, rng=0)
    assert d.method == "interval"
    assert d.min_distance is None
    np.testing.assert_allclose(
        d.points[:, 0], a + np.arange(1, 5) * (b - a) / 5, rtol=1e-14, atol=1e-14 * abs(b)
    )
    assert d.points.shape == (4, 1)
    assert d.provenance.rng is not None  # rng normalized and recorded even without draws


@pytest.mark.parametrize(
    ("n", "N", "q"), [(2, 25, 5), (2, 10, 3), (3, 125, 5), (3, 124, 4), (1, 7, 7), (2, 1, 1)]
)
def test_gauss_hermite_design(n, N, q):
    sp = asub.Gaussian(np.zeros(n + 1), np.eye(n + 1))
    sub = asub.subspace_from_basis(np.eye(n + 1)[:, :n], space=sp)
    dom = asub.active_domain(sub)
    assert dom.kind == "unbounded"
    d = asub.active_design(domain=dom, n_points=N)
    nodes, _ = _quadrature.tensor_rule([_quadrature.gauss_hermite(q)] * n)
    assert d.method == "gauss_hermite"
    assert d.points.shape == (q**n, n)
    np.testing.assert_array_equal(d.points, nodes)


# ---------------------------------------------------------------------------
# V09.12 conditional samples with an inert input [C-20], [D-103]
# ---------------------------------------------------------------------------


def _c20_subspace(inert: bool):
    rng = np.random.default_rng(0)
    W1 = np.linalg.qr(rng.normal(size=(4, 2)))[0]
    if inert:
        W1 = np.vstack((W1, np.zeros((1, 2))))
    return asub.subspace_from_basis(W1, space=_ref(W1.shape[0]))


def test_conditional_samples_inert(seed):
    sub = _c20_subspace(True)
    dom = asub.active_domain(sub)
    assert dom.n_vertices == 8
    model = asub.Model(space=sub.space, value=lambda X: X[:, 0] + X[:, 4] ** 2)
    n_mc = 200
    cs = asub.conditional_samples(
        domain=dom, model=model, design=np.zeros((1, 2)), n_mc=n_mc, rng=seed
    )
    assert cs.is_vertex.sum() == 8
    assert cs.Y.shape == (9, 2)
    np.testing.assert_array_equal(cs.Y[1:], dom.vertices)
    assert np.all(cs.counts == n_mc)
    se = math.sqrt(4 / 45 / n_mc)
    for v in range(8):
        rows = cs.index == 1 + v
        Xv = cs.X[rows]
        np.testing.assert_array_equal(Xv[:, :4], np.tile(dom.vertex_preimages[v, :4], (n_mc, 1)))
        assert Xv[:, 4].min() < -0.8
        assert Xv[:, 4].max() > 0.8
        target = dom.vertex_preimages[v, 0] + 1 / 3
        assert abs(cs.means[1 + v] - target) <= 5.25 * se
    np.testing.assert_allclose(cs.noise_variance, cs.variances / n_mc, rtol=1e-15)


def test_conditional_samples_unique_preimage(seed):
    sub = _c20_subspace(False)
    dom = asub.active_domain(sub)
    model = asub.Model(space=sub.space, value=lambda X: X[:, 0] + X[:, 3] ** 2)
    cs = asub.conditional_samples(
        domain=dom, model=model, design=np.zeros((1, 2)), n_mc=5, rng=seed
    )
    v = cs.is_vertex
    assert v.sum() == dom.n_vertices
    assert np.all(cs.counts[v] == 1)
    assert np.all(cs.variances[v] == 0)
    pre = dom.vertex_preimages
    np.testing.assert_allclose(cs.means[v], pre[:, 0] + pre[:, 3] ** 2, rtol=1e-14, atol=1e-14)
