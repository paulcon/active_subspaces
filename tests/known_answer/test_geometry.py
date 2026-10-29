"""Known-answer tests of active_subspaces.geometry (spec 08 §8): exact zonotopes and boundary
behavior.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest
import scipy.optimize
from scipy.spatial import ConvexHull

import active_subspaces as asub
from active_subspaces.errors import InputError


def _sub(m, n, seed=0, space=None):
    rng = np.random.default_rng(seed)
    space = asub.Uniform.reference(m) if space is None else space
    return asub.subspace_from_basis(rng.standard_normal((m, n)), space=space)


def _corner_images(W1):
    m = W1.shape[0]
    return np.array(list(itertools.product((-1.0, 1.0), repeat=m))) @ W1


def _same_rows(A, B, tol):
    if A.shape != B.shape:
        return False
    D = np.max(np.abs(A[:, None, :] - B[None, :, :]), axis=-1)
    return bool(np.all(D.min(axis=1) <= tol) and np.all(D.min(axis=0) <= tol))


def _hull_planes(points):
    """Distinct facet planes (unit normal, offset) of the convex hull, from
    ConvexHull.equations.
    """
    eq = ConvexHull(points).equations  # a^T y + b <= 0  ->  nu = a, offset = -b
    planes = np.hstack([eq[:, :-1], -eq[:, -1:]])
    out = []
    for p in planes:
        if not any(np.max(np.abs(p - q)) <= 1e-9 for q in out):
            out.append(p)
    return np.array(out)


def _check_against_brute_force(dom):
    W1 = dom.subspace.W1
    Yc = _corner_images(W1)
    hull = ConvexHull(Yc)
    scale = 1.0 + np.max(np.abs(Yc))
    assert _same_rows(dom.vertices, Yc[hull.vertices], 1e-12 * scale)
    mine = np.hstack([dom.facet_normals, dom.facet_offsets[:, None]])
    assert _same_rows(mine, _hull_planes(Yc), 1e-9 * scale)


def _check_h_representation(dom):
    n = dom.n
    A, b, V = dom.facet_normals, dom.facet_offsets, dom.vertices
    tol = 1e-12 * (1.0 + b.max())
    slack = A @ V.T - b[:, None]
    assert np.all(slack <= tol)
    assert np.all(np.sum(np.abs(slack) <= 1e-10 * (1.0 + b.max()), axis=1) >= n)
    np.testing.assert_allclose(np.linalg.norm(A, axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        dom.vertex_preimages @ dom.subspace.W1, V, rtol=0, atol=1e-12 * (1 + b.max())
    )
    assert len({tuple(r) for r in dom.vertex_preimages}) == dom.n_vertices


# ---------------------------------------------------------------------------
# Interval (n = 1)
# ---------------------------------------------------------------------------


def test_interval_endpoints_and_preimages():
    sub = _sub(5, 1, seed=1)
    w = sub.W1[:, 0]
    dom = asub.active_domain(sub)
    h = np.sum(np.abs(w))
    assert dom.kind == "interval"
    assert dom.general_position is True
    np.testing.assert_allclose(dom.vertices, [[-h], [h]], rtol=1e-12)
    np.testing.assert_array_equal(dom.vertex_preimages, [-np.sign(w), np.sign(w)])
    np.testing.assert_allclose(dom.vertex_preimages @ w, [-h, h], rtol=1e-12)
    np.testing.assert_array_equal(dom.facet_normals, [[-1.0], [1.0]])
    np.testing.assert_allclose(dom.facet_offsets, [h, h], rtol=1e-12)


def test_interval_zero_entry():
    a = np.array([1.0, -2.0, 0.0, 0.5])
    dom = asub.active_domain(asub.subspace_from_basis(a[:, None], space=asub.Uniform.reference(4)))
    w = dom.subspace.W1[:, 0]
    sgn = np.sign(a) * np.sign(w[1] / a[1])  # the sign rule may flip the direction
    np.testing.assert_array_equal(dom.vertex_preimages, [-sgn, sgn])
    assert dom.vertex_preimages[0, 2] == 0
    np.testing.assert_allclose(
        dom.vertices[:, 0], [-np.sum(np.abs(w)), np.sum(np.abs(w))], rtol=1e-12
    )
    np.testing.assert_allclose(dom.vertex_preimages @ w, dom.vertices[:, 0], rtol=1e-12)
    assert dom.general_position is False


# ---------------------------------------------------------------------------
# Zonotopes against brute force
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("m", [5, 8, 12])
def test_zonotope_against_convex_hull(m, n):
    dom = asub.active_domain(_sub(m, n, seed=10 * m + n))
    assert dom.kind == "zonotope"
    _check_against_brute_force(dom)
    _check_h_representation(dom)
    assert dom.n_vertices == asub.zonotope_vertex_count(m, n)
    assert dom.n_facets == 2 * len(list(itertools.combinations(range(m), n - 1)))
    assert dom.general_position is True


@pytest.mark.parametrize(("m", "n"), [(3, 2), (4, 3), (5, 4)])
def test_zonotope_small_m(m, n):
    dom = asub.active_domain(_sub(m, n, seed=3))
    _check_against_brute_force(dom)
    assert dom.n_vertices == asub.zonotope_vertex_count(m, n)


def test_n2_counterclockwise_order_and_edges():
    dom = asub.active_domain(_sub(7, 2, seed=4))
    V = dom.vertices
    x, y = V[:, 0], V[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area > 0
    E = np.roll(V, -1, axis=0) - V
    cross = E[:, 0] * np.roll(E, -1, axis=0)[:, 1] - E[:, 1] * np.roll(E, -1, axis=0)[:, 0]
    assert np.all(cross > 0)  # strictly convex, counter-clockwise
    # facet k is the edge v_k -> v_{k+1}: both endpoints on it
    for k in range(dom.n_facets):
        for v in (V[k], V[(k + 1) % len(V)]):
            assert abs(dom.facet_normals[k] @ v - dom.facet_offsets[k]) <= 1e-12 * (
                1 + dom.facet_offsets.max()
            )


def test_n2_first_vertex_is_minus_sum_of_reduced_generators():
    sub = _sub(6, 2, seed=8)
    dom = asub.active_domain(sub)
    W1 = sub.W1
    # orient each generator into the upper half-plane; v_0 = -sum of oriented generators
    G = np.where(((W1[:, 1] < 0) | ((W1[:, 1] == 0) & (W1[:, 0] < 0)))[:, None], -W1, W1)
    np.testing.assert_allclose(dom.vertices[0], -G.sum(axis=0), rtol=1e-12)


def test_large_m_n2():
    sub = _sub(1000, 2, seed=1)
    t0 = time.perf_counter()
    dom = asub.active_domain(sub)
    elapsed = time.perf_counter() - t0
    assert dom.n_vertices == 2000
    assert dom.n_facets == 2000
    x, y = dom.vertices[:, 0], dom.vertices[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# Non-general position [C-20]
# ---------------------------------------------------------------------------


def test_repro_c20_zero_row():
    rng = np.random.default_rng(0)
    W1 = np.linalg.qr(rng.normal(size=(4, 2)))[0]
    W1 = np.vstack((W1, np.zeros((1, 2))))
    W1 /= np.linalg.norm(W1, axis=0)
    dom = asub.active_domain(asub.subspace_from_basis(W1, space=asub.Uniform.reference(5)))
    assert dom.n_vertices == 8
    assert len({tuple(r) for r in dom.vertex_preimages}) == 8
    assert np.all(dom.vertex_preimages[:, 4] == 0)
    assert dom.general_position is False
    _check_against_brute_force(dom)
    _check_h_representation(dom)


def test_duplicated_row_n2():
    rng = np.random.default_rng(2)
    B = rng.standard_normal((5, 2))
    B = np.vstack([B, -2.0 * B[1]])  # parallel to row 1
    dom = asub.active_domain(asub.subspace_from_basis(B, space=asub.Uniform.reference(6)))
    assert dom.n_vertices == 2 * (6 - 1)
    assert dom.general_position is False
    _check_against_brute_force(dom)
    # the two parallel inputs move together, with opposite signs
    assert np.all(dom.vertex_preimages[:, 1] == -dom.vertex_preimages[:, 5])


def test_n3_coplanar_rows():
    rng = np.random.default_rng(3)
    B = rng.standard_normal((7, 3))
    plane = rng.standard_normal((2, 3))
    B[:3] = rng.standard_normal((3, 2)) @ plane  # three generators in one plane
    dom = asub.active_domain(asub.subspace_from_basis(B, space=asub.Uniform.reference(7)))
    _check_against_brute_force(dom)
    _check_h_representation(dom)
    assert dom.general_position is False
    assert dom.n_vertices < asub.zonotope_vertex_count(7, 3)


def test_n4_with_zero_and_parallel_rows():
    rng = np.random.default_rng(4)
    B = rng.standard_normal((8, 4))
    B[2] = 0.0
    B[5] = 3.0 * B[0]
    dom = asub.active_domain(asub.subspace_from_basis(B, space=asub.Uniform.reference(8)))
    _check_against_brute_force(dom)
    _check_h_representation(dom)
    assert np.all(dom.vertex_preimages[:, 2] == 0)
    assert dom.general_position is False


def test_n3_vertex_order_is_lexicographic():
    dom = asub.active_domain(_sub(6, 3, seed=6))
    P = dom.vertex_preimages
    keys = [tuple(r) for r in P]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Entry point and physical spaces
# ---------------------------------------------------------------------------


def test_entry_point_basis_matches_active_subspace(uniform_space):
    space = uniform_space(4, seed=3)
    a = np.array([1.0, -0.5, 2.0, 0.25])
    via_basis = asub.active_domain(asub.subspace_from_basis(a[:, None], space=space))
    X = space.sample(10, rng=1)
    grads = np.tile(a, (10, 1))
    via_as = asub.active_domain(asub.active_subspace(space=space, X=X, grads=grads, n=1))
    np.testing.assert_allclose(via_basis.vertices, via_as.vertices, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        via_basis.facet_normals, via_as.facet_normals, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(via_basis.facet_offsets, via_as.facet_offsets, rtol=1e-12)


# ---------------------------------------------------------------------------
# Inverse map: boundary points, outside points, exact chord [D-95]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [3, 4])  # exact interval (d = 1) and hit-and-run (d = 2)
def test_inverse_map_at_vertex(m):
    sub = _sub(m, 2, seed=m)
    dom = asub.active_domain(sub)
    assert dom.general_position
    with pytest.warns(asub.AccuracyWarning) as rec:
        res = asub.inverse_map(sub, Y=dom.vertices[[0, 2]], n_draws=3, rng=1)
    codes = [w.message.code for w in rec]
    assert codes == ["accuracy.degenerate_fiber"]
    assert rec[0].message.details["rows"] == [0, 1]
    assert res.degenerate.tolist() == [True, True]
    expected = np.repeat(dom.vertex_preimages[[0, 2]], 3, axis=0)
    np.testing.assert_allclose(res.X, expected, atol=1e-9)


@pytest.mark.parametrize("m", [3, 4])
def test_inverse_map_outside(m):
    sub = _sub(m, 2, seed=m)
    dom = asub.active_domain(sub)
    Y = np.vstack([0.5 * dom.vertices[1], 1.01 * dom.vertices[1]])
    with pytest.raises(InputError) as e:
        asub.inverse_map(sub, Y=Y, rng=1)
    assert e.value.code == "input.out_of_domain"
    assert e.value.details["row"] == 1


def test_exact_interval_fixed_coordinate_outside():
    # x_3 is fixed by y (w2 has a zero entry): y requiring |x_3| > 1 is outside
    W = np.linalg.qr(np.random.default_rng(1).standard_normal((2, 2)))[0]
    basis = np.zeros((3, 2))
    basis[:2, 0] = W[:, 0]
    basis[2, 1] = 1.0
    sub = asub.subspace_from_basis(basis, space=asub.Uniform.reference(3))
    assert np.min(np.abs(sub.W2[:, 0])) < 1e-12
    y_in = np.array([0.0, 0.0])
    y_out = sub.W1.T @ np.array([0.0, 0.0, 1.2])
    res = asub.inverse_map(sub, Y=y_in[None, :], n_draws=5, rng=0)
    assert np.all(np.abs(res.X) <= 1)
    with pytest.raises(InputError) as e:
        asub.inverse_map(sub, Y=y_out[None, :], rng=0)
    assert e.value.code == "input.out_of_domain"


def test_exact_interval_chord_with_s_outside_cube():
    """The gate Blocker: s = W1 y is not clipped; the chord matches an LP over t."""
    sub = _sub(3, 2, seed=11)
    W1, w2 = sub.W1, sub.W2[:, 0]
    gen = np.random.default_rng(5)
    found = 0
    for _ in range(200):
        x0 = gen.uniform(-1, 1, 3)
        y = W1.T @ x0
        s = W1 @ y
        if np.max(np.abs(s)) <= 1:
            continue
        found += 1
        lims = []
        for sign in (1.0, -1.0):
            r = scipy.optimize.linprog(
                [sign],
                A_ub=np.concatenate([w2, -w2])[:, None],
                b_ub=np.concatenate([1 - s, 1 + s]),
                bounds=[(None, None)],
            )
            lims.append(sign * r.fun)
        res = asub.inverse_map(sub, Y=y[None, :], n_draws=400, rng=found)
        t = (res.X - s) @ w2
        assert np.all(t >= lims[0] - 1e-12)
        assert np.all(t <= lims[1] + 1e-12)
        span = lims[1] - lims[0]
        assert t.min() - lims[0] < 0.05 * span
        assert lims[1] - t.max() < 0.05 * span
        if found == 5:
            break
    assert found == 5
