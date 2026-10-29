"""Unit tests of active_subspaces.geometry (spec 08): API, errors, determinism, serialization."""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import geometry as geo
from active_subspaces.errors import (
    InputError,
    NumericalError,
    SolverError,
    UnsupportedError,
)
from active_subspaces.solvers import ScipyBackend, SolverOutput


def _sub(m, n, seed=0, space=None):
    rng = np.random.default_rng(seed)
    space = asub.Uniform.reference(m) if space is None else space
    return asub.subspace_from_basis(rng.standard_normal((m, n)), space=space)


class _Mixed(asub.Uniform):
    kind = "mixed"


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_all_is_sorted_and_reexported():
    assert geo.__all__ == tuple(sorted(geo.__all__))
    assert set(geo.__all__) == {
        "active_domain",
        "zonotope_vertex_count",
        "in_active_domain",
        "forward_map",
        "inverse_map",
        "ActiveDomain",
        "ActiveCoordinates",
        "InverseMapResult",
    }
    for name in geo.__all__:
        assert getattr(asub, name) is getattr(geo, name)


# ---------------------------------------------------------------------------
# zonotope_vertex_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("m", "n", "expected"),
    [
        (2, 1, 2),
        (3, 2, 6),
        (5, 2, 10),
        (4, 3, 14),
        (30, 3, 872),
        (100, 3, 9902),
        (30, 4, 8180),
        (1000, 2, 2000),
    ],
)
def test_zonotope_vertex_count(m, n, expected):
    out = asub.zonotope_vertex_count(m, n)
    assert out == expected
    assert type(out) is int


def test_zonotope_vertex_count_exact_for_large_m():
    assert asub.zonotope_vertex_count(1000, 4) == 2 * (
        1 + 999 + 999 * 998 // 2 + 999 * 998 * 997 // 6
    )
    assert asub.zonotope_vertex_count(np.int64(5), np.int32(2)) == 10


@pytest.mark.parametrize(("m", "n"), [(1, 1), (3, 0), (3, 3), (3, 2.0), (True, 1), ("4", 2)])
def test_zonotope_vertex_count_bad(m, n):
    with pytest.raises(InputError) as e:
        asub.zonotope_vertex_count(m, n)
    assert e.value.code == "input.parameter"


# ---------------------------------------------------------------------------
# Entry point, dimension and space [D-13, D-85, D-92]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", ["active_domain", "forward_map", "inverse_map"])
def test_not_a_subspace_result(func):
    kwargs = (
        {"X": np.zeros((1, 3))}
        if func == "forward_map"
        else ({"Y": np.zeros((1, 1))} if func == "inverse_map" else {})
    )
    with pytest.raises(InputError) as e:
        getattr(asub, func)(np.eye(3)[:, :1], **kwargs)
    assert e.value.code == "input.parameter"


def test_n_equals_m():
    sub = _sub(3, 3)
    for call in (
        lambda: asub.active_domain(sub),
        lambda: asub.inverse_map(sub, Y=np.zeros((1, 3)), rng=0),
    ):
        with pytest.raises(InputError) as e:
            call()
        assert e.value.code == "input.parameter"
    coords = asub.forward_map(sub, X=np.zeros((4, 3)))
    assert coords.Y.shape == (4, 3)
    assert coords.Z.shape == (4, 0)


def test_unsupported_space_kind():
    sub = asub.subspace_from_basis(np.eye(4)[:, :2], space=_Mixed(-np.ones(4), np.ones(4)))
    with pytest.raises(UnsupportedError) as e:
        asub.active_domain(sub)
    assert e.value.code == "unsupported.space"
    with pytest.raises(UnsupportedError) as e:
        asub.inverse_map(sub, Y=np.zeros((1, 2)), rng=0)
    assert e.value.code == "unsupported.space"
    coords = asub.forward_map(sub, X=np.zeros((2, 4)))
    assert coords.Y.shape == (2, 2)


@pytest.mark.parametrize("tol", [0.0, -1.0, 1.0, np.nan, "x", True])
def test_active_domain_bad_tol(tol):
    with pytest.raises(InputError) as e:
        asub.active_domain(_sub(3, 2), tol=tol)
    assert e.value.code == "input.parameter"


# ---------------------------------------------------------------------------
# ActiveDomain
# ---------------------------------------------------------------------------


def test_gaussian_domain_unbounded(gaussian_space):
    sub = _sub(4, 2, space=gaussian_space(4))
    dom = asub.active_domain(sub)
    assert dom.kind == "unbounded"
    assert dom.vertices is None
    assert dom.facet_normals is None
    assert dom.general_position is None
    assert dom.n_vertices == 0
    assert dom.n_facets == 0
    assert dom.vertex_preimages_physical is None
    assert dom.m == 4
    assert dom.n == 2
    assert asub.in_active_domain(dom, Y=1e6 * np.ones((3, 2))).all()


def test_domain_properties_and_physical_preimages(uniform_space):
    space = uniform_space(5)
    dom = asub.active_domain(_sub(5, 2, space=space))
    assert dom.kind == "zonotope"
    assert dom.n_vertices == 10
    assert dom.n_facets == 10
    P = dom.vertex_preimages_physical
    assert P.shape == (10, 5)
    assert np.all(P >= space.lower)
    assert np.all(P <= space.upper)
    corners = np.where(dom.vertex_preimages > 0, space.upper, space.lower)
    np.testing.assert_allclose(P, corners, rtol=1e-12, atol=0)


def test_domain_roundtrip_and_readonly(assert_roundtrip, assert_readonly, gaussian_space):
    for sub in (_sub(4, 1), _sub(5, 2), _sub(6, 3), _sub(4, 2, space=gaussian_space(4))):
        dom = asub.active_domain(sub)
        assert_roundtrip(dom)
        assert_readonly(dom)


def test_domain_check_rejects_inconsistent():
    dom = asub.active_domain(_sub(4, 2))
    with pytest.raises(AssertionError):
        geo.ActiveDomain(
            subspace=dom.subspace,
            kind="zonotope",
            vertices=dom.vertices,
            vertex_preimages=dom.vertex_preimages,
            facet_normals=dom.facet_normals,
            facet_offsets=0.5 * dom.facet_offsets,
            general_position=True,
            tol=1e-10,
            provenance=dom.provenance,
        )
    with pytest.raises(AssertionError):
        geo.ActiveDomain(
            subspace=dom.subspace,
            kind="interval",
            vertices=dom.vertices,
            vertex_preimages=dom.vertex_preimages,
            facet_normals=dom.facet_normals,
            facet_offsets=dom.facet_offsets,
            general_position=True,
            tol=1e-10,
            provenance=dom.provenance,
        )


def test_domain_check_linear_cost_branch(monkeypatch):
    dom = asub.active_domain(_sub(7, 3, seed=2))
    monkeypatch.setattr(geo, "_PAIRWISE_CHECK", 0)
    assert asub.from_dict(dom.to_dict()) == dom
    with pytest.raises(AssertionError):
        geo.ActiveDomain(
            subspace=dom.subspace,
            kind="zonotope",
            vertices=dom.vertices,
            vertex_preimages=dom.vertex_preimages,
            facet_normals=dom.facet_normals,
            facet_offsets=0.9 * dom.facet_offsets,
            general_position=True,
            tol=1e-10,
            provenance=dom.provenance,
        )


def test_active_domain_bitwise_stable(monkeypatch):
    sub = _sub(9, 4, seed=5)
    a = asub.active_domain(sub)
    b = asub.active_domain(sub)
    assert a == b
    monkeypatch.setattr(geo, "_SUBSET_BATCH", 3)
    c = asub.active_domain(sub)
    assert a == c


def test_active_domain_no_randomness():
    state = np.random.get_state()  # noqa: NPY002 (checks the legacy global state, spec 05 §8)
    asub.active_domain(_sub(6, 3))
    after = np.random.get_state()  # noqa: NPY002 (checks the legacy global state, spec 05 §8)
    assert np.array_equal(state[1], after[1])


# ---------------------------------------------------------------------------
# in_active_domain
# ---------------------------------------------------------------------------


def test_in_active_domain_vertices_and_scaled():
    for m, n in ((4, 1), (6, 2), (6, 3), (7, 4)):
        dom = asub.active_domain(_sub(m, n, seed=m + n))
        assert asub.in_active_domain(dom, Y=dom.vertices).all()
        assert not asub.in_active_domain(dom, Y=1.001 * dom.vertices).any()
        assert asub.in_active_domain(dom, Y=1.001 * dom.vertices, tol=0.01).all()
        assert asub.in_active_domain(dom, Y=np.zeros((1, n))).all()


def test_in_active_domain_1d_and_errors():
    dom = asub.active_domain(_sub(4, 1))
    h = dom.vertices[1, 0]
    out = asub.in_active_domain(dom, Y=np.array([0.0, 0.9 * h, 1.1 * h]))
    assert out.tolist() == [True, True, False]
    assert out.dtype == np.bool_
    with pytest.raises(InputError) as e:
        asub.in_active_domain(dom.subspace, Y=np.zeros((1, 1)))
    assert e.value.code == "input.parameter"
    with pytest.raises(InputError) as e:
        asub.in_active_domain(dom, Y=np.zeros((1, 2)))
    assert e.value.code == "input.shape"
    with pytest.raises(InputError) as e:
        asub.in_active_domain(dom, Y=np.array([[np.nan]]))
    assert e.value.code == "input.nonfinite"
    with pytest.raises(InputError) as e:
        asub.in_active_domain(dom, Y=np.zeros((1, 1)), tol=-1.0)
    assert e.value.code == "input.parameter"


# ---------------------------------------------------------------------------
# forward_map
# ---------------------------------------------------------------------------


def test_forward_map(uniform_space, gaussian_space):
    for space in (uniform_space(5), gaussian_space(5)):
        sub = _sub(5, 2, space=space)
        X = space.sample(20, rng=1)
        coords = asub.forward_map(sub, X=X)
        assert isinstance(coords, asub.ActiveCoordinates)
        assert isinstance(coords, tuple)
        x = space.to_normalized(X)
        np.testing.assert_allclose(np.hstack(coords), x @ sub.W, rtol=1e-12, atol=1e-12)
        assert coords.Y.flags.writeable
        assert coords.Z.flags.writeable
        np.testing.assert_allclose(coords.Y @ sub.W1.T + coords.Z @ sub.W2.T, x, atol=1e-12)


def test_forward_map_errors(uniform_space):
    space = uniform_space(3)
    sub = _sub(3, 1, space=space)
    with pytest.raises(InputError) as e:
        asub.forward_map(sub, X=space.upper[None, :] + 1.0)
    assert e.value.code == "input.out_of_space"
    with pytest.raises(InputError) as e:
        asub.forward_map(sub, X=np.zeros((2, 4)))
    assert e.value.code == "input.shape"


# ---------------------------------------------------------------------------
# inverse_map: validation, results, determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_draws": 0},
        {"n_draws": 1.5},
        {"burn_in": -1},
        {"thin": 0},
        {"thin": True},
        {"boundary_tol": 0.0},
        {"backend": "cvxpy"},
        {"rng": -1},
    ],
)
def test_inverse_map_bad_parameters(kwargs):
    with pytest.raises(InputError) as e:
        asub.inverse_map(_sub(5, 2), Y=np.zeros((2, 2)), **{"rng": 0, **kwargs})
    assert e.value.code == "input.parameter"


def test_inverse_map_bad_backend_before_any_draw():
    gen = np.random.default_rng(3)
    state = gen.bit_generator.state
    with pytest.raises(InputError) as e:
        asub.inverse_map(_sub(5, 2), Y=np.zeros((1, 2)), backend="other", rng=gen)
    assert e.value.code == "input.parameter"
    assert gen.bit_generator.state == state


@pytest.mark.parametrize(
    ("Y", "code"),
    [
        (np.zeros((1, 3)), "input.shape"),
        (np.zeros((0, 2)), "input.shape"),
        (np.array([[np.inf, 0.0]]), "input.nonfinite"),
    ],
)
def test_inverse_map_bad_Y(Y, code):
    with pytest.raises(InputError) as e:
        asub.inverse_map(_sub(5, 2), Y=Y, rng=0)
    assert e.value.code == code


@pytest.mark.parametrize(
    ("m", "n", "sampler"),
    [
        (4, 3, "exact_interval"),
        (5, 2, "hit_and_run"),
        (4, 1, "hit_and_run"),
        (2, 1, "exact_interval"),
    ],
)
def test_inverse_map_uniform_result(
    m, n, sampler, uniform_space, assert_roundtrip, assert_readonly
):
    space = uniform_space(m)
    sub = _sub(m, n, seed=m, space=space)
    Xs = space.sample(4, rng=2)
    Y = asub.forward_map(sub, X=Xs).Y
    res = asub.inverse_map(sub, Y=Y, n_draws=3, rng=11)
    assert res.sampler == sampler
    assert res.X.shape == (12, m)
    assert res.Y.shape == (4, n)
    assert res.index.dtype == np.int64
    assert res.index.tolist() == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
    ]
    assert not res.degenerate.any()
    d = m - n
    if sampler == "hit_and_run":
        assert (res.burn_in, res.thin) == (10 * d, d)
    else:
        assert res.burn_in is None
        assert res.thin is None
    assert np.all(np.abs(res.X) <= 1.0)
    back = asub.forward_map(sub, X=res.X_physical).Y
    np.testing.assert_allclose(back, Y[res.index], atol=1e-10)
    Xp = res.X_physical
    assert np.all(Xp >= space.lower)
    assert np.all(Xp <= space.upper)
    np.testing.assert_allclose(res.Z, res.X @ sub.W2)
    assert res.provenance.rng is not None
    params = res.provenance.parameters
    assert params["n_draws"] == 3
    assert params["backend"] == "scipy"
    assert "Y" in params
    assert_roundtrip(res)
    assert_readonly(res)


def test_inverse_map_gaussian_result(gaussian_space, assert_roundtrip):
    space = gaussian_space(4)
    sub = _sub(4, 2, space=space)
    Y = np.array([[0.0, 0.0], [5.0, -3.0]])
    res = asub.inverse_map(sub, Y=Y, n_draws=4, burn_in=3, thin=2, rng=1)
    assert res.sampler == "gaussian"
    assert res.burn_in is None
    assert res.thin is None
    np.testing.assert_allclose(res.X @ sub.W1, Y[res.index], atol=1e-12)
    np.testing.assert_allclose(asub.forward_map(sub, X=res.X_physical).Y, Y[res.index], atol=1e-10)
    assert_roundtrip(res)


def test_inverse_map_1d_Y():
    sub = _sub(3, 1)
    a = asub.inverse_map(sub, Y=np.array([0.1, -0.2]), n_draws=2, rng=4)
    b = asub.inverse_map(sub, Y=np.array([[0.1], [-0.2]]), n_draws=2, rng=4)
    assert a == b
    assert a.Y.shape == (2, 1)


def test_inverse_map_same_seed_bitwise():
    for m, n in ((4, 3), (6, 2)):
        sub = _sub(m, n, seed=1)
        Y = 0.4 * asub.active_domain(sub).vertices[:3]
        a = asub.inverse_map(sub, Y=Y, n_draws=5, rng=123)
        b = asub.inverse_map(sub, Y=Y, n_draws=5, rng=123)
        assert a == b
        assert np.array_equal(a.X, b.X)
        c = asub.inverse_map(sub, Y=Y, n_draws=5, rng=124)
        assert not np.array_equal(a.X, c.X)


def test_inverse_map_stream_layout():
    """Point k uses children[k]; hit-and-run draws dirs/steps from children[k].spawn(2)."""
    sub = _sub(4, 3, seed=2)
    Y = 0.3 * asub.active_domain(sub).vertices[:2]
    res = asub.inverse_map(sub, Y=Y, n_draws=4, rng=9)
    children = np.random.default_rng(9).spawn(2)
    W1, w2 = sub.W1, sub.W2[:, 0]
    for k in range(2):
        s = W1 @ Y[k]
        a, c = (-1 - s) / w2, (1 - s) / w2
        lo, hi = np.max(np.minimum(a, c)), np.min(np.maximum(a, c))
        t = lo + children[k].random(4) * (hi - lo)
        np.testing.assert_allclose(
            res.X[4 * k : 4 * k + 4], np.clip(s + t[:, None] * w2, -1, 1), rtol=0, atol=1e-14
        )


@pytest.mark.parametrize("chunk", [1, 7, None])
def test_hit_and_run_chunking_bitwise(monkeypatch, chunk):
    sub = _sub(6, 2, seed=3)
    Y = 0.5 * asub.active_domain(sub).vertices[::3]
    ref = asub.inverse_map(sub, Y=Y, n_draws=3, rng=5)
    T = ref.burn_in + 3 * ref.thin
    monkeypatch.setattr(geo, "_HR_CHUNK", T if chunk is None else chunk)
    assert np.array_equal(asub.inverse_map(sub, Y=Y, n_draws=3, rng=5).X, ref.X)


@pytest.mark.parametrize("block", [1, 2])
def test_hit_and_run_block_size_bitwise(monkeypatch, block):
    sub = _sub(6, 2, seed=3)
    Y = 0.5 * asub.active_domain(sub).vertices[::2]
    monkeypatch.setattr(geo, "_HR_BLOCK", Y.shape[0])
    ref = asub.inverse_map(sub, Y=Y, n_draws=2, rng=5)
    monkeypatch.setattr(geo, "_HR_BLOCK", block)
    assert np.array_equal(asub.inverse_map(sub, Y=Y, n_draws=2, rng=5).X, ref.X)


def test_correlated_draws_warning():
    sub = _sub(5, 2)
    with pytest.warns(asub.AccuracyWarning) as rec:
        res = asub.inverse_map(sub, Y=np.zeros((1, 2)), n_draws=3, thin=1, rng=0)
    assert [w.message.code for w in rec] == ["accuracy.correlated_draws"]
    assert res.thin == 1
    asub.inverse_map(sub, Y=np.zeros((1, 2)), n_draws=1, thin=1, rng=0)  # one draw: no warning


# ---------------------------------------------------------------------------
# Backends [D-28]
# ---------------------------------------------------------------------------


class _Mock:
    name = "mock"

    def __init__(self, status=None):
        self.status = status
        self.calls = 0

    def lp(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        self.calls += 1
        if self.status is None:
            return ScipyBackend().lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
        return SolverOutput(
            x=None, objective=None, status=self.status, message="mock", n_iter=None
        )

    def qp(self, *args, **kwargs):
        raise AssertionError("geometry never solves a QP")


def test_backend_receives_every_lp():
    sub = _sub(5, 2)
    mock = _Mock()
    Y = 0.3 * asub.active_domain(sub).vertices[:4]
    res = asub.inverse_map(sub, Y=Y, backend=mock, rng=0)
    assert mock.calls == 4
    assert res.provenance.parameters["backend"] == "mock"
    assert res == asub.inverse_map(sub, Y=Y, rng=0)


def test_backend_infeasible_and_failed():
    sub = _sub(5, 2)
    with pytest.raises(InputError) as e:
        asub.inverse_map(sub, Y=np.zeros((1, 2)), backend=_Mock("infeasible"), rng=0)
    assert e.value.code == "input.out_of_domain"
    assert e.value.details["row"] == 0
    assert isinstance(e.value.__cause__, SolverError)
    assert e.value.__cause__.code == "solver.infeasible"
    with pytest.raises(SolverError) as e:
        asub.inverse_map(sub, Y=np.zeros((1, 2)), backend=_Mock("failed"), rng=0)
    assert e.value.code == "solver.failed"


def test_backend_not_called_for_other_samplers():
    mock = _Mock("failed")
    asub.inverse_map(_sub(4, 3), Y=np.zeros((1, 3)), backend=mock, rng=0)
    assert mock.calls == 0


def test_infeasible_draw_check(monkeypatch):
    sub = _sub(5, 2)
    real = geo._walk_block
    monkeypatch.setattr(geo, "_walk_block", lambda *a: 1.5 * real(*a) + 0.5)
    with pytest.raises(NumericalError) as e:
        asub.inverse_map(sub, Y=0.2 * np.ones((1, 2)), n_draws=2, rng=0)
    assert e.value.code == "numerical.infeasible_draw"


def test_hit_and_run_repeated_calls_bitwise():
    """Repeated calls with one seed agree bitwise on a long chain (m - n = 9).

    The Chebyshev start once used a BLAS gemv whose rounding depends on memory alignment
    (Accelerate); the chain amplified it to ~1e-7 on alternate calls [D-15].
    """
    wing = asub.test_functions.builtin_models().get("wing_weight")
    X = wing.space.sample(300, rng=0)
    sub = asub.active_subspace(grads=asub.evaluate_grads(wing, X=X), space=wing.space, n=1)
    Y = asub.active_quadrature_rule(asub.active_domain(sub), n_nodes=8).nodes
    first = asub.inverse_map(sub, Y=Y, n_draws=50, rng=1).X
    for _ in range(5):
        assert np.array_equal(asub.inverse_map(sub, Y=Y, n_draws=50, rng=1).X, first)
