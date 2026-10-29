"""Tests of the shared test helpers (spec 17 §8: metrics, tolerances, statistics, seeds)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import active_subspaces as asub
from helpers import isolation, metrics, seeds, stats, tolerances


@pytest.mark.parametrize("theta", [1e-12, 1e-6, 0.3, np.pi / 2])
def test_planted_angles(theta):
    m = 6
    e1, e2 = np.eye(m)[:, 0], np.eye(m)[:, 1]
    v = np.cos(theta) * e1 + np.sin(theta) * e2

    def tol(x):
        return 1e-15 + 1e-14 * abs(x)

    assert abs(metrics.subspace_distance(e1, v) - np.sin(theta)) <= tol(np.sin(theta))
    ang = metrics.principal_angles(e1, v)
    assert ang.shape == (1,)
    assert abs(ang[0] - theta) <= tol(theta)


def test_distance_invariances_and_dimensions():
    gen = np.random.default_rng(0)
    A = gen.standard_normal((7, 3))
    R = np.linalg.qr(gen.standard_normal((3, 3)))[0]
    assert metrics.subspace_distance(A, -A @ R) <= 1e-14
    assert metrics.subspace_distance(A, A[:, :2]) == 1.0
    B = gen.standard_normal((7, 3))
    ang = metrics.principal_angles(A, B)
    assert np.all(np.diff(ang) >= 0)
    assert abs(metrics.subspace_distance(A, B) - np.sin(ang[-1])) <= 1e-14
    assert metrics.principal_angles(A[:, :2], B).shape == (2,)


def test_rank_deficient_fails():
    A = np.ones((5, 2))
    with pytest.raises(AssertionError, match="rank-deficient"):
        metrics.orth(A)


def test_eigvec_distances_sign_free():
    gen = np.random.default_rng(1)
    W = np.linalg.qr(gen.standard_normal((5, 5)))[0]
    assert np.all(metrics.eigvec_distances(-W * 3.0, W) <= 1e-15)
    d = metrics.eigvec_distances(W[:, [1, 0]], W[:, :2])
    assert np.allclose(d, 1.0)


def test_assert_close_boundary_and_nan():
    d = np.array([1.0, -2.0, 0.0])
    a = d.copy()
    a[0] = d[0] + 2e-12
    tolerances.assert_close(a, d, rtol=1e-12)  # scale 2: bound 2e-12
    a[0] = np.nextafter(d[0] + 2e-12 * 1.0000001, np.inf)
    with pytest.raises(AssertionError):
        tolerances.assert_close(a, d, rtol=1e-12)
    tolerances.assert_close([np.nan, 1.0], [np.nan, 1.0], rtol=0)
    with pytest.raises(AssertionError, match="NaN"):
        tolerances.assert_close([np.nan, 1.0], [0.0, 1.0], rtol=1)
    with pytest.raises(AssertionError, match="dtype"):
        tolerances.assert_close(np.ones(2, dtype=np.float32), np.ones(2), rtol=1)


def test_eig_and_lstsq_tolerances():
    C = np.diag([4.0, 1.0, 0.0])
    assert tolerances.eig_tolerance(C, 0) == 1e-12
    assert tolerances.lstsq_tolerance(np.eye(3)) == 1e-12
    with pytest.raises(AssertionError, match="condition"):
        tolerances.lstsq_tolerance(np.diag([1.0, 1e-9]), kappa_max=1e8)


def test_statistical_bounds():
    assert stats.z_bound(1) == 5.0
    assert stats.z_bound(2) == 5.03
    assert stats.z_bound(10) == 5.33
    assert stats.t_bound(19) == 7.07
    assert stats.binomial_bounds(200, 0.90, 0.99) == (157, 200)
    mean_b, (lo, hi) = stats.standardized_moment_bounds(200)
    # §3.3 item 5 formula (alpha/4 per tail): [0.5746, 1.5875]; the printed [0.58, 1.57] uses
    # alpha/2
    assert round(mean_b, 3) == 0.356
    assert round(lo, 4) == 0.5746
    assert round(hi, 4) == 1.5875
    stats.assert_gof(1e-6)
    with pytest.raises(AssertionError):
        stats.assert_gof(9e-7)


def test_pot_quantile_recovers_exponential():
    T = np.random.default_rng(
        seeds.check_seed("tests/unit/test_helpers.py::test_pot_quantile_recovers_exponential")
    ).exponential(size=100_000)
    q, se = stats.pot_quantile(T)
    assert abs(q - np.log(1e6)) <= 3 * se


def test_check_seed_table():
    assert (
        seeds.check_seed("tests/statistical/test_gradfree.py::test_ridge_recovery", "ridge")
        == 116652295
    )
    assert seeds.check_seed("a::b[1-2]", "x") == seeds.check_seed("a::b", "x")
    assert seeds.stat_seed("any", "UC-01/sample") == 101
    assert seeds.stat_seed("any", "UC-02/bootstrap/1") == 1


def test_random_mode_streams():
    a = seeds.stat_seed("t::x", "a", entropy=12345)
    assert a == seeds.stat_seed("t::x", "a", entropy=12345)
    assert a != seeds.stat_seed("t::x", "a", entropy=12346)
    assert a != seeds.stat_seed("t::x", "b", entropy=12345)
    assert seeds.stat_seed("t::x", "UC-01/sample", entropy=2**127 + 5) != 101


def test_stat_rng_fixture_fixed_mode(stat_rng, request):
    ss = stat_rng("data")
    assert isinstance(ss, np.random.SeedSequence)
    if request.config.getoption("--stat-seeds") == "fixed":
        assert ss.entropy == seeds.check_seed(request.node.nodeid, "data")
        assert stat_rng("UC-03/sir").entropy == 303
    assert stat_rng("data") is not ss
    assert stat_rng("data").entropy == ss.entropy


def test_factories_cover_registry_and_roundtrip(assert_roundtrip, assert_readonly):
    from helpers import factories

    registered = {c.__name__ for c in _registered_result_types()}
    assert registered <= set(factories.FACTORIES), registered - set(factories.FACTORIES)
    for name in sorted(factories.FACTORIES):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", asub.ExperimentalWarning)
            r = factories.make(name)
            assert type(r).__name__ == name
            assert_roundtrip(r)
        assert_readonly(r)


def _registered_result_types():
    seen, todo = set(), [asub.Result]
    while todo:
        for sub in todo.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                todo.append(sub)
    return {
        c for c in seen if getattr(c, "TYPE", None) and c.__module__.startswith("active_subspaces")
    }


def test_each_test_runs_in_an_empty_working_directory(tmp_path):
    """Spec 15 §5.4 [R-02]: the autouse fixture gives a fresh empty cwd, separate from tmp_path."""
    cwd = Path.cwd()
    assert cwd != tmp_path
    assert list(cwd.iterdir()) == []


def test_empty_working_directory_fails_on_a_written_file(tmp_path):
    with pytest.raises(AssertionError, match=r"stray\.txt"):
        with isolation.empty_working_directory(tmp_path):
            Path("stray.txt").write_text("x")
    assert Path.cwd() != tmp_path
    (tmp_path / "stray.txt").unlink()
    with isolation.empty_working_directory(tmp_path):
        pass


def test_figures_closed():
    """Spec 15 §5.4: pyplot figures opened in a test are closed; earlier figures are kept."""
    plt = pytest.importorskip("matplotlib.pyplot")
    keep = plt.figure()
    try:
        with isolation.figures_closed():
            fig = plt.figure()
            assert fig.number in plt.get_fignums()
        assert fig.number not in plt.get_fignums()
        assert keep.number in plt.get_fignums()
    finally:
        plt.close(keep)
