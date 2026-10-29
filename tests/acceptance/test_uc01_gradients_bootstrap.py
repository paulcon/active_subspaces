"""UC-01: active subspace from model gradients, with bootstrap and EVG (spec 01 §5; spec 17
V01.1–V01.9).

KA-1 on ``U5`` and ``G5``, ``κ = (2, 1, 0.1, 0.05, 0.01)``. Public API only.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import active_subspaces as asub
from helpers.metrics import eigvec_distances, subspace_distance
from helpers.tolerances import assert_eigenvalues_close
from helpers.usecases import KAPPA, HandMap, ka1, space, uc01_mc, uc01_model

Q, K = ka1(5, KAPPA)
NODE = {"U5": 1 / np.sqrt(3), "G5": 1.0}  # 2-point Gauss–Legendre / Gauss–Hermite nodes
GAMMA = {"U5": 1 / 3, "G5": 1.0}  # reference variance


def _exact(name: str, **kw) -> asub.SubspaceResult:
    sp, model = space(name), uc01_model(name)
    nodes = np.array(list(itertools.product([-NODE[name], NODE[name]], repeat=5)))
    Gq = asub.evaluate_grads(model, X=sp.from_normalized(nodes))
    return asub.active_subspace(grads=Gq, space=sp, quad_weights=np.full(32, 1 / 32), **kw)


@pytest.mark.parametrize("name", ["U5", "G5"])
def test_exact_rule_eigenpairs(name):
    """V01.1: eigenvalues κ²γ to F12/λ_1; eigenvectors q_i to 1e-10 (Davis–Kahan, spec 17)."""
    sub = _exact(name)
    assert_eigenvalues_close(sub.eigenvalues, KAPPA**2 * GAMMA[name], rtol=1e-12)
    assert np.all(eigvec_distances(sub.eigenvectors, Q) <= 1e-10)


@pytest.mark.parametrize("name", ["U5", "G5"])
def test_exact_rule_dimension(name):
    """V01.2: EVG linear gap gives n = 1 [D-79]; EVG(gap="log") gives n = 2 [D-38]."""
    assert _exact(name).n == 1
    assert _exact(name, dimension_rule=asub.subspaces.EVG(gap="log")).n == 2


@pytest.mark.parametrize("name", ["U5", "G5"])
def test_exact_rule_physical_view(name):
    """V01.3: physical view spans A^{-T} q_1 [D-11]; to 1e-10."""
    sub = _exact(name)
    target = np.linalg.solve(HandMap.of(name).A.T, Q[:, 0])
    assert subspace_distance(sub.physical_directions(), target) <= 1e-10
    assert subspace_distance(space(name).directions_to_physical(sub.W1), target) <= 1e-10


def test_mc_shapes_and_dimension(stat_rng):
    """V01.4: replicate shapes (200, 5) and (200, 4) [D-30]; n == 1."""
    *_, sub = uc01_mc(stat_rng)
    assert sub.eigenvalues.shape == (5,)
    assert sub.eigenvectors.shape == (5, 5)
    assert sub.bootstrap_eigenvalues.shape == (200, 5)
    assert sub.bootstrap_distances.shape == (200, 4)
    assert sub.n == 1


def test_mc_subspace(stat_rng):
    """V01.5: T = dist(Ŵ_1, q_1), M = 1000. Construction 6 (S(pot)), calibrated bound 0.15
    (spec 17 §5.4).
    """
    *_, sub = uc01_mc(stat_rng)
    assert subspace_distance(sub.W1, Q[:, 0]) <= 0.15


def test_mc_lambda1(stat_rng):
    """V01.6: T = |λ̂_1 − 4/3| / sd(bootstrap λ_1). Construction 1, z = 6 >= t_199 = 5.05."""
    *_, sub = uc01_mc(stat_rng)
    sd = np.std(sub.bootstrap_eigenvalues[:, 0], ddof=1)
    assert abs(sub.eigenvalues[0] - 4 / 3) <= 6 * sd


def test_intervals(stat_rng):
    """V01.7: percentile interval ``lo <= hi``; ``kind="minmax"`` equals the replicate column
    min/max [N-08].
    """
    *_, sub = uc01_mc(stat_rng)
    lo, hi = asub.bootstrap_interval(sub.bootstrap_eigenvalues, kind="percentile", level=0.95)
    assert lo.shape == hi.shape == (5,)
    assert np.all(lo <= hi)
    mm = asub.bootstrap_interval(sub.bootstrap_eigenvalues, kind="minmax")
    assert np.array_equal(mm[0], sub.bootstrap_eigenvalues.min(axis=0))
    assert np.array_equal(mm[1], sub.bootstrap_eigenvalues.max(axis=0))


def test_psd_eigenvalues(stat_rng):
    """V01.8: AS eigenvalues >= −1e-14 λ_1 [D-31]."""
    *_, sub = uc01_mc(stat_rng)
    for s in (sub, _exact("U5"), _exact("G5")):
        assert np.all(s.eigenvalues >= -1e-14 * s.eigenvalues[0])


def test_determinism(stat_rng):
    """V01.9: same seeds give == results; an int seed and ``default_rng`` of it give == results
    [D-15]; the legacy
    global RNG is never touched (autouse check) [R-03]."""
    _, _, G, sub = uc01_mc(stat_rng)
    assert uc01_mc(stat_rng)[3] == sub
    seed = int(stat_rng("UC-01/bootstrap").entropy)
    a = asub.active_subspace(grads=G, space=space("U5"), nboot=200, rng=seed)
    b = asub.active_subspace(
        grads=G, space=space("U5"), nboot=200, rng=np.random.default_rng(seed)
    )
    assert a == b == sub
