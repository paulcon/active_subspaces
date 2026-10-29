"""UC-02: a data set of gradients with several outputs (spec 01 §5; spec 17 V01.10–V01.15).

A synthetic as-data-set in ``tmp_path`` on ``R7``: two ridges along ``d_j = A^{-1} c_j`` (KA-2,
KA-8).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import plotting
from helpers.metrics import subspace_distance
from helpers.usecases import R7, HandMap, expect_warning, space

IN_LABELS = ("Mach", "alt", "rho", "aoa", "u1", "u2", "u3")
OUT_LABELS = ("heat", "drag")
C1 = np.array([1, 0.5, 0, -0.3, 0, 0.1, 0]) / np.linalg.norm([1, 0.5, 0, -0.3, 0, 0.1, 0])
C2 = np.array([0, 0, 1, 0, 0.5, 0, -1]) / np.linalg.norm([0, 0, 1, 0, 0.5, 0, -1])
C = (C1, C2)
HM = HandMap.of("R7")


def _write_dataset(d, stat_rng) -> None:
    X = space("R7").sample(300, rng=stat_rng("UC-02/sample"))
    x = HM.x(X)
    t1, t2 = x @ C1, x @ C2
    F = np.column_stack([np.exp(0.5 * t1), t2**2 + t2])
    G = np.hstack(
        [HM.grad(np.outer(0.5 * np.exp(0.5 * t1), C1)), HM.grad(np.outer(2 * t2 + 1, C2))]
    )
    for fname, arr in (("inputs", X), ("outputs", F), ("gradients", G)):
        np.savetxt(d / f"{fname}.txt", arr, delimiter=",")


@pytest.fixture
def dataset(tmp_path, stat_rng):
    _write_dataset(tmp_path, stat_rng)
    X = np.loadtxt(tmp_path / "inputs.txt", delimiter=",")
    F = np.loadtxt(tmp_path / "outputs.txt", delimiter=",", ndmin=2)
    G = np.loadtxt(tmp_path / "gradients.txt", delimiter=",")
    sp = asub.Uniform(R7["lower"], R7["upper"], names=IN_LABELS)
    return X, F, G, sp


def _per_output(dataset, stat_rng):
    _, _, G, sp = dataset
    m = sp.dim
    return [
        asub.active_subspace(
            grads=G[:, j * m : (j + 1) * m],
            space=sp,
            nboot=500,
            rng=stat_rng(f"UC-02/bootstrap/{j}"),
            qoi=label,
        )
        for j, label in enumerate(OUT_LABELS)
    ]


def test_per_output_subspace(dataset, stat_rng):
    """V01.10: n == 1; dist(W_1, c_j) <= 1e-10 (F10); λ̂_2..7 <= 1e-12 λ̂_1 (exact rank one,
    KA-2, KA-8).
    """
    for j, sub in enumerate(_per_output(dataset, stat_rng)):
        assert sub.n == 1
        assert subspace_distance(sub.W1, C[j]) <= 1e-10
        assert np.all(np.abs(sub.eigenvalues[1:]) <= 1e-12 * sub.eigenvalues[0])


def test_physical_view(dataset, stat_rng):
    """V01.11: ``directions_to_physical(W_1)`` spans d_j = A^{-1} c_j [D-11]; F10."""
    sp = dataset[3]
    for j, sub in enumerate(_per_output(dataset, stat_rng)):
        d_j = np.linalg.solve(HM.A, C[j])
        assert subspace_distance(sp.directions_to_physical(sub.W1), d_j) <= 1e-10


def test_labels(dataset, stat_rng):
    """V01.12: ``sub.qoi`` is the output label [D-69]; eigenvector tick labels are the input
    names. Plots run.
    """
    X, F, _, _ = dataset
    for j, (label, sub) in enumerate(zip(OUT_LABELS, _per_output(dataset, stat_rng), strict=True)):
        assert sub.qoi == label
        plotting.plot_eigenvalues(sub)
        _, axes = plotting.plot_eigenvectors(sub)
        assert [t.get_text() for t in axes[0].get_xticklabels()] == list(IN_LABELS)
        plotting.plot_sufficient_summary(asub.with_dimension(sub, 2), X=X, f=F[:, j])


def test_values_only_ols(dataset):
    """V01.13: OLS from values only: n == 1; T = dist(W_1, c_1), construction 6, calibrated bound
    0.1.

    OLS on a uniform space issues ``validity.assumption`` by spec 07 §3.5 [D-41] (spec 01 lists no
    warning).
    """
    X, F, _, sp = dataset
    with expect_warning(asub.ValidityWarning, "validity.assumption"):
        sub_v = asub.active_subspace(X=X, f=F[:, 0], space=sp, estimator="ols", qoi=OUT_LABELS[0])
    assert sub_v.n == 1
    assert subspace_distance(sub_v.W1, C1) <= 0.1


def test_vector_qoi_and_shape_errors(dataset):
    """V01.14: all outputs at once: ``f=F`` (k=2) -> ``input.vector_qoi`` [D-08]; ``grads=G``
    (M, km) -> ``input.shape``.
    """
    X, F, G, sp = dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asub.ValidityWarning)
        with pytest.raises(asub.InputError) as e:
            asub.active_subspace(X=X, f=F, space=sp, estimator="ols")
    assert e.value.code == "input.vector_qoi"
    with pytest.raises(asub.InputError) as e:
        asub.active_subspace(grads=G, space=sp)
    assert e.value.code == "input.shape"


def test_wrong_bounds(dataset):
    """V01.15: a point outside the declared bounds -> ``input.out_of_space`` naming the row
    [A-01], [C-05].
    """
    X, F, _, _ = dataset
    upper = list(R7["upper"])
    upper[0] = 2.43  # typo for 24.3
    with pytest.raises(asub.InputError):
        asub.Uniform(R7["lower"], upper)  # lower > upper: rejected at construction
    upper = list(R7["upper"])
    upper[2] = 0.0019  # typo for 0.002: some rows fall outside
    sp_bad = asub.Uniform(R7["lower"], upper, names=IN_LABELS)
    row = int(np.flatnonzero(X[:, 2] > 0.0019)[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asub.ValidityWarning)
        with pytest.raises(asub.InputError) as e:
            asub.active_subspace(X=X, f=F[:, 0], space=sp_bad, estimator="ols")
    assert e.value.code == "input.out_of_space"
    assert e.value.details["row"] == row
