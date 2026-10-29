"""UC-05: diagnostic plots, sufficient summary, and a new n (spec 01 §5; spec 17 V01.26–V01.29).

``sub``, ``X``, ``f`` from the UC-01 Monte Carlo run.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import plotting
from helpers.usecases import HandMap, uc01_mc


@pytest.fixture
def run(stat_rng):
    X, f, _, sub = uc01_mc(stat_rng)
    return X, f, sub


def test_eigenvalue_plot(run):
    """V01.26: log y-scale; the line's y-data equal |λ̂| bitwise [D-31]. The subspace-error
    plot runs.
    """
    _, _, sub = run
    _, ax = plotting.plot_eigenvalues(sub)
    assert ax.get_yscale() == "log"
    line = ax.lines[0]
    assert np.array_equal(line.get_xdata(), np.arange(1, 6))
    assert np.array_equal(line.get_ydata(), np.abs(sub.eigenvalues))
    _, ax = plotting.plot_subspace_errors(sub)
    assert ax.get_yscale() == "log"


def test_summary_1d(run):
    """V01.27: offsets column 0 equals x(ξ_i)^T ŵ_1 with x by hand (1e-12); column 1 equals f
    bitwise.
    """
    X, f, sub = run
    _, ax = plotting.plot_sufficient_summary(sub, X=X, f=f)
    off = ax.collections[0].get_offsets()
    y = HandMap.of("U5").x(X) @ sub.W1[:, 0]
    assert np.max(np.abs(off[:, 0] - y)) <= 1e-12 * np.max(np.abs(y))
    assert np.array_equal(np.asarray(off[:, 1]), f)


def test_summary_2d(run):
    """V01.27: two-column offsets equal x(ξ_i)^T Ŵ_1 (1e-12); the colour array equals f bitwise."""
    X, f, sub = run
    sub2 = asub.with_dimension(sub, 2)
    _, ax = plotting.plot_sufficient_summary(sub2, X=X, f=f)
    sc = ax.collections[0]
    Y = HandMap.of("U5").x(X) @ sub2.W1
    assert np.max(np.abs(sc.get_offsets() - Y)) <= 1e-12 * np.max(np.abs(Y))
    assert np.array_equal(np.asarray(sc.get_array()), f)


def test_new_dimension(run):
    """V01.28 [A-05], [R-09]: ``with_dimension(sub, 2)`` returns a new result; ``sub`` is
    unchanged.
    """
    _, _, sub = run
    sub2 = asub.with_dimension(sub, 2)
    assert sub2.n == 2
    assert sub.n == 1
    assert np.array_equal(sub2.eigenvalues, sub.eigenvalues)
    assert np.array_equal(sub2.W, sub.W)


def test_labels_bytes_side_effects(run, tmp_path, monkeypatch):
    """V01.29 [R-02], [D-06]: tick labels are the input names; PNG bytes; no file written, no
    output,
    and importing ``active_subspaces.plotting`` does not import ``matplotlib.pyplot``."""
    X, f, sub = run
    monkeypatch.chdir(tmp_path)
    _, axes = plotting.plot_eigenvectors(sub, indices=[0, 1])
    assert len(axes) == 2
    assert [t.get_text() for t in axes[-1].get_xticklabels()] == list(
        sub.space.names
    )  # bottom panel (spec 13)
    fig2, _ = plotting.plot_sufficient_summary(asub.with_dimension(sub, 2), X=X, f=f)
    png = plotting.figure_to_bytes(fig2, format="png")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert os.listdir(tmp_path) == []
    code = "import sys, active_subspaces.plotting; assert 'matplotlib.pyplot' not in sys.modules"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True)
