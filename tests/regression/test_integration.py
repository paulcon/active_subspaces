"""Regressions of active_subspaces.integration against 2016 behaviour (spec 10 §7; spec 17 V10.25).

No baseline files exist yet (spec 15 §5.13); the 2016 computation is restated inline:
``av_integrate``
on an unbounded domain was ``sum_i w_i g(y_i)`` over the Gauss-Hermite rule built by
``np.linalg.eig``
of the Jacobi matrix (``utils/quadrature.g1d``).
"""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import integration as I


def _gauss_hermite_2016(N: int) -> tuple[np.ndarray, np.ndarray]:
    b = np.sqrt(np.arange(1, N, dtype=float))
    J = np.diag(b, 1) + np.diag(b, -1)
    t, V = np.linalg.eig(J)
    t, w = np.real(t), np.real(V[0, :]) ** 2
    order = np.argsort(t)
    return t[order], w[order] / w.sum()


def test_baselines_av_unbounded(seed):
    """integration/av_unbounded: av_integrate of a polynomial surrogate, Gaussian, n = 1, N_q =
    10 (F12).
    """
    m = 4
    gen = np.random.default_rng(seed)
    space = asub.Gaussian.reference(m)
    sub = asub.subspace_from_basis(gen.standard_normal((m, 1)), space=space)
    dom = asub.active_domain(sub)
    X = gen.standard_normal((60, m))
    y = X @ np.asarray(sub.W1)[:, 0]
    s = asub.fit_surrogate(
        X=X, f=np.exp(0.7 * y) + 0.1 * X[:, 1], subspace=sub, kind="poly", degree=3
    )
    res = I.integrate_surrogate(s, domain=dom, n_nodes=10)
    t, w = _gauss_hermite_2016(10)
    ref = float(np.dot(w, s.evaluate(t[:, None])))
    assert res.estimate == pytest.approx(ref, rel=1e-12)


def test_gaussian_node_count_2016_bug():
    """2016 took floor(N ** (1/n)) in floating point (8 ** (1/3) could give 1); v1 uses the
    exact integer root.
    """
    dom = asub.active_domain(
        asub.subspace_from_basis(np.eye(4)[:, :3], space=asub.Gaussian.reference(4))
    )
    assert I.active_quadrature_rule(dom, n_nodes=8).nodes.shape[0] == 8
    assert I.active_quadrature_rule(dom, n_nodes=125).nodes.shape[0] == 125
