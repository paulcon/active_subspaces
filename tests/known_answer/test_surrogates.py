"""Known answers for spec 09 surrogates (V09.1-V09.8)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg
import scipy.stats
from scipy.spatial.distance import cdist

import active_subspaces as asub
from active_subspaces import surrogates as su


def _ref(m):
    return asub.Uniform(-np.ones(m), np.ones(m))


def _uniform(m, seed):
    gen = np.random.default_rng(seed)
    lower = gen.uniform(-5, 5, m)
    return asub.Uniform(lower, lower + 10.0 ** gen.uniform(-1, 1, m))


def _gaussian(m, seed):
    gen = np.random.default_rng(seed)
    B = gen.standard_normal((m, m))
    return asub.Gaussian(gen.standard_normal(m), B @ B.T + m * np.eye(m))


def _subspace(space, n, seed):
    """An eigenvalue-carrying subspace from gradients of a random quadratic."""
    gen = np.random.default_rng(seed)
    m = space.dim
    A = gen.standard_normal((m, m))
    A = A @ A.T
    X = space.sample(4 * m + 10, rng=seed)
    x = space.to_normalized(X)
    G = space.grads_to_physical(x @ A)
    return asub.active_subspace(grads=G, space=space, n=n)


# ---------------------------------------------------------------------------
# V09.1 polynomial known answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("d", [0, 1, 2, 3, 4])
def test_polynomial_exact(k, d):
    gen = np.random.default_rng(10 * k + d)
    idx = su._indices(k, d)
    p = idx.shape[0]
    assert p == math.comb(k + d, k)
    beta = gen.standard_normal(p)
    M = 2 * p
    X = gen.uniform(-1, 1, (M, k))
    f = su._basis(X, idx) @ beta
    s = asub.fit_surrogate(X=X, f=f, space=_ref(k), subspace=None, kind="poly", degree=d)
    assert np.array_equal(s.indices, idx)
    B = su._basis(X, idx)
    cond = np.linalg.cond(B / np.linalg.norm(B, axis=0))
    tol = max(1e-10, 10 * cond * np.finfo(float).eps)
    np.testing.assert_allclose(s.coefficients, beta, rtol=tol, atol=tol * np.max(np.abs(beta)))
    scale = np.max(np.abs(f))
    assert np.max(np.abs(s.loo_residuals)) <= 1e-9 * scale
    if d > 0:
        assert s.loo_r_squared >= 1 - 1e-12


def test_index_order():
    assert su._indices(2, 2).tolist() == [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]
    i3 = su._indices(3, 2).tolist()
    assert i3[:4] == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert i3[4:] == [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]


# ---------------------------------------------------------------------------
# V09.2 sample weights [C-04]
# ---------------------------------------------------------------------------


def test_sample_weights():
    gen = np.random.default_rng(1)
    k, d, M = 2, 2, 30
    X = gen.uniform(-1, 1, (M, k))
    f = np.sin(X[:, 0] * 2) + X[:, 1] ** 3
    w = gen.uniform(0.1, 3.0, M)
    sp = _ref(k)
    s = asub.fit_surrogate(
        X=X, f=f, space=sp, subspace=None, kind="poly", degree=d, sample_weights=w
    )
    B = su._basis(X, s.indices)
    ref = np.linalg.lstsq(np.sqrt(w)[:, None] * B, np.sqrt(w) * f, rcond=None)[0]
    np.testing.assert_allclose(s.coefficients, ref, rtol=1e-12, atol=1e-12 * np.max(np.abs(ref)))
    np.testing.assert_array_equal(s.f, f)  # raw f stored
    # integer weight 2 equals duplicating the row
    w2 = np.ones(M)
    w2[3] = 2.0
    s2 = asub.fit_surrogate(
        X=X, f=f, space=sp, subspace=None, kind="poly", degree=d, sample_weights=w2
    )
    sd = asub.fit_surrogate(
        X=np.vstack([X, X[3]]),
        f=np.append(f, f[3]),
        space=sp,
        subspace=None,
        kind="poly",
        degree=d,
    )
    np.testing.assert_allclose(s2.coefficients, sd.coefficients, rtol=1e-12, atol=1e-13)
    # equal weights equal no weights
    s3 = asub.fit_surrogate(
        X=X, f=f, space=sp, subspace=None, kind="poly", degree=d, sample_weights=np.full(M, 4.2)
    )
    s4 = asub.fit_surrogate(X=X, f=f, space=sp, subspace=None, kind="poly", degree=d)
    np.testing.assert_allclose(s3.coefficients, s4.coefficients, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(s3.loo_residuals, s4.loo_residuals, rtol=1e-10, atol=1e-13)
    assert s3.loo_r_squared == pytest.approx(s4.loo_r_squared, rel=1e-12)


# ---------------------------------------------------------------------------
# V09.3 quadratic terms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3])
def test_quadratic_terms(k):
    gen = np.random.default_rng(3)
    c = 0.7
    a = gen.standard_normal(k)
    H = gen.standard_normal((k, k))
    H = H + H.T
    X = gen.uniform(-1, 1, (40, k))
    f = c + X @ a + 0.5 * np.einsum("ij,jk,ik->i", X, H, X)
    s = asub.fit_surrogate(X=X, f=f, space=_ref(k), subspace=None, kind="poly", degree=2)
    c2, a2, H2 = asub.quadratic_terms(s)
    assert c2 == pytest.approx(c, rel=1e-10)
    np.testing.assert_allclose(a2, a, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(H2, H, rtol=1e-10, atol=1e-10)
    # private core used by spec 10's lift
    idx, coef = su._poly_fit(X, f, degree=2)
    _c3, _a3, H3 = su._quadratic_terms(idx, coef)
    np.testing.assert_allclose(H3, H, rtol=1e-10, atol=1e-10)
    # degree 1: H = 0
    s1 = asub.fit_surrogate(X=X, f=f, space=_ref(k), subspace=None, kind="poly", degree=1)
    assert not asub.quadratic_terms(s1)[2].any()


# ---------------------------------------------------------------------------
# V09.4 RBF objective [D-36]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("noise", [False, True])
def test_rbf_objective(noise):
    sp = _uniform(4, 0)
    sub = _subspace(sp, 2, 0)
    X = sp.sample(30, rng=5)
    y = sp.to_normalized(X) @ sub.W1
    f = np.sin(y[:, 0]) + y[:, 1] ** 2 + 0.05 * np.cos(7 * X[:, 3])
    v = np.full(30, 0.01) if noise else None
    for u in (-2.0, -0.3, 0.5):
        s = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="rbf", log10_eta=u, noise_variance=v)
        Y, M = s.Y, s.n_samples
        ls = s.length_scales
        K = np.exp(-cdist(Y / np.sqrt(ls), Y / np.sqrt(ls), "sqeuclidean")) + np.diag(s.nugget)
        B = su._basis(Y, s.indices)
        Ki = np.linalg.inv(K)
        beta = np.linalg.solve(B.T @ Ki @ B, B.T @ Ki @ f)
        r = f - B @ beta
        sig2 = r @ Ki @ r / M
        logpdf = scipy.stats.multivariate_normal(mean=B @ beta, cov=sig2 * K).logpdf(f)
        J = -logpdf - 0.5 * M * (1 + math.log(2 * math.pi))
        assert s.objective == pytest.approx(J, rel=1e-10)
        assert s.amplitude == pytest.approx(sig2, rel=1e-9)
        np.testing.assert_allclose(s.tail_coefficients, beta, rtol=1e-8, atol=1e-10)


def test_rbf_length_scales_and_nugget():
    sp = _uniform(5, 2)
    sub = _subspace(sp, 2, 2)
    X = sp.sample(30, rng=1)
    f = np.sum(sp.to_normalized(X), axis=1) ** 2
    s = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="rbf", log10_eta=-1.0)
    lam = np.abs(sub.eigenvalues)
    eta = 0.1
    np.testing.assert_allclose(s.length_scales, eta * lam.sum() / lam[:2], rtol=1e-14)
    np.testing.assert_allclose(s.nugget, max(eta * lam[2:].sum() / lam.sum(), 1e-6), rtol=1e-14)
    # "basis" subspace and subspace=None: isotropic, floor nugget [D-102]
    bsub = asub.subspace_from_basis(np.eye(5)[:, :2], space=sp)
    sb = asub.fit_surrogate(X=X, f=f, subspace=bsub, kind="rbf", log10_eta=-1.0)
    np.testing.assert_allclose(sb.length_scales, [eta, eta], rtol=1e-14)
    assert np.all(sb.nugget == 1e-6)
    sn = asub.fit_surrogate(X=X, f=f, space=sp, subspace=None, kind="rbf", log10_eta=-1.0)
    np.testing.assert_allclose(sn.length_scales, np.full(5, eta), rtol=1e-14)
    assert np.all(sn.nugget == 1e-6)
    # given noise variances: delta = v / s_f^2, floored
    v = np.linspace(0, 1, 30)
    sv = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="rbf", log10_eta=-1.0, noise_variance=v)
    sf2 = np.mean((f - f.mean()) ** 2)
    np.testing.assert_allclose(sv.nugget, np.maximum(v / sf2, 1e-6), rtol=1e-14)
    assert sv.training_variance == pytest.approx(sf2, rel=1e-14)


def test_rbf_search_is_grid_then_refine():
    sp = _ref(1)
    Y = np.linspace(-1, 1, 15)[:, None]
    f = np.sin(3 * Y[:, 0])
    s = asub.fit_surrogate(X=Y, f=f, space=sp, subspace=None, kind="rbf", degree=1)
    grid = np.linspace(-10, 1, 45)
    Jg = [
        asub.fit_surrogate(
            X=Y, f=f, space=sp, subspace=None, kind="rbf", degree=1, log10_eta=u
        ).objective
        for u in grid
    ]
    g = int(np.argmin(Jg))
    assert s.objective <= Jg[g]
    assert grid[max(g - 1, 0)] <= s.log10_eta <= grid[min(g + 1, 44)]
    again = asub.fit_surrogate(X=Y, f=f, space=sp, subspace=None, kind="rbf", degree=1)
    assert again == s


def test_rbf_reproduces_tail_polynomial():
    """UC-09 relies on this: f in the span of the quadratic tail gives alpha = 0 and zero
    residuals.
    """
    sp = _uniform(4, 3)
    sub = _subspace(sp, 1, 3)
    X = sp.sample(25, rng=3)
    y = (sp.to_normalized(X) @ sub.W1)[:, 0]
    f = 2 * y**2 + math.sqrt(2) * y
    s = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="rbf", degree=2)
    scale = np.max(np.abs(f))
    assert np.max(np.abs(s.residuals)) <= 1e-10 * scale
    assert np.max(np.abs(s.radial_coefficients)) <= 1e-8 * scale
    Xt = sp.sample(10, rng=4)
    yt = (sp.to_normalized(Xt) @ sub.W1)[:, 0]
    np.testing.assert_allclose(
        asub.predict(s, Xt), 2 * yt**2 + math.sqrt(2) * yt, rtol=1e-8, atol=1e-8 * scale
    )


# ---------------------------------------------------------------------------
# V09.5 scale equivariance [C-18]
# ---------------------------------------------------------------------------


def test_rbf_scale_equivariance():
    sp = _ref(1)
    Y = np.linspace(-1, 1, 25)[:, None]
    Yt = np.linspace(-1, 1, 200)[:, None]
    base = asub.fit_surrogate(
        X=Y,
        f=np.sin(3 * Y[:, 0]),
        space=sp,
        subspace=None,
        kind="rbf",
        noise_variance=np.full(25, 1e-2),
    )
    p0 = asub.predict(base, Yt)
    err0 = np.linalg.norm(p0 - np.sin(3 * Yt[:, 0])) / np.linalg.norm(np.sin(3 * Yt[:, 0]))
    for a, shift in ((100.0, 0.0), (1e-3, 0.0), (1.0, 7.0)):
        s = asub.fit_surrogate(
            X=Y,
            f=a * np.sin(3 * Y[:, 0]) + shift,
            space=sp,
            subspace=None,
            kind="rbf",
            noise_variance=np.full(25, a * a * 1e-2),
        )
        assert s.log10_eta == pytest.approx(base.log10_eta, rel=1e-10, abs=1e-10)
        np.testing.assert_allclose(
            asub.predict(s, Yt), a * p0 + shift, rtol=1e-10, atol=1e-10 * (abs(a) + abs(shift))
        )
        truth = a * np.sin(3 * Yt[:, 0])
        err = np.linalg.norm(asub.predict(s, Yt) - shift - truth) / np.linalg.norm(truth)
        assert err == pytest.approx(err0, rel=1e-6)
    assert err0 < 0.1


# ---------------------------------------------------------------------------
# V09.6 LOO by brute force
# ---------------------------------------------------------------------------


def _poly_brute(s):
    Y, f, w = s.Y, s.f, s.sample_weights
    out = np.empty(s.n_samples)
    for i in range(s.n_samples):
        keep = np.arange(s.n_samples) != i
        idx, coef = su._poly_fit(Y[keep], f[keep], s.degree, w[keep])
        out[i] = f[i] - su._basis(Y[i : i + 1], idx)[0] @ coef
    return out


def _rbf_brute(s):
    Y, f, ls, delta = s.Y, s.f, s.length_scales, s.nugget
    out = np.empty(s.n_samples)
    for i in range(s.n_samples):
        keep = np.arange(s.n_samples) != i
        Yk = Y[keep]
        K = su._rbf_kernel(Yk, Yk, ls) + np.diag(delta[keep])
        B = su._basis(Yk, s.indices)
        Ki = np.linalg.inv(K)
        beta = np.linalg.solve(B.T @ Ki @ B, B.T @ Ki @ f[keep])
        alpha = Ki @ (f[keep] - B @ beta)
        g = (
            su._rbf_kernel(Y[i : i + 1], Yk, ls)[0] @ alpha
            + su._basis(Y[i : i + 1], s.indices)[0] @ beta
        )
        out[i] = f[i] - g
    return out


@pytest.mark.parametrize("kind", ["poly", "rbf"])
def test_loo_brute_force(kind):
    sp = _uniform(4, 1)
    sub = _subspace(sp, 2, 1)
    X = sp.sample(24, rng=2)
    x = sp.to_normalized(X)
    f = np.exp(x[:, 0]) + np.sin(2 * x[:, 1]) + 0.1 * x[:, 3]
    if kind == "poly":
        w = np.random.default_rng(0).uniform(0.5, 2.0, 24)
        w[5] = 0.0
        s = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2, sample_weights=w)
        brute = _poly_brute(s)
        assert s.loo_residuals[5] == pytest.approx(s.residuals[5], rel=1e-12)
        sw = w
    else:
        s = asub.fit_surrogate(
            X=X, f=f, subspace=sub, kind="rbf", degree=1, noise_variance=np.full(24, 1e-3)
        )
        brute = _rbf_brute(s)
        sw = np.ones(24)
    np.testing.assert_allclose(
        s.loo_residuals, brute, rtol=1e-8, atol=1e-8 * np.max(np.abs(brute))
    )
    fbar = np.sum(sw * f) / np.sum(sw)
    r2 = 1 - np.sum(sw * brute**2) / np.sum(sw * (f - fbar) ** 2)
    assert s.loo_r_squared == pytest.approx(r2, rel=1e-8)


def test_rbf_residuals_identity():
    sp = _uniform(3, 4)
    sub = _subspace(sp, 2, 4)
    X = sp.sample(20, rng=6)
    f = np.cos(sp.to_normalized(X).sum(axis=1))
    s = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="rbf", noise_variance=np.full(20, 0.01))
    np.testing.assert_allclose(s.residuals, f - s.evaluate(s.Y), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("kind", ["poly", "rbf"])
def test_loo_undefined_when_m_equals_p(kind):
    gen = np.random.default_rng(0)
    X = gen.uniform(-1, 1, (6, 2))
    f = gen.standard_normal(6)
    with pytest.warns(asub.AccuracyWarning) as rec:
        s = asub.fit_surrogate(X=X, f=f, space=_ref(2), subspace=None, kind=kind, degree=2)
    assert [w.message.code for w in rec] == ["accuracy.loo_undefined"]
    assert s.loo_residuals is None
    assert s.loo_r_squared is None


# ---------------------------------------------------------------------------
# V09.7 gradients
# ---------------------------------------------------------------------------


def _central(fun, X, h):
    G = np.empty_like(X)
    for j in range(X.shape[1]):
        E = np.zeros_like(X)
        E[:, j] = h[j]
        G[:, j] = (fun(X + E) - fun(X - E)) / (2 * h[j])
    return G


@pytest.mark.filterwarnings(r"ignore:.*\[accuracy\.flat_objective\]")  # plateau of J [D-192]
@pytest.mark.parametrize("kind", ["poly", "rbf"])
@pytest.mark.parametrize("space_kind", ["uniform", "gaussian", "none"])
def test_gradients(kind, space_kind):
    m = 4
    sp = _uniform(m, 7) if space_kind != "gaussian" else _gaussian(m, 7)
    sub = None if space_kind == "none" else _subspace(sp, 2, 7)
    X = sp.sample(40, rng=8)
    x = sp.to_normalized(X)
    f = np.sin(x[:, 0] + 0.5 * x[:, 1]) + 0.3 * x[:, 2] ** 2
    kw = {"space": sp} if sub is None else {}
    s = asub.fit_surrogate(
        X=X, f=f, subspace=sub, kind=kind, degree=3 if kind == "poly" else 2, **kw
    )
    k = s.active_dim
    gen = np.random.default_rng(9)
    Yt = gen.uniform(-0.8, 0.8, (6, k))
    Gy = s.gradient(Yt)
    np.testing.assert_allclose(
        Gy, _central(s.evaluate, Yt, np.full(k, 1e-5)), rtol=1e-6, atol=1e-6 * np.max(np.abs(Gy))
    )
    if isinstance(sp, asub.Uniform):
        Xt = sp.from_normalized(gen.uniform(-0.8, 0.8, (6, m)))
        h = 1e-6 * (sp.upper - sp.lower)
    else:
        Xt = sp.sample(6, rng=10)
        h = 1e-5 * np.sqrt(np.diag(sp.cov))
    Gx = asub.predict_gradient(s, Xt)
    np.testing.assert_allclose(
        Gx,
        _central(lambda Z: asub.predict(s, Z), Xt, h),
        rtol=1e-6,
        atol=1e-6 * np.max(np.abs(Gx)),
    )


# ---------------------------------------------------------------------------
# V09.8 ridge fit quality
# ---------------------------------------------------------------------------


def test_ridge_fit_quality():
    m = 10
    gen = np.random.default_rng(20261028)
    sp = _uniform(m, 11)
    a = gen.standard_normal(m) / (sp.upper - sp.lower)
    a *= 1.5 / np.linalg.norm(a * (sp.upper - sp.lower) / 2)
    X = sp.sample(100, rng=gen)
    f = np.sin(X @ a)
    sub = asub.subspace_from_basis(a[:, None], space=sp)
    rbf = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="rbf")
    poly = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=5)
    assert rbf.loo_r_squared >= 0.99
    assert poly.loo_r_squared >= 0.99


# ---------------------------------------------------------------------------
# As-data-sets port (Phase 5b): D-192, D-193/D-204, D-200
# ---------------------------------------------------------------------------


def _ridge_data(space, M=40, seed=0):
    gen = np.random.default_rng(seed)
    X = space.sample(M, rng=gen)
    x = space.to_normalized(X)
    a = np.array([1.0, 0.5, 0.2])[: space.dim]
    f = np.sin(x @ a) + 0.05 * gen.standard_normal(M)
    G = space.grads_to_physical(np.cos(x @ a)[:, None] * a)
    return X, f, G


@pytest.mark.parametrize("kind", ["uniform", "gaussian"])
def test_rbf_published_bounds(kind):
    """With f_variance the search interval is CDW (4.13) with eta = 2 alpha and C1 per space;
    explicit bounds override; an empty interval warns [D-192], [D-204] (09 §8, V09.18).
    """
    space = asub.Uniform.reference(3) if kind == "uniform" else asub.Gaussian.reference(3)
    X, f, G = _ridge_data(space)
    sub = asub.active_subspace(grads=G, space=space, n=1)
    total = float(np.sum(np.abs(sub.eigenvalues)))
    c1 = 4.0 / math.pi**2 if kind == "uniform" else 1.0
    sig2 = 0.5 * c1 * total  # inside the interval
    rs = asub.fit_surrogate(X=X, f=f, subspace=sub, kind="rbf", f_variance=sig2)
    p = rs.provenance.parameters
    assert p["log10_eta_bounds_source"] == "f_variance"
    assert p["f_variance"] == sig2
    np.testing.assert_allclose(
        p["log10_eta_bounds"], [math.log10(2 * sig2 / total), math.log10(2 * c1)], rtol=1e-14
    )
    lo, hi = p["log10_eta_bounds"]
    assert lo - 1e-12 <= rs.log10_eta <= hi + 1e-12
    # Explicit bounds or log10_eta take precedence; f_variance is only recorded.
    given = asub.fit_surrogate(
        X=X, f=f, subspace=sub, f_variance=sig2, log10_eta_bounds=(-3.0, 0.0)
    )
    assert given.provenance.parameters["log10_eta_bounds"] == [-3.0, 0.0]
    assert given.provenance.parameters["log10_eta_bounds_source"] == "given"
    fixed = asub.fit_surrogate(X=X, f=f, subspace=sub, f_variance=sig2, log10_eta=-0.5)
    assert fixed.log10_eta == -0.5
    # Empty interval: collapses to its upper end, with accuracy.empty_bounds.
    with pytest.warns(asub.AccuracyWarning, match=r"\[accuracy\.empty_bounds\]$") as rec:
        empty = asub.fit_surrogate(X=X, f=f, subspace=sub, f_variance=10 * c1 * total)
    assert empty.log10_eta == pytest.approx(math.log10(2 * c1), rel=1e-15)
    w = next(r.message for r in rec if r.message.code == "accuracy.empty_bounds")
    assert w.details["C1"] == c1
    # Without eigenvalues (all inputs) the default interval applies.
    none = asub.fit_surrogate(X=X, f=f, space=space, subspace=None, f_variance=sig2)
    assert none.provenance.parameters["log10_eta_bounds"] == [-10.0, 1.0]


def test_flat_objective_warning():
    """J flat over >= 4 decades around the selection warns; a unimodal J does not [D-192]."""
    grid = np.linspace(-10.0, 1.0, 45)
    with pytest.warns(asub.AccuracyWarning, match="flat_objective"):
        su._warn_flat(grid, np.full(45, -13.41), 0)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        su._warn_flat(grid, (grid + 3.0) ** 2, int(np.argmin((grid + 3.0) ** 2)))
        flat3 = np.where(np.abs(grid + 5.0) <= 1.5, 0.0, 1.0)  # flat over 3 decades only
        su._warn_flat(grid, flat3, int(np.argmin(flat3)))


def test_poincare_constant():
    """C1 = 4/pi^2 on [-1,1]^m for every m, 1 for N(0, I) [D-204]; sin(pi x1 / 2) attains it."""
    from active_subspaces.spaces import _poincare_constant

    for m in (1, 3, 10, 100):
        assert _poincare_constant(asub.Uniform.reference(m)) == 4.0 / math.pi**2
        assert _poincare_constant(asub.Gaussian.reference(m)) == 1.0
    t, w = np.polynomial.legendre.leggauss(40)
    w = w / 2
    f, df = np.sin(math.pi * t / 2), (math.pi / 2) * np.cos(math.pi * t / 2)
    var = w @ f**2 - (w @ f) ** 2
    assert var == pytest.approx(4.0 / math.pi**2 * (w @ df**2), rel=1e-12)


def _weighted_quadratic(seed, M=30):
    gen = np.random.default_rng(seed)
    space = asub.Uniform.reference(2)
    X = space.sample(M, rng=gen)
    f = 1.0 + X[:, 0] - 0.5 * X[:, 1] ** 2 + 0.1 * gen.standard_normal(M)
    s = gen.uniform(0.5, 2.0, M)
    s[3] = 0.0
    return space, X, f, s


@pytest.mark.parametrize("kind", ["prediction", "confidence"])
@pytest.mark.parametrize("side", ["two", "lower", "upper"])
def test_polynomial_intervals(kind, side):
    """Closed-form least-squares limits with weights; one-sided quantiles [D-200] (V09.19)."""
    space, X, f, s = _weighted_quadratic(1)
    sur = asub.fit_surrogate(
        X=X, f=f, space=space, subspace=None, kind="poly", degree=2, sample_weights=s
    )
    Xn = space.sample(7, rng=5)
    res = asub.predict_interval(sur, Xn, level=0.9, kind=kind, side=side)
    B = su._basis(np.asarray(sur.Y), sur.indices)
    nu = int(np.sum(s > 0)) - B.shape[1]
    r = f - B @ sur.coefficients
    s2 = np.sum(s * r * r) / nu
    b = su._basis(Xn, sur.indices)
    q = np.einsum("ij,jk,ik->i", b, np.linalg.inv(B.T @ (s[:, None] * B)), b)
    c = 1.0 if kind == "prediction" else 0.0
    est = b @ sur.coefficients
    if side == "two":
        t = scipy.stats.t.ppf(0.95, nu)
        lo, hi = est - t * np.sqrt(s2 * (c + q)), est + t * np.sqrt(s2 * (c + q))
    elif side == "lower":
        t = scipy.stats.t.ppf(0.9, nu)
        lo, hi = est - t * np.sqrt(s2 * (c + q)), np.full(7, np.inf)
    else:
        t = scipy.stats.t.ppf(0.9, nu)
        lo, hi = np.full(7, -np.inf), est + t * np.sqrt(s2 * (c + q))
    np.testing.assert_allclose(res.lower, lo, rtol=1e-10)
    np.testing.assert_allclose(res.upper, hi, rtol=1e-10)
    np.testing.assert_allclose(res.estimate, est, rtol=1e-12)
    assert res.dof == nu
    assert res.residual_scale == pytest.approx(math.sqrt(s2), rel=1e-12)
    assert (res.level, res.kind, res.side) == (0.9, kind, side)
    lo2, hi2 = sur.interval(space.to_normalized(Xn), level=0.9, kind=kind, side=side)
    np.testing.assert_array_equal(lo2, res.lower)
    np.testing.assert_array_equal(hi2, res.upper)
    assert not res.lower.flags.writeable


def test_interval_errors():
    """RBF -> not_polynomial; no degrees of freedom -> no_dof; bad options [D-200]."""
    space, X, f, _ = _weighted_quadratic(2)
    rbf = asub.fit_surrogate(
        X=X, f=f, space=space, subspace=None, kind="rbf", degree=1, log10_eta=0.0
    )
    with pytest.raises(asub.InputError) as e:
        asub.predict_interval(rbf, X)
    assert e.value.details["reason"] == "not_polynomial"
    with pytest.warns(asub.AccuracyWarning, match="loo_undefined"):  # M = p
        exact = asub.fit_surrogate(
            X=X[:6], f=f[:6], space=space, subspace=None, kind="poly", degree=2
        )
    with pytest.raises(asub.InputError) as e:
        asub.predict_interval(exact, X[:2])
    assert e.value.details["reason"] == "no_dof"
    poly = asub.fit_surrogate(X=X, f=f, space=space, subspace=None, kind="poly", degree=1)
    for bad in ({"level": 1.0}, {"kind": "mean"}, {"side": "both"}):
        with pytest.raises(asub.InputError):
            asub.predict_interval(poly, X, **bad)
