"""Contract tests for ``active_subspaces.subspaces`` (spec 07 §4-§6; spec 17 V07.7, .11, .15,
.17-.19, .21, .30).
"""

from __future__ import annotations

import concurrent.futures
import itertools
import warnings

import numpy as np
import pytest
import scipy.linalg

import active_subspaces as asub
from active_subspaces import subspaces as S
from active_subspaces.errors import (
    AccuracyWarning,
    InputError,
    NumericalError,
    ValidityWarning,
)

M, m = 60, 3


def _data(seed: int = 0, space=None):
    space = space or asub.Gaussian.reference(m)
    gen = np.random.default_rng(seed)
    X = space.sample(M, rng=gen)
    x = space.to_normalized(X)
    f = x[:, 0] + 0.5 * x[:, 1] ** 2 + 0.1 * x[:, 2] ** 3
    G = np.column_stack([np.ones(M), x[:, 1], 0.3 * x[:, 2] ** 2])
    return space, X, f, space.grads_to_physical(G)


def _codes(record) -> list[str]:
    return [w.message.code for w in record if isinstance(w.message, asub.ActiveSubspacesWarning)]


def _run(**kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = S.active_subspace(**kw)
    return res, _codes(rec)


# --- slicing (V07.7) ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("f", "n_slices", "sizes"),
    [
        (np.arange(10.0), 5, [2, 2, 2, 2, 2]),
        (np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3.0]), 5, [2, 3, 2, 3]),  # ties across boundaries
        (np.full(12, 4.0), 3, [12]),  # all equal: one slice
        (
            np.array([0, 1, 1, 1, 1, 2, 3, 4.0]),
            4,
            [6, 2],
        ),  # singleton merged into the preceding slice
        (np.arange(5.0), 4, [3, 2]),  # first-slice singleton merged forward
        (np.arange(11.0), 3, [3, 4, 4]),  # M not divisible by S
    ],
    ids=["distinct", "ties", "all_equal", "singleton", "first_singleton", "remainder"],
)
def test_slicing(f, n_slices, sizes):
    perm = np.random.default_rng(1).permutation(f.size)
    fp = f[perm]
    slices = S._slices(fp, n_slices)
    assert [s.size for s in slices] == sizes
    assert np.array_equal(np.sort(np.concatenate(slices)), np.arange(f.size))
    for a, b in itertools.pairwise(slices):
        assert fp[a].max() < fp[b].min()  # ordered, and ties never split


def test_sir_and_save_share_slices(monkeypatch):
    space, X, f, _ = _data()
    calls = []
    real = S._slices

    def spy(fv, n_slices):
        out = real(fv, n_slices)
        calls.append([s.copy() for s in out])
        return out

    monkeypatch.setattr(S, "_slices", spy)
    S.active_subspace(space=space, X=X, f=f, estimator=S.SIR(n_slices=5), n=1)
    S.active_subspace(space=space, X=X, f=f, estimator=S.SAVE(n_slices=5), n=1)
    assert len(calls) == 2
    assert all(np.array_equal(a, b) for a, b in zip(*calls, strict=False))


# --- result contract ----------------------------------------------------------------------------


def test_result_fields_and_views():
    space, _X, _f, G = _data()
    res = S.active_subspace(space=space, grads=G, n=2)
    assert isinstance(res, asub.SubspaceResult)
    assert res.estimator == "as"
    assert res.estimator_options == ()
    assert res.dimension_rule == "fixed"
    assert res.dimension_options == ()
    assert res.criterion is None
    assert res.W is res.eigenvectors
    assert res.W1.shape == (m, 2)
    assert res.W2.shape == (m, 1)
    assert res.n_samples == M
    assert res.nboot == 0
    assert res.bootstrap_eigenvalues is None
    assert res.qoi == "f"
    assert res.provenance.rng is None
    assert res.provenance.function == "active_subspaces.active_subspace"
    assert np.allclose(res.W.T @ res.W, np.eye(m), atol=1e-12)
    W = res.W
    j = np.argmax(np.abs(W), axis=0)
    assert np.all(W[j, np.arange(m)] > 0)  # sign rule [D-81]
    assert np.all(np.diff(np.abs(res.eigenvalues)) <= 0)
    pd = res.physical_directions()
    assert pd.shape == (m, 2)
    assert res.physical_directions(3).shape == (m, 3)
    with pytest.raises(InputError) as e:
        res.physical_directions(0)
    assert e.value.code == "input.parameter"


def test_n_equals_m_and_m_equals_one():
    space, _X, _f, G = _data()
    res = S.active_subspace(space=space, grads=G, n=m)
    assert res.W2.shape == (m, 0)
    s1 = asub.Uniform.reference(1)
    g = np.random.default_rng(0).uniform(-1, 1, (20, 1))
    r1 = S.active_subspace(space=s1, grads=g)
    assert r1.n == 1
    assert r1.criterion is None
    assert r1.eigenvectors.tolist() == [[1.0]]
    r2 = S.active_subspace(space=s1, grads=g, dimension_rule="li", nboot=4, rng=0)
    assert r2.n == 1
    assert r2.criterion is None
    assert r2.bootstrap_distances.shape == (4, 0)
    assert S.dimension_criterion(r2, rule="li")[2] == 1


def test_round_trip_and_readonly(assert_roundtrip, assert_readonly):
    space, X, f, G = _data(space=asub.Uniform([0, 1, -5], [2, 3, 5]))
    with pytest.warns(ValidityWarning):
        r1 = S.active_subspace(
            space=space,
            X=X,
            f=f,
            estimator=S.SIR(n_slices=5),
            dimension_rule="li",
            nboot=5,
            rng=3,
            qoi="y",
        )
    r2 = S.active_subspace(space=space, X=X, f=f, estimator="opg", nboot=3, rng=1)
    r3 = S.subspace_from_basis(np.array([[1.0], [2.0], [0.0]]), space=space)
    r4 = S.active_subspace(space=space, grads=G, dimension_rule=S.EVG(gap="log"))
    r5 = S.with_dimension(r2, dimension_rule="rs")
    for r in (r1, r2, r3, r4, r5):
        assert_roundtrip(r)
        assert_readonly(r)
    assert r1.estimator_options == (("n_slices", 5),)
    assert r2.estimator_options == (("p", None),)
    assert r4.dimension_options == (("gap", "log"),)


def test_no_mutation():
    space, X, f, G = _data(space=asub.Uniform([0, 1, -5], [2, 3, 5]))
    w = np.random.default_rng(2).uniform(0.5, 1, M)
    w /= w.sum()
    arrays = [X, f, G, w]
    before = [a.copy() for a in arrays]
    for name in S.ESTIMATORS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S.active_subspace(
                space=space, X=X, f=f, grads=G, quad_weights=w, estimator=name, nboot=2, rng=0
            )
        for a, b in zip(arrays, before, strict=False):
            assert np.array_equal(a, b)
            assert a.dtype == b.dtype


def test_determinism_fixed_seed():
    space, _X, _f, G = _data()
    a = S.active_subspace(space=space, grads=G, dimension_rule="rs", nboot=20, rng=7)
    b = S.active_subspace(space=space, grads=G, dimension_rule="rs", nboot=20, rng=7)
    assert a == b
    assert a.provenance.rng is not None
    assert np.random.default_rng(7).bit_generator.state == S._random.replay(a).bit_generator.state


# --- registries and resolution
# --------------------------------------------------------------------


def test_registries():
    assert set(S.ESTIMATORS) == {
        "as",
        "nas",
        "asx",
        "nasx",
        "ols",
        "qphd",
        "sir",
        "phd",
        "save",
        "ss",
        "opg",
    }
    assert set(S.DIMENSION_RULES) == {"evg", "rs", "li"}
    with pytest.raises(TypeError):
        S.ESTIMATORS["mine"] = S.AS  # type: ignore[index]
    for name, cls in S.ESTIMATORS.items():
        assert cls.name == name
        assert cls.needs_gradients == ("grads" in cls.requires)
        assert cls.psd == (name not in ("asx", "nasx", "phd"))
    assert S.SIR().options() == {"n_slices": 10}
    assert S.EVG().options() == {"gap": "linear"}
    assert S.LI().options() == {}
    space, _X, _f, G = _data()
    assert S.active_subspace(space=space, grads=G, estimator="AS", n=1).estimator == "as"
    assert "AS" not in asub.__all__
    assert "active_subspace" in asub.__all__


@pytest.mark.parametrize(
    "kw",
    [
        {"estimator": "mave"},
        {"estimator": lambda d: d},
        {"estimator": type("Mine", (S.AS,), {"__slots__": ()})()},
        {"dimension_rule": "bic"},
        {"n": 1, "dimension_rule": "evg"},
        {"n": 0},
        {"n": m + 1},
        {"n": 1.5},
        {"nboot": -1},
        {"qoi": ""},
        {"executor": object()},
    ],
)
def test_parameter_errors(kw):
    space, X, f, G = _data()
    with pytest.raises(InputError) as e:
        S.active_subspace(space=space, X=X, f=f, grads=G, **kw)
    assert e.value.code == "input.parameter"


def test_option_and_data_errors():
    space, X, f, G = _data()
    for bad in (
        lambda: S.SIR(n_slices=1),
        lambda: S.SAVE(n_slices=1.5),
        lambda: S.EVG(gap="cubic"),
        lambda: S.OPG(p=1),
    ):
        with pytest.raises(InputError) as e:
            bad()
        assert e.value.code == "input.parameter"
    with pytest.raises(InputError) as e:
        S.active_subspace(X=X, grads=G)
    assert e.value.code == "input.parameter"  # space missing
    with pytest.raises(InputError) as e:
        S.active_subspace(space=space, X=X, estimator="ols")
    assert e.value.code == "input.parameter"
    assert e.value.details["missing"] == ["f"]
    for p in (m, M + 1):
        with pytest.raises(InputError) as e:
            S.active_subspace(space=space, X=X, f=f, estimator=S.OPG(p=p), n=1)
        assert e.value.code == "input.parameter"
    with pytest.raises(InputError) as e:
        S.active_subspace(space=space, X=X[:9], f=f[:9], estimator="qphd", n=1)
    assert e.value.code == "input.too_few_samples"
    with pytest.raises(InputError) as e:
        S.active_subspace(space=space, X=X[:19], f=f[:19], estimator="sir", n=1)
    assert e.value.code == "input.too_few_samples"
    Xd = np.repeat(X[:2], 10, axis=0)
    with pytest.raises(InputError) as e:
        S.active_subspace(space=space, X=Xd, f=np.arange(20.0), estimator="opg", n=1)
    assert e.value.code == "input.too_few_samples"
    with pytest.raises(InputError) as e:
        S.active_subspace(space=space, X=X, f=f[:-1], n=1, estimator="ols")
    assert e.value.code == "input.size_mismatch"


def test_no_bootstrap_errors():
    space, _X, _f, G = _data()
    for rule in ("rs", "li", S.RS()):
        with pytest.raises(InputError) as e:
            S.active_subspace(space=space, grads=G, dimension_rule=rule)
        assert e.value.code == "input.no_bootstrap"
    res = S.active_subspace(space=space, grads=G)
    for call in (
        lambda: S.with_dimension(res, dimension_rule="li"),
        lambda: S.dimension_criterion(res, rule="rs"),
        lambda: S.mean_distance(res),
    ):
        with pytest.raises(InputError) as e:
            call()
        assert e.value.code == "input.no_bootstrap"


def test_basis_result_rules():
    space = asub.Gaussian.reference(m)
    b = S.subspace_from_basis(np.eye(m)[:, :1], space=space)
    assert b.estimator == "basis"
    assert np.all(np.isnan(b.eigenvalues))
    assert b.n_samples == 0
    assert S.with_dimension(b, 2).n == 2
    for call in (
        lambda: S.with_dimension(b, dimension_rule="evg"),
        lambda: S.dimension_criterion(b, rule="evg"),
    ):
        with pytest.raises(InputError) as e:
            call()
        assert e.value.code == "input.parameter"
    with pytest.raises(InputError) as e:
        S.subspace_from_basis(np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]), space=space)
    assert e.value.code == "input.parameter"
    with pytest.raises(InputError):
        S.subspace_from_basis(np.ones((m + 1, 1)), space=space)


def test_with_dimension_arguments():
    space, _X, _f, G = _data()
    res = S.active_subspace(space=space, grads=G)
    for kw in ({}, {"n": 1, "dimension_rule": "evg"}):
        with pytest.raises(InputError) as e:
            S.with_dimension(res, **kw)
        assert e.value.code == "input.parameter"
    with pytest.raises(InputError):
        S.with_dimension(res, m + 1)
    r = S.with_dimension(res, 2)
    assert r.n == 2
    assert r.dimension_rule == "fixed"
    assert r.criterion is None
    assert r.provenance.function == "active_subspaces.with_dimension"
    assert r.provenance.parameters["source"]["function"] == "active_subspaces.active_subspace"
    assert r.provenance.rng is None


# --- with_dimension equivalence (V07.15) ---------------------------------------------------------


@pytest.mark.parametrize("rule", ["evg", S.EVG(gap="log"), "rs", "li", 2])
def test_with_dimension_equivalence(rule):
    space, _X, _f, G = _data()
    base = S.active_subspace(space=space, grads=G, n=1, nboot=15, rng=11)
    if isinstance(rule, int):
        direct = S.active_subspace(space=space, grads=G, n=rule, nboot=15, rng=11)
        via = S.with_dimension(base, rule)
    else:
        direct = S.active_subspace(space=space, grads=G, dimension_rule=rule, nboot=15, rng=11)
        via = S.with_dimension(base, dimension_rule=rule)
    assert via == direct


# --- bootstrap (V07.17-V07.19) -------------------------------------------------------------------


def test_bootstrap_streams():
    space, _X, _f, G = _data()
    w = np.random.default_rng(5).uniform(0.5, 1.0, M)
    w /= w.sum()
    B, seed = 6, 1234
    with pytest.warns(ValidityWarning) as rec:
        res = S.active_subspace(space=space, grads=G, quad_weights=w, n=1, nboot=B, rng=seed)
    assert _codes(rec) == ["validity.bootstrap_weights"]
    children = np.random.default_rng(seed).spawn(B)
    for b in range(B):
        idx = children[b].integers(0, M, M)
        wb = w[idx] / w[idx].sum()
        rb = S.active_subspace(space=space, grads=G[idx], quad_weights=wb, n=1)
        assert np.array_equal(rb.eigenvalues, res.bootstrap_eigenvalues[b])
        for k in range(1, m):
            dist = np.linalg.norm(res.W[:, :k].T @ rb.W[:, k:], 2)
            det = abs(np.linalg.det(res.W[:, :k].T @ rb.W[:, :k]))
            assert res.bootstrap_distances[b, k - 1] == pytest.approx(dist, abs=1e-14)
            assert res.bootstrap_dets[b, k - 1] == pytest.approx(det, abs=1e-14)
    assert np.array_equal(S.mean_distance(res), res.bootstrap_distances.mean(axis=0))


def test_bootstrap_workers():
    space, X, f, _G = _data()
    kw = {
        "space": space,
        "X": X,
        "f": f,
        "estimator": "opg",
        "dimension_rule": "li",
        "nboot": 9,
        "rng": 99,
    }
    ref = S.active_subspace(**kw)
    for workers in (1, 2, 7):
        with concurrent.futures.ThreadPoolExecutor(workers) as ex:
            assert S.active_subspace(executor=ex, **kw) == ref
    with concurrent.futures.ProcessPoolExecutor(2) as ex:
        assert S.active_subspace(executor=ex, **kw) == ref


def test_interval_helpers():
    reps = np.random.default_rng(0).standard_normal((50, 4))
    pi = S.bootstrap_interval(reps)
    assert pi.shape == (2, 4)
    assert np.allclose(pi, np.quantile(reps, [0.025, 0.975], axis=0), rtol=0, atol=1e-14)
    pi = S.bootstrap_interval(reps, level=0.5)
    assert np.allclose(pi, np.quantile(reps, [0.25, 0.75], axis=0), rtol=0, atol=1e-14)
    mm = S.bootstrap_interval(reps, kind="minmax")
    assert np.array_equal(mm, np.vstack([reps.min(0), reps.max(0)]))
    assert S.bootstrap_interval(reps[:, 0]).shape == (2, 1)
    for kw in ({"level": 0.0}, {"level": 1.0}, {"kind": "bca"}):
        with pytest.raises(InputError) as e:
            S.bootstrap_interval(reps, **kw)
        assert e.value.code == "input.parameter"
    with pytest.raises(InputError) as e:
        S.bootstrap_interval(np.zeros((0, 3)))
    assert e.value.code == "input.parameter"


def test_zero_weight_replicate():
    space, _X, _f, G = _data()
    w = np.zeros(M)
    w[0] = 1.0
    with pytest.warns(ValidityWarning), pytest.raises(NumericalError) as e:
        S.active_subspace(space=space, grads=G, quad_weights=w, n=1, nboot=20, rng=0)
    assert e.value.code == "numerical.singular"
    assert e.value.details["reason"] == "zero_weight_replicate"


# --- warnings (V07.21) and numerical errors ------------------------------------------------------

_SPACE_WARN = {
    ("asx", "uniform"),
    ("nasx", "uniform"),
    ("ols", "uniform"),
    ("qphd", "uniform"),
    ("sir", "uniform"),
    ("phd", "uniform"),
    ("save", "uniform"),
    ("ss", "uniform"),
    ("ss", "gaussian"),
}
_RULE_OK = {
    "evg": set(S.ESTIMATORS),
    "rs": {"as", "opg"},
    "li": set(S.ESTIMATORS) - {"asx", "nasx", "phd"},
}


@pytest.mark.parametrize("kind", ["uniform", "gaussian"])
@pytest.mark.parametrize("estimator", sorted(S.ESTIMATORS))
def test_validity_estimator_space(estimator, kind):
    space = asub.Uniform.reference(m) if kind == "uniform" else asub.Gaussian.reference(m)
    space, X, f, G = _data(space=space)
    _, codes = _run(space=space, X=X, f=f, grads=G, estimator=estimator, n=1)
    assert codes.count("validity.assumption") == ((estimator, kind) in _SPACE_WARN)
    assert "validity.bootstrap_weights" not in codes


@pytest.mark.parametrize("rule", ["evg", "rs", "li"])
@pytest.mark.parametrize("estimator", sorted(S.ESTIMATORS))
def test_validity_rule_estimator(estimator, rule):
    space, X, f, G = _data()
    res, codes = _run(
        space=space, X=X, f=f, grads=G, estimator=estimator, dimension_rule=rule, nboot=2, rng=0
    )
    rule_warn = estimator not in _RULE_OK[rule]
    space_warn = (estimator, "gaussian") in _SPACE_WARN
    assert codes.count("validity.assumption") == (rule_warn or space_warn)
    _, codes2 = _run(space=space, X=X, f=f, grads=G, estimator=estimator, n=1, nboot=2, rng=0)
    assert codes2.count("validity.assumption") == space_warn  # a fixed n never warns
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        S.with_dimension(res, dimension_rule=rule)
    assert _codes(rec).count("validity.assumption") == rule_warn


def test_bootstrap_weights_warning_only_when_nonuniform():
    space, _X, _f, G = _data()
    _, codes = _run(space=space, grads=G, quad_weights=np.full(M, 1 / M), n=1, nboot=2, rng=0)
    assert codes == []
    w = np.linspace(1, 2, M)
    _, codes = _run(space=space, grads=G, quad_weights=w / w.sum(), n=1)
    assert codes == []
    _, codes = _run(space=space, grads=G, quad_weights=w / w.sum(), n=1, nboot=2, rng=0)
    assert codes == ["validity.bootstrap_weights"]


def test_accuracy_warnings():
    space, X, f, _G = _data()
    with pytest.warns(AccuracyWarning) as rec:
        res = S.active_subspace(space=space, grads=np.zeros((M, m)), nboot=3, rng=0)
    assert _codes(rec) == ["accuracy.degenerate_matrix"]
    assert rec[0].message.details["replicates"] == 3
    assert np.array_equal(res.eigenvectors, np.eye(m))
    assert np.all(res.eigenvalues == 0)
    Xc = X.copy()
    Xc[:, 1] = Xc[:, 0]
    with pytest.warns(AccuracyWarning) as rec:
        S.active_subspace(space=space, X=Xc, f=f, estimator="ols", n=1, nboot=2, rng=0)
    assert _codes(rec) == ["accuracy.rank_deficient"]
    assert rec[0].message.details == {"estimator": "ols", "data": True, "replicates": 2}


def test_zero_dimension_warning(make_provenance):
    space = asub.Gaussian.reference(4)
    B = 5
    hand = S.SubspaceResult(
        space=space,
        estimator="as",
        estimator_options=(),
        eigenvalues=np.array([1e-3, 1e-4, 1e-5, 1e-6]),
        eigenvectors=np.eye(4),
        n=1,
        dimension_rule="fixed",
        dimension_options=(),
        criterion=None,
        n_samples=10,
        nboot=B,
        bootstrap_eigenvalues=np.zeros((B, 4)),
        bootstrap_distances=np.ones((B, 3)),
        bootstrap_dets=np.zeros((B, 3)),
        bootstrap_eigenvectors=None,
        provenance=make_provenance(),
    )
    with pytest.warns(AccuracyWarning) as rec:
        r = S.with_dimension(hand, dimension_rule="li")
    assert _codes(rec) == ["accuracy.zero_dimension"]
    assert r.n == 1
    assert r.criterion.shape == (4,)
    assert S.dimension_criterion(hand, rule="li")[2] == 0


def test_numerical_errors():
    space, X, f, G = _data()
    with pytest.raises(NumericalError) as e:
        S.active_subspace(space=space, grads=1e200 * G, n=1)
    assert e.value.code == "numerical.nonfinite"
    with pytest.warns(ValidityWarning), pytest.raises(NumericalError) as e:
        S.active_subspace(space=space, X=X, f=1e160 * f, estimator="ss", n=1)
    assert e.value.code == "numerical.nonfinite"


# --- GradientResult as grads [D-76], [D-88]
# -------------------------------------------------------


def test_gradient_result_space():
    space = asub.Gaussian.reference(m)
    space, X, f, _ = _data(space=space)
    gr = asub.local_linear_gradients(X=X, f=f, space=space, qoi="q")
    r = S.active_subspace(grads=gr, estimator="asx", n=1)
    ref = S.active_subspace(space=space, X=gr.points, grads=gr.grads, estimator="asx", n=1)
    assert r.qoi == "q"
    assert np.array_equal(r.eigenvectors, ref.eigenvectors)
    assert np.array_equal(r.eigenvalues, ref.eigenvalues)
    assert S.active_subspace(grads=gr, space=asub.Gaussian.reference(m), n=1).space == space
    with pytest.raises(InputError) as e:
        S.active_subspace(grads=gr, space=asub.Uniform.reference(m), n=1)
    assert e.value.code == "input.parameter"
    assert e.value.details["reason"] == "space_mismatch"
    with pytest.raises(InputError) as e:
        S.active_subspace(grads=gr, X=X[:-1], n=1)
    assert e.value.code == "input.size_mismatch"


def test_ladle_truncation_large_m():
    m12 = 12
    space = asub.Gaussian.reference(m12)
    G = np.random.default_rng(0).standard_normal((40, m12)) * np.linspace(3, 0.1, m12)
    res = S.active_subspace(space=space, grads=G, dimension_rule="li", nboot=4, rng=0)
    kn = int(np.floor(m12 / np.log(m12)))
    assert res.criterion.shape == (kn + 1,)
    assert 1 <= res.n <= kn
    assert res.bootstrap_dets.shape == (4, m12 - 1)


def test_distance_and_det_match_definitions():
    """The smaller-side rewrites equal ||P[:k, k:]||_2 and |det P[:k, :k]| to rounding [D-178]."""
    from active_subspaces.subspaces import _distance_and_det

    gen = np.random.default_rng(5)
    for m in (2, 3, 8, 13):
        W = np.linalg.qr(gen.standard_normal((m, m)))[0]
        Wb = np.linalg.qr(W + 0.05 * gen.standard_normal((m, m)))[0]
        P = W.T @ Wb
        for k in range(1, m):
            d, t = _distance_and_det(P, k)
            d0 = np.linalg.norm(P[:k, k:], 2)
            assert abs(d - d0) <= 1e-13 * max(d0, 1e-300)
            assert abs(t - abs(np.linalg.det(P[:k, :k]))) <= 1e-13


# ---------------------------------------------------------------------------
# As-data-sets port (Phase 5b): D-196, D-197, V1-11
# ---------------------------------------------------------------------------


def test_bootstrap_vectors():
    """Stored replicate eigenvectors are sign-aligned; k = 0 is bitwise unchanged [D-196]
    (07 §8 bootstrap item 5, V07.37).
    """
    space, _, _, G = _data(3)
    base = asub.active_subspace(grads=G, space=space, nboot=12, rng=4, n=2)
    sub = asub.active_subspace(grads=G, space=space, nboot=12, rng=4, n=2, bootstrap_vectors=2)
    assert base.bootstrap_eigenvectors is None
    assert sub.bootstrap_eigenvectors.shape == (12, m, 2)
    for name in (
        "eigenvalues",
        "eigenvectors",
        "bootstrap_eigenvalues",
        "bootstrap_distances",
        "bootstrap_dets",
    ):
        np.testing.assert_array_equal(getattr(sub, name), getattr(base, name))
    V = sub.bootstrap_eigenvectors
    inner = np.einsum("ij,bij->bj", sub.W[:, :2], V)
    assert np.all(inner >= 0)
    np.testing.assert_allclose(
        sub.bootstrap_distances[:, 0], np.sqrt(np.clip(1 - inner[:, 0] ** 2, 0, None)), atol=1e-12
    )
    # Replicate b by hand from its own child stream (spec 05 layout), then aligned.
    child = np.random.default_rng(4).spawn(12)[5]
    idx = child.integers(0, M, size=M)
    by_hand = asub.active_subspace(grads=G[idx], space=space, n=2)
    s = np.sign(sub.W[:, :2].T @ by_hand.W[:, :2]).diagonal()
    np.testing.assert_allclose(V[5], by_hand.W[:, :2] * s, atol=1e-12)
    # A rule may pick n < k; the k columns are kept, and with_dimension keeps them too.
    ruled = asub.active_subspace(grads=G, space=space, nboot=12, rng=4, bootstrap_vectors=3)
    assert ruled.bootstrap_eigenvectors.shape[2] == 3
    np.testing.assert_array_equal(
        asub.with_dimension(ruled, 1).bootstrap_eigenvectors, ruled.bootstrap_eigenvectors
    )
    assert asub.from_dict(sub.to_dict()) == sub
    for bad in (
        {"bootstrap_vectors": 1},
        {"bootstrap_vectors": 3, "nboot": 2, "n": 2},
        {"bootstrap_vectors": m + 1, "nboot": 2},
        {"bootstrap_vectors": -1, "nboot": 2},
    ):
        with pytest.raises(InputError) as e:
            asub.active_subspace(grads=G, space=space, **{"n": 1, **bad})
        assert e.value.code == "input.parameter"


def test_bootstrap_vectors_unaligned_warning():
    """Near-repeated eigenvalues make replicates rotate: accuracy.unaligned_replicates [D-196]."""
    space = asub.Gaussian.reference(3)
    G = np.random.default_rng(0).standard_normal((200, 3)) * np.array([1.0, 0.995, 0.01])
    with pytest.warns(AccuracyWarning) as rec:
        asub.active_subspace(grads=G, space=space, n=2, nboot=40, rng=1, bootstrap_vectors=2)
    assert "accuracy.unaligned_replicates" in _codes(rec)
    G2 = G * np.array([1.0, 0.3, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sub = asub.active_subspace(
            grads=G2, space=space, n=1, nboot=40, rng=1, bootstrap_vectors=1
        )
    assert np.all(np.einsum("i,bi->b", sub.W[:, 0], sub.bootstrap_eigenvectors[:, :, 0]) > 0.5)


def test_orient():
    """orient re-signs the leading columns by a criterion; replicates and physical
    directions follow [D-197] (07 §8 KA-16, V07.34).
    """
    space = asub.Uniform([0.0, -2.0, 1.0], [1.0, 2.0, 5.0])
    gen = np.random.default_rng(8)
    X = space.sample(200, rng=gen)
    x = space.to_normalized(X)
    a = np.array([-1.0, 0.4, 0.2])
    f = x @ a + 0.1 * x[:, 1] ** 2
    G = space.grads_to_physical(
        np.tile(a, (200, 1)) + np.column_stack([np.zeros(200), 0.2 * x[:, 1], np.zeros(200)])
    )
    sub = asub.active_subspace(grads=G, space=space, n=2, nboot=8, rng=0, bootstrap_vectors=2)
    flipped = asub.orient(sub, by="reference", reference=-sub.W)
    np.testing.assert_array_equal(flipped.W[:, :2], -sub.W[:, :2])
    np.testing.assert_array_equal(flipped.W[:, 2], sub.W[:, 2])
    np.testing.assert_array_equal(flipped.bootstrap_eigenvectors, -sub.bootstrap_eigenvectors)
    np.testing.assert_array_equal(flipped.physical_directions(), -sub.physical_directions())
    for name in ("eigenvalues", "bootstrap_distances", "bootstrap_dets", "n", "criterion"):
        np.testing.assert_array_equal(getattr(flipped, name), getattr(sub, name))
    assert flipped.provenance.function == "active_subspaces.orient"
    assert flipped.provenance.parameters["signs"] == [-1, -1]
    assert asub.orient(flipped, by="reference", reference=sub) == sub
    # Mean gradient: w1^T a > 0; covariance: positive correlation of y1 with f.
    mg = asub.orient(sub, by="mean_gradient", grads=G, columns=1)
    assert mg.W[:, 0] @ a > 0
    cov = asub.orient(sub, by="covariance", X=X, f=f, columns=1)
    y = x @ cov.W[:, 0]
    assert np.cov(y, f)[0, 1] > 0
    assert asub.orient(cov, by="covariance", X=X, f=f, columns=1) == cov  # idempotent
    # The OLS sign is the covariance sign for uniform weights.
    with pytest.warns(ValidityWarning):
        ols = asub.active_subspace(X=X, f=f, space=space, estimator="ols", n=1)
    np.testing.assert_array_equal(asub.orient(ols, by="covariance", X=X, f=f).W, ols.W)
    # A criterion of exactly zero keeps the sign.
    zero = np.zeros((3, 1))
    zero[:, 0] = np.cross(sub.W[:, 0], sub.W[:, 1])  # orthogonal to w1
    assert asub.orient(sub, by="reference", reference=zero, columns=1) == sub
    # Errors.
    with pytest.raises(InputError) as e:
        asub.orient(sub, by="reference", reference=sub.W, X=X)
    assert e.value.details["reason"] == "not_for_criterion"
    with pytest.raises(InputError):
        asub.orient(sub, by="covariance", X=X)
    with pytest.raises(InputError):
        asub.orient(sub, by="reference", reference=sub.W[:, :1])  # c = n = 2 > 1 column
    with pytest.raises(InputError) as e:
        asub.orient(
            sub,
            by="reference",
            reference=asub.subspace_from_basis(np.eye(3)[:, :1], space=asub.Uniform.reference(3)),
        )
    assert e.value.details["reason"] == "space_mismatch"
    with pytest.raises(InputError):
        asub.orient(sub, by="nonsense", reference=sub.W)


def test_lstsq_qr():
    """The QR solve equals gelsd on full-rank data; an ill-conditioned or rank-deficient design
    takes the gelsd route, and rank deficiency still warns (V1-11)
    (07 §8 KA-14, V07.40).
    """
    gen = np.random.default_rng(1)
    A = gen.standard_normal((200, 10))
    b = gen.standard_normal(200)
    coef, bad = S._lstsq(A, b, "test")
    assert not bad
    ref = np.linalg.lstsq(A, b, rcond=None)[0]
    np.testing.assert_allclose(coef, ref, rtol=1e-10, atol=1e-12)
    A[:, 3] = A[:, 2] + 1e-12 * gen.standard_normal(200)  # cond ~ 1e12: gelsd route
    coef, _ = S._lstsq(A, b, "test")
    ref = scipy.linalg.lstsq(A, b, lapack_driver="gelsd", check_finite=False)[0]
    np.testing.assert_array_equal(coef, ref)
    A[:, 3] = A[:, 2]  # duplicated column: rank 9, gelsd route
    coef, bad = S._lstsq(A, b, "test")
    assert bad
    space = asub.Gaussian.reference(3)
    X = space.sample(30, rng=0)
    X[:, 2] = X[:, 1]
    with pytest.warns(AccuracyWarning, match="rank-deficient"):
        asub.active_subspace(X=X, f=X[:, 0], space=space, estimator="ols", n=1)


def test_as_inplace_scaling_never_touches_inputs():
    """Without a bootstrap AS scales its own normalized copy in place, in row blocks; the
    caller's arrays and a GradientResult's stored arrays are unchanged, and the result equals
    the bootstrap path bitwise [D-202].
    """
    M3 = S._ROW_BLOCK + 17  # crosses a block boundary
    G = np.random.default_rng(9).standard_normal((M3, 3))
    G0 = G.copy()
    sp = asub.Uniform.reference(3)  # identity map: the normalized array is a copy, not G itself
    a = asub.active_subspace(grads=G, space=sp, n=1)
    np.testing.assert_array_equal(G, G0)
    b = asub.active_subspace(grads=G, space=sp, n=1, nboot=1, rng=0)
    np.testing.assert_array_equal(a.eigenvalues, b.eigenvalues)
    np.testing.assert_array_equal(a.eigenvectors, b.eigenvectors)
    gr = asub.local_linear_gradients(X=sp.sample(40, rng=0), f=np.arange(40.0), space=sp)
    stored = np.array(gr.grads, copy=True)
    asub.active_subspace(grads=gr, n=1)
    np.testing.assert_array_equal(gr.grads, stored)


def test_review_followups_bootstrap_and_warnings():
    """Executor independence with bootstrap_vectors; unaligned_replicates is issued last
    (07 §6); orient records input digests [D-196], [D-197].
    """
    space, X, f, G = _data(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AccuracyWarning)  # w2, w3 rotate in this example
        serial = asub.active_subspace(
            grads=G, space=space, n=2, nboot=6, rng=3, bootstrap_vectors=2
        )
        with concurrent.futures.ThreadPoolExecutor(3) as ex:
            threaded = asub.active_subspace(
                grads=G, space=space, n=2, nboot=6, rng=3, bootstrap_vectors=2, executor=ex
            )
    assert threaded == serial
    # Warning order: rank_deficient before unaligned_replicates.
    Xd = np.column_stack([X[:, 0], X[:, 0], X[:, 1]])
    with pytest.warns(AccuracyWarning) as rec:
        asub.active_subspace(
            X=Xd, f=f, space=space, estimator="ols", n=2, nboot=10, rng=0, bootstrap_vectors=2
        )
    codes = [c for c in _codes(rec) if c.startswith("accuracy.")]
    assert codes == ["accuracy.rank_deficient", "accuracy.unaligned_replicates"]
    o = asub.orient(serial, by="mean_gradient", grads=G)
    assert o.provenance.parameters["grads"] == asub._validation.digest(np.asarray(G))
    o2 = asub.orient(serial, by="reference", reference=serial.W)
    assert "reference" in o2.provenance.parameters


def test_qphd_solve_routes_agree():
    """QPHD's C is the same through the QR and gelsd least-squares routes; a rank-deficient
    QPHD design takes gelsd and warns (V1-11).
    """
    space = asub.Gaussian.reference(3)
    X = space.sample(80, rng=4)
    f = X[:, 0] ** 2 + X[:, 1] * X[:, 2] + 0.1 * X[:, 0]
    d = S._Arrays(X, f, None, np.full(80, 1 / 80), np.eye(3))
    C_qr = S.QPHD()._matrix(d, {})

    def gelsd_only(A, b, what):
        coef, _, rank, _ = S.scipy.linalg.lstsq(A, b, lapack_driver="gelsd")
        return coef, int(rank) < A.shape[1]

    orig = S._lstsq
    try:
        S._lstsq = gelsd_only
        C_gelsd = S.QPHD()._matrix(d, {})
    finally:
        S._lstsq = orig
    np.testing.assert_allclose(C_qr, C_gelsd, rtol=1e-10, atol=1e-12 * np.abs(C_qr).max())
    Xr = X.copy()
    Xr[:, 2] = Xr[:, 1]
    with pytest.warns(AccuracyWarning, match="rank-deficient"):
        asub.active_subspace(X=Xr, f=f, space=space, estimator="qphd", n=1)
