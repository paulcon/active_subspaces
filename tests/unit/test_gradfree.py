"""Contracts of spec 14 B (active_subspaces.experimental): import warning, result, streams,
errors (V14.15, V14.18, V14.21-V14.23).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import _random

with warnings.catch_warnings():
    warnings.simplefilter("ignore", asub.ExperimentalWarning)
    from active_subspaces import experimental
    from active_subspaces.experimental import (
        SketchedSubspace,
        gradfree,
        sketched_active_subspace,
        subspace_from_sketches,
    )

A = np.array([0.3, -0.5, 0.2, 0.7, 0.1])


def _ridge(X):
    return np.sin((X * A[: X.shape[1]]).sum(axis=1))


def _run(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": ":".join(sys.path), "PATH": ""},
    )


def test_experimental_warning():
    """V14.15 [D-73]: no warning on `import active_subspaces`; one experimental.module at the
    import line; none again.
    """
    out = _run(
        """
        import json, sys, warnings
        warnings.simplefilter("always")
        with warnings.catch_warnings(record=True) as w0:
            import active_subspaces
        assert "active_subspaces.experimental" not in sys.modules
        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always")
            import active_subspaces.experimental
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            import active_subspaces.experimental
            from active_subspaces.experimental import gradfree
        print(json.dumps({
            "w0": len(w0), "w2": len(w2),
            "w1": [(type(x.message).__name__, getattr(x.message, "code", None), x.filename,
                    x.lineno) for x in w1],
        }))
        """
    )
    rec = json.loads(out.stdout)
    assert rec["w0"] == 0
    assert rec["w2"] == 0
    assert len(rec["w1"]) == 1
    name, code, filename, lineno = rec["w1"][0]
    assert (name, code, filename, lineno) == (
        "ExperimentalWarning",
        "experimental.module",
        "<string>",
        9,
    )


def test_fresh_process_load(tmp_path):
    """V14.23 [D-141]: loading a saved SketchedSubspace in a process that never imported
    experimental succeeds and warns once.
    """
    res = subspace_from_sketches(
        sketches=np.eye(3)[:, :2],
        directions=np.tile(np.eye(3)[:, :2], (3, 1, 1)),
        space=asub.Uniform.reference(3),
    )
    path = tmp_path / "s.npz"
    res.save(path)
    out = _run(
        """
        import json, sys, warnings
        import active_subspaces as asub
        assert "active_subspaces.experimental" not in sys.modules
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = asub.load(sys.argv[1])
        print(json.dumps({"type": type(r).__name__,
                          "codes": [getattr(x.message, "code", None) for x in w],
                          "fields": r.to_dict()["fields"]}))
        """,
        str(path),
    )
    rec = json.loads(out.stdout)
    assert rec["type"] == "SketchedSubspace"
    assert rec["codes"] == ["experimental.module"]
    assert rec["fields"] == res.to_dict()["fields"]


def test_exports_and_core_independence():
    assert (
        experimental.__all__
        == gradfree.__all__
        == ("SketchedSubspace", "sketched_active_subspace", "subspace_from_sketches")
    )
    assert "experimental" not in asub.__all__
    for name in experimental.__all__:
        assert getattr(experimental, name) is getattr(gradfree, name)


def test_unpacking_serialization(assert_roundtrip, assert_readonly):
    res = sketched_active_subspace(_ridge, space=asub.Uniform.reference(5), k=2, M=30, rng=1)
    W, V, s = res
    assert W is res.W
    assert V is res.V
    assert s is res.s
    assert W.shape == (5, 5)
    assert V.shape == (30, 5)
    assert s.shape == (5,)
    assert res.n_evaluations == 90
    assert res.design == "gaussian"
    assert res.method == "projection"
    assert res.rank is res.residual is res.iterations is res.stop_reason is None
    assert res.provenance.function == "active_subspaces.experimental.sketched_active_subspace"
    assert_roundtrip(res)
    assert_readonly(res)
    als = sketched_active_subspace(
        _ridge, space=asub.Uniform.reference(5), k=2, M=30, rng=1, method="alternating", rank=1
    )
    assert als.rank == 1
    assert als.stop_reason in ("tolerance", "stalled")
    assert_roundtrip(als)


def test_no_debiasing():
    """V14.18 [D-35, N-18]: s is the singular values of G divided by M."""
    space = asub.Uniform.reference(5)
    res = sketched_active_subspace(_ridge, space=space, k=2, M=40, rng=3)
    g = _random.replay(res)
    child = _random.spawn(g, 2)
    X = space.sample(40, rng=child[0], normalized=True)
    E = child[1].standard_normal((40, 5, 2))
    P = X[:, None, :] + 1e-6 * E.transpose(0, 2, 1)
    D = (_ridge(P.reshape(-1, 5)).reshape(40, 2) - _ridge(X)[:, None]) / 1e-6
    G = gradfree._projection(D, E)
    sig = np.linalg.svd(G, full_matrices=False)[1]
    assert np.array_equal(res.s, sig / 40)


def test_als_stopping():
    """V14.21: max_iter=1 -> 'max_iter' with one convergence.max_iter warning; exact low rank
    -> 'tolerance'.
    """
    space = asub.Uniform.reference(5)
    with pytest.warns(asub.ConvergenceWarning) as rec:
        res = sketched_active_subspace(
            _ridge,
            space=space,
            k=2,
            M=50,
            rng=2,
            method="alternating",
            rank=1,
            max_iter=1,
            tol=1e-12,
        )
    assert len(rec) == 1
    assert rec[0].message.code == "convergence.max_iter"
    assert set(rec[0].message.details) == {"residual", "iterations"}
    assert res.stop_reason == "max_iter"
    assert res.iterations == 1
    gen = np.random.default_rng(0)
    a = gen.standard_normal(5)
    E = gen.standard_normal((60, 5, 2))
    D = np.einsum("imk,m->ik", E, a)
    ex = subspace_from_sketches(
        sketches=D, directions=E, space=space, method="alternating", rank=1
    )
    assert ex.stop_reason == "tolerance"
    assert ex.residual < 1e-3


def test_stall():
    """A residual plateau stops with 'stalled' (no warning) once 5 iterations exist."""
    gen = np.random.default_rng(5)
    E = gen.standard_normal((80, 6, 2))
    D = gen.standard_normal((80, 2))
    res = subspace_from_sketches(
        sketches=D,
        directions=E,
        space=asub.Uniform.reference(6),
        method="alternating",
        rank=1,
        stall_tol=1e-2,
    )
    assert res.stop_reason == "stalled"
    assert res.iterations >= 5


def test_data_level_equivalence():
    """V14.22: end-to-end equals subspace_from_sketches on sketches rebuilt from the documented
    stream layout.
    """
    space = asub.Uniform.reference(5)
    for design, method, rank in (
        ("gaussian", "projection", None),
        ("orthogonal", "alternating", 2),
        ("bernoulli", "projection", None),
    ):
        res = sketched_active_subspace(
            _ridge, space=space, k=3, M=25, rng=11, design=design, method=method, rank=rank, h=1e-5
        )
        gen, _ = _random.generator(11)
        child = gen.spawn(2)
        X = space.sample(25, rng=child[0], normalized=True)
        g = child[1]
        if design == "gaussian":
            E = g.standard_normal((25, 5, 3))
        elif design == "bernoulli":
            E = 2.0 * g.integers(0, 2, size=(25, 5, 3)) - 1.0
        else:
            Z = g.standard_normal((25, 5, 3))
            E = np.empty_like(Z)
            for i in range(25):
                Q, R = np.linalg.qr(Z[i])
                sgn = np.where(np.diagonal(R) < 0, -1.0, 1.0)
                E[i] = Q * sgn
        P = X[:, None, :] + 1e-5 * E.transpose(0, 2, 1)
        assert np.all(np.abs(P) <= 1)
        D = 1.0 * (_ridge(P.reshape(-1, 5)).reshape(25, 3) - _ridge(X)[:, None]) / 1e-5
        ref = subspace_from_sketches(
            sketches=D, directions=E, space=space, method=method, rank=rank
        )
        for name in ("W", "V", "s"):
            np.testing.assert_allclose(getattr(res, name), getattr(ref, name), rtol=0, atol=0)
        assert (res.rank, res.residual, res.iterations, res.stop_reason) == (
            ref.rank,
            ref.residual,
            ref.iterations,
            ref.stop_reason,
        )


def test_determinism_and_workers(seed):
    space = asub.Uniform.reference(5)
    a = sketched_active_subspace(_ridge, space=space, k=2, M=20, rng=seed)
    b = sketched_active_subspace(_ridge, space=space, k=2, M=20, rng=seed)
    assert a == b
    X = space.sample(20, rng=1)
    c = sketched_active_subspace(_ridge, space=space, k=2, X=X, rng=seed)
    with ThreadPoolExecutor(3) as ex:
        d = sketched_active_subspace(
            _ridge, space=space, k=2, X=X, rng=seed, executor=ex, batch_size=7
        )
        e = sketched_active_subspace(
            _ridge, space=space, k=2, M=20, rng=seed, executor=ex, batch_size=4
        )
    assert c == d
    assert a == e
    assert d.provenance.parameters["batch_size"] == 7
    model = asub.Model(space=space, value=_ridge, name="ridge", qoi="q")
    f = sketched_active_subspace(model, k=2, M=20, rng=seed)
    assert f.qoi == "q"
    assert f.provenance.parameters["model"] == "ridge"
    assert np.array_equal(f.W, a.W)


def test_backward_steps_at_the_boundary():
    """A forward point outside the box is replaced by the backward one; linear f gives exact
    directional derivatives.
    """
    space = asub.Uniform.reference(3)
    a = np.array([1.0, -2.0, 0.5])
    X = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.2], [0.0, 0.0, 1.0]])
    res = sketched_active_subspace(
        lambda X: X @ a, space=space, k=3, X=X, design="orthogonal", rng=0, h=1e-3
    )
    # the corner case: a direction with mixed signs leaves the box both ways
    with pytest.raises(asub.InputError) as ei:
        sketched_active_subspace(
            lambda X: X @ a, space=space, k=3, X=np.ones((1, 3)), design="orthogonal", rng=0
        )
    assert ei.value.code == "input.out_of_space"
    assert {"sample", "direction"} <= set(ei.value.details)
    G = res.W @ np.diag(3 * res.s) @ res.V.T
    np.testing.assert_allclose(G, np.tile(a[:, None], (1, 3)), rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    ("kw", "code"),
    [
        ({"X": np.zeros((3, 3)), "M": 3}, "input.parameter"),
        ({"X": None, "M": None}, "input.parameter"),
        ({"k": 0}, "input.parameter"),
        ({"k": 4}, "input.parameter"),
        ({"method": "alternating"}, "input.parameter"),
        ({"rank": 1}, "input.parameter"),
        ({"method": "alternating", "rank": 4}, "input.parameter"),
        ({"design": "sobol"}, "input.parameter"),
        ({"h": 0.0}, "input.parameter"),
        ({"tol": -1.0}, "input.parameter"),
        ({"max_iter": 0}, "input.parameter"),
        ({"M": 0}, "input.parameter"),
        ({"X": np.full((2, 3), 2.0), "M": None}, "input.out_of_space"),
        ({"X": np.zeros((2, 2)), "M": None}, "input.shape"),
        ({"h": 3.0, "design": "orthogonal"}, "input.out_of_space"),
        ({"qoi": ""}, "input.parameter"),
    ],
)
def test_input_errors(kw, code):
    kw = {"k": 2, "M": 5, "rng": 0, **kw}
    with pytest.raises(asub.InputError) as ei:
        sketched_active_subspace(_ridge, space=asub.Uniform.reference(3), **kw)
    assert ei.value.code == code


def test_space_rules():
    space = asub.Uniform.reference(3)
    model = asub.Model(space=space, value=_ridge)
    with pytest.raises(asub.InputError) as ei:
        sketched_active_subspace(model, space=asub.Uniform.reference(4), k=1, M=3)
    assert ei.value.details["reason"] == "space_mismatch"
    with pytest.raises(asub.InputError):
        sketched_active_subspace(_ridge, k=1, M=3)
    with pytest.raises(asub.ModelError) as ei:
        sketched_active_subspace(
            lambda X: np.full(X.shape[0], np.inf), space=space, k=1, M=3, rng=0
        )
    assert ei.value.code == "model.nonfinite"


def test_from_sketches_errors():
    space = asub.Uniform.reference(3)
    D, E = np.zeros((4, 2)), np.zeros((4, 3, 2))
    for kw, code in (
        ({"sketches": np.zeros(4)}, "input.shape"),
        ({"sketches": np.zeros((4, 4)), "directions": np.zeros((4, 3, 4))}, "input.shape"),
        ({"directions": np.zeros((4, 3, 3))}, "input.shape"),
        ({"directions": np.zeros((5, 3, 2))}, "input.size_mismatch"),
        ({"sketches": np.full((4, 2), np.nan)}, "input.nonfinite"),
        ({"directions": np.full((4, 3, 2), np.inf)}, "input.nonfinite"),
        ({"method": "alternating", "rank": 5}, "input.parameter"),
    ):
        with pytest.raises(asub.InputError) as ei:
            subspace_from_sketches(**{"sketches": D, "directions": E, "space": space, **kw})
        assert ei.value.code == code


def test_numerical_errors(monkeypatch):
    def bad_svd(*a, **k):
        raise np.linalg.LinAlgError("boom")

    monkeypatch.setattr(gradfree.np.linalg, "svd", bad_svd)
    with pytest.raises(asub.NumericalError) as ei:
        subspace_from_sketches(
            sketches=np.ones((2, 1)),
            directions=np.ones((2, 2, 1)),
            space=asub.Uniform.reference(2),
        )
    assert ei.value.code == "numerical.singular"
    monkeypatch.undo()
    with pytest.raises(asub.NumericalError) as ei:
        subspace_from_sketches(
            sketches=np.full((2, 1), 1e308),
            directions=np.full((2, 2, 1), 1e-308),
            space=asub.Uniform.reference(2),
        )
    assert ei.value.code == "numerical.nonfinite"


def test_invariants(make_provenance):
    space = asub.Uniform.reference(2)
    good = {
        "W": np.eye(2),
        "V": np.eye(2),
        "s": [2.0, 1.0],
        "method": "projection",
        "design": "user",
        "k": 1,
        "n_samples": 2,
        "rank": None,
        "h": None,
        "n_evaluations": 0,
        "residual": None,
        "iterations": None,
        "stop_reason": None,
        "space": space,
    }
    SketchedSubspace(**good, provenance=make_provenance())
    for change in (
        {"s": [1.0, 2.0]},
        {"s": [1.0, -1.0]},
        {"W": 2 * np.eye(2)},
        {"rank": 1},
        {"h": 1e-6},
        {"design": "gaussian"},
        {"k": 3},
        {"V": np.eye(3)[:, :2]},
        {"method": "alternating"},
    ):
        with pytest.raises((AssertionError, TypeError)):
            SketchedSubspace(**{**good, **change}, provenance=make_provenance())
