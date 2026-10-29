"""Unit tests of the model contract and evaluation (spec 12 §4.1-4.4, §5.1-5.4; V12.12,
V12.15-V12.22).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing
import pickle
import threading

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces.errors import InputError, ModelError, UnsupportedError
from active_subspaces.models import (
    Model,
    ReferenceResults,
    as_model,
    evaluate,
    evaluate_grads,
    evaluate_multi,
    evaluate_with_grads,
    pointwise,
)
from helpers import spec12_models as h


def _ctx():
    return multiprocessing.get_context("spawn")


@pytest.fixture(scope="module")
def process_pool():
    with cf.ProcessPoolExecutor(max_workers=2, mp_context=_ctx()) as pool:
        yield pool


def _row_space(M: int, m: int = 2) -> asub.Uniform:
    return asub.Uniform(np.zeros(m), np.full(m, float(M)))


def _row_points(M: int, m: int = 2) -> np.ndarray:
    X = np.zeros((M, m))
    X[:, 0] = np.arange(M)
    return X


# ---------------------------------------------------------------------------
# Model construction (§5.1) and describe (§5.5)
# ---------------------------------------------------------------------------


def test_model_basics(uniform_space):
    sp = uniform_space(3)
    m = Model(
        space=sp, value=h.strict_value, name="my_model", input_descriptions=["a", "b", "c"], cost=2
    )
    assert m.dim == 3
    assert not m.has_grad
    assert m.input_descriptions == ("a", "b", "c")
    assert m.cost == 2.0
    assert isinstance(m.cost, float)
    assert m != Model(space=sp, value=h.strict_value, name="my_model")  # identity equality
    hash(m)
    assert Model(space=sp, value_and_grad=h.BatchDotBoth(h.BatchDot(3))).has_grad
    d = m.describe()
    json.dumps(d, allow_nan=False)
    assert d["space"] == sp.to_dict()
    assert d["has_grad"] is False
    assert d["has_value_and_grad"] is False
    assert d["reference_results"] is None
    assert d["input_descriptions"] == ["a", "b", "c"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"space": "not a space", "value": h.strict_value},
        {"value": None},
        {"grad": h.strict_grad},
        {"value": 3},
        {"value": h.strict_value, "grad": "x"},
        {"value": h.strict_value, "name": "Bad-Name"},
        {"value": h.strict_value, "name": "a" * 65},
        {"value": h.strict_value, "input_descriptions": ("a",)},
        {"value": h.strict_value, "cost": -1.0},
        {"value": h.strict_value, "cost": float("inf")},
        {"value": h.strict_value, "batch_size": 0},
        {"value": h.strict_value, "batch_size": 2.5},
        {"value": h.strict_value, "qoi": ""},
    ],
)
def test_model_checks(kwargs):
    kwargs = {"space": asub.Uniform.reference(2), **kwargs}
    with pytest.raises(InputError) as e:
        Model(**kwargs)
    assert e.value.code == "input.parameter"


def test_model_reference_results_checks():
    from active_subspaces import test_functions as tf

    rr = tf.BOREHOLE.reference_results
    sp = tf.BOREHOLE.space
    Model(space=sp, value=tf.borehole, name="borehole", reference_results=rr)
    Model(space=sp, value=tf.borehole, reference_results=rr)
    for kwargs in (
        {"name": "other"},
        {"qoi": "g"},
        {"space": asub.Uniform.reference(3)},
    ):
        args = {"space": sp, "value": tf.borehole, "reference_results": rr, **kwargs}
        with pytest.raises(InputError) as e:
            Model(**args)
        assert e.value.code == "input.parameter"


# ---------------------------------------------------------------------------
# Reference results (V12.12)
# ---------------------------------------------------------------------------


def test_reference_results_stored(assert_roundtrip, assert_readonly):
    from active_subspaces import _reference_data
    from active_subspaces import test_functions as tf

    for m in (tf.BOREHOLE, tf.OTL_CIRCUIT, tf.PISTON, tf.ROBOT_ARM, tf.WING_WEIGHT):
        rr = m.reference_results
        assert isinstance(rr, ReferenceResults)
        assert rr.model == m.name
        assert not rr.exact
        assert rr.n_samples == _reference_data.N_SAMPLES
        assert rr.seed == 2016
        assert rr.eigenvalues.shape == (m.dim,)
        assert rr.eigenvectors.shape == (m.dim, m.dim)
        assert rr.to_dict() == _reference_data.REFERENCE_RESULTS[m.name]
        assert_readonly(rr)
    assert_roundtrip(tf.BOREHOLE.reference_results)
    assert_roundtrip(tf.QUADRATIC_GAUSSIAN_10.reference_results)
    assert_roundtrip(tf.RIDGE_SIN_UNIFORM_10.reference_results)


def test_reference_results_invariants(make_provenance):
    W = np.eye(2)
    base = {
        "model": "x",
        "exact": True,
        "eigenvalues": np.array([2.0, 1.0]),
        "eigenvectors": W,
        "eigenvalue_se": None,
        "mean": 0.0,
        "mean_se": None,
        "variance": 1.0,
        "n_samples": None,
        "seed": None,
        "provenance": make_provenance(),
    }
    ReferenceResults(**base)
    bad = [
        {"eigenvalues": np.array([1.0, 2.0])},
        {"eigenvalues": np.array([1.0, -1.0])},
        {"eigenvectors": np.array([[1.0, 1.0], [0.0, 1.0]])},
        {"eigenvectors": np.eye(3)},
        {"exact": False},
        {"n_samples": 10},
        {"variance": -1.0},
        {"exact": False, "eigenvalue_se": np.ones(3), "mean_se": 0.1, "n_samples": 10, "seed": 1},
    ]
    for change in bad:
        with pytest.raises(AssertionError):
            ReferenceResults(**{**base, **change})
    d = ReferenceResults(**base).to_dict()
    d["fields"]["eigenvalues"]["data"] = [1.0, 2.0]
    with pytest.raises(asub.SchemaError) as e:
        asub.from_dict(d)
    assert e.value.code == "schema.invalid"


# ---------------------------------------------------------------------------
# Worker independence (V12.15), one contract (V12.16), no mutation (V12.17)
# ---------------------------------------------------------------------------


def test_worker_independence(process_pool):
    m, M = 5, 23
    space = asub.Uniform(-np.ones(m), np.ones(m))
    inner = h.BatchDot(m)
    models = [
        Model(space=space, value=inner, grad=h.BatchDotGrad(inner)),
        Model(space=space, value=inner, value_and_grad=h.BatchDotBoth(inner)),
    ]
    X = space.sample(M, rng=1)
    for model in models:
        for b in (1, 7, M):
            ref = (evaluate(model, X=X, batch_size=b), evaluate_grads(model, X=X, batch_size=b))
            ref_both = evaluate_with_grads(model, X=X, batch_size=b)
            executors = [cf.ThreadPoolExecutor(n) for n in (1, 2, 7)]
            try:
                for ex in [*executors, process_pool]:
                    f = evaluate(model, X=X, executor=ex, batch_size=b)
                    g = evaluate_grads(model, X=X, executor=ex, batch_size=b)
                    fb, gb = evaluate_with_grads(model, X=X, executor=ex, batch_size=b)
                    assert np.array_equal(f, ref[0])
                    assert np.array_equal(g, ref[1])
                    assert np.array_equal(fb, ref_both[0])
                    assert np.array_equal(gb, ref_both[1])
            finally:
                for ex in executors:
                    ex.shutdown()


def test_one_contract(process_pool):
    space = asub.Uniform.reference(1)
    model = Model(space=space, value=h.strict_value, grad=h.strict_grad)
    X = np.linspace(-1, 1, 9)  # (M,) with m = 1 [D-64]
    with cf.ThreadPoolExecutor(3) as tp:
        for ex in (None, tp, process_pool):
            for b in (1, 4, 9, None):
                f, g = evaluate_with_grads(model, X=X, executor=ex, batch_size=b)
                assert f.shape == (9,)
                assert g.shape == (9, 1)
                assert f.flags.writeable
                assert f.dtype == np.float64


def test_no_mutation():
    space = asub.Uniform.reference(3)
    X = space.sample(10, rng=0)
    X0 = X.copy()
    model = Model(space=space, value=h.mutating_value)
    with cf.ThreadPoolExecutor(2) as tp:
        for ex in (None, tp):
            for b in (1, 3, None):
                f = evaluate(model, X=X, executor=ex, batch_size=b)
                assert np.array_equal(X, X0)
                assert np.all(np.isfinite(f))


def test_outputs_are_new_arrays():
    space = asub.Uniform.reference(2)
    X = space.sample(4, rng=0)
    out = np.arange(4.0)
    f = evaluate(lambda Z: out, X=X, space=space)
    assert f is not out
    assert not np.shares_memory(f, out)
    assert f.flags.writeable
    f = evaluate(lambda Z: np.arange(4).reshape(4, 1), X=X, space=space)
    assert f.dtype == np.float64
    assert f.shape == (4,)


# ---------------------------------------------------------------------------
# Failure policy (V12.18)
# ---------------------------------------------------------------------------


def test_failure_raised_row_37(process_pool):
    M = 50
    space = _row_space(M)
    X = _row_points(M)
    model = Model(space=space, value=h.RowModel(fail_rows=[37]), name="rows")
    with cf.ThreadPoolExecutor(3) as tp:
        for ex in (None, tp, process_pool):
            for b in (None, 1, 10):
                with pytest.raises(ModelError) as e:
                    evaluate(model, X=X, executor=ex, batch_size=b)
                err = e.value
                assert err.code == "model.raised"
                assert isinstance(err.__cause__, ValueError)
                start, stop = err.details["batch"]
                assert start <= 37 < stop
                assert err.details["model"] == "rows"
                assert err.details["callable"] == "value"
                assert err.details["exception"] == "ValueError"


def test_failure_first_batch_in_batch_order(process_pool):
    M, b = 60, 10
    space = _row_space(M)
    X = _row_points(M)
    model = Model(space=space, value=h.RowModel(fail_rows=[25, 55], slow_rows=[25], delay=0.3))
    with cf.ThreadPoolExecutor(7) as tp7, cf.ThreadPoolExecutor(2) as tp2:
        for ex in (None, tp7, tp2, process_pool):
            with pytest.raises(ModelError) as e:
                evaluate(model, X=X, executor=ex, batch_size=b)
            assert e.value.details["batch"] == [20, 30]


class _RecordingExecutor(cf.ThreadPoolExecutor):
    def __init__(self):
        super().__init__(max_workers=1)
        self.futures = []

    def submit(self, fn, /, *args, **kwargs):
        fut = super().submit(fn, *args, **kwargs)
        self.futures.append(fut)
        return fut


def test_failure_cancels_pending_futures():
    M = 40
    space = _row_space(M)
    X = _row_points(M)
    model = Model(
        space=space, value=h.RowModel(fail_rows=[0], slow_rows=list(range(1, M)), delay=0.02)
    )
    ex = _RecordingExecutor()
    try:
        with pytest.raises(ModelError):
            evaluate(model, X=X, executor=ex, batch_size=1)
        assert len(ex.futures) == M
        assert sum(f.cancelled() for f in ex.futures) >= M - 3
    finally:
        ex.shutdown(wait=True)


def _interrupt(X):
    raise KeyboardInterrupt


def test_keyboard_interrupt_propagates():
    space = asub.Uniform.reference(2)
    X = space.sample(4, rng=0)
    with pytest.raises(KeyboardInterrupt):
        evaluate(_interrupt, X=X, space=space)
    with cf.ThreadPoolExecutor(2) as tp, pytest.raises(KeyboardInterrupt):
        evaluate(_interrupt, X=X, space=space, executor=tp, batch_size=1)


def test_output_errors():
    M = 20
    space = _row_space(M)
    X = _row_points(M)
    with pytest.raises(ModelError) as e:
        evaluate(Model(space=space, value=h.RowModel(nan_rows=[13])), X=X, batch_size=5)
    assert e.value.code == "model.nonfinite"
    assert e.value.details["row"] == 13
    assert e.value.details["batch"] == [10, 15]
    cases = [
        (lambda Z: np.ones((Z.shape[0], 2)), "model.shape"),
        (lambda Z: np.ones(Z.shape[0] + 1), "model.shape"),
        (lambda Z: np.array(["a"] * Z.shape[0]), "model.dtype"),
        (lambda Z: np.ones(Z.shape[0], dtype=bool), "model.dtype"),
        (lambda Z: np.ones(Z.shape[0], dtype=complex), "model.dtype"),
        (lambda Z: [[1.0], [1.0, 2.0]], "model.shape"),
    ]
    for fn, code in cases:
        with pytest.raises(ModelError) as e:
            evaluate(fn, X=X, space=space)
        assert e.value.code == code
    both_bad = [
        (lambda Z: Z.sum(1), "model.shape"),
        (lambda Z: (Z.sum(1),), "model.shape"),
        (lambda Z: (Z.sum(1), np.ones((Z.shape[0], 3))), "model.shape"),
        (lambda Z: (Z.sum(1), np.full(Z.shape, np.inf)), "model.nonfinite"),
    ]
    for fn, code in both_bad:
        with pytest.raises(ModelError) as e:
            evaluate_with_grads(Model(space=space, value_and_grad=fn), X=X)
        assert e.value.code == code
        assert e.value.details["callable"] == "value_and_grad"
    # only the requested part is validated (§5.2)
    model = Model(space=space, value_and_grad=lambda Z: (Z.sum(1), np.full(Z.shape, np.nan)))
    assert evaluate(model, X=X).shape == (M,)
    model = Model(space=space, value_and_grad=lambda Z: (np.full(Z.shape[0], np.nan), Z))
    assert evaluate_grads(model, X=X).shape == (M, 2)


def test_killed_worker_gives_model_executor():
    space = asub.Uniform.reference(2)
    X = space.sample(4, rng=0)
    with (
        cf.ProcessPoolExecutor(max_workers=1, mp_context=_ctx()) as pool,
        pytest.raises(ModelError) as e,
    ):
        evaluate(h.die, X=X, space=space, executor=pool)
    assert e.value.code == "model.executor"
    assert isinstance(e.value.__cause__, cf.BrokenExecutor)


def test_unpicklable_model_in_process_pool(process_pool):
    space = asub.Uniform.reference(2)
    X = space.sample(4, rng=0)
    with pytest.raises(ModelError) as e:
        evaluate(lambda Z: Z.sum(1), X=X, space=space, executor=process_pool)
    assert e.value.code == "model.raised"
    assert e.value.__cause__ is not None


def test_shut_down_executor():
    space = asub.Uniform.reference(2)
    X = space.sample(4, rng=0)
    tp = cf.ThreadPoolExecutor(1)
    tp.shutdown()
    with pytest.raises(ModelError) as e:
        evaluate(h.strict_value, X=X, space=space, executor=tp)
    assert e.value.code == "model.raised"
    assert isinstance(e.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Callable selection (V12.19)
# ---------------------------------------------------------------------------


class _Counter:
    def __init__(self, fn, log, label):
        self.fn, self.log, self.label = fn, log, label

    def __call__(self, X):
        self.log.append((self.label, X.shape[0]))
        return self.fn(X)


def _counted(space, *, value=True, grad=True, both=True):
    log: list = []
    kw = {}
    if value:
        kw["value"] = _Counter(lambda X: X.sum(1), log, "value")
    if grad:
        kw["grad"] = _Counter(lambda X: 2 * X, log, "grad")
    if both:
        kw["value_and_grad"] = _Counter(lambda X: (X.sum(1), 2 * X), log, "both")
    return Model(space=space, **kw), log


def test_callable_selection():
    space = asub.Uniform.reference(2)
    X = space.sample(6, rng=0)
    cases = [
        ({}, "value", ["value"]),
        ({}, "grad", ["grad"]),
        ({}, "both", ["both"]),
        ({"value": False}, "value", ["both"]),
        ({"grad": False}, "grad", ["both"]),
        ({"both": False}, "both", ["value", "grad", "value", "grad"]),
    ]
    run = {"value": evaluate, "grad": evaluate_grads, "both": evaluate_with_grads}
    for flags, request, expected in cases:
        model, log = _counted(space, **flags)
        run[request](model, X=X, batch_size=3)
        assert [entry[0] for entry in log] == (expected if len(expected) == 4 else expected * 2)
    model, log = _counted(space, grad=False, both=False)
    for fn in (evaluate_grads, evaluate_with_grads):
        with pytest.raises(UnsupportedError) as e:
            fn(model, X=X)
        assert e.value.code == "unsupported.model_gradient"
    with pytest.raises(UnsupportedError):
        evaluate_grads(model, X=np.full((2, 2), np.nan))  # raised before X is even validated
    assert log == []


# ---------------------------------------------------------------------------
# Bare callables and as_model (V12.20)
# ---------------------------------------------------------------------------


def test_bare_callables_as_model(uniform_space):
    space = uniform_space(3)
    X = space.sample(11, rng=2)
    model = Model(space=space, value=h.strict_value)
    assert np.array_equal(evaluate(h.strict_value, X=X, space=space), evaluate(model, X=X))
    assert np.array_equal(evaluate(model, X=X, space=space), evaluate(model, X=X))
    with pytest.raises(InputError) as e:
        evaluate(h.strict_value, X=X)
    assert e.value.code == "input.parameter"
    for fn in (evaluate, evaluate_grads, evaluate_with_grads):
        with pytest.raises(InputError) as e:
            fn(model, X=X, space=uniform_space(3, seed=5))
        assert e.value.code == "input.parameter"
        assert e.value.details["reason"] == "space_mismatch"
    with pytest.raises(UnsupportedError):
        evaluate_grads(h.strict_value, X=X, space=space)
    assert as_model(model) is model
    assert as_model(model, space=space) is model
    same = asub.Uniform(space.lower.copy(), space.upper.copy())
    assert as_model(model, space=same) is model
    with pytest.raises(InputError) as e:
        as_model(model, space=asub.Gaussian.reference(3))
    assert e.value.details["reason"] == "space_mismatch"
    wrapped = as_model(h.strict_value, space=space)
    assert isinstance(wrapped, Model)
    assert wrapped.value is h.strict_value
    assert wrapped.name is None
    assert wrapped.qoi == "f"
    for bad in (3, None, "model"):
        with pytest.raises(InputError) as e:
            as_model(bad, space=space)
        assert e.value.code == "input.parameter"
    with pytest.raises(InputError):
        as_model(h.strict_value, space="space")


def test_argument_errors():
    space = asub.Uniform.reference(2)
    X = space.sample(5, rng=0)
    model = Model(space=space, value=h.strict_value)
    for kwargs in (
        {"executor": "pool"},
        {"batch_size": 0},
        {"batch_size": 1.5},
        {"batch_size": True},
    ):
        with pytest.raises(InputError) as e:
            evaluate(model, X=X, **kwargs)
        assert e.value.code == "input.parameter"
    for Xbad, code in (
        (np.full((2, 2), 5.0), "input.out_of_space"),
        (np.full((2, 3), 0.0), "input.shape"),
        (np.array([[np.nan, 0.0]]), "input.nonfinite"),
        (np.array([["a", "b"]]), "input.dtype"),
        (np.zeros((0, 2)), "input.shape"),
    ):
        with pytest.raises(InputError) as e:
            evaluate(model, X=Xbad)
        assert e.value.code == code


# ---------------------------------------------------------------------------
# batch_size (V12.21)
# ---------------------------------------------------------------------------


def test_batch_size():
    space = asub.Uniform.reference(2)
    X = space.sample(20, rng=0)
    log: list = []
    model = Model(space=space, value=_Counter(lambda Z: Z.sum(1), log, "value"), batch_size=7)
    evaluate(model, X=X)
    assert [k for _, k in log] == [7, 7, 6]
    log.clear()
    evaluate(model, X=X, batch_size=3)
    assert [k for _, k in log] == [3] * 6 + [2]
    with pytest.raises(InputError) as e:
        evaluate(model, X=X, batch_size=8)
    assert e.value.code == "input.parameter"
    log.clear()
    evaluate(Model(space=space, value=_Counter(lambda Z: Z.sum(1), log, "value")), X=X)
    assert [k for _, k in log] == [20]


# ---------------------------------------------------------------------------
# pointwise and pickling (V12.22)
# ---------------------------------------------------------------------------


def test_pointwise_and_pickling(process_pool):
    space = asub.Uniform.reference(3)
    X = space.sample(5, rng=0)
    pv = pointwise(h.point_value)
    pg = pointwise(h.point_grad, output="grad")
    pb = pointwise(h.point_both, output="value_and_grad")
    assert pv(X).shape == (5,)
    assert pg(X).shape == (5, 3)
    fb, gb = pb(X)
    assert fb.shape == (5,)
    assert gb.shape == (5, 3)
    model = Model(space=space, value=pv, grad=pg)
    model_b = Model(space=space, value_and_grad=pb)
    f, g = evaluate_with_grads(model, X=X)
    assert np.allclose(f, np.sum(X**2, axis=1))
    assert np.allclose(g, 2 * X)
    for wrapped in (pv, pg, pb):
        back = pickle.loads(pickle.dumps(wrapped))
        assert back.output == wrapped.output
    for mdl in (model, model_b):
        ref = evaluate_with_grads(mdl, X=X, batch_size=2)
        out = evaluate_with_grads(mdl, X=X, batch_size=2, executor=process_pool)
        assert np.array_equal(ref[0], out[0])
        assert np.array_equal(ref[1], out[1])
    # a non-scalar return from a "value" function is reported as model.raised
    with pytest.raises(ModelError) as e:
        evaluate(pointwise(h.point_grad), X=X, space=space)
    assert e.value.code == "model.raised"
    with pytest.raises(InputError):
        pointwise(h.point_value, output="hessian")
    with pytest.raises(InputError):
        pointwise(3)


def test_pointwise_passes_fresh_rows():
    seen = []

    def fun(x):
        assert x.shape == (2,)
        assert x.flags.writeable
        seen.append(x)
        x[:] = 0.0
        return 1.0

    X = np.ones((3, 2))
    pointwise(fun)(X)
    assert np.array_equal(X, np.ones((3, 2)))
    assert len({id(s) for s in seen}) == 3


def test_builtin_models_pickle_and_evaluate_in_processes(process_pool):
    from active_subspaces import test_functions as tf

    for name in tf.builtin_models():
        model = tf.builtin_models()[name]
        back = pickle.loads(pickle.dumps(model))
        X = model.space.sample(12, rng=3)
        ref = evaluate_with_grads(model, X=X, batch_size=5)
        for mdl, ex in ((back, None), (model, process_pool)):
            out = evaluate_with_grads(mdl, X=X, batch_size=5, executor=ex)
            assert np.array_equal(ref[0], out[0])
            assert np.array_equal(ref[1], out[1])


def test_thread_safety_of_concurrent_calls():
    space = asub.Uniform.reference(2)
    X = space.sample(100, rng=0)
    lock = threading.Lock()
    calls = []

    def fn(Z):
        with lock:
            calls.append(Z.shape[0])
        return Z.sum(1)

    with cf.ThreadPoolExecutor(4) as tp:
        f = evaluate(fn, X=X, space=space, executor=tp, batch_size=9)
    assert sorted(calls) == sorted([9] * 11 + [1])
    assert np.array_equal(f, X.sum(1))


def test_evaluate_multi():
    """D-199: (k, P) values return as (M, P); other shapes raise model.shape; grads ignored."""
    space = _row_space(23)
    X = space.sample(23, rng=1)
    P = 3
    calls: list[int] = []

    def value(Z):
        calls.append(Z.shape[0])
        return np.stack([Z[:, 0] * (p + 1) for p in range(P)], axis=1)

    def boom(Z):
        raise AssertionError("gradient callable must not be called")

    ref = np.stack([X[:, 0] * (p + 1) for p in range(P)], axis=1)
    out = evaluate_multi(value, X=X, outputs=P, space=space)
    assert out.shape == (23, P)
    assert out.dtype == np.float64
    assert out.flags.writeable
    assert out.tobytes() == ref.tobytes()
    assert calls == [23]
    for b in (1, 5, 23):
        with cf.ThreadPoolExecutor(3) as ex:
            got = evaluate_multi(value, X=X, outputs=P, space=space, executor=ex, batch_size=b)
        assert got.tobytes() == ref.tobytes()
    both = Model(space=space, value=value, grad=boom)
    assert evaluate_multi(both, X=X, outputs=P).tobytes() == ref.tobytes()
    vg = Model(space=space, value_and_grad=lambda Z: (value(Z), np.full(Z.shape, np.nan)))
    assert evaluate_multi(vg, X=X, outputs=P).tobytes() == ref.tobytes()
    for fn in (
        lambda Z: Z[:, 0],
        lambda Z: np.ones((Z.shape[0], 1)),
        lambda Z: np.ones((Z.shape[0], P + 1)),
        lambda Z: np.ones((Z.shape[0], P, 1)),
        lambda Z: np.ones((Z.shape[0] + 1, P)),
    ):
        with pytest.raises(ModelError) as e:
            evaluate_multi(fn, X=X, outputs=P, space=space)
        assert e.value.code == "model.shape"
    with pytest.raises(ModelError) as e:
        evaluate_multi(lambda Z: np.full((Z.shape[0], P), np.nan), X=X, outputs=P, space=space)
    assert e.value.code == "model.nonfinite"
    with pytest.raises(ModelError) as e:
        evaluate_multi(lambda Z: np.array([["a"] * P] * Z.shape[0]), X=X, outputs=P, space=space)
    assert e.value.code == "model.dtype"
    with pytest.raises(ModelError) as e:  # a (k, 1) value is not flattened
        evaluate(lambda Z: np.ones((Z.shape[0], P)), X=X, space=space)
    assert e.value.code == "model.shape"
    for bad in (0, -2, 1.5, True, "3", None):
        with pytest.raises(InputError) as e:
            evaluate_multi(value, X=X, outputs=bad, space=space)
        assert e.value.code == "input.parameter"
