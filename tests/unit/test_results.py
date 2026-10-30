"""Tests for spec 04: result base, immutability, equality, serialization, provenance."""

from __future__ import annotations

import dataclasses
import io
import json
import pickle
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

import active_subspaces as asub
from active_subspaces import _random, _validation, results
from active_subspaces.results import QoIResult, Result
from active_subspaces.results import _invariant as invariant
from active_subspaces.results import _make_provenance as make_provenance
from active_subspaces.results import _register_result as register_result
from active_subspaces.spaces import InputSpace

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "data" / "schemas"  # tests/data/schemas (04 §8)


@register_result
class _TInner(Result):
    TYPE = "_TestInner"
    SCHEMA_VERSION = 1
    value: float


@register_result
class _TAll(QoIResult):
    TYPE = "_TestAllKinds"
    SCHEMA_VERSION = 1
    W: NDArray[np.float64]
    idx: NDArray[np.int64]
    mask: NDArray[np.bool_]
    n: int
    tol: float
    flag: bool
    label: str
    method: Literal["a", "b"]
    maybe: NDArray[np.float64] | None
    floats: tuple[float, ...]
    arrays: tuple[NDArray[np.float64], ...]
    pair: tuple[int, str]
    space: InputSpace
    inner: _TInner
    note: str | None = None

    @property
    def W1(self):
        return self.W[:, : self.n]

    def _check(self):
        invariant(self.W.ndim == 2 and self.W.shape[0] == self.space.dim, "W must be (m, k)")
        invariant(1 <= self.n <= self.W.shape[1], "bad n")


@register_result
class _TVersioned(Result):
    TYPE = "_TestVersioned"
    SCHEMA_VERSION = 3
    scale: float


@results._register_migration("_TestVersioned", 1)
def _mig1(fields):
    return {"factor": fields["old_factor"]}


@results._register_migration("_TestVersioned", 2)
def _mig2(fields):
    return {"scale": fields["factor"] * 2}


def _prov(**kw):
    return make_provenance("test_function", kw.pop("parameters", {"n": 1}), **kw)


def _make(**over):
    space = asub.Uniform([0.0, 1.0, 2.0], [1.0, 3.0, 7.0])
    kw = {
        "W": np.linspace(-1, 1, 6).reshape(3, 2) + np.array([[np.pi, 0], [0, 0], [0, 0]]),
        "idx": np.array([3, -2, 7]),
        "mask": np.array([True, False]),
        "n": 1,
        "tol": float("inf"),
        "flag": True,
        "label": "NaN",  # a string that looks like a non-finite float
        "method": "b",
        "maybe": None,
        "floats": (1.0, float("nan"), -float("inf"), 1e-300),
        "arrays": (np.ones(2), np.array([[np.nan, 2.0]])),
        "pair": (4, "x"),
        "space": space,
        "inner": _TInner(value=0.1, provenance=_prov()),
        "qoi": "f",
        "provenance": _prov(space=space),
    }
    kw.update(over)
    return _TAll(**kw)


# ---------------------------------------------------------------------------
# Construction, immutability
# ---------------------------------------------------------------------------


def test_fields_coerced():
    r = _make(
        n=np.int64(2),
        tol=np.float32(0.5),
        flag=np.bool_(False),
        floats=[1, 2],
        W=[[1, 2], [3, 4], [5, 6]],
    )
    assert type(r.n) is int
    assert type(r.tol) is float
    assert type(r.flag) is bool
    assert r.floats == (1.0, 2.0)
    assert all(type(x) is float for x in r.floats)
    assert r.W.dtype == np.float64
    assert r.W.flags.c_contiguous
    assert r.idx.dtype == np.int64
    assert r.mask.dtype == np.bool_
    assert r.qoi == "f"


def test_immutability(assert_readonly):
    W = np.arange(6.0).reshape(3, 2)
    r = _make(W=W)
    assert_readonly(r)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.n = 2
    with pytest.raises(ValueError, match="read-only"):
        r.W[0, 0] = 1.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        r.W.flags.writeable = True
    with pytest.raises(ValueError, match="read-only"):
        r.arrays[0][0] = 5.0
    W[0, 0] = 99.0
    assert r.W[0, 0] == 0.0
    assert not r.W1.flags.writeable
    assert r.W1.shape == (3, 1)


def test_invariant_failure_in_constructor_is_assertion():
    with pytest.raises(AssertionError):
        _make(n=5)
    with pytest.raises(AssertionError):
        _make(qoi="")


def test_bad_types_raise_type_error():
    with pytest.raises(TypeError):
        _make(n=1.5)
    with pytest.raises(TypeError):
        _make(method="c")
    with pytest.raises(TypeError):
        _make(space="space")
    with pytest.raises(TypeError):
        _make(provenance={"x": 1})


def test_abstract_bases_not_instantiable():
    with pytest.raises(TypeError):
        Result(provenance=_prov())


def test_registry():
    types = results._registered_types()
    assert types["_TestAllKinds"] is _TAll
    assert types["_TestInner"] is _TInner
    assert len(set(types)) == len(types)
    with pytest.raises(TypeError):

        @register_result
        class _Dup(Result):
            TYPE = "_TestInner"
            SCHEMA_VERSION = 1
            value: float

    with pytest.raises(TypeError):

        @register_result
        class _NoType(Result):
            value: float

    assert results._registered_types()["_TestInner"] is _TInner


def test_undecorated_subclass_rejected():
    @dataclasses.dataclass(frozen=True)
    class _Plain(Result):
        TYPE = "_TestPlainNotRegisteredProperly"
        SCHEMA_VERSION = 1
        value: float

    with pytest.raises(TypeError, match="_register_result"):
        _Plain(value=1.0, provenance=_prov())
    results._REGISTRY.pop("_TestPlainNotRegisteredProperly", None)


# ---------------------------------------------------------------------------
# Equality, hash, repr
# ---------------------------------------------------------------------------


def test_equality_excludes_provenance():
    a = _make()
    b = _make(provenance=_prov(parameters={"other": 2}))
    assert a.provenance != b.provenance
    assert a == b


def test_single_ulp_change_unequal():
    a = _make()
    W = np.array(a.W)
    W[1, 1] = np.nextafter(W[1, 1], np.inf)
    assert _make(W=W) != a
    assert _make(floats=(1.0, float("nan"), -float("inf"), np.nextafter(1e-300, 1))) != a


def test_nan_equals_nan_and_type_sensitivity():
    assert _make() == _make()
    assert _make(maybe=np.array([np.nan])) == _make(maybe=np.array([np.nan]))
    assert _make(maybe=np.array([np.nan])) != _make()
    assert _make(note="x") != _make()
    assert _make(idx=np.array([1, 2])) != _make(idx=np.array([[1, 2]]))
    assert _make() != _TInner(value=0.1, provenance=_prov())
    assert _make(inner=_TInner(value=0.2, provenance=_prov())) != _make()


def test_not_hashable():
    with pytest.raises(TypeError):
        hash(_make())
    with pytest.raises(TypeError):
        hash(_TInner(value=1.0, provenance=_prov()))


def test_repr_is_summary():
    big = np.arange(30000.0).reshape(3, 10000)
    r = _make(W=big)
    text = repr(r)
    assert len(text) < 1500
    assert "float64" in text
    assert "(3, 10000)" in text
    assert "…" in text
    assert "provenance" not in text
    assert "<_TestInner>" in text


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_round_trips(assert_roundtrip):
    assert_roundtrip(_make())
    assert_roundtrip(_make(maybe=np.array([1.0, np.inf]), note="hello", qoi="g"))
    assert_roundtrip(_TInner(value=float("nan"), provenance=_prov()))


def test_to_dict_layout_and_strict_json():
    r = _make()
    d = r.to_dict()
    assert set(d) == {"type", "schema_version", "library_version", "fields", "provenance"}
    assert d["type"] == "_TestAllKinds"
    assert d["schema_version"] == 1
    assert d["library_version"] == asub.__version__
    json.dumps(d, allow_nan=False)
    W = d["fields"]["W"]
    assert W == {
        "__ndarray__": 1,
        "dtype": "float64",
        "shape": [3, 2],
        "data": r.W.ravel().tolist(),
    }
    assert d["fields"]["tol"] == "Infinity"
    assert d["fields"]["label"] == "NaN"
    assert d["fields"]["floats"][1:3] == ["NaN", "-Infinity"]
    back = asub.from_dict(json.loads(json.dumps(d)))
    assert back.label == "NaN"
    assert back.tol == float("inf")
    db = r.to_dict(binary=True)
    assert "b64" in db["fields"]["W"]
    assert "data" not in db["fields"]["W"]
    json.dumps(db, allow_nan=False)


def test_floats_bitwise_exact():
    W = np.random.default_rng(0).standard_normal((3, 2)) * 10.0 ** np.random.default_rng(
        1
    ).integers(-300, 300, (3, 2))
    r = _make(W=W)
    for binary in (False, True):
        back = asub.from_dict(json.loads(json.dumps(r.to_dict(binary=binary))))
        assert back.W.tobytes() == r.W.tobytes()


def test_serialization_deterministic():
    r = _make()
    a = json.dumps(r.to_dict()["fields"])
    b = json.dumps(asub.from_dict(r.to_dict()).to_dict()["fields"])
    assert a == b


def test_decoded_arrays_read_only():
    r = asub.from_dict(_make().to_dict(binary=True))
    for arr in (r.W, r.idx, r.mask, r.arrays[1]):
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            arr.flags.writeable = True
    buf = io.BytesIO()
    _make().save(buf)
    buf.seek(0)
    loaded = asub.load(buf)
    with pytest.raises(ValueError, match="WRITEABLE"):
        loaded.W.flags.writeable = True


def test_frozen_array_base_cannot_be_made_writeable():
    r = _make()
    for arr in (r.W, r.idx, r.mask, r.W1):
        a = arr
        while a is not None and isinstance(a, np.ndarray):
            with pytest.raises(ValueError, match="WRITEABLE"):
                a.flags.writeable = True
            a = a.base
    from active_subspaces._validation import freeze_array

    for x in (np.zeros((0, 3)), 2.5, np.arange(6.0).reshape(2, 3).T):
        f = freeze_array(x)
        assert f.flags.c_contiguous
        assert not f.flags.writeable
        np.testing.assert_array_equal(f, x)


def test_npz_wrong_dtype_array_rejected():
    buf = io.BytesIO()
    _make().save(buf)
    buf.seek(0)
    with np.load(buf) as z:
        payload = {k: z[k] for k in z.files}
    meta = json.loads(str(payload["__meta__"][()]))
    key = meta["fields"]["idx"]["__npz__"]
    payload[key] = payload[key].astype(np.float64) + 0.5
    bad = io.BytesIO()
    np.savez(bad, **payload)
    bad.seek(0)
    with pytest.raises(asub.SchemaError) as e:
        asub.load(bad)
    assert e.value.code == "schema.invalid"


def test_pickle_preserves_read_only():
    r = pickle.loads(pickle.dumps(_make()))
    assert r == _make()
    with pytest.raises(ValueError, match="WRITEABLE"):
        r.W.flags.writeable = True


def test_subclass_from_dict_checks_type():
    d = _make().to_dict()
    assert _TAll.from_dict(d) == _make()
    with pytest.raises(asub.SchemaError) as e:
        _TInner.from_dict(d)
    assert e.value.code == "schema.invalid"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda d: d.update(type="NoSuchType"), "schema.unknown_type"),
        (lambda d: d.update(schema_version=2), "schema.too_new"),
        (lambda d: d.update(schema_version=0), "schema.invalid"),
        (lambda d: d.update(schema_version=True), "schema.invalid"),
        (lambda d: d.pop("library_version"), "schema.invalid"),
        (lambda d: d.update(extra=1), "schema.invalid"),
        (lambda d: d["fields"].update(unknown=1), "schema.invalid"),
        (lambda d: d["fields"].pop("note"), "schema.invalid"),
        (lambda d: d["fields"].update(n=5), "schema.invalid"),  # invariant
        (lambda d: d["fields"].update(n="1"), "schema.invalid"),
        (lambda d: d["fields"].update(n=True), "schema.invalid"),
        (lambda d: d["fields"].update(method="c"), "schema.invalid"),
        (lambda d: d["fields"].update(floats=["nan"]), "schema.invalid"),
        (lambda d: d["fields"]["W"].update(dtype="float32"), "schema.invalid"),
        (lambda d: d["fields"]["W"].update(shape=[2, 2]), "schema.invalid"),
        (lambda d: d["fields"]["W"].update(data=[1.0] * 5 + [True]), "schema.invalid"),
        (lambda d: d["fields"]["idx"].update(data=[1.5, 2, 3]), "schema.invalid"),
        # a well-formed array of the wrong dtype is rejected, not cast
        (
            lambda d: d["fields"]["idx"].update(dtype="float64", data=[1.7, 2.9, 3.0]),
            "schema.invalid",
        ),
        (lambda d: d["fields"]["W"].update(dtype="bool", data=[True] * 6), "schema.invalid"),
        (lambda d: d["fields"]["mask"].update(dtype="int64", data=[1, 0]), "schema.invalid"),
        (lambda d: d["fields"]["arrays"][0].update(dtype="int64", data=[1, 1]), "schema.invalid"),
        (lambda d: d["fields"]["space"].update(type="Mixed"), "schema.invalid"),
        (lambda d: d["fields"]["inner"].update(type="_TestAllKinds"), "schema.invalid"),
        (lambda d: d["provenance"].pop("created"), "schema.invalid"),
        (lambda d: d.update(fields=[1]), "schema.invalid"),
    ],
)
def test_from_dict_errors(mutate, code):
    d = json.loads(json.dumps(_make().to_dict()))
    mutate(d)
    with pytest.raises(asub.SchemaError) as e:
        asub.from_dict(d)
    assert e.value.code == code
    assert isinstance(e.value, ValueError)


def test_from_dict_rejects_non_mapping():
    with pytest.raises(asub.SchemaError):
        asub.from_dict([1, 2])
    with pytest.raises(asub.SchemaError):
        asub.from_dict({"type": 3})


def test_bad_base64():
    d = _make().to_dict(binary=True)
    d["fields"]["W"]["b64"] = "!!!notbase64"
    with pytest.raises(asub.SchemaError):
        asub.from_dict(d)
    d["fields"]["W"]["b64"] = "AAAA"
    with pytest.raises(asub.SchemaError):
        asub.from_dict(d)


def test_migrations():
    old = {
        "type": "_TestVersioned",
        "schema_version": 1,
        "library_version": "1.0.0",
        "fields": {"old_factor": 1.5},
        "provenance": _prov().to_dict(),
    }
    r = asub.from_dict(old)
    assert r.scale == 3.0
    assert type(r) is _TVersioned
    current = r.to_dict()
    assert current["schema_version"] == 3
    assert current["fields"] == {"scale": 3.0}


def test_schema_fixtures_load_forever():
    """Every committed fixture tests/data/schemas/<TYPE>/v<k>.json still loads (component types
    add them).
    """
    if not SCHEMA_DIR.is_dir():
        pytest.skip("no schema fixtures yet")
    for path in sorted(SCHEMA_DIR.glob("*/v*.json")):
        d = json.loads(path.read_text())
        assert d["type"] == path.parent.name
        assert re.fullmatch(r"v\d+\.json", path.name)
        r = asub.from_dict(d)
        assert type(r).TYPE == path.parent.name


# ---------------------------------------------------------------------------
# .npz files
# ---------------------------------------------------------------------------


def test_save_writes_exactly_the_given_path(tmp_path):
    p = tmp_path / "result.bin"
    _make().save(p)
    assert sorted(x.name for x in tmp_path.iterdir()) == ["result.bin"]
    assert asub.load(p) == _make()
    assert asub.load(str(p)) == _make()


def test_npz_contents(tmp_path):
    p = tmp_path / "r.npz"
    _make().save(p)
    with np.load(p, allow_pickle=False) as z:
        assert "__meta__" in z.files
        meta = json.loads(str(z["__meta__"][()]))
        assert meta["fields"]["W"] == {"__npz__": "a0"}
        assert all(k == "__meta__" or re.fullmatch(r"a\d+", k) for k in z.files)
        np.testing.assert_array_equal(z["a0"], _make().W)


_SIDE_EFFECTS: list = []


def _side_effect():
    _SIDE_EFFECTS.append("unpickled!")
    return "payload"


class _Evil:
    def __reduce__(self):
        return (_side_effect, ())


def test_object_array_rejected_without_unpickling(tmp_path):
    p = tmp_path / "evil.npz"
    good = _make().to_dict()
    arr = np.empty(1, dtype=object)
    arr[0] = _Evil()
    np.savez(p, __meta__=np.array(json.dumps(good)), a0=arr)
    with pytest.raises(asub.SchemaError) as e:
        asub.load(p)
    assert e.value.code == "schema.invalid"
    assert _SIDE_EFFECTS == []


def test_pickle_file_rejected(tmp_path):
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps(_Evil()))
    with pytest.raises(asub.SchemaError):
        asub.load(p)
    assert _SIDE_EFFECTS == []


@pytest.mark.parametrize(
    "payload",
    [
        {"a0": np.ones(2)},  # no __meta__
        {"__meta__": np.array([1.0])},
        {"__meta__": np.array("not json")},
    ],
)
def test_malformed_npz(tmp_path, payload):
    p = tmp_path / "bad.npz"
    np.savez(p, **payload)
    with pytest.raises(asub.SchemaError):
        asub.load(p)


def test_npz_and_garbage_errors(tmp_path):
    p = tmp_path / "x.npy"
    np.save(p, np.ones(3))
    with pytest.raises(asub.SchemaError):
        asub.load(p)
    g = tmp_path / "garbage.npz"
    g.write_bytes(b"PK\x03\x04garbage")
    with pytest.raises(asub.SchemaError):
        asub.load(g)
    with pytest.raises(FileNotFoundError):
        asub.load(tmp_path / "missing.npz")
    with pytest.raises(FileNotFoundError):
        _make().save(tmp_path / "no_such_dir" / "r.npz")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_make_provenance_contents():
    import platform

    import scipy

    space = asub.Gaussian([0.0, 1.0], [1.0, 2.0])
    gen, desc = _random.generator(123)
    X = np.ones((4, 2))
    p = make_provenance(
        "active_subspace",
        {
            "n": np.int64(2),
            "tol": float("inf"),
            "X": X,
            "fn": np.sin,
            "method": "ols",
            "nested": {"a": (1, 2)},
        },
        space=space,
        rng_state=desc,
        digests={"grads": {"shape": [4, 2], "dtype": "float64", "sha256": "0" * 64}},
    )
    assert p.function == "active_subspaces.active_subspace"
    assert p.library_version == asub.__version__
    assert p.numpy_version == np.__version__
    assert p.scipy_version == scipy.__version__
    assert p.python_version == platform.python_version()
    assert p.platform == platform.platform()
    assert p.created.endswith("+00:00")
    params = p.parameters
    assert params["n"] == 2
    assert params["tol"] == "Infinity"
    assert params["method"] == "ols"
    assert params["X"]["sha256"]
    assert params["X"]["shape"] == [
        4,
        2,
    ]
    assert "sin" in params["fn"]["callable"]
    assert params["grads"]["sha256"] == "0" * 64
    assert space_from(p.input_space) == space
    assert p.rng["kind"] == "seed_sequence"
    assert p.rng["entropy"] == "123"
    with pytest.raises(TypeError):
        params["n"] = 3
    with pytest.raises(TypeError):
        params["nested"]["a"] = 3
    with pytest.raises(TypeError):
        params["nested"]["a"].append(3)
    with pytest.raises(TypeError):
        p.input_space["mean"]["data"][0] = 9.0
    assert p.input_space == space.to_dict()  # spec 04 §4.3: the attribute is space.to_dict()
    assert isinstance(p.input_space["mean"]["data"], list)
    json.dumps(p.to_dict(), allow_nan=False)
    assert results.Provenance.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    assert results.Provenance.from_dict(p.to_dict(binary=True)) == p
    assert pickle.loads(pickle.dumps(p)) == p
    assert pickle.loads(pickle.dumps(p)).input_space == space.to_dict()
    assert results.Provenance.from_dict(p.to_dict(binary=True)).input_space == space.to_dict()
    with pytest.raises(TypeError):
        make_provenance("f", rng_state=gen)


def space_from(d):
    from active_subspaces.spaces import _space_from_dict

    return _space_from_dict(d)


def test_provenance_survives_round_trips(tmp_path):
    r = _make()
    assert asub.from_dict(r.to_dict()).provenance == r.provenance
    p = tmp_path / "r.npz"
    r.save(p)
    assert asub.load(p).provenance == r.provenance


def test_provenance_from_dict_errors():
    d = _prov().to_dict()
    for bad in (
        {**d, "extra": 1},
        {**d, "function": 3},
        {**d, "parameters": [1]},
        {**d, "rng": "x"},
        "nope",
    ):
        with pytest.raises(asub.SchemaError):
            results.Provenance.from_dict(bad)


# ---------------------------------------------------------------------------
# Experimental lazy type tags [D-141]
# ---------------------------------------------------------------------------

_LAZY_SRC = """
import numpy as np
from numpy.typing import NDArray
from active_subspaces.results import Result, _register_result

IMPORTS = []
IMPORTS.append(1)

@_register_result
class LazyThing(Result):
    TYPE = "_TestLazyThing"
    SCHEMA_VERSION = 1
    values: NDArray[np.float64]
"""


def test_experimental_tag_table():
    assert dict(results._EXPERIMENTAL_TYPES) == {
        "SketchedSubspace": "active_subspaces.experimental.gradfree"
    }
    with pytest.raises(TypeError):
        results._EXPERIMENTAL_TYPES["x"] = "y"  # type: ignore[index]


def test_lazy_tag_imports_module_on_demand(tmp_path, monkeypatch):
    import importlib
    import sys

    modname = "_as_lazy_fake_module"
    (tmp_path / f"{modname}.py").write_text(_LAZY_SRC)
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module(modname)
    r = mod.LazyThing(values=np.arange(3.0), provenance=_prov())
    d = json.loads(json.dumps(r.to_dict()))
    buf = io.BytesIO()
    r.save(buf)
    try:
        # unregister and forget the module, as in a fresh process
        results._REGISTRY.pop("_TestLazyThing")
        del sys.modules[modname]
        with pytest.raises(asub.SchemaError) as e:
            asub.from_dict(d)
        assert e.value.code == "schema.unknown_type"
        monkeypatch.setattr(results, "_EXPERIMENTAL_TYPES", {"_TestLazyThing": modname})
        back = asub.from_dict(d)
        assert modname in sys.modules
        assert type(back).__module__ == modname
        np.testing.assert_array_equal(back.values, r.values)
        results._REGISTRY.pop("_TestLazyThing")
        del sys.modules[modname]
        buf.seek(0)
        assert type(asub.load(buf)).TYPE == "_TestLazyThing"
        assert modname in sys.modules
    finally:
        results._REGISTRY.pop("_TestLazyThing", None)
        sys.modules.pop(modname, None)


def test_results_exports():
    assert results.__all__ == ("Provenance", "QoIResult", "Result", "from_dict", "load")
    for name in (
        "invariant",
        "register_result",
        "register_migration",
        "registered_types",
        "make_provenance",
        "freeze_array",
        "encode_array",
        "decode_array",
        "summarize",
        "InvariantError",
    ):
        assert not hasattr(results, name), name


def test_migration_subspace_v1():
    """A version-1 SubspaceResult, written by the development build, loads as version 2 with
    bootstrap_eigenvectors None and every other field unchanged [D-196] (04 §8, V04.8).
    """
    path = SCHEMA_DIR / "SubspaceResult" / "v1.json"
    d = json.loads(path.read_text())
    assert d["schema_version"] == 1
    assert "bootstrap_eigenvectors" not in d["fields"]
    r = asub.from_dict(d)
    assert r.bootstrap_eigenvectors is None
    assert r.nboot == 5
    assert r.estimator == "ols"
    np.testing.assert_array_equal(
        r.eigenvectors, _validation.decode_array(d["fields"]["eigenvectors"])
    )
    np.testing.assert_array_equal(
        r.bootstrap_distances, _validation.decode_array(d["fields"]["bootstrap_distances"])
    )
    again = r.to_dict()
    assert again["schema_version"] == 2
    assert asub.from_dict(again) == r


def test_migration_rbf_v1():
    """A version-1 RBFSurrogate loads with training_variance = the stored f_variance [D-206]."""
    d = json.loads((SCHEMA_DIR / "RBFSurrogate" / "v1.json").read_text())
    assert d["schema_version"] == 1
    old = d["fields"]["f_variance"]
    r = asub.from_dict(d)
    assert r.training_variance == old
    assert not hasattr(r, "f_variance")
    again = r.to_dict()
    assert again["schema_version"] == 2
    assert "f_variance" not in again["fields"]
    assert asub.from_dict(again) == r


def test_product_space_roundtrip():
    """A SubspaceResult on a Product round-trips through to_dict, JSON and .npz [D-198]."""
    sp = asub.Product([asub.Uniform([0.0, 1.0], [2.0, 5.0]), asub.Gaussian([0.5], [0.25])])
    G = np.random.default_rng(0).standard_normal((30, 3))
    r = asub.active_subspace(grads=G, space=sp, n=1, nboot=3, rng=0, bootstrap_vectors=1)
    assert asub.from_dict(json.loads(json.dumps(r.to_dict(), allow_nan=False))) == r
    buf = io.BytesIO()
    r.save(buf)
    buf.seek(0)
    back = asub.load(buf)
    assert back == r
    assert back.space.blocks == r.space.blocks
